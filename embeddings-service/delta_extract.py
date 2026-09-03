"""Deterministic structural-edge extractor for delta batches (task D1, 2026-09-02).

Emits ONLY edge types already present in the namespace, and only in the direction and
between the file kinds the existing graph already uses. The conventions below were read
off the graph, not chosen:

    IMPORTS     java -> java   (2667 edges)   resolved by fully-qualified package path
                ts   -> api    (150 edges)    the generated client is ONE collapsed node,
                                              so TS import edges all land on it; there are
                                              no ts -> ts IMPORTS anywhere in the graph
    INJECTS     java -> java   (638)          `private final Type x;` fields + @Autowired
                ts   -> ts     (116)          relative imports that resolve to an indexed
                                              .ts file (this is what the original pass
                                              labelled INJECTS, e.g. error.interceptor.ts
                                              -> auth-api.service.ts)
    EXTENDS     java -> java   (315)
    IMPLEMENTS  java -> java   (43)
    TESTED_BY   subject -> test                deterministic name pairing only
                                              (Foo.java/FooUnitTest.java, x.ts/x.spec.ts)

Semantic edges the original pass produced by judgement -- CONSTRAINS, USES, PERFORMS,
MODIFIES, CALLS, ACCESSES, VALIDATES, AFFECTS, TRIGGERS -- are NOT emitted here: they are
not mechanically derivable and guessing them would corrupt the meta-path priors.

The lexical regexes are imported from the frozen socket builder rather than re-written,
so an edge and the T-socket prose that describes it can never disagree.

Usage:
    python delta_extract.py --batch 2026-09-02              # extract + write
    python delta_extract.py --batch 2026-09-02 --dry        # report only
"""

import argparse
import os
import re
import sys

from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_sockets import (  # noqa: E402  -- the frozen spec is the source of truth
    JAVA_EXTENDS,
    JAVA_IMPLEMENTS,
    JAVA_INJECT,
    read_source,
)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7611")
NEO4J_AUTH = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASS", "password"))
NAMESPACE = os.environ.get("EMBED_NAMESPACE", "CheckItOutV3")

JAVA_IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;", re.M)
TS_IMPORT = re.compile(r"""(?:from|import)\s*\(?\s*['"]([^'"]+)['"]""")
INTERNAL_JAVA_ROOT = "com.sm."
IDENT = re.compile(r"[A-Za-z_]\w*")

TS_EXT = (".ts", ".tsx", ".mts", ".js")


# ---------------------------------------------------------------------------------
# target index, built from the graph itself
# ---------------------------------------------------------------------------------

def build_index(session, ns):
    rows = session.run(
        "MATCH (n:EntityDetail {namespace:$ns}) "
        "WHERE n.file_path IS NOT NULL "
        "RETURN n.file_path AS p, n.name AS name", ns=ns).data()
    java_fqn, java_simple, ts_path, api_node = {}, {}, {}, None
    for r in rows:
        p = r["p"].replace("\\", "/")
        if p.endswith(".java"):
            for marker in ("/src/main/java/", "/src/test/java/"):
                if marker in p:
                    tail = p.split(marker, 1)[1][: -len(".java")]
                    java_fqn[tail.replace("/", ".")] = p
            java_simple.setdefault(os.path.basename(p)[: -len(".java")], []).append(p)
        elif p.endswith(TS_EXT):
            ts_path[os.path.splitext(p)[0]] = p
        elif p.endswith("/src/app/api"):
            api_node = p
    return java_fqn, java_simple, ts_path, api_node


def resolve_simple(name, java_simple):
    """Simple class name -> file path, only when unambiguous."""
    hits = java_simple.get(name)
    return hits[0] if hits and len(hits) == 1 else None


def base_types(clause):
    """Supertype names from an extends/implements clause, generics discarded.

    `BaseService<Content, ContentDtoIn, ContentDtoOut, Long>` is ONE supertype,
    BaseService -- the type arguments are not superclasses. Splitting naively on every
    identifier invents EXTENDS edges to the DTOs, so commas are split at angle-bracket
    depth 0 only and each part is truncated at its first '<'.
    """
    parts, depth, cur = [], 0, []
    for ch in clause:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur))

    names = []
    for part in parts:
        head = part.split("<", 1)[0].strip()
        if not head:
            continue
        m = IDENT.findall(head.rsplit(".", 1)[-1])   # com.foo.Bar -> Bar
        if m:
            names.append(m[0])
    return names


def resolve_ts(spec, src_path, ts_path, api_node, fe_root):
    """Resolve a TS import specifier to an indexed file, mirroring tsc resolution."""
    if spec.startswith("."):
        base = os.path.normpath(os.path.join(os.path.dirname(src_path), spec))
    elif spec.startswith(("@app/", "app/", "src/")):
        rel = spec.split("/", 1)[1] if spec.startswith("@app/") else spec
        rel = rel[4:] if rel.startswith("src/") else rel
        base = os.path.normpath(os.path.join(fe_root, "src", rel))
    else:
        return None, None                      # third-party package
    base = base.replace("\\", "/")
    if api_node and (base == api_node or base.startswith(api_node + "/")):
        return api_node, "IMPORTS"             # the collapsed generated client
    for cand in (base, base + "/index"):
        if cand in ts_path:
            return ts_path[cand], "INJECTS"
    return None, None


# ---------------------------------------------------------------------------------
# per-file extraction
# ---------------------------------------------------------------------------------

def extract(path, idx, fe_root):
    java_fqn, java_simple, ts_path, api_node = idx
    p = path.replace("\\", "/")
    content = read_source(p)
    edges = set()
    if not content:
        return edges

    if p.endswith(".java"):
        for imp in JAVA_IMPORT.findall(content):
            if not imp.startswith(INTERNAL_JAVA_ROOT):
                continue
            tgt = java_fqn.get(imp)
            if tgt is None:                    # static import: drop the member
                tgt = java_fqn.get(imp.rsplit(".", 1)[0])
            if tgt and tgt != p:
                edges.add(("IMPORTS", tgt))
        for raw in JAVA_INJECT.findall(content):
            for token in IDENT.findall(raw):
                tgt = resolve_simple(token, java_simple)
                if tgt and tgt != p:
                    edges.add(("INJECTS", tgt))
        for m in JAVA_EXTENDS.findall(content):
            for token in base_types(m):
                tgt = resolve_simple(token, java_simple)
                if tgt and tgt != p:
                    edges.add(("EXTENDS", tgt))
        for m in JAVA_IMPLEMENTS.findall(content):
            for token in base_types(m):
                tgt = resolve_simple(token, java_simple)
                if tgt and tgt != p:
                    edges.add(("IMPLEMENTS", tgt))

    elif p.endswith(TS_EXT):
        for spec in TS_IMPORT.findall(content):
            tgt, rel = resolve_ts(spec, p, ts_path, api_node, fe_root)
            if tgt and tgt != p:
                edges.add((rel, tgt))

    return edges


def test_subject(path, ts_path, java_simple):
    """Deterministic test -> subject pairing; returns the SUBJECT path or None."""
    p = path.replace("\\", "/")
    b = os.path.basename(p)
    if p.endswith(".java"):
        stem = b[: -len(".java")]
        for suffix in ("UnitTest", "IntegrationTest", "IT", "Test"):
            if stem.endswith(suffix) and len(stem) > len(suffix):
                return resolve_simple(stem[: -len(suffix)], java_simple)
        return None
    for suffix in (".unit.spec.ts", ".spec.ts"):
        if b.endswith(suffix):
            cand = p[: -len(suffix)]
            return ts_path.get(cand)
    return None


ALLOWED_RELS = ("IMPORTS", "INJECTS", "EXTENDS", "IMPLEMENTS", "TESTED_BY")

# One statement per type: the relationship type cannot be parameterised, and a literal
# per-type MERGE is far easier to audit than a UNION subquery. ALLOWED_RELS is a closed
# whitelist, so no edge type outside the graph's existing vocabulary can be written.
WRITE = """
UNWIND $rows AS row
MATCH (a:EntityDetail {namespace:$ns, file_path: row.src})
MATCH (b:EntityDetail {namespace:$ns, file_path: row.tgt})
MERGE (a)-[:%s]->(b)
RETURN count(*) AS written
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="delta_batch marker to extract for")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--reverse", action="store_true",
                    help="also scan already-indexed files for references INTO the batch. "
                         "Without this a new node only gets its outgoing edges and can "
                         "land isolated -- e.g. app.config.ts registering a new "
                         "interceptor, or a DTO carrying a new @VimeoUrls annotation.")
    args = ap.parse_args()

    fe_root = "C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield"
    drv = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with drv.session() as s:
        idx = build_index(s, NAMESPACE)
        java_fqn, java_simple, ts_path, api_node = idx
        print(f"index: {len(java_fqn)} java fqn, {len(ts_path)} ts files, "
              f"api node {'found' if api_node else 'MISSING'}")
        batch = [r["p"] for r in s.run(
            "MATCH (n:EntityDetail {namespace:$ns, delta_batch:$b}) "
            "RETURN n.file_path AS p", ns=NAMESPACE, b=args.batch).data()]
        print(f"batch files: {len(batch)}")

        rows, per_rel = [], {}
        for p in batch:
            for rel, tgt in extract(p, idx, fe_root):
                rows.append({"src": p, "tgt": tgt, "rel": rel})
                per_rel[rel] = per_rel.get(rel, 0) + 1
            subj = test_subject(p, ts_path, java_simple)
            if subj and subj != p:
                rows.append({"src": subj, "tgt": p, "rel": "TESTED_BY"})
                per_rel["TESTED_BY"] = per_rel.get("TESTED_BY", 0) + 1

        if args.reverse:
            batch_set = set(batch)
            everything = [r["p"] for r in s.run(
                "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.file_path IS NOT NULL "
                "RETURN n.file_path AS p", ns=NAMESPACE).data()]
            rev = 0
            for p in everything:
                if p in batch_set:
                    continue                    # forward pass already covered these
                for rel, tgt in extract(p, idx, fe_root):
                    if tgt in batch_set:
                        rows.append({"src": p, "tgt": tgt, "rel": rel})
                        per_rel[rel] = per_rel.get(rel, 0) + 1
                        rev += 1
                subj = test_subject(p, ts_path, java_simple)
                if subj in batch_set and subj != p:
                    rows.append({"src": subj, "tgt": p, "rel": "TESTED_BY"})
                    per_rel["TESTED_BY"] = per_rel.get("TESTED_BY", 0) + 1
                    rev += 1
            print(f"reverse pass over {len(everything)} indexed files: +{rev} edges")

        print(f"extracted {len(rows)} edges: {per_rel}")
        if args.dry:
            for r in rows[:40]:
                print(f"  {r['rel']:<11} {os.path.basename(r['src'])} -> "
                      f"{os.path.basename(r['tgt'])}")
            return 0

        for rel in ALLOWED_RELS:
            sub = [r for r in rows if r["rel"] == rel]
            if not sub:
                continue
            res = s.run(WRITE % rel, ns=NAMESPACE, rows=sub).single()
            print(f"  {rel:<11} extracted {len(sub):4d}  write returned {res['written']}")

        # VERIFY BY FOLLOW-UP READ -- a write's return value is not evidence.
        seen = s.run("""
            UNWIND $rows AS row
            MATCH (a:EntityDetail {namespace:$ns, file_path: row.src})
                  -[r]->(b:EntityDetail {namespace:$ns, file_path: row.tgt})
            WHERE type(r) = row.rel
            RETURN count(DISTINCT [row.src, row.tgt, row.rel]) AS c
            """, ns=NAMESPACE, rows=rows).single()["c"]
        want = len({(r["src"], r["tgt"], r["rel"]) for r in rows})
        print(f"VERIFY: {seen}/{want} distinct edges present on re-read "
              f"-- {'OK' if seen == want else 'MISMATCH'}")
    drv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
