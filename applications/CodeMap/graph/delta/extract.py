"""Deterministic delta extraction: a checkout + a change list -> pack.next (entities, edges, lbdb) + delta.json.

    python graph/delta/extract.py --name backend --repo-dir <checkout> --base <sha> --head <sha>
                                  [--pack graph/pack] [--out delta/<run>] [--backlog graph/delta/backlog.jsonl] [--date D]
    python graph/delta/extract.py --name backend --repo-dir <checkout> --changed-list changes.txt ...   # "A|M|D<TAB>path" lines

No model runs here. What a machine can decide, it decides: eligibility by repository rules, the
entity row for an added file (type by the documented suffix/annotation heuristic, a content
fingerprint, line count), the five structural edge types (IMPORTS, INJECTS, EXTENDS, IMPLEMENTS,
TESTED_BY) re-extracted for added and modified files, the purge of deleted files, the churn that
decides delta vs full, and the rebuilt LadybugDB. What only a mind can decide — which subsystem a
new file belongs to, the twelve semantic edge types — is left as `unassigned` for the proposal step
and carried forward untouched for unchanged files.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
REPOS = json.load(open(os.path.join(HERE, "repos.json"), encoding="utf-8"))
ENT_COLS = ["name", "file_path", "entity_type", "subsystem", "curated", "layer", "local_height", "entry_point",
            "spines", "line_count", "fingerprint", "delta_batch"]
STRUCTURAL = ("IMPORTS", "INJECTS", "EXTENDS", "IMPLEMENTS", "TESTED_BY")

JAVA_IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;", re.M)
TS_IMPORT = re.compile(r"""(?:from|import)\s*\(?\s*['"]([^'"]+)['"]""")
JAVA_EXTENDS = re.compile(r"\b(?:class|interface)\s+\w+[^{]*?\bextends\s+([\w<>,.\s]+?)(?:\bimplements\b|\{)")
JAVA_IMPLEMENTS = re.compile(r"\bimplements\s+([\w<>,.\s]+?)\{")
JAVA_INJECT = re.compile(r"^\s*(?:private|protected)\s+final\s+([\w<>\[\],.]+)\s+\w+\s*;", re.M)
IDENT = re.compile(r"[A-Za-z_]\w*")
INTERNAL_JAVA_ROOT = "com.sm."
TS_EXT = (".ts", ".tsx", ".mts", ".js")


# ----------------------------------------------------------------------------- paths

def canon(name, rel):
    return REPOS["repos"][name]["prefix"] + rel.replace("\\", "/").lstrip("/")


def eligible(name, rel):
    r = REPOS["repos"][name]
    p = rel.replace("\\", "/")
    if any(p.startswith(x) or f"/{x}" in p for x in r["exclude"]):
        return False
    if r.get("roots") and not any(p.startswith(x) for x in r["roots"]):
        return False
    return any(p.endswith(ext) for ext in r["include"])


def read_source(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def fingerprint(content):
    data = content.encode("utf-8", "ignore")
    return f"size:{len(data)}|lines:{len(content.splitlines())}|sha:{hashlib.sha256(data).hexdigest()[:8]}"


# ----------------------------------------------------------------------------- entity type heuristic (Hypatia.md:305-356)

def entity_type(rel, content):
    """Most specific rule wins; the documented Hypatia heuristic, made deterministic."""
    base = os.path.basename(rel)
    stem = base.rsplit(".", 1)[0]
    low = rel.lower()
    if base.endswith((".yml", ".yaml", ".properties", ".xml")):
        return "Context"
    if base.endswith(".feature"):
        return "Rule"
    if re.search(r"(Test|IT|Spec)$", stem) or ".spec." in base or "/src/test/" in "/" + low or "@Test" in content:
        return "Rule"
    if "@EventListener" in content or "ApplicationEvent" in content or stem.endswith(("Event", "EventListener")):
        return "Event"
    if re.search(r"Security|Guard|Filter|Interceptor", stem) or "@PreAuthorize" in content or "@EnableWebSecurity" in content:
        return "Rule"
    if stem.endswith(("Controller", "Resource", "Handler")) or "@RestController" in content or "@RequestMapping" in content \
            or base.endswith((".component.ts", ".component.html", ".page.ts", ".routes.ts")):
        return "Actor"
    if stem.endswith(("Config", "Configuration")) or "@Configuration" in content or base.endswith(".config.ts"):
        return "Context"
    if stem.endswith(("Service", "Manager", "Facade", "CronJob", "Job", "Processor")) or "@Service" in content \
            or base.endswith(".service.ts"):
        return "Process"
    return "Resource"


# ----------------------------------------------------------------------------- index from the pack

def build_index(ents):
    java_fqn, java_simple, ts_path, api_node = {}, {}, {}, None
    for e in ents:
        p = e["file_path"]
        if p.endswith(".java"):
            for marker in ("/src/main/java/", "/src/test/java/"):
                if marker in p:
                    java_fqn[p.split(marker, 1)[1][:-5].replace("/", ".")] = p
            java_simple.setdefault(os.path.basename(p)[:-5], []).append(p)
        elif p.endswith(TS_EXT):
            ts_path[p.rsplit(".", 1)[0]] = p
        elif p.endswith("/src/app/api") or e["name"] == "api":
            api_node = p
    return java_fqn, java_simple, ts_path, api_node


def resolve_simple(token, java_simple):
    cands = java_simple.get(token)
    return cands[0] if cands and len(cands) == 1 else None


def base_types(clause):
    depth, parts, cur = 0, [], []
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
        if head:
            m = IDENT.findall(head.rsplit(".", 1)[-1])
            if m:
                names.append(m[0])
    return names


def resolve_ts(spec, src_path, ts_path, api_node, fe_root):
    if spec.startswith("."):
        base = os.path.normpath(os.path.join(os.path.dirname(src_path), spec))
    elif spec.startswith(("@app/", "app/", "src/")):
        rel = spec.split("/", 1)[1] if spec.startswith("@app/") else spec
        rel = rel[4:] if rel.startswith("src/") else rel
        base = os.path.normpath(os.path.join(fe_root, "src", rel))
    else:
        return None, None
    base = base.replace("\\", "/")
    if api_node and (base == api_node or base.startswith(api_node + "/")):
        return api_node, "IMPORTS"
    for cand in (base, base + "/index"):
        if cand in ts_path:
            return ts_path[cand], "INJECTS"
    return None, None


def extract_edges(canon_path, content, idx, fe_root):
    java_fqn, java_simple, ts_path, api_node = idx
    p = canon_path
    edges = set()
    if p.endswith(".java"):
        for imp in JAVA_IMPORT.findall(content):
            if not imp.startswith(INTERNAL_JAVA_ROOT):
                continue
            tgt = java_fqn.get(imp) or java_fqn.get(imp.rsplit(".", 1)[0])
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


def test_subject(canon_path, ts_path, java_simple):
    p = canon_path
    b = os.path.basename(p)
    if p.endswith(".java"):
        stem = b[:-5]
        for suffix in ("UnitTest", "IntegrationTest", "IT", "Test"):
            if stem.endswith(suffix) and len(stem) > len(suffix):
                return resolve_simple(stem[:-len(suffix)], java_simple)
        return None
    for suffix in (".unit.spec.ts", ".spec.ts"):
        if b.endswith(suffix):
            return ts_path.get(p[:-len(suffix)])
    return None


# ----------------------------------------------------------------------------- change lists

def git_changes(repo_dir, base, head):
    out = subprocess.run(["git", "-C", repo_dir, "diff", "--name-status", "-M", f"{base}..{head}"],  # NOSONAR - argv list; see sonar-project.properties
                         capture_output=True, text=True, check=True).stdout
    return parse_changes(out)


def parse_changes(text):
    """'A\\tpath' / 'M\\tpath' / 'D\\tpath' / 'R100\\told\\tnew' -> [(status, path, old_path|None)]"""
    rows = []
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2 or not parts[0]:
            continue
        st = parts[0][0]
        if st == "R" and len(parts) >= 3:
            rows.append(("D", parts[1], None))
            rows.append(("A", parts[2], parts[1]))
        elif st in ("A", "M", "D", "C", "T"):
            rows.append(("A" if st == "C" else ("M" if st == "T" else st), parts[1], None))
    return rows


# ----------------------------------------------------------------------------- the delta

def run(name, repo_dir, changes, pack_dir, out_dir, backlog_rows=None, date=None, churn_full=None):
    repo = REPOS["repos"][name]
    churn_full = REPOS.get("churn_full_reindex", 0.1) if churn_full is None else churn_full
    date = date or time.strftime("%Y-%m-%d", time.gmtime())
    ents = list(csv.DictReader(open(os.path.join(pack_dir, "entities.csv"), encoding="utf-8")))
    edges = list(csv.DictReader(open(os.path.join(pack_dir, "edges.csv"), encoding="utf-8")))
    by_path = {e["file_path"]: e for e in ents}
    hyper_rows = list(csv.DictReader(open(os.path.join(pack_dir, "hyperedges.csv"), encoding="utf-8"))) \
        if os.path.exists(os.path.join(pack_dir, "hyperedges.csv")) else []
    fe_root = repo["prefix"].rstrip("/")

    # backlog misses are added files that were never indexed
    todo = {}
    for st, rel, old in changes:
        if eligible(name, rel):
            todo[rel] = st
    for row in backlog_rows or []:
        p = (row.get("path") or "")
        pref = name + "/"
        if p.startswith(pref):
            rel = p[len(pref):]
            if eligible(name, rel) and rel not in todo and canon(name, rel) not in by_path:
                todo[rel] = "A"

    added, modified, deleted, unchanged, skipped = [], [], [], [], []
    for rel, st in sorted(todo.items()):
        cp = canon(name, rel)
        full = os.path.join(repo_dir, rel)
        if st == "D" or (st != "D" and not os.path.exists(full)):
            if cp in by_path:
                deleted.append(cp)
            else:
                skipped.append({"path": rel, "why": "deleted, not indexed"})
            continue
        content = read_source(full)
        fp = fingerprint(content)
        if cp in by_path:
            if by_path[cp]["fingerprint"] == fp:
                unchanged.append(cp)
            else:
                modified.append((cp, content, fp))
        else:
            added.append((cp, rel, content, fp))

    # 1. entities
    for cp in deleted:
        by_path.pop(cp, None)
    for cp, content, fp in modified:
        e = by_path[cp]
        e["fingerprint"] = fp
        e["line_count"] = str(len(content.splitlines()))
        e["delta_batch"] = date
    new_rows = []
    for cp, rel, content, fp in added:
        row = {"name": os.path.basename(rel), "file_path": cp, "entity_type": entity_type(rel, content), "subsystem": "",
               "curated": "", "layer": "", "local_height": "", "entry_point": "False", "spines": "[]",
               "line_count": str(len(content.splitlines())), "fingerprint": fp, "delta_batch": date}
        row["layer"] = row["entity_type"]
        by_path[cp] = row
        new_rows.append(row)
    ents_next = [by_path[e["file_path"]] for e in ents if e["file_path"] in by_path] + new_rows
    # 2. edges: purge deleted endpoints; drop structural out-edges of modified/added files; re-extract
    name_of = {e["file_path"]: e["name"] for e in ents_next}
    ambiguous = {n for n in set(name_of.values()) if sum(1 for v in name_of.values() if v == n) > 1}
    del_names = {os.path.basename(cp) for cp in deleted}
    touched = {cp for cp, *_ in modified} | {cp for cp, *_ in added}
    touched_names = {os.path.basename(cp) for cp in touched}
    kept = []
    removed = 0
    for ed in edges:
        if ed["src"] in del_names or ed["dst"] in del_names:
            removed += 1
            continue
        if ed["src"] in touched_names and ed["rel"] in STRUCTURAL:
            removed += 1
            continue
        kept.append(ed)
    idx = build_index(ents_next)
    new_edges = []
    for cp, content in [(cp, c) for cp, c, _ in modified] + [(cp, c) for cp, _, c, _ in added]:
        src_name = os.path.basename(cp)
        for rel_t, tgt in sorted(extract_edges(cp, content, idx, fe_root)):
            tgt_name = name_of.get(tgt)
            if tgt_name and tgt_name not in ambiguous:
                new_edges.append({"src": src_name, "dst": tgt_name, "rel": rel_t})
        subj = test_subject(cp, idx[2], idx[1])
        if subj and name_of.get(subj):
            new_edges.append({"src": name_of[subj], "dst": src_name, "rel": "TESTED_BY"})
    seen = {(e["src"], e["dst"], e["rel"]) for e in kept}
    for e in new_edges:
        k = (e["src"], e["dst"], e["rel"])
        if k not in seen:
            kept.append(e)
            seen.add(k)
    # 3. hyperedges: drop members that vanished (structure only; recomputation is a full reindex)
    hyper_next = []
    for h in hyper_rows:
        text = json.dumps(h)
        if any(n in text for n in del_names):
            continue
        hyper_next.append(h)
    # 4. churn + mode
    n_before = len(ents)
    churn = (len(added) + len(modified) + len(deleted)) / max(1, n_before)
    mode = "full" if churn > churn_full else "delta"
    os.makedirs(out_dir, exist_ok=True)
    delta = {"schema": 1, "repo": name, "date": date, "mode": mode, "churn": round(churn, 4), "churn_full_reindex": churn_full,
             "counts": {"before": n_before, "after": len(ents_next), "added": len(added), "modified": len(modified),
                        "deleted": len(deleted), "unchanged": len(unchanged), "skipped": len(skipped),
                        "edges_before": len(edges), "edges_removed": removed, "edges_added": len(kept) - (len(edges) - removed),
                        "edges_after": len(kept), "hyperedges_before": len(hyper_rows), "hyperedges_after": len(hyper_next)},
             "added": [{"file_path": cp, "name": os.path.basename(rel), "entity_type": by_path[cp]["entity_type"],
                        "line_count": by_path[cp]["line_count"],
                        "edges_out": [e for e in new_edges if e["src"] == os.path.basename(cp)][:20]} for cp, rel, _, _ in added],
             "modified": [cp for cp, *_ in modified], "deleted": deleted, "skipped": skipped,
             "unassigned": [cp for cp, *_ in added]}
    with open(os.path.join(out_dir, "delta.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(delta, f, indent=1, sort_keys=True)
    if mode == "full":
        return delta
    # 5. pack.next
    nxt = os.path.join(out_dir, "pack.next")
    if os.path.isdir(nxt):
        shutil.rmtree(nxt)
    os.makedirs(nxt)
    with open(os.path.join(nxt, "entities.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ENT_COLS)
        w.writeheader()
        for e in ents_next:
            w.writerow({k: e.get(k, "") for k in ENT_COLS})
    with open(os.path.join(nxt, "edges.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["src", "dst", "rel"])
        w.writeheader()
        for e in kept:
            w.writerow({"src": e["src"], "dst": e["dst"], "rel": e["rel"]})
    if hyper_rows:
        with open(os.path.join(nxt, "hyperedges.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(hyper_rows[0].keys()))
            w.writeheader()
            for h in hyper_next:
                w.writerow(h)
    # the manifest rides along (indexed_sha of the other repository, the version to bump) and so does an
    # earlier delta invalidation, which the next apply extends rather than replaces
    for name_ in ("l1_master.json", "l2_navigators.jsonl", "mfq.jsonl", "codemap_vocab.gbnf", "DIALECT_NOTES.md", "manifest.json", "INVALIDATED_delta.json"):
        src = os.path.join(pack_dir, name_)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(nxt, name_))
    build_lbdb(nxt)
    return delta


# ----------------------------------------------------------------------------- lbdb

def build_lbdb(pack_dir):
    """Rebuild codemap.lbdb and edges_lb.csv from entities.csv + edges.csv (import_ladybug.py, without the gold check)."""
    import real_ladybug as lb
    ents = list(csv.DictReader(open(os.path.join(pack_dir, "entities.csv"), encoding="utf-8")))
    from collections import Counter
    counts = Counter(e["name"] for e in ents)
    n2p = {e["name"]: e["file_path"] for e in ents if counts[e["name"]] == 1}
    skipped = 0
    with open(os.path.join(pack_dir, "edges_lb.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "rel"])
        for r in csv.DictReader(open(os.path.join(pack_dir, "edges.csv"), encoding="utf-8")):
            if r["src"] in n2p and r["dst"] in n2p:
                w.writerow([n2p[r["src"]], n2p[r["dst"]], r["rel"]])
            else:
                skipped += 1
    dbp = os.path.join(pack_dir, "codemap.lbdb")
    for stale in (dbp, dbp + ".wal", dbp + ".lock"):
        if os.path.isdir(stale):
            shutil.rmtree(stale, ignore_errors=True)
        elif os.path.exists(stale):
            os.remove(stale)
    db = lb.Database(dbp)
    conn = lb.Connection(db)
    conn.execute("CREATE NODE TABLE Entity(name STRING, file_path STRING, entity_type STRING, subsystem INT64, curated INT64, "
                 "layer STRING, local_height DOUBLE, entry_point BOOLEAN, spines STRING, line_count INT64, fingerprint STRING, "
                 "delta_batch STRING, PRIMARY KEY (file_path))")
    conn.execute("CREATE REL TABLE Dep(FROM Entity TO Entity, rel STRING)")
    conn.execute(f'COPY Entity FROM "{os.path.join(pack_dir, "entities.csv").replace(chr(92), "/")}" (HEADER=true)')
    conn.execute(f'COPY Dep FROM "{os.path.join(pack_dir, "edges_lb.csv").replace(chr(92), "/")}" (HEADER=true)')
    n = conn.execute("MATCH (e:Entity) RETURN count(*)").get_next()[0]
    m = conn.execute("MATCH ()-[r:Dep]->() RETURN count(*)").get_next()[0]
    del conn, db
    return {"entities": n, "dep_edges": m, "edges_skipped_ambiguous": skipped}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(REPOS["repos"]))
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--base")
    ap.add_argument("--head")
    ap.add_argument("--changed-list")
    ap.add_argument("--pack", default=os.path.join(R, "graph", "pack"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--backlog", default=os.path.join(HERE, "backlog.jsonl"))
    ap.add_argument("--date", default=None)
    a = ap.parse_args(argv)
    if a.changed_list:
        changes = parse_changes(open(a.changed_list, encoding="utf-8").read())
    elif a.base and a.head:
        changes = git_changes(a.repo_dir, a.base, a.head)
    else:
        raise SystemExit("give --base/--head or --changed-list")
    backlog = [json.loads(l) for l in open(a.backlog, encoding="utf-8") if l.strip()] if os.path.exists(a.backlog) else []
    out = a.out or os.path.join(R, "delta", time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    delta = run(a.name, a.repo_dir, changes, a.pack, out, backlog, a.date)
    print(json.dumps({k: delta[k] for k in ("repo", "mode", "churn", "counts")}, indent=1))
    print(f"out: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
