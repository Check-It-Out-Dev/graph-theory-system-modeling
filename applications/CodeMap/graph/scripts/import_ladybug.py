# CodeMap migration spike: pack -> LadybugDB (real-ladybug, Kuzu-successor, embedded MIT).
# Loads entities + edges, then re-runs gold M03 (UserRepository blast radius) on Ladybug and
# compares the canonical fingerprint against the Neo4j gold — the migration acceptance test.
#
# Usage: PYTHONUTF8=1 python import_ladybug.py
# Dialect notes discovered here feed the recipe cypher_ladybug fields (Q2 catalogue).

import csv, hashlib, json, os, shutil, sys

import real_ladybug as lb

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.abspath(os.path.join(HERE, "..", "pack"))
DBP = os.path.join(PACK, "codemap.lbdb")


def canon(rows):
    rows = sorted([str(c) for c in r] for r in rows)
    return rows, hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()[:16]


def main():
    ents = list(csv.DictReader(open(os.path.join(PACK, "entities.csv"), encoding="utf-8")))
    from collections import Counter
    name_count = Counter(e["name"] for e in ents)
    ambiguous = {n for n, c in name_count.items() if c > 1}
    print(f"PK: file_path ({len(ambiguous)} ambiguous names: {sorted(ambiguous)})")
    n2p = {e["name"]: e["file_path"] for e in ents if e["name"] not in ambiguous}

    # rel-table COPY wants (FROM, TO, props) order; keys are file_paths now
    skipped = 0
    with open(os.path.join(PACK, "edges_lb.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "rel"])
        for r in csv.DictReader(open(os.path.join(PACK, "edges.csv"), encoding="utf-8")):
            if r["src"] in n2p and r["dst"] in n2p:
                w.writerow([n2p[r["src"]], n2p[r["dst"]], r["rel"]])
            else:
                skipped += 1
    print(f"edges skipped (ambiguous-name endpoints): {skipped}")

    for stale in (DBP, DBP + ".wal", DBP + ".lock"):
        if os.path.isdir(stale):
            shutil.rmtree(stale, ignore_errors=True)
        elif os.path.exists(stale):
            os.remove(stale)
    db = lb.Database(DBP)
    conn = lb.Connection(db)
    conn.execute(
        "CREATE NODE TABLE Entity(name STRING, file_path STRING, entity_type STRING, "
        "subsystem INT64, curated INT64, layer STRING, local_height DOUBLE, entry_point BOOLEAN, "
        "spines STRING, line_count INT64, fingerprint STRING, delta_batch STRING, "
        "PRIMARY KEY (file_path))")
    conn.execute("CREATE REL TABLE Dep(FROM Entity TO Entity, rel STRING)")
    conn.execute(f'COPY Entity FROM "{os.path.join(PACK, "entities.csv").replace(chr(92), "/")}" (HEADER=true)')
    conn.execute(f'COPY Dep FROM "{os.path.join(PACK, "edges_lb.csv").replace(chr(92), "/")}" (HEADER=true)')

    n = conn.execute("MATCH (e:Entity) RETURN count(*)").get_next()[0]
    m = conn.execute("MATCH ()-[r:Dep]->() RETURN count(*)").get_next()[0]
    print(f"loaded: {n} entities, {m} dep edges")
    assert n == len(ents)

    # gold M03 on Ladybug — dialect: single Dep table with rel property (vs typed rels in Neo4j)
    res = conn.execute(
        "MATCH (a:Entity)-[r:Dep]->(b:Entity) WHERE b.name = 'UserRepository.java' "
        "RETURN DISTINCT a.name, r.rel, a.subsystem")
    rows = []
    while res.has_next():
        a, rel, sub = res.get_next()
        rows.append([a, rel, sub])
    rows, fp = canon(rows)
    print(f"M03 on Ladybug: {len(rows)} rows, fingerprint {fp}")
    gold = [json.loads(l) for l in open(os.path.join(PACK, "mfq.jsonl"), encoding="utf-8")
            if '"GR01"' in l][0]
    ref = gold["gold_fingerprint"]
    print(f"M03 Neo4j gold fingerprint: {ref} -> {'MATCH' if fp == ref else 'MISMATCH'}")
    dialect_notes = [
        "package: real-ladybug (PyPI); import real_ladybug; API Database/Connection (Kuzu style)",
        "typed rels -> single Dep table with rel STRING property; queries use r.rel = 'X' instead of [:X]",
        "COPY rel tables requires (FROM, TO, props) column order",
        "COPY paths must use forward slashes on Windows (backslash = parser escape)",
        "node PK = file_path (2 duplicate basenames found in 1374: UnifiedStorageConfiguration.java x2, HashingUtilUnitTest.java x2); ambiguous-name edges skipped and counted",
        f"gold M03 fingerprint {'REPRODUCED' if fp == ref else 'NOT reproduced — investigate'} on v0.15.3",
        "CASE WHEN inside an aggregate (count/sum) returns wrong numbers (bisected "
        "2026-09-03: count(CASE WHEN b THEN 1 END) = 1 where count(*) WHERE b = 300); "
        "plain CASE, count(DISTINCT) and key-grouped count(*) are all correct — "
        "conditional counts must be separate WHERE-filtered queries",
    ]
    with open(os.path.join(PACK, "DIALECT_NOTES.md"), "w", encoding="utf-8") as f:
        f.write("# Ladybug dialect notes (spike)\n\n" + "\n".join(f"- {x}" for x in dialect_notes) + "\n")
    return 0 if fp == ref else 1


if __name__ == "__main__":
    sys.exit(main())
