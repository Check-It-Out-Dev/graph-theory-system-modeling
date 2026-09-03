# CodeMap export tool — Ladybug authoring store -> portable pack -> embedded runtime DB.
# Exports the FULL logical schema (graph/SCHEMA.md) as CSV/JSONL + manifest. Embeddings are
# deliberately NOT exported (two-embedder law: runtime re-embeds with the bundled model).
# Since 2026-09-02 the authoring source is the MIT Ladybug store (graph/authoring/);
# acceptance for the migration was per-file hash equality against the Neo4j-era pack.
#
# Usage: PYTHONUTF8=1 python export_pack.py [--out ../pack]

import argparse, csv, hashlib, json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "authoring")))
from ladybug_store import Store


def w_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "pack"))
    out = os.path.abspath(ap.parse_args().out)
    os.makedirs(out, exist_ok=True)
    counts = {}
    if True:
        s = Store(read_only=True)
        cur = {r["nid"]: r["sub"] for r in s.q(
            "MATCH (sn:Nav)-[:Member]->(n:Entity) "
            "WHERE sn.role IS NULL OR NOT sn.role IN ['MERGED','GROUP'] "
            "RETURN n.nid AS nid, sn.sub_id AS sub")}
        nodes = s.q(
            "MATCH (n:Entity) RETURN n.nid AS nid, n.name AS name, "
            "n.file_path AS file_path, n.entity_type AS entity_type, n.sub AS subsystem, "
            "n.layer AS layer, n.local_height AS local_height, "
            "n.entry_point AS entry_point, n.spine_membership AS spines, "
            "n.props AS props, n.delta_batch AS delta_batch")
        nodes.sort(key=lambda n: n["nid"])  # determinism at the boundary
        for n in nodes:
            p = json.loads(n.pop("props") or "{}")
            n["line_count"] = p.get("line_count", 0) or 0
            n["fingerprint"] = p.get("content_fingerprint")
            # store "" == Neo4j NULL; "True"/"False" strings == the original booleans
            n["entry_point"] = ({"": None, "True": True, "False": False}
                                .get(n["entry_point"], n["entry_point"]))
            sp = n["spines"]
            n["spines"] = json.loads(sp) if sp and sp.lstrip()[:1] == "[" else sp
            n["curated"] = cur.get(n["nid"], n["subsystem"])
        counts["entities"] = w_csv(os.path.join(out, "entities.csv"),
            ["name", "file_path", "entity_type", "subsystem", "curated", "layer", "local_height",
             "entry_point", "spines", "line_count", "fingerprint", "delta_batch"],
            [[n["name"], n["file_path"], n["entity_type"], n["subsystem"], n["curated"], n["layer"],
              n["local_height"], n["entry_point"], json.dumps(n["spines"] or []),
              n["line_count"], n["fingerprint"], n["delta_batch"]] for n in nodes])
        edges = s.q(
            "MATCH (a:Entity)-[r:Dep]->(b:Entity) "
            "RETURN a.nid AS aid, b.nid AS bid, a.name AS src, r.rel AS rel, b.name AS dst")
        edges.sort(key=lambda e: (e["aid"], e["bid"], e["rel"]))
        counts["edges"] = w_csv(os.path.join(out, "edges.csv"), ["src", "rel", "dst"],
                                [[e["src"], e["rel"], e["dst"]] for e in edges])
        hyper = s.q(
            "MATCH (h:Hyperedge) RETURN h.key AS key, h.metapath AS metapath, "
            "h.hub_name AS hub, h.arity AS arity, h.idf AS idf, h.prior AS prior, "
            "h.members AS members")
        hyper.sort(key=lambda h: str(h["key"]))
        counts["hyperedges"] = w_csv(os.path.join(out, "hyperedges.csv"),
            ["key", "metapath", "hub", "arity", "idf", "prior", "members"],
            [[h["key"], h["metapath"], h["hub"], h["arity"], h["idf"], h["prior"],
              json.dumps(json.loads(h["members"] or "null") or [])] for h in hyper])
        # MERGED navigators are EXPORTED, not filtered out. They hold no members and never
        # appear in subsystem_index, so they cannot be routed to — but mfq stamps and curation
        # history still name sub_id 13 and 18, and the L1 caveat promises those stamps resolve.
        # That promise was true in the authoring graph and false in the pack while the runtime
        # only ever reads the pack. Each one carries `superseded_by` (13 -> 4, 18 -> 177) read
        # from the [:SUPERSEDED_BY] edge, so a stale stamp resolves to its absorber offline.
        # Nav full property maps reconstructed: core columns + JSON-field columns
        # + props remainder. Of the five JSON fields, caveats/responsibilities carry ONE
        # jdump layer (json.loads yields the list); spines/entry_points/contracts carry
        # TWO — a pre-migration authoring artifact, uniform across all 32 Nav rows and
        # already inside the 2026-09-02 ClueSnap bodies — so one loads leaves them as
        # JSON STRINGS in l2_navigators.jsonl. DO NOT normalize while model v1 is
        # frozen: pack bytes are part of the freeze bundle (ledger L13). Keys are
        # written sort_keys=True — the DETERMINISM LAW extended to l2/l1 (the old byte
        # order was accidental Neo4j map order; acceptance for the migration is
        # canonical-content equality on these two files, hash equality on the CSVs).
        JF = ("spines", "entry_points", "contracts", "caveats", "responsibilities")
        navs = s.q("MATCH (sn:Nav) RETURN sn.*")
        navs.sort(key=lambda r: r["sn.sub_id"])
        parent_of = {r["c"]: r["p"] for r in s.q(
            "MATCH (a:Nav)-[:GuidesChild]->(b:Nav) "
            "RETURN a.sub_id AS p, b.sub_id AS c")}
        children_of, sup_of = {}, {}
        for r in s.q("MATCH (a:Nav)-[:GuidesChild]->(b:Nav) "
                     "RETURN a.sub_id AS p, b.sub_id AS c"):
            children_of.setdefault(r["p"], []).append(r["c"])
        for r in s.q("MATCH (a:Nav)-[:SupersededBy]->(b:Nav) "
                     "RETURN a.sub_id AS a, b.sub_id AS b"):
            sup_of.setdefault(r["a"], []).append(r["b"])
        with open(os.path.join(out, "l2_navigators.jsonl"), "w", encoding="utf-8") as f:
            for r in navs:
                sid = r["sn.sub_id"]
                d = json.loads(r["sn.props"] or "{}")
                for col in ("name", "role", "ai_summary", "clue_version",
                            "clue_body_status", "generated_by", "clue_delta_batch",
                            "dossier_fingerprint", "size", "external_ratio"):
                    v = r.get(f"sn.{col}")
                    if v is not None:
                        d[col] = v
                for col in JF:
                    v = json.loads(r.get(f"sn.{col}") or "null")
                    if v is not None:
                        d[col] = v
                d["sub_id"] = sid
                d["generated_at"] = str(d.get("generated_at"))
                d["parent"] = parent_of.get(sid)
                d["children"] = sorted(children_of.get(sid, []))
                d["superseded_by"] = sorted(sup_of.get(sid, []))
                d["routable"] = d.get("role") != "MERGED"
                f.write(json.dumps(d, ensure_ascii=False, default=str,
                                   sort_keys=True) + "\n")
        counts["l2"] = len(navs)
        m = s.one("MATCH (nm:Master) RETURN nm.*")
        with open(os.path.join(out, "l1_master.json"), "w", encoding="utf-8") as f:
            d = json.loads((m or {}).get("nm.props") or "{}")
            for col, key in (("nm.ai_summary", "ai_summary"),
                             ("nm.subsystem_index", "subsystem_index")):
                if (m or {}).get(col) is not None:
                    d[key] = m[col]
            gc = json.loads((m or {}).get("nm.global_caveats") or "null")
            if gc is not None:
                d["global_caveats"] = gc
            json.dump(d, f, ensure_ascii=False, indent=1, default=str, sort_keys=True)
        counts["l1"] = 1 if m else 0

    # MFQ + recipes ride from eval (they are part of the pack per SCHEMA)
    qdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "eval", "q"))
    mfq_src = os.path.join(qdir, "mfq_all.jsonl")
    with open(mfq_src, encoding="utf-8") as f, \
         open(os.path.join(out, "mfq.jsonl"), "w", encoding="utf-8") as g:
        n = 0
        for line in f:
            g.write(line)
            n += 1
    counts["mfq"] = n

    files = sorted(f for f in os.listdir(out) if f != "manifest.json")
    manifest = {
        "schema_version": "1.0", "dialect_source": "neo4j-5",
        "dialect_target": "ladybug (import pending spike)",
        "codebase": "checkItOut", "packed_at_note": "see git commit date",
        "embedder": {"note": "embeddings intentionally excluded; runtime re-embeds "
                             "(two-embedder law, SCHEMA.md section 4)"},
        "counts": counts,
        "files": {f: hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest()[:16]
                  for f in files},
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("PACK OK:", json.dumps(counts))

    # vocabulary grammar is derived FROM the pack — regenerate with every export so the
    # decode-time vocabulary can never go stale against the entities/navigators it constrains
    import gbnf_vocab
    gbnf_vocab.main()


if __name__ == "__main__":
    sys.exit(main())
