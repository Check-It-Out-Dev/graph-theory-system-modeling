# CodeMap AUTHORING store on LadybugDB (MIT) — the Ladybug-first generation.
# This replaces Neo4j as the system of record for the authoring graph: entities with
# their THREE embedding lenses, typed dependency edges, the curated navigation tree,
# hyperedge cohorts, bi-temporal clue snapshots, curation decisions and the diagnostic
# layer. The runtime pack remains a downstream EXPORT of this store.
#
# Design laws carried over:
# - VERTEX IDENTITY IS nid (explicit INT64 PK) — a file name is a label, never a key
#   (ledger L11; two duplicate basenames exist in the corpus by right).
# - Supersession, never erasure: ClueSnap rows are append-only bi-temporal.
# - Provenance on every write (source column / delta_batch).
# - Long-tail properties ride in a `props` JSON column so migration is LOSSLESS while
#   every script-queried field stays a real typed column.
#
# Dialect notes vs Neo4j (kept in sync with graph/pack/DIALECT_NOTES.md):
# - No id(n): use the explicit nid column everywhere.
# - One Dep rel table with a `rel STRING` column instead of 17 typed rel types
#   (recipes filter on the column; runtime pack does the same).
# - Parameterized CREATE loops instead of COPY (corpus is 1.4k nodes — seconds).
#
# Usage: PYTHONUTF8=1 python ladybug_store.py init|info [--db <path>]

import argparse
import json
import os

import real_ladybug as lb

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "checkitout.lbdb")

DDL = [
    # -- L3: the corpus ---------------------------------------------------------------
    """CREATE NODE TABLE IF NOT EXISTS Entity(
        nid INT64 PRIMARY KEY, name STRING, entity_type STRING, sub INT64,
        file_path STRING, repo STRING,
        layer STRING, entry_point STRING, spine_membership STRING, local_height DOUBLE,
        clue_version STRING, delta_batch STRING,
        sem_emb FLOAT[], beh_emb FLOAT[], str_emb FLOAT[],
        props STRING)""",
    """CREATE REL TABLE IF NOT EXISTS Dep(FROM Entity TO Entity,
        rel STRING, source STRING)""",
    """CREATE REL TABLE IF NOT EXISTS Violation(FROM Entity TO Entity, source STRING)""",
    # -- L2/L1: the navigation tree ---------------------------------------------------
    """CREATE NODE TABLE IF NOT EXISTS Nav(
        sub_id INT64 PRIMARY KEY, name STRING, role STRING, routable BOOLEAN,
        parent INT64, size INT64, external_ratio DOUBLE,
        ai_summary STRING, clue_version STRING, clue_body_status STRING,
        generated_by STRING, clue_delta_batch STRING, dossier_fingerprint STRING,
        spines STRING, entry_points STRING, contracts STRING, caveats STRING,
        responsibilities STRING, props STRING)""",
    """CREATE NODE TABLE IF NOT EXISTS Master(
        mid INT64 PRIMARY KEY, ai_summary STRING, subsystem_index STRING,
        global_caveats STRING, props STRING)""",
    """CREATE REL TABLE IF NOT EXISTS Guides(FROM Master TO Nav)""",
    """CREATE REL TABLE IF NOT EXISTS GuidesChild(FROM Nav TO Nav)""",
    """CREATE REL TABLE IF NOT EXISTS Member(FROM Nav TO Entity)""",
    # -- cohorts ----------------------------------------------------------------------
    """CREATE NODE TABLE IF NOT EXISTS Hyperedge(
        key STRING PRIMARY KEY, metapath STRING, hub_nid INT64, hub_name STRING,
        hubsub INT64, idf DOUBLE, arity INT64, prior DOUBLE, members STRING,
        source STRING, props STRING)""",
    """CREATE REL TABLE IF NOT EXISTS SupersededBy(FROM Nav TO Nav)""",
    # -- bi-temporal clue history (supersession, never erasure) -----------------------
    """CREATE NODE TABLE IF NOT EXISTS ClueSnap(
        snap_id STRING PRIMARY KEY, sub_id INT64, taken_at STRING,
        t_created STRING, t_expired STRING, superseded_by STRING, body STRING)""",
    """CREATE NODE TABLE IF NOT EXISTS CurationDecision(
        did STRING PRIMARY KEY, batch STRING, decided_at STRING, body STRING)""",
    # -- diagnostic layer (SCHEMA 6b): observations per (functional, scope, run) ------
    """CREATE NODE TABLE IF NOT EXISTS Diag(
        oid STRING PRIMARY KEY, functional STRING, scope STRING,
        value_num DOUBLE, value_json STRING, run_id STRING, computed_at STRING)""",
    # -- store metadata ---------------------------------------------------------------
    """CREATE NODE TABLE IF NOT EXISTS Meta(k STRING PRIMARY KEY, v STRING)""",
]


class Store:
    def __init__(self, path=None, read_only=False):
        self.path = path or os.environ.get("CODEMAP_AUTHORING_DB") or DEFAULT_DB
        self.db = lb.Database(self.path, read_only=read_only)
        self.conn = lb.Connection(self.db)
        if not read_only:  # writers always see the full schema (idempotent DDL+ALTERs)
            self.init_schema()

    def init_schema(self):
        for ddl in DDL:
            self.conn.execute(ddl)
        # idempotent column additions (writers may extend the Entity annotation set)
        for col, typ in (("org_version", "STRING"), ("org_at", "STRING"),
                         ("curated_sub", "INT64")):
            try:
                self.conn.execute(f"ALTER TABLE Entity ADD {col} {typ}")
            except RuntimeError:
                pass  # already present
        return self

    def q(self, cypher, params=None):
        """Run one statement -> list of dict rows."""
        res = self.conn.execute(cypher, parameters=params or {})
        cols = res.get_column_names()
        out = []
        while res.has_next():
            out.append(dict(zip(cols, res.get_next())))
        return out

    def one(self, cypher, params=None):
        rows = self.q(cypher, params)
        return rows[0] if rows else None

    def create(self, table, data, pk):
        """The armored writer — THE one implementation every script uses (L11).

        Ladybug 0.15.3 hazards it guards (bisected 2026-09-02):
        - a bound None or EMPTY list is untypeable -> omitted from the property map;
        - a bound STRING that LOOKS like a list/struct literal is silently parsed and
          re-serialized in Kuzu notation ('["a"]' -> '[a]', JSON destroyed; '[]'
          crashes) -> such values travel as escaped in-query literals, which the
          grammar guarantees are STRINGs;
        - a bound LIST into a STRING column gets the same notation cast -> callers
          jdump lists first (the [-prefix then routes through the armor)."""
        d = {k: v for k, v in data.items()
             if v is not None and not (isinstance(v, list) and not v)}
        params, lits = {}, {}
        for k, v in d.items():
            if isinstance(v, str) and v.lstrip()[:1] in ("[", "{"):
                lits[k] = "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"
            else:
                params[k] = v
        keys = ", ".join([f"{k}:${k}" for k in params]
                         + [f"{k}:{v}" for k, v in lits.items()])
        self.conn.execute(f"CREATE (:{table} {{{keys}}})", parameters=params)

    @staticmethod
    def lit(v):
        """Escaped in-query string literal — the armor for JSON-looking values."""
        return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"

    def merge_props(self, table, pk, pkval, updates):
        """Read-modify-write of the `props` JSON column (long-tail fields live there).
        The merged JSON travels as an armored literal — never a parameter."""
        row = self.one(f"MATCH (n:{table}) WHERE n.{pk} = $v RETURN n.props AS p",
                       dict(v=pkval))
        cur = json.loads((row or {}).get("p") or "{}")
        cur.update(updates)
        blob = json.dumps(cur, ensure_ascii=False, default=str)
        self.conn.execute(
            f"MATCH (n:{table}) WHERE n.{pk} = $v SET n.props = {self.lit(blob)}",
            parameters=dict(v=pkval))

    def counts(self):
        tabs = ["Entity", "Nav", "Master", "Hyperedge", "ClueSnap",
                "CurationDecision", "Diag", "Meta"]
        out = {t: self.one(f"MATCH (n:{t}) RETURN count(n) AS c")["c"] for t in tabs}
        out["Dep"] = self.one("MATCH (:Entity)-[r:Dep]->(:Entity) RETURN count(r) AS c")["c"]
        out["Violation"] = self.one(
            "MATCH (:Entity)-[r:Violation]->(:Entity) RETURN count(r) AS c")["c"]
        out["Member"] = self.one("MATCH (:Nav)-[r:Member]->(:Entity) RETURN count(r) AS c")["c"]
        out["Guides"] = self.one("MATCH (:Master)-[r:Guides]->(:Nav) RETURN count(r) AS c")["c"]
        out["GuidesChild"] = self.one(
            "MATCH (:Nav)-[r:GuidesChild]->(:Nav) RETURN count(r) AS c")["c"]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "info"])
    ap.add_argument("--db", default=None)
    a = ap.parse_args()
    s = Store(a.db)
    if a.cmd == "init":
        s.init_schema()
        print("schema ready:", s.path)
    print(json.dumps(s.counts(), indent=1))


if __name__ == "__main__":
    main()
