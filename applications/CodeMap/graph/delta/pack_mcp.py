"""Read-only stdio MCP over a pack directory (pack.next during a delta): Grothendieck's hands.

    CODEMAP_PACK_DIR=<pack.next> python graph/delta/pack_mcp.py

Tools: pack_subsystem(id) — the L2 navigator record; pack_entity(name) — the entity row and its edges
in both directions with the neighbours' subsystems; pack_folder(path_prefix) — entities under a path
and their subsystem mix; pack_cypher(stmt) — one read-only statement with LIMIT. Nothing here writes.
"""

import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)

from remote import mcp  # noqa: E402

TOOLS = [
    {"name": "pack_subsystem", "description": "The L2 navigator record of a subsystem id (name, role, summary, responsibilities, entry points, size, parent).",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}},
    {"name": "pack_entity", "description": "An entity row (by exact name) with its edges in both directions and the neighbours' subsystems.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "pack_folder", "description": "Entities whose path contains the given fragment, with the subsystem mix of that folder.",
     "inputSchema": {"type": "object", "properties": {"fragment": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["fragment"]}},
    {"name": "pack_cypher", "description": "One read-only openCypher statement over the pack (Entity, Dep{rel}); LIMIT enforced.",
     "inputSchema": {"type": "object", "properties": {"stmt": {"type": "string"}}, "required": ["stmt"]}},
]
FORBID = re.compile(r"\b(create|merge|delete|detach|set|drop|alter|copy|call|load)\b", re.I)


class Pack:
    def __init__(self, pack_dir=None):
        self.dir = pack_dir or os.environ.get("CODEMAP_PACK_DIR") or os.path.join(R, "graph", "pack")
        self.ents = list(csv.DictReader(open(os.path.join(self.dir, "entities.csv"), encoding="utf-8")))
        self.by_name = {}
        for e in self.ents:
            self.by_name.setdefault(e["name"], e)
        self.edges = list(csv.DictReader(open(os.path.join(self.dir, "edges.csv"), encoding="utf-8")))
        self.out_e, self.in_e = {}, {}
        for ed in self.edges:
            self.out_e.setdefault(ed["src"], []).append((ed["dst"], ed["rel"]))
            self.in_e.setdefault(ed["dst"], []).append((ed["src"], ed["rel"]))
        self.l2 = {}
        p = os.path.join(self.dir, "l2_navigators.jsonl")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    nav = json.loads(line)
                    self.l2[str(nav.get("sub_id"))] = nav
        self._conn = None

    def subsystem(self, sid):
        nav = self.l2.get(str(sid))
        if not nav:
            return {"error": f"no subsystem {sid}"}
        keep = ("sub_id", "name", "role", "ai_summary", "responsibilities", "entry_points", "size", "parent", "children", "layer_profile", "caveats")
        return {k: nav.get(k) for k in keep if k in nav}

    def entity(self, name):
        e = self.by_name.get(name)
        if not e:
            cands = [n for n in self.by_name if name.lower() in n.lower()][:10]
            return {"error": f"unknown entity {name}", "candidates": cands}
        sub_of = lambda n: (self.by_name.get(n) or {}).get("curated") or (self.by_name.get(n) or {}).get("subsystem")
        return {"entity": {k: e.get(k) for k in ("name", "file_path", "entity_type", "subsystem", "curated", "layer", "line_count", "delta_batch")},
                "out": [{"to": d, "rel": r, "sub": sub_of(d)} for d, r in self.out_e.get(name, [])][:60],
                "in": [{"from": s, "rel": r, "sub": sub_of(s)} for s, r in self.in_e.get(name, [])][:60]}

    def folder(self, fragment, limit=40):
        rows = [e for e in self.ents if fragment.replace("\\", "/") in e["file_path"]]
        mix = {}
        for e in rows:
            s = e.get("curated") or e.get("subsystem") or "unassigned"
            mix[s] = mix.get(s, 0) + 1
        return {"n": len(rows), "subsystem_mix": dict(sorted(mix.items(), key=lambda kv: -kv[1])),
                "entities": [{"name": e["name"], "sub": e.get("curated") or e.get("subsystem") or "", "type": e["entity_type"]} for e in rows[:limit]]}

    def cypher(self, stmt):
        q = stmt.strip().rstrip(";")
        if ";" in q or FORBID.search(q):
            return {"error": "read-only: one MATCH ... RETURN statement"}
        if not re.search(r"\blimit\s+\d+", q, re.I):
            q += " LIMIT 50"
        try:
            import real_ladybug as lb
            if self._conn is None:
                self._conn = lb.Connection(lb.Database(os.path.join(self.dir, "codemap.lbdb"), read_only=True))
            res = self._conn.execute(q)
            cols = res.get_column_names()
            rows = []
            while res.has_next() and len(rows) < 50:
                rows.append([str(v) for v in res.get_next()])
            return {"columns": cols, "rows": rows}
        except Exception as ex:
            return {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}

    def call(self, name, args):
        if name == "pack_subsystem":
            out = self.subsystem(args.get("id"))
        elif name == "pack_entity":
            out = self.entity(args.get("name") or "")
        elif name == "pack_folder":
            out = self.folder(args.get("fragment") or "", int(args.get("limit") or 40))
        elif name == "pack_cypher":
            out = self.cypher(args.get("stmt") or "")
        else:
            out = {"error": f"unknown tool {name}"}
        return json.dumps(out, ensure_ascii=False), "error" in out


def serve(stdin=None, stdout=None, pack=None):
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    pack = pack or Pack()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        _, reply = mcp.dispatch_body(msg, pack.call, tools=TOOLS, server_name="codemap-pack")
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
