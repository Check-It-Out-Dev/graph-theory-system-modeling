"""The navigator's hands: a stdio MCP that exposes the engine to the `claude -p` navigator.

    python -m remote.engine_mcp        (spawned by Claude Code from the mcp-config the server builds)

Tools: engine_step(dsl) — one CMDSL verb; engine_cypher(stmt) — guarded read-only openCypher over
the pack (big_tier.run_cypher: forbidden clauses rejected, LIMIT enforced); engine_open(name) — a
pointer. Every call is appended to CODEMAP_TRACE_FILE so the server can record the trajectory of
the answer it is about to emit. Newline-delimited JSON-RPC 2.0, no SSE, no sessions.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "app"))

from remote import mcp, pointers  # noqa: E402

TOOLS = [
    {"name": "engine_step",
     "description": "One CMDSL verb against the code graph: map() / enter(sub) / find(term) / impact(entity[,depth]) / "
                    "flow(entity[,depth]) / seam(subA,subB) / cohort(entity) / spine(sub) / health(kind) / read(entity). "
                    "Entity names and sub ids must be copied exactly from a previous result or the question.",
     "inputSchema": {"type": "object", "properties": {"dsl": {"type": "string"}}, "required": ["dsl"]}},
    {"name": "engine_cypher",
     "description": "One read-only openCypher statement over the pack (nodes :Entity {name, entity_type, layer, curated, "
                    "local_height, entry_point, line_count}; edges (:Entity)-[r:Dep {rel}]->(:Entity)). Always LIMIT; "
                    "prefer count(*) / ORDER BY over listing. No CASE inside count()/sum() (dialect bug).",
     "inputSchema": {"type": "object", "properties": {"stmt": {"type": "string"}}, "required": ["stmt"]}},
    {"name": "engine_open",
     "description": "Exact entity name -> pointer (repo, relative path, subsystem, lines). Never file content.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
]


class EngineTools:
    def __init__(self, engine=None, trace_file=None):
        if engine is None:
            from engine import Engine
            engine = Engine()
        self.engine = engine
        self.by_name = {e["name"]: e for e in engine.ents}
        self.trace_file = trace_file

    def _trace(self, tool, arg, ok):
        if not self.trace_file:
            return
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"tool": tool, "arg": arg, "ok": ok}, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def call(self, name, args):
        if name == "engine_step":
            return self.step(args.get("dsl") or "")
        if name == "engine_cypher":
            return self.cypher(args.get("stmt") or "")
        if name == "engine_open":
            return self.open(args.get("name") or "")
        return json.dumps({"error": f"unknown tool {name}"}), True

    def step(self, dsl):
        from dsl import execute, ParseError
        from engine import DslError
        try:
            res = execute(self.engine, dsl.strip())
        except (ParseError, DslError) as e:
            self._trace("engine_step", dsl, False)
            return json.dumps({"error": str(e)}), True
        if res.get("kind") == "pointer":
            ent = self.by_name.get(res.get("file"))
            if ent:
                res["pointer"] = pointers.to_pointer(ent)
            res.pop("path", None)
        self._trace("engine_step", dsl, True)
        return json.dumps(res, ensure_ascii=False), False

    def cypher(self, stmt):
        import big_tier
        text = big_tier.run_cypher(self.engine, stmt.strip())
        ok = not text.startswith("ERROR:")
        self._trace("engine_cypher", stmt, ok)
        return text, not ok

    def open(self, name):
        ent = self.by_name.get(name.strip())
        if not ent:
            cands = [n for n in self.by_name if name.strip().lower() in n.lower()][:10]
            self._trace("engine_open", name, False)
            return json.dumps({"error": f"unknown entity {name!r}", "candidates": cands}), True
        self._trace("engine_open", name, True)
        return json.dumps(pointers.to_pointer(ent), ensure_ascii=False), False


def serve(stdin=None, stdout=None, tools=None):
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    tools = tools or EngineTools(trace_file=os.environ.get("CODEMAP_TRACE_FILE"))
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        _, reply = mcp.dispatch_body(msg, tools.call, tools=TOOLS, server_name="codemap-engine")
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
