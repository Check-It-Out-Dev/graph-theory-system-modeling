"""Direct, read-only access to the CodeMap graph in LadybugDB: one Cypher statement in, rows out.

    python remote/ladybug_mcp.py        (spawned by Claude Code from an MCP config; CODEMAP_PACK_DIR names the pack)

This is the raw graph, not a grammar: no verbs, no rewriting of the question, no forced LIMIT. The
database is opened with `read_only=True`, so a write is refused by LadybugDB itself rather than by a
keyword filter. The only conveniences are the ones a reader of a workspace needs:

- paths: the pack stores the authoring box's absolute prefixes; results show them as `backend/...` and
  `frontend/...` (the layout of the workspace), and a string literal in a statement that starts with
  `backend/` or `frontend/` is translated back before it runs, so `e.file_path STARTS WITH 'backend/src/'`
  works as written;
- caps: at most 200 rows and 40,000 characters per answer, with `truncated` and the full `row_count`
  when a result is larger, and a 30-second query timeout.

Newline-delimited JSON-RPC 2.0 over stdio, one tool: `graph_query(statement)`.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
sys.path.insert(0, R)

from remote import mcp  # noqa: E402

MAX_ROWS = 200
MAX_CHARS = 40000
TIMEOUT_MS = 30000

TOOLS = [
    {"name": "graph_query",
     "description": ("Run ONE read-only Cypher statement against the CodeMap graph of checkItOut (LadybugDB). "
                     "Tables: node Entity(name, file_path PK, entity_type, subsystem, curated, layer, local_height, entry_point, "
                     "spines, line_count, fingerprint, delta_batch) and relationship Dep(rel) from Entity to Entity; the edge "
                     "kind is the property r.rel (IMPORTS, INJECTS, CALLS, TRIGGERS, ...), never a label. Paths read and write "
                     "as backend/... and frontend/.... Returns columns and rows (at most 200 rows, 40,000 characters)."),
     "inputSchema": {"type": "object", "properties": {"statement": {"type": "string", "description": "one Cypher statement"}},
                     "required": ["statement"]}},
]


def prefixes():
    """-> [(stored prefix, workspace prefix)] from graph/delta/repos.json."""
    with open(os.path.join(R, "graph", "delta", "repos.json"), encoding="utf-8") as f:
        repos = json.load(f)["repos"]
    return [(cfg["prefix"], f"{name}/") for name, cfg in sorted(repos.items())]


def to_workspace(value, pairs):
    if isinstance(value, str):
        for stored, short in pairs:
            if value.startswith(stored):
                return short + value[len(stored):]
            if stored in value:
                value = value.replace(stored, short)
        return value
    if isinstance(value, list):
        return [to_workspace(v, pairs) for v in value]
    if isinstance(value, dict):
        return {k: to_workspace(v, pairs) for k, v in value.items()}
    return value


LITERAL = re.compile(r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")")


def to_stored(statement, pairs):
    """Translate string literals that start with a workspace prefix back to the stored prefix."""
    def fix(m):
        lit = m.group(0)
        quote, body = lit[0], lit[1:-1]
        for stored, short in pairs:
            if body.startswith(short):
                return quote + stored + body[len(short):] + quote
        return lit
    return LITERAL.sub(fix, statement)


def one_statement(statement):
    """-> the statement without a trailing ';', or None when it holds more than one."""
    s = statement.strip().rstrip(";").strip()
    outside = LITERAL.sub("''", s)
    return None if ";" in outside else s


class GraphTools:
    def __init__(self, pack_dir=None, connection=None):
        self.pairs = prefixes()
        if connection is not None:
            self.con = connection
            return
        import real_ladybug as lb
        pack = pack_dir or os.environ.get("CODEMAP_PACK_DIR") or os.path.join(R, "graph", "pack")
        self.db = lb.Database(os.path.join(pack, "codemap.lbdb"), read_only=True)
        self.con = lb.Connection(self.db)
        try:
            self.con.set_query_timeout(TIMEOUT_MS)
        except Exception:  # an older binding without timeouts still answers
            pass

    def call(self, name, args):
        if name != "graph_query":
            return json.dumps({"error": f"unknown tool {name}"}), True
        return self.query(args.get("statement") or "")

    def query(self, statement):
        s = one_statement(statement or "")
        if not s:
            return json.dumps({"error": "send exactly one non-empty Cypher statement per call"}), True
        try:
            res = self.con.execute(to_stored(s, self.pairs))
            columns = list(res.get_column_names())
            rows = []
            total = 0
            while res.has_next():
                row = res.get_next()
                total += 1
                if len(rows) < MAX_ROWS:
                    rows.append(to_workspace(list(row), self.pairs))
        except Exception as ex:  # the database's own message is the most useful answer to a bad statement
            return json.dumps({"error": str(ex).strip().splitlines()[0][:600]}), True
        out = {"columns": columns, "rows": rows, "row_count": total, "truncated": total > len(rows)}
        text = json.dumps(out, ensure_ascii=False, default=str)
        while len(text) > MAX_CHARS and out["rows"]:
            out["rows"] = out["rows"][: max(1, len(out["rows"]) // 2)] if len(out["rows"]) > 1 else []
            out["truncated"] = True
            text = json.dumps(out, ensure_ascii=False, default=str)
        return text, False


def serve(stdin=None, stdout=None, tools=None):
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    tools = tools or GraphTools()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        _, reply = mcp.dispatch_body(msg, tools.call, tools=TOOLS, server_name="codemap-graph")
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
