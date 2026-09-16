"""The MCP surface: JSON-RPC 2.0 over Streamable HTTP, tools only. Pure `dispatch` for tests.

Protocol: `initialize` (echo the client's version when we know it), `notifications/initialized`
(202, no body), `ping`, `tools/list`, `tools/call`. Everything else is -32601. Tool bodies live in
tools.py; this module only knows their names and input schemas, so the contract is testable
without an engine.
"""

import json

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26")
SERVER_INFO = {"name": "codemap", "version": "1.2.0"}
FEEDBACK_TAGS = ["wrong", "incomplete", "hallucinated", "slow", "great", "should_have_abstained",
                 "should_have_answered", "pointer_wrong", "pointer_verified"]

TOOLS = [
    {"name": "codemap_ask",
     "description": "Ask the CodeMap navigator a question about the checkItOut code (backend + frontend). "
                    "Returns a short answer plus POINTERS (repo, path, subsystem); open the files yourself. "
                    "Reuse context_id for follow-ups in the same conversation.",
     "inputSchema": {"type": "object", "properties": {
         "q": {"type": "string", "description": "the question, in your own words"},
         "context_id": {"type": "string", "description": "conversation id to continue (omit to start one)"},
         "tier": {"type": "string", "enum": ["auto", "deep"], "description": "deep = Opus navigator (costs more credits)"}},
         "required": ["q"]}},
    {"name": "codemap_step",
     "description": "Run one CMDSL verb directly against the graph (free): map() / enter(id) / find(term) / "
                    "impact(name[,depth]) / flow(name[,depth]) / seam(a,b) / cohort(name) / spine(id) / "
                    "health(kind) / read(name). Entity names must be exact names from a previous result.",
     "inputSchema": {"type": "object", "properties": {"dsl": {"type": "string"}}, "required": ["dsl"]}},
    {"name": "codemap_open",
     "description": "Resolve an exact entity name to a pointer (repo, relative path, subsystem, clue). Never returns file content.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "codemap_search",
     "description": "Semantic search over entity sockets (Qwen3 embedding + reranker): text in, top-k pointers out. "
                    "Use when you do not know the repository vocabulary yet.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"},
                                                     "k": {"type": "integer", "minimum": 1, "maximum": 20}},
                     "required": ["text"]}},
    {"name": "codemap_feedback",
     "description": "Rate an answer AFTER verifying at least one pointer in your checkout. Exactly one of rating (1-5) or vote (up|down).",
     "inputSchema": {"type": "object", "properties": {
         "request_id": {"type": "string"}, "rating": {"type": "integer", "minimum": 1, "maximum": 5},
         "vote": {"type": "string", "enum": ["up", "down"]},
         "tags": {"type": "array", "items": {"type": "string", "enum": FEEDBACK_TAGS}},
         "comment": {"type": "string", "maxLength": 300},
         "verified": {"type": "boolean", "description": "you opened at least one pointer and it was right"}},
         "required": ["request_id"]}},
    {"name": "codemap_miss",
     "description": "Report a file you had to find WITHOUT the graph (grep, guess). Feeds the saturation backlog so the next reindex covers it.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "why": {"type": "string", "maxLength": 200}},
                     "required": ["path"]}},
    {"name": "codemap_status",
     "description": "Server status: pack/prompt versions, your remaining daily credits, queue depth.",
     "inputSchema": {"type": "object", "properties": {}}},
]
TOOL_NAMES = [t["name"] for t in TOOLS]


def _err(mid, code, message, data=None):
    e = {"code": code, "message": message}
    if data is not None:
        e["data"] = data
    return {"jsonrpc": "2.0", "id": mid, "error": e}


def _ok(mid, result):
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def dispatch(msg, call_tool, tools=None, server_name=None):
    """One JSON-RPC message -> (http_status, reply_or_None). `call_tool(name, args) -> (text, is_error)`.
    `tools` defaults to the remote surface; the engine MCP passes its own list."""
    tools = TOOLS if tools is None else tools
    names = [t["name"] for t in tools]
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return 400, _err(None, -32600, "invalid request")
    method = msg.get("method")
    mid = msg.get("id")
    if method is None:
        return 400, _err(mid, -32600, "missing method")
    if mid is None:  # notification
        return 202, None
    if method == "initialize":
        want = (msg.get("params") or {}).get("protocolVersion")
        ver = want if want in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return 200, _ok(mid, {"protocolVersion": ver, "capabilities": {"tools": {"listChanged": False}},
                             "serverInfo": dict(SERVER_INFO, name=server_name or SERVER_INFO["name"]),
                             "instructions": "Ask with codemap_ask; verify pointers in your checkout; "
                                             "rate with codemap_feedback."})
    if method == "ping":
        return 200, _ok(mid, {})
    if method == "tools/list":
        return 200, _ok(mid, {"tools": tools})
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        if name not in names:
            return 200, _err(mid, -32602, f"unknown tool {name}")
        args = params.get("arguments") or {}
        try:
            text, is_error = call_tool(name, args)
        except Exception as ex:  # a tool failure is a tool result, not a protocol error
            text, is_error = json.dumps({"error": f"{type(ex).__name__}: {ex}"}), True
        res = {"content": [{"type": "text", "text": text}]}
        if is_error:
            res["isError"] = True
        return 200, _ok(mid, res)
    return 200, _err(mid, -32601, f"method not found: {method}")


def dispatch_body(body, call_tool, tools=None, server_name=None):
    """A body may be one message or a batch; a batch reply is a list."""
    if isinstance(body, list):
        replies = []
        for m in body:
            _, r = dispatch(m, call_tool, tools, server_name)
            if r is not None:
                replies.append(r)
        return (200, replies) if replies else (202, None)
    return dispatch(body, call_tool, tools, server_name)
