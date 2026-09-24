"""Does Erdős do what his manual says? Deterministic checks on one run's transcript and answer, one per rule.

    PYTHONUTF8=1 python eval/erdos/erdos_adherence.py --label <label> [--arm erdos]      one table per run

The manual states its rules as behaviour a transcript can show. Each check reads the stream-json events the
runner recorded and the answer, needs no model, and returns a value in [0, 1] with a sentence saying what was
seen (counts only, no code names, so a prompt optimiser can read it without learning the answers). The mean is
the adherence score; it sits beside the judge's verdict on the answer itself.

| check | manual rule | value |
|---|---|---|
| graph_pass | core rule 1 and <graph_pass> | half: graph queries before the first file tool (three or more count as full); half: the share of the five query kinds the pass asks for (entry points, dependents, dependencies, coupling or seams, behaviour edges) issued before the first file tool |
| key_files_read | core rule 2 | the share of the answer key's must_find files the run opened with Read |
| facts_grounded | core rule 3 | the share of FACT evidence lines (labelled one by one, under a FACT heading, or in a table row) citing a file or class the run opened with Read or saw in a Grep or Glob result |
| one_pass | core rule 4 | 1 when the completion marker was written and nothing ran after it |
| contract | <answer_contract> | the share of: the six sections, the marker, and an answer free of subsystem ids and graph wording |

`trace_text` renders the same transcript for the judge: every graph query with the rows it returned, and the
files opened, in order.
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import erdos_phases  # noqa: E402

GRAPH_PREFIXES = erdos_phases.GRAPH_PREFIXES
FILE_TOOLS = erdos_phases.FILE_TOOLS
KINDS = ("entry_points", "dependents", "dependencies", "coupling", "behaviour")
SECTIONS = ("Problem", "Where it lives today", "Proposed change", "Plan", "Risks and invariants", "Evidence")
BEHAVIOUR_RELS = ("PERFORMS", "MODIFIES", "ACCESSES", "CALLS", "TRIGGERS", "INITIATES", "AFFECTS", "CONSTRAINS", "VALIDATES")
FILE_RX = re.compile(r"[A-Za-z0-9_.\-]+\.(?:java|ts|html|scss|css|yml|yaml|xml|properties|feature|sql|js|mjs|json|md)\b")
LABEL_RX = re.compile(r"\b(FACT|INFERENCE|HYPOTHESIS)\b")
IDENT_RX = re.compile(r"\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b")              # a class name such as UserService
# (the same words as `[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+`: the tail's class absorbs every later capital, so one
#  group suffices, and the nested quantifier that could backtrack exponentially is gone)
CODE_IDENT_RX = re.compile(r"`([A-Z][A-Za-z0-9_]*)(?:[.#(][^`]*)?`")          # `Guard`, `UserService.delete`
TELL_RX = re.compile(r"(?<![\w`\]])\[\d{1,3}\]|\bsubsystems?\s+\[?\d{1,3}\b|\bgraph_query\b|\bthe graph\b", re.I)
EDGE_RX = re.compile(r"\((\w*)(?::Entity)?\s*(\{[^}]*\})?\)\s*<?-\s*\[[^\]]*\]\s*->?\s*\((\w*)(?::Entity)?\s*(\{[^}]*\})?\)")
WORKSPACE_RX = re.compile(r"[\\/](backend|frontend)[\\/](.*)$")


# ----------------------------------------------------------------------------- transcript

def tool_uses(events):
    """-> [(name, input, result_text)] in call order; each result joined to its call by tool_use_id."""
    uses, results = [], {}
    for _, e in events:
        if e.get("type") == "assistant":
            for block in (e.get("message") or {}).get("content") or []:
                if block.get("type") == "tool_use":
                    uses.append((block.get("id"), block.get("name") or "?", block.get("input") or {}))
        elif e.get("type") == "user":
            for block in (e.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    c = block.get("content")
                    text = "".join(x.get("text", "") for x in c if isinstance(x, dict)) if isinstance(c, list) else str(c or "")
                    results[block.get("tool_use_id")] = text
    return [(name, inp, results.get(uid, "")) for uid, name, inp in uses]


def workspace_path(path):
    m = WORKSPACE_RX.search(path or "")
    return f"{m.group(1)}/{m.group(2)}".replace("\\", "/") if m else (path or "").replace("\\", "/")


def _filtered(var, props, statement):
    if props and re.search(r"\b(?:file_path|name)\s*:", props):
        return True
    return bool(var) and bool(re.search(rf"\b{re.escape(var)}\.(?:file_path|name)\s*(?:=|IN\b|STARTS WITH|ENDS WITH|CONTAINS|=~)",
                                        statement, re.I))


def query_kinds(statement):
    """-> the kinds of graph question a Cypher statement asks, from its shape."""
    s = " ".join((statement or "").split())
    kinds = set()
    asked = re.split(r"\bRETURN\b", s, maxsplit=1, flags=re.I)[0]            # a filter, not a returned column
    if re.search(r"\bentry_point\b", asked):
        kinds.add("entry_points")
    behaviour = any(re.search(rf"\b{rel}\b", s) for rel in BEHAVIOUR_RELS)
    if behaviour:
        kinds.add("behaviour")
    for m in EDGE_RX.finditer(s):
        left, lprops, right, rprops = m.group(1), m.group(2), m.group(3), m.group(4)
        if "SHORTEST" in s.upper():
            continue
        reverse = "<-" in m.group(0)
        src, sprops, dst, dprops = (right, rprops, left, lprops) if reverse else (left, lprops, right, rprops)
        src_f, dst_f = _filtered(src, sprops, s), _filtered(dst, dprops, s)
        if dst_f and not src_f:
            kinds.add("dependents")
        elif src_f and not dst_f:
            kinds.add("dependencies")
        both_subsystems = all(v and re.search(rf"\b{re.escape(v)}\.(?:curated|subsystem)\b", s) for v in (src, dst))
        if both_subsystems and not behaviour:
            kinds.add("coupling")
    return kinds


# ----------------------------------------------------------------------------- answer

def sections_present(answer):
    return [name for name in SECTIONS if re.search(rf"^\s*#{{1,4}}\s*\**{re.escape(name)}\b", answer or "", re.I | re.M)]


def evidence_lines(answer):
    """-> the lines of the Evidence section (up to the next heading or the marker)."""
    m = re.search(r"^\s*#{1,4}\s*\**Evidence\b.*$", answer or "", re.I | re.M)
    if not m:
        return []
    tail = answer[m.end():]
    stop = re.search(r"^\s*#{1,4}\s|" + re.escape(erdos_phases.SOLVED), tail, re.M)
    return [line for line in (tail[:stop.start()] if stop else tail).splitlines() if line.strip()]


def labelled_evidence(answer):
    """-> [(label, line)] for the Evidence section's bullets and table rows. A line's own label wins; otherwise
    the label of the heading above it (for example **FACT (lines read)**) applies."""
    current, out = None, []
    for line in evidence_lines(answer):
        stripped = line.strip()
        labels = LABEL_RX.findall(stripped)
        item = stripped.startswith(("-", "*", "|")) and not stripped.startswith("**") or bool(re.match(r"^\d+\.", stripped))
        if labels and not item:
            current = labels[0]
            continue
        label = labels[0] if labels else current
        if item and label and not re.match(r"^\|[\s:|-]*\|?$", stripped):
            out.append((label, stripped))
    return out


def cited_names(line):
    """-> the lower-case file stems and class names a line cites."""
    names = {os.path.splitext(f)[0].lower() for f in FILE_RX.findall(line)}
    return names | {w.lower() for w in IDENT_RX.findall(line)} | {w.lower() for w in CODE_IDENT_RX.findall(line)}


# ----------------------------------------------------------------------------- checks

def check_run(events, answer, key):
    """-> {"checks": {name: {"value", "seen"}}, "score"}."""
    uses = tool_uses(events)
    first_file = next((i for i, (name, _, _) in enumerate(uses) if name in FILE_TOOLS), len(uses))
    graph = [(i, inp.get("statement") or "") for i, (name, inp, _) in enumerate(uses) if name.startswith(GRAPH_PREFIXES)]
    before = [s for i, s in graph if i < first_file]
    kinds = set().union(*(query_kinds(s) for s in before)) if before else set()
    covered = [k for k in KINDS if k in kinds]
    checks = {"graph_pass": {
        "value": round(0.5 * min(1.0, len(before) / 3) + 0.5 * len(covered) / len(KINDS), 3),
        "seen": (f"{len(before)} of {len(graph)} graph queries came before the first file tool; kinds covered before files: "
                 f"{', '.join(covered) or 'none'}; missing: {', '.join(k for k in KINDS if k not in kinds) or 'none'}")}}

    read = {os.path.basename(workspace_path(inp.get("file_path", ""))).lower() for name, inp, _ in uses if name == "Read"}
    must = [m["name"].lower() for m in (key or {}).get("must_find") or []]
    hits = sum(1 for m in must if m in read)
    checks["key_files_read"] = {"value": round(hits / len(must), 3) if must else 0.0,
                                "seen": f"{hits} of the {len(must)} key files were opened with Read; {len(read)} files opened in all"}

    seen_in_search = {name.lower() for tool, _, result in uses if tool in ("Grep", "Glob") for name in FILE_RX.findall(result)}
    opened = {os.path.splitext(name)[0] for name in read | seen_in_search}
    facts = [line for label, line in labelled_evidence(answer) if label == "FACT"]
    grounded = sum(1 for line in facts if cited_names(line) & opened)
    checks["facts_grounded"] = {"value": round(grounded / len(facts), 3) if facts else 0.0,
                                "seen": (f"{grounded} of {len(facts)} FACT lines cite a file the run opened or saw in a search"
                                         if facts else "the Evidence section has no FACT lines")}

    split = erdos_phases.split(events)
    one_pass = split["marker"] and split["verify"]["calls"] == 0
    checks["one_pass"] = {"value": 1.0 if one_pass else 0.0,
                          "seen": ("the answer was written once and nothing ran after the marker" if one_pass else
                                   ("no completion marker" if not split["marker"] else
                                    f"{split['verify']['calls']} calls ran after the marker"))}

    present = sections_present(answer)
    tells = len(TELL_RX.findall(answer or ""))
    parts = len(present) + (1 if split["marker"] else 0) + (1 if tells == 0 else 0)
    checks["contract"] = {"value": round(parts / (len(SECTIONS) + 2), 3),
                          "seen": (f"{len(present)} of {len(SECTIONS)} sections, marker {'present' if split['marker'] else 'missing'}, "
                                   f"{tells} subsystem ids or graph mentions in the answer")}
    return {"checks": checks, "score": round(sum(c["value"] for c in checks.values()) / len(checks), 3)}


# ----------------------------------------------------------------------------- the judge's view of the run

def trace_text(events, max_chars=16000, rows_shown=4):
    uses = tool_uses(events)
    first_file = next((i for i, (name, _, _) in enumerate(uses) if name in FILE_TOOLS), len(uses))
    graph_lines, reads, searches = [], [], []
    for i, (name, inp, result) in enumerate(uses):
        if name.startswith(GRAPH_PREFIXES):
            statement = " ".join((inp.get("statement") or json.dumps(inp)).split())
            try:
                out = json.loads(result)
            except ValueError:
                out = {"error": result[:200]}
            if out.get("error"):
                got = f"error: {str(out['error'])[:160]}"
            else:
                rows = json.dumps(out.get("rows", [])[:rows_shown], ensure_ascii=False)
                got = f"{out.get('row_count', len(out.get('rows', [])))} rows, first {min(rows_shown, len(out.get('rows', [])))}: {rows[:400]}"
            graph_lines.append(f"{len(graph_lines) + 1}. ({'before' if i < first_file else 'after'} the first file tool) "
                               f"{statement[:500]} -> {got}")
        elif name == "Read":
            reads.append(workspace_path(inp.get("file_path", "")))
        elif name in ("Grep", "Glob"):
            searches.append(f"{name} {str(inp.get('pattern', ''))[:80]}")
    before = sum(1 for line in graph_lines if "(before" in line)
    head = [f"Graph queries: {len(graph_lines)} in all, {before} before the first file tool."]
    tail = ["", f"Files opened with Read, in order ({len(reads)}): " + ", ".join(reads),
            "", f"Searches ({len(searches)}): " + "; ".join(searches)]
    text = "\n".join(head + graph_lines + tail)
    if len(text) > max_chars:                               # keep every query line, shorten what they returned
        graph_lines = [re.sub(r"first \d+: .*$", "rows omitted", line) for line in graph_lines]
        text = "\n".join(head + graph_lines + tail)[:max_chars]
    return text


def main(argv=None):
    import take_gold
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", default="erdos")
    a = ap.parse_args(argv)
    runs = json.load(open(os.path.join(HERE, "runs", f"{a.label}.json"), encoding="utf-8"))
    R = os.path.dirname(os.path.dirname(HERE))
    print("| problem | score | " + " | ".join(("graph_pass", "key_files_read", "facts_grounded", "one_pass", "contract")) + " |")
    print("|---|---|---|---|---|---|---|")
    for row in sorted((r for r in runs["rows"] if r["arm"] == a.arm), key=lambda r: r["problem"]):
        events = erdos_phases.load(os.path.join(HERE, "runs", a.label, f"{row['problem']}.{a.arm}.events.jsonl"))
        answer = open(os.path.join(R, row["answer_file"]), encoding="utf-8").read()
        res = check_run(events, answer, take_gold.load(row["problem"]))
        print(f"| {row['problem']} | {res['score']} | " + " | ".join(str(c["value"]) for c in res["checks"].values()) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
