# CMDSL v0.1 — parser + GBNF emitter (docs/03-dsl-interface.md). 13 verbs, one expression
# per turn, no nesting. The grammar is the model's whole action space; llama.cpp GBNF makes
# invalid syntax unrepresentable.

import re

VERBS = {
    "map": 0, "enter": 1, "find": 1, "impact": (1, 2), "flow": (1, 2), "seam": 2,
    "cohort": 1, "spine": 1, "health": 1, "read": 1, "cache": 1, "answer": 1, "pass": 1,
}
# The whitespace is stripped before matching rather than inside the pattern. `\s*(.*?)\s*`
# with re.S lets `.` match spaces too, so the engine has two ways to account for every space
# around the arguments and backtracks through them -- polynomial in the length of the run
# (py/polynomial-redos, and the model is what supplies these strings). Stripping first leaves
# one parse for any input, and the recorded corpus says the verdicts are identical.
_RX = re.compile(r"^([a-z]+)\s*\((.*)\)$", re.S)


class ParseError(Exception):
    pass


def parse(expr):
    m = _RX.match((expr or "").strip())
    if not m:
        raise ParseError(f"not a CMDSL expression: {expr!r} — form: verb(args)")
    verb, raw = m.group(1), m.group(2).strip()
    if verb not in VERBS:
        raise ParseError(f"unknown verb '{verb}' — verbs: {', '.join(sorted(VERBS))}")
    args = [] if raw == "" else [a.strip().strip('"\'') for a in _split(raw)]
    arity = VERBS[verb]
    ok = (len(args) == arity) if isinstance(arity, int) else (arity[0] <= len(args) <= arity[1])
    if not ok:
        raise ParseError(f"{verb}() takes {arity} arg(s), got {len(args)}")
    return verb, args


def _split(raw):
    # split on commas not inside quotes; answer()/pass() take one free-text arg
    parts, buf, q = [], "", None
    for ch in raw:
        if q:
            buf += ch
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
            buf += ch
        elif ch == ",":
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)
    return parts


def execute(engine, expr):
    """One loop step: DSL in -> result dict out (terminal verbs set done=True)."""
    verb, args = parse(expr)
    if verb == "answer":
        return dict(kind="answer", text=args[0], done=True)
    if verb == "pass":
        return dict(kind="pass", reason=args[0], done=True,
                    note="honest abstention — logged for the risk-coverage curve")
    fn = getattr(engine, verb)
    out = fn(*args)
    out["done"] = False
    out["dsl"] = f"{verb}({', '.join(args)})"
    return out


def emit_gbnf():
    """The llama.cpp grammar: the model physically cannot emit invalid CMDSL."""
    return r'''root ::= verb0 | verb1 | verb2
verb0 ::= "map()"
verb1 ::= v1name "(" arg ")"
v1name ::= "enter" | "find" | "cohort" | "spine" | "health" | "read" | "cache" | "answer" | "pass" | "impact" | "flow"
verb2 ::= v2name "(" arg "," ws arg ")"
v2name ::= "seam" | "impact" | "flow"
arg ::= [^,()]+
ws ::= " "?
'''


if __name__ == "__main__":
    import sys
    if "--gbnf" in sys.argv:
        print(emit_gbnf())
