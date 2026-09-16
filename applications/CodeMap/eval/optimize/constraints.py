"""What a navigator prompt candidate may not lose, checked before any candidate is scored.

The optimiser edits the prose of `prompts/navigator/template.md`; the data the template renders (L1
index, L2 navigators, curation notes) is not prose and is not optimised. A candidate that drops a
placeholder, a verb of the DSL, or the answer contract's JSON block would score well on nothing and
break the runtime, so it is refused here and reported back to the reflector as feedback.
"""

import os
import re
import sys

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(R, "app"))

PLACEHOLDERS = ("{{L1_INDEX}}", "{{CAVEATS}}", "{{L2_NAVIGATORS}}", "{{CURATION_NOTES}}")
MAX_TEMPLATE_WORDS = 3000          # the template alone (the rendered prompt adds ~10k tokens of pack data)
MIN_TEMPLATE_WORDS = 150


TERMINAL_VERBS = {"answer", "pass", "cache"}  # the answer contract, not engine calls; the navigator answers in prose + JSON


def verbs():
    """The engine verbs the navigator drives through the loopback MCP: the DSL minus the terminal verbs."""
    import dsl
    return sorted(set(dsl.VERBS) - TERMINAL_VERBS)


def check(template_text):
    """-> list of problems; empty means the candidate is admissible."""
    problems = []
    t = template_text or ""
    for ph in PLACEHOLDERS:
        n = t.count(ph)
        if n != 1:
            problems.append(f"placeholder {ph} appears {n} times (must be exactly once)")
    for v in verbs():
        if not re.search(r"\b" + re.escape(v) + r"\s*\(", t):
            problems.append(f"verb {v}() missing from the verb table")
    if "```json" not in t or '"pointers"' not in t or '"terminal"' not in t:
        problems.append("the answer contract (a ```json block with terminal and pointers) is missing")
    words = len(t.split())
    if words > MAX_TEMPLATE_WORDS:
        problems.append(f"template {words} words > {MAX_TEMPLATE_WORDS}")
    if words < MIN_TEMPLATE_WORDS:
        problems.append(f"template {words} words < {MIN_TEMPLATE_WORDS}")
    if "TODO" in t or "lorem ipsum" in t.lower():
        problems.append("placeholder text")
    idx = [t.index(ph) for ph in PLACEHOLDERS if t.count(ph) == 1]
    if idx != sorted(idx):
        problems.append("placeholders out of order (L1 index, caveats, L2 navigators, curation notes)")
    return problems
