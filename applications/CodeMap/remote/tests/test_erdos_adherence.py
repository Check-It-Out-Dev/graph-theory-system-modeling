"""Does Erdős do what his manual says: the query classifier on the manual's own recipes, each adherence check on a
synthetic run, the evidence formats answers use, and the trace the judge reads."""

import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(os.path.dirname(R))
sys.path.insert(0, os.path.join(R, "eval", "erdos"))

import erdos_adherence as A  # noqa: E402

WS = "C:\\Users\\x\\erdos-ws\\"


def _use(uid, name, inp):
    return {"type": "assistant", "message": {"id": f"m-{uid}", "content": [{"type": "tool_use", "id": uid, "name": name, "input": inp}]}}


def _result(uid, text):
    return {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": uid, "content": [{"type": "text", "text": text}]}]}}


def _text(mid, text):
    return {"type": "assistant", "message": {"id": mid, "content": [{"type": "text", "text": text}]}}


ANSWER = """## Problem
Pilot payments.
## Where it lives today
The switch.
## Proposed change
A list.
## Plan
1. Add it.
## Risks and invariants
None.
## Evidence
- FACT: the guard refuses to boot (`backend/src/Guard.java`, lines 1-9).
- FACT: the poller runs hourly (`Poller.java`).
- INFERENCE: nothing else reads the flag.

**FACT (lines read)**
- `Guard` is loaded only when the flag is off.
=== ANSWER COMPLETE ==="""

KEY = {"must_find": [{"name": "Guard.java", "path": "backend/src/Guard.java"}, {"name": "Other.java", "path": "backend/src/Other.java"}]}


def _events(graph_first=True):
    rows = json.dumps({"columns": ["e.file_path"], "rows": [["backend/src/Guard.java"]], "row_count": 1, "truncated": False})
    graph = [(1.0, _use("g1", "mcp__graph__graph_query", {"statement": "MATCH (d:Entity)-[r:Dep]->(e:Entity {file_path: 'backend/src/Guard.java'}) RETURN d.file_path"})),
             (1.1, _result("g1", rows)),
             (1.2, _use("g2", "mcp__graph__graph_query", {"statement": "MATCH (e:Entity) WHERE coalesce(e.curated, e.subsystem) = 11 AND e.entry_point RETURN e.file_path"})),
             (1.3, _result("g2", rows))]
    files = [(2.0, _use("r1", "Read", {"file_path": WS + "backend\\src\\Guard.java"})), (2.1, _result("r1", "class Guard {}"))]
    body = graph + files if graph_first else files + graph
    return [(0.0, {"type": "system", "subtype": "init"})] + body + [(3.0, _text("m9", ANSWER)),
                                                                    (3.1, {"type": "result", "subtype": "success", "is_error": False})]


class ClassifierTests(unittest.TestCase):
    def test_the_manuals_recipes_classify_as_their_intent(self):
        body = open(os.path.join(ROOT, ".agents", "skills", "erdos-architect", "SKILL.md"), encoding="utf-8").read()
        recipes = {m.group(1): m.group(2) for m in re.finditer(r'<recipe intent="([^"]+)">\n(.*?)\n', body)}
        expected = {"who depends on a file: the impact of changing it": {"dependents"},
                    "dependents within two hops, nearest first": {"dependents"},
                    "what a file depends on": {"dependencies"},
                    "entry points of a subsystem, callers first": {"entry_points"},
                    "where a subsystem couples to the others": {"coupling"},
                    "the seam between two subsystems, file by file": {"coupling"},
                    "behaviour edges around a subsystem: who performs, writes, publishes, constrains": {"behaviour"},
                    "find files by name or word": set()}
        for intent, kinds in expected.items():
            self.assertEqual(A.query_kinds(recipes[intent]), kinds, intent)


class CheckTests(unittest.TestCase):
    def test_a_run_that_follows_the_manual(self):
        res = A.check_run(_events(), ANSWER, KEY)
        c = res["checks"]
        self.assertEqual(c["graph_pass"]["value"], round(0.5 * 2 / 3 + 0.5 * 2 / 5, 3))   # two queries, two kinds before files
        self.assertIn("missing: dependencies, coupling, behaviour", c["graph_pass"]["seen"])
        self.assertEqual(c["key_files_read"]["value"], 0.5)
        self.assertEqual(c["facts_grounded"]["value"], round(2 / 3, 3))                    # Poller.java was never opened
        self.assertEqual(c["one_pass"]["value"], 1.0)
        self.assertEqual(c["contract"]["value"], 1.0)
        self.assertEqual(res["score"], round(sum(x["value"] for x in c.values()) / 5, 3))
        for check in c.values():
            self.assertNotIn("Guard", check["seen"])                                      # counts only, no code names

    def test_reading_before_querying_empties_the_graph_pass(self):
        self.assertEqual(A.check_run(_events(graph_first=False), ANSWER, KEY)["checks"]["graph_pass"]["value"], 0.0)

    def test_the_contract_counts_ids_and_graph_wording(self):
        told = ANSWER.replace("The switch.", "The switch lives in subsystem [11]; the graph shows it.")
        self.assertLess(A.check_run(_events(), told, KEY)["checks"]["contract"]["value"], 1.0)

    def test_evidence_in_a_table_keeps_its_labels(self):
        table = "## Evidence\n| claim | label |\n|---|---|\n| `Guard.java` refuses | FACT |\n| nothing else reads it | INFERENCE |\n"
        self.assertEqual([label for label, _ in A.labelled_evidence(table)], ["FACT", "INFERENCE"])


class TraceTests(unittest.TestCase):
    def test_the_trace_shows_order_rows_and_files(self):
        text = A.trace_text(_events())
        self.assertIn("Graph queries: 2 in all, 2 before the first file tool.", text)
        self.assertIn("1 rows, first 1:", text)
        self.assertIn("Files opened with Read, in order (1): backend/src/Guard.java", text)


if __name__ == "__main__":
    unittest.main()
