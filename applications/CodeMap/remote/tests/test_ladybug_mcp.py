"""Direct read-only graph access: workspace paths both ways, one statement per call, and, on the real pack,
the database refusing writes, the row cap with the full count, and the parser's own error text."""

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)

from remote import ladybug_mcp as L  # noqa: E402

PACK = os.path.join(R, "graph", "pack")


class PureTests(unittest.TestCase):
    def test_paths_translate_both_ways(self):
        pairs = L.prefixes()
        stored_be = dict((short, stored) for stored, short in pairs)["backend/"]
        self.assertEqual(L.to_workspace(stored_be + "src/A.java", pairs), "backend/src/A.java")
        self.assertEqual(L.to_workspace(["x", {"p": stored_be + "B.java"}], pairs), ["x", {"p": "backend/B.java"}])
        stmt = "MATCH (e:Entity) WHERE e.file_path STARTS WITH 'backend/src/' AND e.name <> \"frontend/x\" RETURN e"
        out = L.to_stored(stmt, pairs)
        self.assertIn("'" + stored_be + "src/'", out)
        self.assertNotIn("'backend/src/'", out)
        self.assertNotIn('"frontend/x"', out)

    def test_one_statement(self):
        self.assertEqual(L.one_statement("MATCH (n) RETURN n;"), "MATCH (n) RETURN n")
        self.assertIsNone(L.one_statement("MATCH (n) RETURN n; MATCH (m) RETURN m"))
        self.assertEqual(L.one_statement("MATCH (n) WHERE n.name = 'a;b' RETURN n"), "MATCH (n) WHERE n.name = 'a;b' RETURN n")
        self.assertEqual(L.one_statement("  ;  "), "")

    def test_the_tool_contract(self):
        self.assertEqual([t["name"] for t in L.TOOLS], ["graph_query"])
        self.assertEqual(L.TOOLS[0]["inputSchema"]["required"], ["statement"])


@unittest.skipUnless(os.path.exists(os.path.join(PACK, "codemap.lbdb")), "no pack")
class LiteralScannerTests(unittest.TestCase):
    """The one-pass literal scanner that replaced a regex CodeQL flagged as polynomial on caller input."""
    OLD = __import__("re").compile(r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")")

    def test_it_finds_exactly_the_literals_the_old_pattern_found(self):
        import random
        rnd = random.Random(20260924)
        for _ in range(20000):
            s = "".join(rnd.choice("'\"\\;a ") for _ in range(rnd.randint(0, 24)))
            self.assertEqual(L.literal_spans(s), [m.span() for m in self.OLD.finditer(s)], repr(s))

    def test_hostile_input_stays_linear(self):
        import time
        for s in ("'" + "\\'" * 200000, "'a" * 200000, "\"'" * 200000):
            t0 = time.perf_counter()
            L.one_statement(s)
            self.assertLess(time.perf_counter() - t0, 2.0)


class PackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = L.GraphTools(pack_dir=PACK)

    def ask(self, stmt):
        text, err = self.tools.query(stmt)
        return json.loads(text), err

    def test_rows_use_workspace_paths_and_filters_accept_them(self):
        d, err = self.ask("MATCH (e:Entity) WHERE e.file_path STARTS WITH 'backend/src/' RETURN e.file_path ORDER BY e.file_path LIMIT 2")
        self.assertFalse(err)
        self.assertEqual(d["row_count"], 2)
        self.assertTrue(all(r[0].startswith("backend/src/") for r in d["rows"]))

    def test_the_database_refuses_writes(self):
        d, err = self.ask("CREATE (e:Entity {name:'x', file_path:'x'})")
        self.assertTrue(err)
        self.assertIn("read-only", d["error"])

    def test_caps_keep_the_full_count(self):
        d, err = self.ask("MATCH (e:Entity) RETURN e.name")
        self.assertFalse(err)
        self.assertTrue(d["truncated"])
        self.assertEqual(len(d["rows"]), L.MAX_ROWS)
        self.assertGreater(d["row_count"], L.MAX_ROWS)

    def test_a_bad_statement_returns_the_parser_message(self):
        d, err = self.ask("MATCH (e:Entity RETURN e")
        self.assertTrue(err)
        self.assertIn("Parser exception", d["error"])

    def test_mcp_round_trip(self):
        import io
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "graph_query", "arguments": {
                    "statement": "CALL show_tables() RETURN *"}}}]
        out = io.StringIO()
        L.serve(stdin=io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n"), stdout=out, tools=self.tools)
        replies = [json.loads(l) for l in out.getvalue().splitlines()]
        self.assertEqual(replies[0]["result"]["tools"][0]["name"], "graph_query")
        self.assertIn("Entity", json.dumps(replies[1]["result"]))


if __name__ == "__main__":
    unittest.main()
