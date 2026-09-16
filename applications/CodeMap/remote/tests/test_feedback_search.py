"""Feedback validation matrix, the miss backlog, and search with a fake Modal (deterministic
vectors, a 303 redirect the client must follow). No network."""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)

from remote import feedback, search  # noqa: E402

PACK = os.path.join(R, "graph", "pack")
HAS_PACK = os.path.exists(os.path.join(PACK, "entities.csv")) and os.path.exists(os.path.join(PACK, "codemap.lbdb"))


def _rpc(name, mid=1, **args):
    return {"jsonrpc": "2.0", "id": mid, "method": "tools/call", "params": {"name": name, "arguments": args}}


class SeenAndValidationTests(unittest.TestCase):
    def setUp(self):
        self.seen = feedback.Seen(cap=3)
        for i in range(4):
            self.seen.observe({"event_type": "ask", "request_id": f"r{i}", "tier": "nav-sonnet", "user": "owner"})

    def test_ring_is_bounded_and_ask_only(self):
        self.assertIsNone(self.seen.get("r0"))
        self.assertEqual(self.seen.get("r3")["tier"], "nav-sonnet")
        self.seen.observe({"event_type": "feedback", "request_id": "fb"})
        self.assertIsNone(self.seen.get("fb"))

    def test_matrix(self):
        ok = lambda a: feedback.validate_feedback(a, self.seen)[0]
        self.assertIsNone(ok({"request_id": "r3", "rating": 5, "verified": True}))
        self.assertIsNone(ok({"request_id": "r3", "vote": "down", "tags": ["wrong"]}))
        self.assertIn("unknown request_id", ok({"request_id": "r0", "rating": 3}))
        self.assertIn("exactly one", ok({"request_id": "r3"}))
        self.assertIn("exactly one", ok({"request_id": "r3", "rating": 3, "vote": "up"}))
        self.assertIn("1-5", ok({"request_id": "r3", "rating": 6}))
        self.assertIn("1-5", ok({"request_id": "r3", "rating": True}))
        self.assertIn("up or down", ok({"request_id": "r3", "vote": "meh"}))
        self.assertIn("subset", ok({"request_id": "r3", "rating": 2, "tags": ["brilliant"]}))
        self.assertIn("300", ok({"request_id": "r3", "rating": 2, "comment": "x" * 301}))
        self.assertIn("verified=true", ok({"request_id": "r3", "rating": 4}))
        self.assertIsNone(ok({"request_id": "r3", "rating": 3}))


class FakeModal:
    """Embeds by keyword: a text mentioning 'consent' points one way, 'stripe' another; first
    embed POST answers 303 with a Location the client must GET."""

    def __init__(self):
        self.calls = []
        self.redirected = False

    def __call__(self, method, url, body):
        self.calls.append((method, url))
        if url.endswith("/poll"):
            return 200, {}, json.dumps({"embeddings": [self._vec(t) for t in self._pending]})
        if "embeddings" in url:
            self._pending = body["texts"]
            if not self.redirected:
                self.redirected = True
                return 303, {"Location": url + "/poll"}, ""
            return 200, {}, json.dumps({"embeddings": [self._vec(t) for t in body["texts"]]})
        if "reranker" in url:
            q = body["query"].lower()
            heads = [d.split("(")[0].lower() for d in body["documents"]]  # the entity-name part of the socket
            return 200, {}, json.dumps({"scores": [1.0 if ("consent" in q and "consent" in h) or ("stripe" in q and "stripe" in h)
                                                    else 0.01 for h in heads]})
        return 500, {}, "nope"

    @staticmethod
    def _vec(t):
        t = t.split("(")[0].lower()  # the entity-name part of the socket, not the subsystem prose
        return [1.0 if "consent" in t else 0.0, 1.0 if "stripe" in t else 0.0, 0.1]


class SocketTests(unittest.TestCase):
    def test_socket_text_is_words_not_paths(self):
        ent = {"name": "ConsentCookieService.java", "file_path": "C:/x/checkItOut-be2/src/main/java/com/sm/instagram/platform/legal/ConsentCookieService.java",
               "entity_type": "Process", "entry_point": "False"}
        t = search.socket_text(ent, {"name": "Subscriptions, payments & consent", "responsibilities": '["consent enforcement", "Stripe payments"]'})
        self.assertIn("Consent Cookie Service", t)
        self.assertIn("legal", t)
        self.assertNotIn("C:/", t)
        self.assertIn("subsystem Subscriptions", t)


@unittest.skipUnless(HAS_PACK, "graph pack absent")
class ToolsWithPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from remote import server
        cls.server = server
        cls.tmp = tempfile.mkdtemp()
        os.environ["CODEMAP_BACKLOG"] = os.path.join(cls.tmp, "backlog.jsonl")
        cls.sink = []
        cls.app = server.App(token="t", sink=cls.sink, navigator=False)
        cls.app.backlog_path = os.environ["CODEMAP_BACKLOG"]
        cls.fake = FakeModal()
        cls.app.search_index = search.SearchIndex(cls.app, client=search.ModalClient(http=cls.fake), cache_dir=cls.tmp)

    def _call(self, name, user="owner", **args):
        code, r = self.server.handle_mcp(self.app, _rpc(name, **args), {"Authorization": "Bearer t", "X-CodeMap-User": user})
        return json.loads(r["result"]["content"][0]["text"]), r["result"].get("isError", False)

    def test_feedback_end_to_end_and_rest_twin(self):
        out, err = self._call("codemap_ask", q="What does PaymentsDisabledBootGuard do and when does it stop the app from booting?")
        rid = out["request_id"]
        fb, err = self._call("codemap_feedback", request_id=rid, rating=5, verified=True, tags=["great", "pointer_verified"], comment="spot on")
        self.assertFalse(err)
        ev = self.sink[-1]
        self.assertEqual((ev["event_type"], ev["tier"], ev["rating"], ev["verified"], ev["tags"]), ("feedback", "faq", 5, True, ["great", "pointer_verified"]))
        _, err = self._call("codemap_feedback", request_id="nope", rating=1)
        self.assertTrue(err)
        code, obj = self.server.handle_feedback(self.app, {"request_id": rid, "vote": "up"}, {"Authorization": "Bearer t", "X-CodeMap-User": "haiku-pm"})
        self.assertEqual(code, 201)
        code, obj = self.server.handle_feedback(self.app, {"request_id": "ghost", "vote": "up"}, {"Authorization": "Bearer t"})
        self.assertEqual(code, 404)
        self.assertIn("codemap_feedback_total{", self.app.registry.render())

    def test_miss_goes_to_the_backlog(self):
        out, err = self._call("codemap_miss", path="backend/src/main/resources/application.yml", why="grepped for the flag")
        self.assertFalse(err)
        rows = [json.loads(l) for l in open(self.app.backlog_path, encoding="utf-8")]
        self.assertEqual(rows[-1]["path"], "backend/src/main/resources/application.yml")
        self.assertEqual((self.sink[-1]["event_type"], self.sink[-1]["repo"], rows[-1]["repo"]), ("miss", "backend", "backend"))
        _, err = self._call("codemap_miss", path="../../etc/passwd")
        self.assertTrue(err)
        self.assertIn('codemap_miss_total{repo="backend"}', self.app.registry.render())

    def test_search_builds_through_a_303_and_ranks(self):
        idx = self.app.search_index
        self.assertEqual(idx.state, "empty")
        out, err = self._call("codemap_search", text="consent cookie signing")
        self.assertTrue(err)  # building
        idx.wait(30)
        self.assertEqual(idx.state, "ready", idx.error)
        self.assertEqual(len(idx.names), len(self.app.engine.ents))
        self.assertTrue(any(url.endswith("/poll") for _, url in self.fake.calls), "303 was not followed")
        out, err = self._call("codemap_search", text="consent cookie signing", k=3)
        self.assertFalse(err, out)
        names = [p["name"] for p in out["pointers"]]
        self.assertTrue(all("onsent" in n for n in names), names)
        self.assertGreater(out["credits"], 2.0)
        self.assertEqual(self.sink[-1]["tier"], "search")
        # the cache reloads without Modal
        idx2 = search.SearchIndex(self.app, client=search.ModalClient(http=lambda *a: (500, {}, "down")), cache_dir=self.tmp)
        self.assertTrue(idx2.load())
        self.assertEqual(idx2.state, "ready")
        _, err = self._call("codemap_search", text="x", k=0)
        self.assertTrue(err)


if __name__ == "__main__":
    unittest.main()
