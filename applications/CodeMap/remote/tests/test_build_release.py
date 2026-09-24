"""The release keeps the commits the pack indexes: a manifest written by apply is carried, flags add on top."""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "tools", "pack"))

import build_release  # noqa: E402


class IndexedShaTests(unittest.TestCase):
    def test_carried_from_the_pack_and_extended_by_flags(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(build_release.resolve_indexed(d, []), {})
            json.dump({"indexed_sha": {"backend": "ff43730b", "frontend": None}}, open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"))
            self.assertEqual(build_release.resolve_indexed(d, []), {"backend": "ff43730b"})
            self.assertEqual(build_release.resolve_indexed(d, ["frontend=1606647d"]), {"backend": "ff43730b", "frontend": "1606647d"})
            self.assertEqual(build_release.resolve_indexed(d, ["backend=abc", "frontend="]), {"backend": "abc"})


if __name__ == "__main__":
    unittest.main()
