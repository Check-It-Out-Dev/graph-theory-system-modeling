"""Judge calibration (eval/put/put_calibrate.py) and the r5 judge input (put_judge BASE CODE): the agreement numbers on
known pairs, the gap classification against the judge's second pass, and which files a diff exposes as base code."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "eval", "put"))

import put_calibrate  # noqa: E402
import put_judge  # noqa: E402

DIFF = """diff --git a/src/main/java/a/FaqService.java b/src/main/java/a/FaqService.java
index 1..2 100644
--- a/src/main/java/a/FaqService.java
+++ b/src/main/java/a/FaqService.java
@@ -1 +1,2 @@
+    void reorder() {}
diff --git a/src/main/java/a/FaqReorderDtoIn.java b/src/main/java/a/FaqReorderDtoIn.java
new file mode 100644
index 0..3
--- /dev/null
+++ b/src/main/java/a/FaqReorderDtoIn.java
@@ -0,0 +1 @@
+class FaqReorderDtoIn {}
diff --git a/src/test/java/a/FaqServiceUnitTest.java b/src/test/java/a/FaqServiceUnitTest.java
index 4..5 100644
--- a/src/test/java/a/FaqServiceUnitTest.java
+++ b/src/test/java/a/FaqServiceUnitTest.java
@@ -1 +1,2 @@
+    @Test void x() {}
diff --git a/src/main/resources/messages_en.properties b/src/main/resources/messages_en.properties
index 6..7 100644
--- a/src/main/resources/messages_en.properties
+++ b/src/main/resources/messages_en.properties
@@ -1 +1,2 @@
+k=v
"""


class BaseCode(unittest.TestCase):
    def test_only_modified_production_java_files(self):
        self.assertEqual(put_judge.modified_sources(DIFF), ["src/main/java/a/FaqService.java"])

    def test_base_code_carries_the_unchanged_text(self):
        text = put_judge.base_code(DIFF, lambda p: "List<Faq> getFaqsByCategory(Long id) { ... }")
        self.assertIn("src/main/java/a/FaqService.java (before the change)", text)
        self.assertIn("getFaqsByCategory", text)

    def test_no_reader_no_base_code_and_the_prompt_omits_the_section(self):
        self.assertEqual(put_judge.base_code(DIFF, None), "")

    def test_per_file_cap(self):
        text = put_judge.base_code(DIFF, lambda p: "x" * (put_judge.BASE_CAP_FILE + 50))
        self.assertIn("50 characters omitted", text)

    def test_verdict_file_names_follow_the_rubric(self):
        self.assertEqual(put_judge.VERDICT, f"verdict-{put_judge.RUBRIC}.json")
        self.assertTrue(os.path.exists(put_judge.RUBRIC_FILE))


class Compare(unittest.TestCase):
    def rows(self):
        c = ("convention_fit", "design_fit", "test_quality")
        mk = lambda a, b, cc: dict(zip(c, (a, b, cc)))
        return [("A1", "baseline", mk(5, 5, 1), mk(5, 5, 1), mk(5, 5, 1)),
                ("A2", "baseline", mk(3, 5, 4), mk(4, 5, 4), mk(4, 4, 4)),     # conv gap closed by the repeat; design gap repeats
                ("A3", "fixture", mk(2, 4, 3), mk(2, 4, 3), mk(2, 2, 3))]      # design gap crosses the line and repeats

    def test_gaps_are_classified(self):
        out = put_calibrate.compare(self.rows())
        gaps = {(g["anchor"], g["criterion"]): g for g in out["gaps"]}
        self.assertEqual(set(gaps), {("A2", "convention_fit"), ("A2", "design_fit"), ("A3", "design_fit")})
        self.assertTrue(gaps[("A2", "convention_fit")]["repeat_closes"])
        self.assertTrue(gaps[("A2", "convention_fit")]["crosses_line"])
        self.assertFalse(gaps[("A2", "design_fit")]["crosses_line"])
        self.assertFalse(gaps[("A3", "design_fit")]["repeat_closes"])
        self.assertEqual(out["gap_summary"], {"cells": 9, "gaps": 3, "crossing_the_line": 2, "closed_by_the_repeat": 1})

    def test_exact_and_binary_agreement(self):
        out = put_calibrate.compare(self.rows())
        self.assertAlmostEqual(out["criteria"]["test_quality"]["exact"], 1.0)
        self.assertAlmostEqual(out["criteria"]["convention_fit"]["agreement"], 2 / 3, places=3)
        self.assertEqual(out["pooled"]["n"], 9)


if __name__ == "__main__":
    unittest.main()
