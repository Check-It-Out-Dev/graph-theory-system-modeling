"""The deterministic checks of the prompt-under-test instance, on committed fixtures: every reference solution passes
every rule that applies to its task, and every degraded variant (built from a reference by breaking one convention in
a real worktree, `eval/put/put_fixtures.py`) fails the rule it breaks. No model, no Maven, no repository checkout."""

import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "eval", "put"))

import put_checks  # noqa: E402
import put_contract  # noqa: E402
import put_diff  # noqa: E402

INSTANCE = "backend-conventions"
FIX = os.path.join(HERE, "fixtures", "put")


class FixtureRepo:
    """The repository at the base commit, from the fixtures' shared base/ and the recorded exception hierarchy."""

    def __init__(self):
        self.source, self.base_sha = FIX, "fixture"

    def base_text(self, path):
        p = os.path.join(FIX, "base", *path.split("/"))
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return f.read()

    def translatable_types(self):
        with open(os.path.join(FIX, "exception_hierarchy.json"), encoding="utf-8") as f:
            pairs = json.load(f)
        known, grew = {"TranslatableException"}, True
        while grew:
            grew = False
            for child, parent in pairs:
                if parent in known and child not in known:
                    known.add(child)
                    grew = True
        return known


def fixtures():
    out = []
    for name in sorted(os.listdir(FIX)):
        p = os.path.join(FIX, name, "fixture.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out.append((name, json.load(f)))
    return out


class CheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = put_contract.load(INSTANCE)
        cls.tasks = {t["id"]: t for t in put_contract.tasks(INSTANCE)}
        cls.repo = FixtureRepo()
        cls.results = {}
        for name, fx in fixtures():
            cls.results[name] = put_checks.check_run(os.path.join(FIX, name), cls.tasks[fx["task"]], cls.contract,
                                                     cls.repo, write=False)["checks"]

    def test_every_task_has_a_reference_fixture(self):
        refs = {n.split("--")[0] for n in self.results if n.endswith("--reference")}
        self.assertEqual(refs, set(self.tasks))

    def test_every_reference_passes_every_rule_that_applies(self):
        for name, checks in self.results.items():
            if not name.endswith("--reference"):
                continue
            failing = {k: v["seen"] for k, v in checks.items() if v["applicable"] and not v["passed"]}
            self.assertEqual(failing, {}, name)

    def test_every_degraded_variant_fails_the_rule_it_breaks(self):
        for name, fx in fixtures():
            if fx["variant"] == "reference":
                continue
            rule = self.results[name][fx["must_fail"]]
            ref = self.results[f"{fx['task']}--reference"][fx["must_fail"]]
            self.assertTrue(ref["passed"], f"{name}: the reference must pass {fx['must_fail']}")
            self.assertFalse(rule["passed"], f"{name}: {fx['must_fail']} should fail; saw {rule['seen']}")

    def test_a_degraded_variant_breaks_only_its_rule_among_the_conventions(self):
        conventions = set(self.contract["weight_groups"]["conventions_det"])
        for name, fx in fixtures():
            if fx["variant"] == "reference" or fx["must_fail"] not in conventions:
                continue
            others = {k: v["seen"] for k, v in self.results[name].items()
                      if k in conventions and k != fx["must_fail"] and v["applicable"] and not v["passed"]}
            self.assertEqual(others, {}, name)

    def test_rules_that_do_not_apply_are_not_scored(self):
        checks = self.results["stale-ticket-close--reference"]
        self.assertFalse(checks["liquibase_changeset"]["applicable"])
        self.assertIsNone(checks["liquibase_changeset"]["value"])

    def test_no_check_crashes_on_any_fixture(self):
        for name, checks in self.results.items():
            for k, v in checks.items():
                self.assertFalse(str(v["seen"]).startswith("CHECK ERROR"), f"{name} {k}: {v['seen']}")


class JavaReadingTests(unittest.TestCase):
    SRC = """package x;

import a.B;

/** A job. */
@Slf4j
@Component
@RequiredArgsConstructor
public class SampleCronJob {

    private final Service service;

    @Value("${sample.enabled:true}")
    private boolean enabled;

    @Scheduled(cron = "${sample.cron:0 0 3 * * *}")
    @SchedulerLock(
            name = "sample:job",
            lockAtMostFor = "30m"
    )
    public void run() {
        try {
            service.go();
        } catch (Exception e) {
            log.error("x", e);
        }
    }

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT) public void on(Event e) { }
}
"""

    def test_annotations_attach_to_the_declaration_below_them(self):
        ms = {m.name: m for m in put_checks.members(self.SRC)}
        self.assertTrue(ms["SampleCronJob"].has("RequiredArgsConstructor"))
        self.assertTrue(ms["run"].has("Scheduled"))
        self.assertIn('name = "sample:job"', ms["run"].annotation("SchedulerLock"))    # a multi-line annotation, joined
        self.assertIn("catch (Exception e)", ms["run"].body)
        self.assertTrue(ms["on"].has("TransactionalEventListener"))                    # annotation on the same line
        self.assertFalse(ms["enabled"].has("Scheduled"))

    def test_the_diff_parser_numbers_added_lines_in_the_final_file(self):
        patch = ("diff --git a/src/A.java b/src/A.java\nindex 1..2 100644\n--- a/src/A.java\n+++ b/src/A.java\n"
                 "@@ -1,3 +1,4 @@\n line1\n+added2\n line2\n line3\n"
                 "diff --git a/src/B.java b/src/B.java\nnew file mode 100644\n--- /dev/null\n+++ b/src/B.java\n@@ -0,0 +1,2 @@\n+b1\n+b2\n")
        ch = put_diff.parse(patch)
        self.assertEqual(ch["src/A.java"].status, "M")
        self.assertEqual(ch["src/A.java"].added_numbers, {2})
        self.assertEqual(ch["src/B.java"].status, "A")
        self.assertEqual(ch["src/B.java"].added, ["b1", "b2"])

    def test_tool_paths_become_repository_paths(self):
        self.assertEqual(put_checks.rel_path("C:\\Users\\N\\put-ws\\runs\\x\\src\\main\\java\\A.java"), "src/main/java/A.java")
        self.assertEqual(put_checks.rel_path("backend/src/main/java/A.java"), "src/main/java/A.java")


if __name__ == "__main__":
    unittest.main()
