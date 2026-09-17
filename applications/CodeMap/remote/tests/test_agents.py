"""The `.agents/` folder: skills that follow the Agent Skills specification, generated files that
match their sources, a dialect reference that carries the pack's own notes, and a reviewer prompt
that carries the whole Grothendieck skill instead of a cut of the Neo4j-era manual."""

import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(os.path.dirname(R))
sys.path.insert(0, os.path.join(R, "tools", "agents"))
sys.path.insert(0, os.path.join(R, "graph", "delta"))
sys.path.insert(0, os.path.join(R, "app"))

import sync_agents  # noqa: E402

SKILLS = os.path.join(ROOT, ".agents", "skills")
PACK = os.path.join(R, "graph", "pack")
NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class SkillSpecTests(unittest.TestCase):
    def test_every_skill_follows_the_agent_skills_specification(self):
        names = sorted(d for d in os.listdir(SKILLS) if os.path.isdir(os.path.join(SKILLS, d)))
        self.assertEqual(names, ["erdos-architect", "grothendieck-organizer", "hypatia-indexer", "ladybug-graph"])
        for name in names:
            fields, body = sync_agents.skill(name)
            self.assertEqual(fields.get("name"), name, name)                      # name equals the folder
            self.assertRegex(name, NAME)
            self.assertLessEqual(len(name), 64)
            self.assertTrue(0 < len(fields.get("description", "")) <= 1024, name)
            self.assertNotIn('"', fields["description"], name)                    # shims quote it
            self.assertLessEqual(len(fields.get("compatibility", "")), 500, name)
            self.assertLess(len(body.splitlines()), 500, name)                    # progressive disclosure
            for ref in sorted(set(re.findall(r"`((?:\.agents/skills/[a-z-]+/)?references/[a-z-]+\.md)`", body))):
                path = os.path.join(ROOT, ref) if ref.startswith(".agents") else os.path.join(SKILLS, name, ref)
                self.assertTrue(os.path.exists(path), f"{name} names {ref}, which does not exist")

    def test_the_frontmatter_parser(self):
        fields, body = sync_agents.split_frontmatter('---\nname: x\ndescription: "a: b"\nmetadata:\n  version: "1.0"\n---\nbody\n')
        self.assertEqual(fields, {"name": "x", "description": "a: b", "metadata": {"version": "1.0"}})
        self.assertEqual(body, "body\n")


class GeneratedFilesTests(unittest.TestCase):
    def test_generated_files_match_their_sources(self):
        files, skipped = sync_agents.targets(PACK)
        for path, text in sorted(files.items()):
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read().replace("\r\n", "\n"), text, f"{os.path.relpath(path, ROOT)} is stale: run sync_agents.py")
        if skipped:
            self.skipTest("; ".join(skipped))

    def test_every_shim_reads_files_that_exist(self):
        for name, reads in sync_agents.AGENTS.items():
            for rel in reads:
                self.assertTrue(os.path.exists(os.path.join(ROOT, rel)), f"{name} reads {rel}")


class DialectAndReviewerTests(unittest.TestCase):
    def test_the_dialect_reference_carries_the_packs_generated_notes(self):
        notes = os.path.join(PACK, "DIALECT_NOTES.md")
        if not os.path.exists(notes):
            self.skipTest("no pack")
        dialect = open(os.path.join(SKILLS, "ladybug-graph", "references", "dialect.md"), encoding="utf-8").read()
        for line in open(notes, encoding="utf-8"):
            if line.startswith("- "):
                self.assertIn(line.strip(), dialect)

    def test_the_reviewer_gets_the_whole_grothendieck_skill(self):
        import propose
        manual = propose.manual_text()
        self.assertFalse(manual.startswith("---"))
        self.assertIn("## Procedure — placement", manual)
        self.assertIn("## Proposal contract", manual)
        prompt = propose.review_prompt({"repo": "backend", "date": "d", "mode": "delta", "churn": 0.0, "counts": {}}, [], PACK)
        self.assertIn(manual, prompt)
        self.assertNotIn("GrothendieckV5 in MODE delta", prompt)


if __name__ == "__main__":
    unittest.main()
