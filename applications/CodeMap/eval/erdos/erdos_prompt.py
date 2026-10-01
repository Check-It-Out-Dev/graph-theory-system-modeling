"""Erdős's operating manual, rendered: the erdos-architect skill body with its generated references expanded
in place, ending with a checksum line that proves the whole manual reached the model.

    PYTHONUTF8=1 python eval/erdos/erdos_prompt.py [--out path]      prints the version, the checksum and the size

The skill body (hand-written XML) is the component a prompt optimiser changes; the references (the graph tool,
the measured topology, the subsystem map) are generated from the pack by `tools/agents/sync_agents.py` and are
carried unchanged. The committed rendering is `.agents/skills/erdos-architect/CLAUDE.erdos.md`; a harness places
it as the CLAUDE.md of Erdős's starting environment. The version is `erdos@<sha16 of the rendered text>`.
"""

import argparse
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(os.path.dirname(R))
sys.path.insert(0, os.path.join(R, "tools", "agents"))

import sync_agents  # noqa: E402

SKILL_DIR = sync_agents.ERDOS
MANUAL = sync_agents.MANUAL
CHECKSUM = re.compile(r'<manual_checksum value="([0-9a-f]{16})"/>\s*\Z')


def strip_frontmatter(text):
    text = text.replace("\r\n", "\n")
    if text.startswith("---\n"):
        text = text[text.index("\n---\n", 4) + 5:]
    return text.strip()


def skill_body(path=None):
    with open(path or os.path.join(SKILL_DIR, "SKILL.md"), encoding="utf-8") as f:
        return strip_frontmatter(f.read())


def assemble(body=None):
    """-> the manual; `body` replaces the skill body (a candidate from the optimiser)."""
    return sync_agents.render_manual(body)


def checksum(text):
    """-> the checksum on the manual's last line, or None."""
    m = CHECKSUM.search(text or "")
    return m.group(1) if m else None


def version(text):
    return "erdos@" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = assemble()
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print(f"{version(text)} checksum {checksum(text)} {len(text)} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
