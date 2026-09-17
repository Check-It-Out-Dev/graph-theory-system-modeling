"""The Erdős system prompt, assembled: the erdos-architect skill body, then the graph map and the tool
contract it names, attached as data.

    PYTHONUTF8=1 python eval/erdos/erdos_prompt.py [--out path]      prints the size and the version

The skill body is the component a prompt optimiser changes; the graph map and the tool contract are
generated from the pack and the code (`tools/agents/sync_agents.py`) and are carried unchanged. The
version is `erdos@<sha16 of the assembled text>`, recorded with every run.
"""

import argparse
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(os.path.dirname(R))
SKILL_DIR = os.path.join(ROOT, ".agents", "skills", "erdos-architect")
COMMENT = re.compile(r"<!--.*?-->\n?", re.S)


def strip_frontmatter(text):
    text = text.replace("\r\n", "\n")
    if text.startswith("---\n"):
        text = text[text.index("\n---\n", 4) + 5:]
    return text.strip()


def skill_body(path=None):
    with open(path or os.path.join(SKILL_DIR, "SKILL.md"), encoding="utf-8") as f:
        return strip_frontmatter(f.read())


def attachment(name):
    with open(os.path.join(SKILL_DIR, "references", name), encoding="utf-8") as f:
        return COMMENT.sub("", f.read().replace("\r\n", "\n")).strip()


def assemble(body=None):
    """-> the full system prompt; `body` replaces the skill body (a candidate from the optimiser)."""
    parts = [body if body is not None else skill_body(),
             "# Attached: references/graph-map.md", attachment("graph-map.md"),
             "# Attached: references/tools.md", attachment("tools.md")]
    return "\n\n".join(parts) + "\n"


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
    print(f"{version(text)} {len(text)} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
