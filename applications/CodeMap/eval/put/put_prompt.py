"""A prompt under test, rendered for delivery: comments dropped, `<include file="references/..."/>` lines expanded
from the instance's own `prompt/references/`, blank-line runs collapsed, then one checksum line. The coder receives
it as the CLAUDE.md of its session (the route Erdős proved: `--add-dir` + `CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD=1`),
and a preflight asks a small model for the checksum before any run, so a prompt that silently stays out of context
stops the campaign instead of producing numbers.

    PYTHONUTF8=1 python eval/put/put_prompt.py --instance backend-conventions [--version v1] [--out path]
"""

import argparse
import hashlib
import os
import re
import sys

import put_paths
import put_contract
import sync_agents        # tools/agents: expand_includes, COMMENT (the renderer the Erdős manual uses)

CHECKSUM = re.compile(r'<manual_checksum value="([0-9a-f]{16})"/>\s*\Z')


def prompt_dir(instance):
    return os.path.join(put_paths.instance_dir(instance), "prompt")


def render(body, instance):
    """-> the prompt as the coder receives it, ending with its checksum line."""
    text = sync_agents.COMMENT.sub("", body.replace("\r\n", "\n"))
    text = sync_agents.expand_includes(text, base=prompt_dir(instance))
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text + f'<manual_checksum value="{hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]}"/>\n'


def checksum(text):
    m = CHECKSUM.search(text or "")
    return m.group(1) if m else None


def sha16(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def version(body):
    """The version of a candidate is the hash of its body (the part GEPA changes), not of the rendering."""
    return "put@" + sha16(body)


def load_body(instance, name="v1"):
    return put_contract.prompt_text(instance, name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="backend-conventions")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    body = load_body(a.instance, a.version)
    text = render(body, a.instance)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print(f"{version(body)} checksum {checksum(text)} body {len(body)} chars, rendered {len(text)} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
