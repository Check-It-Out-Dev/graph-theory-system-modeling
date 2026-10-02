"""The interactive walkthrough, as a page GitHub Pages can serve.

    python tools/pages/build_walkthrough.py --out site/walkthrough

`docs/walkthrough/one-question-four-hops.html` is committed as page content only — a title, a
stylesheet, the markup and one script — because the place it was first published supplies the
document around it. Pages supplies nothing, so this wraps the committed file, unchanged, in the
few lines a browser needs to read it in standards mode: the doctype, the charset, the viewport
and the same small reset. No model, no network, no dependency. Published by `nightly.yml` under
`/walkthrough/`.
"""

import argparse
import os
import sys

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(os.path.dirname(os.path.dirname(R)), "docs", "walkthrough", "one-question-four-hops.html")

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
:root { color-scheme: light; }
body { margin: 0; font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
img { max-width: 100%; }
[hidden] { display: none !important; }
</style>
"""
TAIL = "\n</html>\n"


def build(out, source=SOURCE):
    """Write `<out>/index.html` and return its path."""
    with open(source, encoding="utf-8") as f:
        content = f.read()
    if content.lstrip().lower().startswith(("<!doctype", "<html")):
        raise ValueError(f"{source} is already a whole document; publish it as it is instead of wrapping it")
    os.makedirs(out, exist_ok=True)  # NOSONAR - operator's own path; see sonar-project.properties
    path = os.path.join(out, "index.html")
    with open(path, "w", encoding="utf-8", newline="\n") as f:  # NOSONAR - operator's own path; see sonar-project.properties
        f.write(HEAD + content + TAIL)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    path = build(args.out)
    print(f"walkthrough: {os.path.relpath(SOURCE)} -> {path} ({os.path.getsize(path)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
