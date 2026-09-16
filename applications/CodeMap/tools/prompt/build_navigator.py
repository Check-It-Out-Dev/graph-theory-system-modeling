"""Build the navigator system prompt: template prose + pack data + curation notes -> prompts/navigator/v<N>.md

    PYTHONUTF8=1 python tools/prompt/build_navigator.py [--version 1] [--template prompts/navigator/template.md]
                                                        [--pack graph/pack] [--check]

The template is what a prompt optimiser may change; the L1 index and L2 navigators are data
rendered from the pack (changing them is a reindex, not a prompt edit); the curation notes are
appended by the delta pipeline. `--check` rebuilds and fails when the committed file differs,
so a stale prompt cannot ship unnoticed. Deterministic: same inputs, same bytes.
"""

import argparse
import hashlib
import json
import os
import sys

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(R, "app"))

NAV_FIELDS = ("role", "ai_summary", "responsibilities", "entry_points", "spines", "caveats")


def sha16(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def render_l2(l2):
    """Compact prose per subsystem, sorted by id; JSON-encoded string fields are decoded first."""
    lines = []
    for sid in sorted(l2, key=lambda k: int(k)):
        nav = l2[sid]
        head = f"[{sid}] {nav.get('name', '?')} ({nav.get('role', '?')}, {nav.get('size', '?')} files"
        if nav.get("parent") not in (None, "", "None"):
            head += f", in group {nav.get('parent')}"
        head += ")"
        lines.append(head)
        summary = nav.get("ai_summary")
        if summary:
            lines.append(f"  summary: {summary}")
        resp = _list(nav.get("responsibilities"))
        if resp:
            lines.append("  does: " + "; ".join(str(x) for x in resp[:6]))
        eps = _list(nav.get("entry_points"))
        if eps:
            lines.append("  entry points: " + ", ".join(str(x) for x in eps[:6]))
        spines = _list(nav.get("spines"))
        if spines:
            lines.append("  spines: " + "; ".join(str(x)[:120] for x in spines[:3]))
        cav = _list(nav.get("caveats"))
        if cav:
            lines.append("  caveats: " + "; ".join(str(x)[:160] for x in cav[:3]))
    return "\n".join(lines)


def _list(v):
    if v in (None, "", "[]"):
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return [v]
    return list(v) if isinstance(v, (list, tuple)) else [v]


def build(template_path, pack_dir, notes_path):
    from engine import Engine
    if pack_dir:
        os.environ["CODEMAP_PACK_DIR"] = pack_dir
    engine = Engine(pack_dir=pack_dir)
    l1 = engine.map()
    template = open(template_path, encoding="utf-8").read()
    # the notes ride the pack (a Release carries the notes its decisions wrote); the repo file is the fallback
    pack_notes = os.path.join(pack_dir, "curation_notes.md") if pack_dir else None
    if pack_notes and os.path.exists(pack_notes):
        notes_path = pack_notes
    notes = open(notes_path, encoding="utf-8").read().strip() if os.path.exists(notes_path) else ""
    caveats = l1.get("caveats")
    if isinstance(caveats, (list, tuple)):
        caveats = " | ".join(str(c) for c in caveats)
    text = (template.replace("{{L1_INDEX}}", (l1.get("index") or "").strip())
                    .replace("{{CAVEATS}}", str(caveats or "none"))
                    .replace("{{L2_NAVIGATORS}}", render_l2(engine.l2))
                    .replace("{{CURATION_NOTES}}", notes))
    return text.replace("\r\n", "\n"), sha16(template)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--template", default=os.path.join(R, "prompts", "navigator", "template.md"))
    ap.add_argument("--notes", default=os.path.join(R, "prompts", "navigator", "curation_notes.md"))
    ap.add_argument("--pack", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(R, "prompts", "navigator", f"v{a.version}.md")
    text, tsha = build(a.template, a.pack, a.notes)
    if a.check:
        cur = open(out, encoding="utf-8").read().replace("\r\n", "\n") if os.path.exists(out) else ""
        if cur != text:
            print(f"STALE: {out} differs from a fresh build (template {tsha})")
            return 1
        print(f"fresh: {out} nav@{sha16(text)} (template {tsha}, {len(text)} chars)")
        return 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {out}: nav@{sha16(text)} template@{tsha} {len(text)} chars ~{len(text)//4} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
