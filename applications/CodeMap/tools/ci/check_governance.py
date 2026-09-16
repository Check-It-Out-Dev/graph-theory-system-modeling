"""The governance-docs gate: the cards exist, carry their required sections, their relative links
resolve, the NIST AI RMF table names all four functions, and the AI Act Art. 50 transparency line is
where a user meets the system. No model, no network; runs in CI and by hand:

    python tools/ci/check_governance.py
"""

import os
import re
import sys

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED = {
    "MODEL_CARD.md": ["Model details", "Intended use", "Out-of-scope", "Training data", "Evaluation", "Limitations", "Versions"],
    "EVAL_CARD.md": ["Corpus", "Oracles", "Judge & calibration", "Metrics", "Not gated"],
    "DATA_CARD.md": ["Sources", "Collection", "PII & consent", "Retention", "Splits"],
    "THREAT_MODEL.md": ["Trust boundaries", "OWASP LLM Top 10 (2025) → controls", "Secrets", "Open risks (owner actions)"],
    "INCIDENTS.md": [],
}
NIST_FUNCTIONS = ("Govern", "Map", "Measure", "Manage")
ART50_LINE = "AI system"  # the transparency line must say so plainly
ART50_HOMES = ("remote/README.md", "app/ui.html")
LINK_RX = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")
FORBIDDEN = ("certified",)


def h2s(text):
    return [m.group(1).strip() for m in re.finditer(r"^## (.+)$", text, re.M)]


def check():
    problems = []
    for name, sections in REQUIRED.items():
        p = os.path.join(R, name)
        if not os.path.exists(p):
            problems.append(f"{name}: missing")
            continue
        text = open(p, encoding="utf-8").read()
        have = h2s(text)
        for s in sections:
            if s not in have:
                problems.append(f"{name}: missing section '## {s}'")
        if name == "INCIDENTS.md" and not re.search(r"^\| date \| what \| impact \| fix \| gate added \|", text, re.M):
            problems.append("INCIDENTS.md: the table header must be | date | what | impact | fix | gate added |")
        for word in FORBIDDEN:
            if re.search(r"\b" + word + r"\b", text, re.I) and "never \"certified\"" not in text and 'never "certified"' not in text:
                problems.append(f"{name}: says '{word}' — say 'aligned with'")
        for link in LINK_RX.findall(text):
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            target = os.path.normpath(os.path.join(os.path.dirname(p), link))
            if not os.path.exists(target):
                problems.append(f"{name}: relative link does not resolve: {link}")
    gov = os.path.join(R, "docs", "07-ai-quality-governance.md")
    text = open(gov, encoding="utf-8").read() if os.path.exists(gov) else ""
    m = re.search(r"^## NIST AI RMF.*?$", text, re.M)
    if not m:
        problems.append("docs/07: no '## NIST AI RMF' section")
    else:
        section = text[m.end():]
        nxt = re.search(r"^## ", section, re.M)
        section = section[:nxt.start()] if nxt else section
        for fn in NIST_FUNCTIONS:
            if not re.search(r"^\| \*\*" + fn + r"\*\*", section, re.M):
                problems.append(f"docs/07: NIST table lacks the {fn} row")
    for home in ART50_HOMES:
        p = os.path.join(R, home)
        if os.path.exists(p) and ART50_LINE not in open(p, encoding="utf-8").read():
            problems.append(f"{home}: no transparency line saying this is an {ART50_LINE} (EU AI Act Art. 50)")
    return problems


def main():
    problems = check()
    for p in problems:
        print("FAIL", p)
    print(f"governance-docs: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
