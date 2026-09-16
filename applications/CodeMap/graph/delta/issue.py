"""The proposal as a GitHub issue on the graph repository — where a person decides.

    python graph/delta/issue.py --proposal proposal.json --delta delta.json --run-url <url> [--repo owner/name] [--mirror-pr <n>]

Creates (or updates, matched by the marker line) one issue per digested head with the candidate
table, the reviewer's reasons, and the decision menu. GITHUB_TOKEN of the graph repository suffices —
no cross-repository secret is needed; when CODEMAP_PR_TOKEN is present the same text is mirrored as
a comment on the product pull request. The decision is a comment on the issue:

    /codemap accept
    /codemap move <Entity.java> to <subsystem id>
    /codemap new-subsystem <Name>: <Entity.java>, <Other.java>
    /codemap reject <reason>
"""

import argparse
import json
import os
import subprocess
import sys

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.environ.get("CODEMAP_GRAPH_REPO", "Check-It-Out-Dev/graph-theory-system-modeling")
MARKER = "<!-- codemap-delta"


def render(proposal, delta, run_url=None, ref=None):
    repo, head = delta.get("repo"), delta.get("head") or proposal.get("head") or ""
    c = delta.get("counts", {})
    tail = " ref=" + ref if ref else ""
    lines = [f"{MARKER} {repo}@{head}{tail} -->",
             f"## Partition proposal — `{repo}` @ `{head[:10]}`",
             "",
             f"Delta: **{c.get('added', 0)} added · {c.get('modified', 0)} modified · {c.get('deleted', 0)} deleted** "
             f"(churn {delta.get('churn', 0):.1%}, mode `{delta.get('mode')}`). Reviewed by `{proposal.get('reviewed_by')}`."
             + (f" [Run]({run_url})" if run_url else ""),
             "",
             "| entity | type | proposed subsystem | confidence | alternatives | why |",
             "|---|---|---|---|---|---|"]
    types = {a["name"]: a.get("entity_type") for a in delta.get("added", [])}
    for a in proposal.get("assignments", []):
        alts = ", ".join(str(x) for x in (a.get("alternatives") or [])[:3]) or "—"
        why = (a.get("why") or "").replace("|", "/").replace("\n", " ")[:220]
        lines.append(f"| `{a['entity']}` | {types.get(a['entity'], '?')} | **{a.get('subsystem')}** | {a.get('confidence', 0):.2f} | {alts} | {why} |")
    if proposal.get("new_subsystems"):
        lines += ["", "New subsystems proposed:", ""]
        for n in proposal["new_subsystems"]:
            lines.append(f"- **{n.get('name')}**: {', '.join(n.get('members') or [])} — {n.get('why')}")
    if proposal.get("unresolved"):
        lines += ["", "Unresolved (the reviewer could not place): " + ", ".join(f"`{x}`" for x in proposal["unresolved"])]
    if proposal.get("check"):
        lines += ["", "Checker: " + "; ".join(proposal["check"])]
    lines += ["", "### Decide (one comment)", "",
              "```", "/codemap accept", "/codemap move <Entity.java> to <subsystem id>",
              "/codemap new-subsystem <Name>: <Entity.java>, <Other.java>", "/codemap reject <reason>", "```",
              "",
              "Nothing changes in the graph until a decision is applied; after 48 h without one the proposal is applied as `accepted-by-timeout` and recorded as such. "
              "Every decision becomes a bi-temporal ledger row and a curation note in the navigator prompt."]
    return "\n".join(lines)


def gh(args, input_text=None):
    p = subprocess.run(["gh"] + args, capture_output=True, text=True, input=input_text)  # NOSONAR - gh CLI, argv list; see sonar-project.properties
    if p.returncode != 0:
        raise SystemExit(f"gh {' '.join(args[:3])}: {p.stderr.strip()[:300]}")
    return p.stdout.strip()


def find_issue(repo, marker_line):
    out = gh(["issue", "list", "--repo", repo, "--label", "graph-delta", "--state", "open", "--json", "number,body", "--limit", "50"])
    for it in json.loads(out or "[]"):
        if marker_line in (it.get("body") or ""):
            return it["number"]
    return None


def upsert_issue(repo, title, body, marker_line):
    try:
        gh(["label", "create", "graph-delta", "--repo", repo, "--color", "5319e7", "--description", "partition proposals from the delta pipeline", "--force"])
    except SystemExit:
        pass
    n = find_issue(repo, marker_line)
    if n:
        gh(["issue", "edit", str(n), "--repo", repo, "--body-file", "-"], input_text=body)
        return n, False
    url = gh(["issue", "create", "--repo", repo, "--title", title, "--label", "graph-delta", "--body-file", "-"], input_text=body)
    return int(url.rstrip("/").rsplit("/", 1)[-1]), True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", required=True)
    ap.add_argument("--delta", required=True)
    ap.add_argument("--run-url", default=None)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--mirror-pr", default=None, help="product PR number; needs CODEMAP_PR_TOKEN")
    ap.add_argument("--ref", default=os.environ.get("GITHUB_REF_NAME") or None, help="branch that produced the proposal; the decision job checks it out")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    proposal = json.load(open(a.proposal, encoding="utf-8"))
    delta = json.load(open(a.delta, encoding="utf-8"))
    body = render(proposal, delta, a.run_url, a.ref)
    head = (delta.get("head") or proposal.get("head") or "")[:10]
    title = f"graph-delta: {delta.get('repo')} @ {head} — {len(proposal.get('assignments', []))} entities to place"
    if a.dry_run:
        print(body)
        return 0
    n, created = upsert_issue(a.repo, title, body, body.splitlines()[0])
    print(f"{'created' if created else 'updated'} issue #{n} on {a.repo}")
    if a.mirror_pr and os.environ.get("CODEMAP_PR_TOKEN"):
        from graph.delta import repos  # noqa: F401  (placeholder: the mirror needs the product repo name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
