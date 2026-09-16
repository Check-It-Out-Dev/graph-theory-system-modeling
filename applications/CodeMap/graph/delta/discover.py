"""Decide which product-repository heads to digest.

    python graph/delta/discover.py --event workflow_dispatch --repo backend --sha <sha> [--base <sha>] [--pr N] --out plan.json
    python graph/delta/discover.py --event schedule --out plan.json            # poll open PRs + main of both repos
    python graph/delta/discover.py --event repository_dispatch --repo backend --sha <sha> --pr 12 --out plan.json

The plan is a GitHub Actions matrix (`include` rows: repo, public, sha, base, pr). The base defaults
to the pack's indexed commit for that repository (manifest.json `indexed_sha`), falling back to the
head's parent, so the first run measures the whole distance since the pack was built. A poll skips
heads already recorded in `graph/delta/seen.json` (committed by the apply step) — nothing runs twice.
Reads the public GitHub API without a token (GITHUB_TOKEN raises the rate limit when present).
"""

import argparse
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
REPOS = json.load(open(os.path.join(HERE, "repos.json"), encoding="utf-8"))["repos"]
API = "https://api.github.com"


def gh(path):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "codemap-discover"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    with urllib.request.urlopen(urllib.request.Request(API + path, headers=headers), timeout=30) as r:  # NOSONAR - GitHub API; see sonar-project.properties
        return json.loads(r.read().decode("utf-8"))


def indexed_sha(name):
    man = os.path.join(R, "graph", "pack", "manifest.json")
    try:
        return (json.load(open(man, encoding="utf-8")).get("indexed_sha") or {}).get(name) or None
    except (OSError, ValueError):
        return None


def parent_of(public, sha):
    try:
        c = gh(f"/repos/{public}/commits/{sha}")
        return (c.get("parents") or [{}])[0].get("sha")
    except Exception:
        return None


def seen():
    p = os.path.join(HERE, "seen.json")
    try:
        return set(json.load(open(p, encoding="utf-8")).get("heads", []))
    except (OSError, ValueError):
        return set()


def row(name, sha, base=None, pr=None):
    public = REPOS[name]["public"]
    base = base or indexed_sha(name) or REPOS[name].get("indexed_sha_initial") or parent_of(public, sha) or ""
    return {"repo": name, "public": public, "sha": sha, "base": base, "pr": str(pr or "")}


def plan(event, repo=None, sha=None, base=None, pr=None, http=gh, seen_heads=None):
    seen_heads = seen() if seen_heads is None else seen_heads
    include = []
    if event in ("workflow_dispatch", "repository_dispatch"):
        if repo and sha:
            include.append(row(repo, sha, base or None, pr or None))
    else:  # schedule: open PRs of both repos, plus main when it moved
        for name, cfg in REPOS.items():
            public = cfg["public"]
            try:
                for p in http(f"/repos/{public}/pulls?state=open&per_page=20"):
                    head = p["head"]["sha"]
                    if p["head"]["repo"] and p["head"]["repo"]["full_name"] != public:
                        continue  # a fork's pull request never runs here
                    if head in seen_heads:
                        continue
                    include.append(row(name, head, p["base"]["sha"], p["number"]))
                main = http(f"/repos/{public}/commits/main")["sha"]
                if main not in seen_heads and main != indexed_sha(name):
                    include.append(row(name, main, None, None))
            except Exception as ex:
                print(f"discover: {public}: {type(ex).__name__}: {ex}", file=sys.stderr)
    return {"event": event, "include": include}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--repo", default="")
    ap.add_argument("--sha", default="")
    ap.add_argument("--base", default="")
    ap.add_argument("--pr", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    p = plan(a.event, a.repo or None, a.sha or None, a.base or None, a.pr or None)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(p, f, indent=1)
    print(json.dumps(p, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
