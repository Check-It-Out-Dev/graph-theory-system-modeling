"""Fetch a graph pack Release into graph/pack (CI, the VPS, a teammate's clone — the same command).

    python tools/pack/fetch_pack.py --latest            # newest pack-* release
    python tools/pack/fetch_pack.py --version 1.0.0
    python tools/pack/fetch_pack.py --check             # is the local pack the latest? (exit 3 when not)

Verifies the tarball against SHA256SUMS before extracting; refuses paths that escape the
destination; leaves a manifest.json whose pack_version the server reports. Stdlib only.
GITHUB_TOKEN (optional) raises the API rate limit.
"""

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.environ.get("CODEMAP_PACK_REPO", "Check-It-Out-Dev/graph-theory-system-modeling")
API = "https://api.github.com"


def _get(url, accept="application/vnd.github+json", timeout=60):
    headers = {"Accept": accept, "User-Agent": "codemap-fetch-pack"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:  # NOSONAR - GitHub API over https; see sonar-project.properties
        return r.read()


def resolve(version=None, repo=REPO):
    """-> release dict for pack-<version>, or the newest pack-* release."""
    if version:
        return json.loads(_get(f"{API}/repos/{repo}/releases/tags/pack-{version}"))
    rels = json.loads(_get(f"{API}/repos/{repo}/releases?per_page=50"))
    packs = [r for r in rels if r.get("tag_name", "").startswith("pack-") and not r.get("draft")]
    if not packs:
        raise SystemExit("no pack-* release found")
    packs.sort(key=lambda r: [int(x) if x.isdigit() else 0 for x in r["tag_name"][5:].split(".")], reverse=True)
    return packs[0]


def asset(release, name_suffix):
    for a in release.get("assets", []):
        if a["name"].endswith(name_suffix):
            return a
    raise SystemExit(f"release {release.get('tag_name')} has no asset ending in {name_suffix}")


def fetch(release, dest, repo=REPO):
    version = release["tag_name"][5:]
    tar_asset = asset(release, ".tar.gz")
    sums = _get(asset(release, "SHA256SUMS")["browser_download_url"], accept="application/octet-stream").decode("utf-8")
    expected = None
    for line in sums.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == tar_asset["name"]:
            expected = parts[0]
    if not expected:
        raise SystemExit("SHA256SUMS does not list the tarball")
    blob = _get(tar_asset["browser_download_url"], accept="application/octet-stream", timeout=300)
    got = hashlib.sha256(blob).hexdigest()
    if got != expected:
        raise SystemExit(f"sha256 mismatch: expected {expected}, got {got}")
    os.makedirs(dest, exist_ok=True)
    dest_abs = os.path.abspath(dest)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            target = os.path.abspath(os.path.join(dest_abs, m.name))
            if not m.isfile() or os.path.dirname(target) != dest_abs or m.name.startswith((".", "/")):
                raise SystemExit(f"refusing tar member {m.name!r}")
        tar.extractall(dest_abs, members=[m for m in tar.getmembers() if m.isfile()], filter="data")
    man = json.load(open(os.path.join(dest, "manifest.json"), encoding="utf-8"))
    return version, man, got


def local_version(dest):
    p = os.path.join(dest, "manifest.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8")).get("pack_version")
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--latest", action="store_true")
    g.add_argument("--version")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--dest", default=os.path.join(R, "graph", "pack"))
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    try:
        rel = resolve(None if (a.latest or a.check) else a.version, a.repo)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"GitHub API {e.code} for {a.repo}: {e.reason}")
    if a.check:
        latest = rel["tag_name"][5:]
        local = local_version(a.dest)
        print(f"local pack {local or 'none'}, latest release {latest}")
        return 0 if local == latest else 3
    if not a.latest and not a.version:
        a.latest = True
    version, man, digest = fetch(rel, a.dest, a.repo)
    print(f"fetched pack {version} into {a.dest}: {json.dumps(man.get('counts'))} sha256 {digest[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
