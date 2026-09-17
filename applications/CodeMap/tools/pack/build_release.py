"""Turn a pack directory into a versioned GitHub Release asset.

    python tools/pack/build_release.py --version 1.0.0 [--pack graph/pack] [--out dist]
                                        [--indexed backend=<sha> --indexed frontend=<sha>] [--publish]

Writes manifest.json (schema 2) INTO the pack dir, then dist/codemap-pack-<v>.tar.gz (pack files at
the top level), dist/SHA256SUMS and dist/RELEASE_NOTES.md; --publish runs `gh release create
pack-<v>`. The pack never enters git: the Release is its home, and the manifest's hashes are how
anyone (CI, the VPS, a teammate) proves they hold the same graph.
"""

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACK_FILES = ["codemap.lbdb", "codemap_vocab.gbnf", "entities.csv", "edges.csv", "edges_lb.csv", "hyperedges.csv",
              "l1_master.json", "l2_navigators.jsonl", "mfq.jsonl", "DIALECT_NOTES.md"]
OPTIONAL_FILES = ["INVALIDATED_delta.json", "curation_notes.md"]
REPO = "Check-It-Out-Dev/graph-theory-system-modeling"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def counts(pack):
    def lines(name):
        p = os.path.join(pack, name)
        if not os.path.exists(p):
            return 0
        with open(p, encoding="utf-8") as f:
            return sum(1 for _ in f)
    return {"entities": max(0, lines("entities.csv") - 1), "edges": max(0, lines("edges.csv") - 1),
            "hyperedges": max(0, lines("hyperedges.csv") - 1), "l2": lines("l2_navigators.jsonl"), "mfq": lines("mfq.jsonl")}


def resolve_indexed(pack, pairs):
    """The commits the pack indexes: the pack's own manifest first (apply wrote the heads it scanned),
    then any `repo=sha` given on the command line on top. Without the carry-over a release rewrote the
    manifest with an empty map and the delta job lost its base (packs 1.0.1 and 1.1.0 shipped `{}`)."""
    indexed = {}
    try:
        with open(os.path.join(pack, "manifest.json"), encoding="utf-8") as f:
            indexed = {k: v for k, v in (json.load(f).get("indexed_sha") or {}).items() if v}
    except (OSError, ValueError):
        pass
    for kv in pairs or []:
        k, _, v = kv.partition("=")
        if v:
            indexed[k] = v
    return indexed


def write_manifest(pack, version, indexed, note=None):
    files = {}
    for name in PACK_FILES + OPTIONAL_FILES:
        p = os.path.join(pack, name)
        if not os.path.exists(p):
            if name in OPTIONAL_FILES:
                continue
            raise SystemExit(f"pack file missing: {name}")
        files[name] = sha256(p)[:16]
    man = {"schema_version": "2.0", "pack_version": version, "codebase": "checkItOut",
           "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
           "indexed_sha": indexed, "counts": counts(pack), "files": files,
           "dialect": "ladybug (real_ladybug 0.15.3)",
           "embedder": {"note": "no embeddings in the pack (authoring only, SCHEMA.md section 4); the remote server "
                                "builds its search index per pack version from entity sockets"},
           "note": note or "built by tools/pack/build_release.py"}
    with open(os.path.join(pack, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1, sort_keys=True)
        f.write("\n")
    return man


def build(pack, version, out, indexed, note=None):
    man = write_manifest(pack, version, indexed, note)
    os.makedirs(out, exist_ok=True)
    tar_path = os.path.join(out, f"codemap-pack-{version}.tar.gz")
    with tarfile.open(tar_path, "w:gz", compresslevel=6) as tar:
        for name in PACK_FILES + OPTIONAL_FILES + ["manifest.json"]:
            if os.path.exists(os.path.join(pack, name)):
                tar.add(os.path.join(pack, name), arcname=name)
    digest = sha256(tar_path)
    with open(os.path.join(out, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{digest}  codemap-pack-{version}.tar.gz\n")
    notes = io.StringIO()
    notes.write(f"# CodeMap graph pack {version}\n\n")
    notes.write("The runtime graph of the checkItOut repositories (backend + frontend) as CodeMap serves it: a LadybugDB "
                "database, the entity/edge/hyperedge CSVs, the L1 index, the L2 navigators, the FAQ bank and the grammar. "
                "Fetch with `python applications/CodeMap/tools/pack/fetch_pack.py --version " + version + "`.\n\n")
    notes.write("| | |\n|---|---|\n")
    for k, v in man["counts"].items():
        notes.write(f"| {k} | {v} |\n")
    notes.write(f"| indexed commits | {json.dumps(indexed)} |\n")
    notes.write(f"| tarball sha256 | `{digest}` |\n")
    notes.write(f"| built | {man['built_at']} |\n\n")
    notes.write("Files (sha256[:16]):\n\n")
    for k, v in man["files"].items():
        notes.write(f"- `{k}` `{v}`\n")
    with open(os.path.join(out, "RELEASE_NOTES.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(notes.getvalue())
    return tar_path, digest, man


def publish(version, out, repo=REPO):
    tag = f"pack-{version}"
    cmd = ["gh", "release", "create", tag, os.path.join(out, f"codemap-pack-{version}.tar.gz"),
           os.path.join(out, "SHA256SUMS"), "--repo", repo, "--title", f"CodeMap graph pack {version}",
           "--notes-file", os.path.join(out, "RELEASE_NOTES.md")]
    subprocess.run(cmd, check=True)  # NOSONAR - gh CLI, argv list; see sonar-project.properties
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--pack", default=os.path.join(R, "graph", "pack"))
    ap.add_argument("--out", default=os.path.join(R, "dist"))
    ap.add_argument("--indexed", action="append", default=[], help="repo=sha (repeatable)")
    ap.add_argument("--note", default=None)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    indexed = resolve_indexed(a.pack, a.indexed)
    tar_path, digest, man = build(a.pack, a.version, a.out, indexed, a.note)
    print(f"built {tar_path} ({os.path.getsize(tar_path)//1024} KB) sha256 {digest}")
    print(json.dumps(man["counts"]))
    if a.publish:
        print("published", publish(a.version, a.out, a.repo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
