"""A full reindex's change list: every eligible file in a checkout, compared with the pack.

    python graph/delta/fullscan.py --name backend --repo-dir C:/ri/backend [--pack graph/pack] --out changes.txt

Writes a name-status list the extractor understands (`A`/`M`/`D` per line) without needing the base
commit's history: a file the pack does not know is `A`, a file it knows is `M` (the extractor keeps
unchanged fingerprints as unchanged), a pack entity whose file is gone is `D`. This is the "full
reindex on the box" the churn threshold asks for when a delta would be too large — run by hand,
never in CI, and always followed by propose → apply → reclue → Release with `indexed_sha` set to the
head that was scanned.
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extract  # noqa: E402


def scan(name, repo_dir, pack_dir):
    prefix = extract.REPOS["repos"][name]["prefix"]
    ents = list(csv.DictReader(open(os.path.join(pack_dir, "entities.csv"), encoding="utf-8")))
    known = {e["file_path"] for e in ents if e["file_path"].startswith(prefix)}
    on_disk = set()
    rows = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "target", "dist", "build", ".angular")]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), repo_dir).replace("\\", "/")
            if not extract.eligible(name, rel):
                continue
            cp = prefix + rel
            on_disk.add(cp)
            rows.append(("M" if cp in known else "A", rel))
    for cp in sorted(known - on_disk):
        rows.append(("D", cp[len(prefix):]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(extract.REPOS["repos"]))
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--pack", default=os.path.join(os.path.dirname(HERE), "pack"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = scan(a.name, a.repo_dir, a.pack)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for st, rel in rows:
            f.write(f"{st}\t{rel}\n")
    counts = {k: sum(1 for st, _ in rows if st == k) for k in ("A", "M", "D")}
    print(f"fullscan {a.name}: {counts} → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
