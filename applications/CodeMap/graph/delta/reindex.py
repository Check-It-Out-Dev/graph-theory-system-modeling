"""The full reindex on the box, as the delta pipeline's own steps run by hand.

    python graph/delta/reindex.py propose --work C:/ri --delta C:/ri/out-frontend/delta.json --pack C:/ri/out-frontend/pack.next [--chunk 45]
    python graph/delta/reindex.py apply   --work C:/ri --pack <pack.next> --version 1.1.0 --by <login>
    python graph/delta/reindex.py reclue  --work C:/ri --pack <pack.next> --version 1.1.0
    python graph/delta/reindex.py release --work C:/ri --pack <pack.next> --version 1.1.0 --old graph/pack [--publish]

`propose` runs Grothendieck (`propose.py`, subscription, read-only pack MCP) over the delta's added
entities in chunks, merges the chunk proposals into `proposal.json` and runs the id checker on the
merged document. `apply` accepts it through `apply.py` (ledger row, curation notes, structural L2,
FAQ invalidation, manifest with `indexed_sha` = the scanned heads). `reclue` runs `reclue.py` over
every subsystem the apply touched. `release` runs the drift pass against the previous pack and
`build_release.py`. Every step leaves its artifact under --work; nothing here calls a model except
through the same scripts CI uses.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable


def sh(args, **kw):
    print("$", " ".join(str(a) for a in args[1:4]), "…")
    return subprocess.run([str(a) for a in args], check=True, **kw)  # NOSONAR - argv lists of our own scripts


def cmd_propose(a):
    delta = json.load(open(a.delta, encoding="utf-8"))
    unassigned = delta.get("unassigned") or []
    chunks = [unassigned[i:i + a.chunk] for i in range(0, len(unassigned), a.chunk)] or [[]]
    merged = {"assignments": [], "new_subsystems": [], "unresolved": [], "reviewed_by": None, "chunks": len(chunks), "check": []}
    for i, chunk in enumerate(chunks, 1):
        names = {os.path.basename(cp) for cp in chunk}
        d = dict(delta, unassigned=chunk, added=[x for x in (delta.get("added") or []) if x.get("name") in names])
        dp = os.path.join(a.work, f"delta-chunk-{i}.json")
        pp = os.path.join(a.work, f"proposal-chunk-{i}.json")
        json.dump(d, open(dp, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
        if not os.path.exists(pp):  # resume: a chunk already reviewed is not reviewed twice
            sh([PY, os.path.join(HERE, "propose.py"), "--delta", dp, "--pack", a.pack, "--out", pp, "--backend", a.backend])
        p = json.load(open(pp, encoding="utf-8"))
        merged["assignments"] += p.get("assignments") or []
        merged["new_subsystems"] += p.get("new_subsystems") or []
        merged["unresolved"] += p.get("unresolved") or []
        merged["reviewed_by"] = p.get("reviewed_by") or merged["reviewed_by"]
        print(f"chunk {i}/{len(chunks)}: {len(p.get('assignments') or [])} assignments, {len(p.get('new_subsystems') or [])} new subsystems, reviewed by {p.get('reviewed_by')}")
    by_entity = {}
    for row in merged["assignments"]:  # one row per entity; the last review wins
        by_entity[row["entity"]] = row
    merged["assignments"] = list(by_entity.values())
    merged["unresolved"] = sorted(set(merged["unresolved"]) - set(by_entity))
    out = os.path.join(a.work, "proposal.json")
    json.dump(merged, open(out, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    r = subprocess.run([PY, os.path.join(HERE, "propose.py"), "--check", "--proposal", out, "--pack", a.pack], capture_output=True, text=True)  # NOSONAR
    print(r.stdout.strip()[-400:], r.stderr.strip()[-400:])
    print(f"merged proposal: {len(merged['assignments'])} assignments, {len(merged['new_subsystems'])} new subsystems, "
          f"{len(merged['unresolved'])} unresolved → {out}; checker exit {r.returncode}")
    return r.returncode


def cmd_apply(a):
    out = os.path.join(a.work, "apply.json")
    sh([PY, os.path.join(HERE, "apply.py"), "--proposal", os.path.join(a.work, "proposal.json"), "--delta", a.delta, "--pack-next", a.pack,
        "--command", "/codemap accept", "--by", a.by, "--version", a.version, "--ledger", os.path.join(R, "graph", "ledger"),
        "--notes", os.path.join(R, "prompts", "navigator", "curation_notes.md"), "--out", out])
    d = json.load(open(out, encoding="utf-8"))
    print(f"applied: {d['assignments']} placed, {d['new_subsystems']} new subsystems, changed {d['changed_subsystems']}, FAQ invalidated {d['mfq_invalidated']}")
    return 0


def cmd_reclue(a):
    d = json.load(open(os.path.join(a.work, "apply.json"), encoding="utf-8"))
    subs = ",".join(str(s) for s in d["changed_subsystems"])
    sh([PY, os.path.join(HERE, "reclue.py"), "--pack", a.pack, "--subsystems", subs, "--version", a.version, "--ledger", os.path.join(R, "graph", "ledger"),
        "--notes", os.path.join(R, "prompts", "navigator", "curation_notes.md"), "--backend", a.backend, "--by", a.by, "--max", str(a.max)])
    return 0


def cmd_release(a):
    sh([PY, os.path.join(HERE, "drift.py"), "--old", a.old, "--new", a.pack, "--out", os.path.join(R, "graph", "ledger", f"{a.version}.drift.json")])
    sh([PY, os.path.join(R, "tools", "prompt", "build_navigator.py"), "--pack", a.pack, "--out", os.path.join(R, "prompts", "navigator", "active.md")])
    args = [PY, os.path.join(R, "tools", "pack", "build_release.py"), "--version", a.version, "--pack", a.pack, "--out", os.path.join(a.work, "dist"),
            "--note", f"full reindex on the box ({a.note})"]
    if a.publish:
        args.append("--publish")
    sh(args)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("propose"); p.add_argument("--work", required=True); p.add_argument("--delta", required=True); p.add_argument("--pack", required=True)
    p.add_argument("--chunk", type=int, default=25); p.add_argument("--backend", default="claude")
    p = sub.add_parser("apply"); p.add_argument("--work", required=True); p.add_argument("--delta", required=True); p.add_argument("--pack", required=True)
    p.add_argument("--version", required=True); p.add_argument("--by", default=os.environ.get("USERNAME") or "owner")
    p = sub.add_parser("reclue"); p.add_argument("--work", required=True); p.add_argument("--pack", required=True); p.add_argument("--version", required=True)
    p.add_argument("--backend", default="claude"); p.add_argument("--by", default=os.environ.get("USERNAME") or "owner"); p.add_argument("--max", type=int, default=40)
    p = sub.add_parser("release"); p.add_argument("--work", required=True); p.add_argument("--pack", required=True); p.add_argument("--version", required=True)
    p.add_argument("--old", required=True); p.add_argument("--note", default="both repositories at their public main heads"); p.add_argument("--publish", action="store_true")
    a = ap.parse_args(argv)
    return {"propose": cmd_propose, "apply": cmd_apply, "reclue": cmd_reclue, "release": cmd_release}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
