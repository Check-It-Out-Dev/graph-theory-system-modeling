"""Apply a decision to pack.next: the ledger row, the curation note, the FAQ invalidation, the structural
L2 fields, the manifest, and the pack ready to be released.

    python graph/delta/apply.py --proposal proposal.json --delta delta.json --pack-next <dir> --command "accept" --by <login>
                                --version 1.0.1 [--ledger graph/ledger] [--notes prompts/navigator/curation_notes.md] [--out apply.json]

Commands: `accept` · `move <Entity> to <sub>` · `new-subsystem <Name>: <A>, <B>` · `reject <reason>` ·
`accept-by-timeout`. A move or a new subsystem applies the rest of the proposal as proposed. Nothing
is invented: every entity and subsystem must exist in pack.next or be created here with >= 2 members.
Bi-temporal: the previous ledger row for an entity gets `t_invalid`, the new one `t_valid`; nothing is
deleted. FAQ entries are invalidated in the pack (`INVALIDATED_delta.json`, read by the engine beside the
curated one) when an entity MOVES between two subsystems they depend on; mere growth of a subsystem
(additions, new subsystems) invalidates only the enumerating archetypes, whose gold lists or counts members.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "tools", "pack"))

REPOS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "repos.json"), encoding="utf-8"))
ENT_COLS = ["name", "file_path", "entity_type", "subsystem", "curated", "layer", "local_height", "entry_point",
            "spines", "line_count", "fingerprint", "delta_batch"]
CMD = re.compile(r"^/?codemap\s+(accept-by-timeout|accept|reject(?:\s+(?P<reason>.*))?|move\s+(?P<ent>\S+)\s+to\s+(?P<sub>\S+)|new-subsystem\s+(?P<name>[^:]+):\s*(?P<members>.+))\s*$", re.I | re.S)


def parse_command(text):
    m = CMD.match((text or "").strip())
    if not m:
        return None
    head = m.group(1).split()[0].lower()
    if head == "accept-by-timeout":
        return {"kind": "accept", "timeout": True}
    if head == "accept":
        return {"kind": "accept"}
    if head == "reject":
        return {"kind": "reject", "reason": (m.group("reason") or "").strip()}
    if head == "move":
        return {"kind": "move", "entity": m.group("ent"), "subsystem": m.group("sub")}
    if head == "new-subsystem":
        return {"kind": "new-subsystem", "name": m.group("name").strip(),
                "members": [x.strip() for x in m.group("members").split(",") if x.strip()]}
    return None


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_pack(pack):
    ents = list(csv.DictReader(open(os.path.join(pack, "entities.csv"), encoding="utf-8")))
    edges = list(csv.DictReader(open(os.path.join(pack, "edges.csv"), encoding="utf-8")))
    l2 = []
    p = os.path.join(pack, "l2_navigators.jsonl")
    if os.path.exists(p):
        l2 = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return ents, edges, l2


def decide(proposal, command):
    """-> (assignments {entity: subsystem}, new_subsystems [ {name, members} ], rejected reason|None)"""
    if command["kind"] == "reject":
        return {}, [], command.get("reason") or "rejected"
    assign = {a["entity"]: str(a["subsystem"]) for a in proposal.get("assignments", []) if a.get("subsystem") not in (None, "")}
    new = list(proposal.get("new_subsystems") or [])
    if command["kind"] == "move":
        if command["entity"] not in assign and not any(command["entity"] == a["entity"] for a in proposal.get("assignments", [])):
            raise SystemExit(f"move: {command['entity']} is not in the proposal")
        assign[command["entity"]] = str(command["subsystem"])
    if command["kind"] == "new-subsystem":
        if len(command["members"]) < 2:
            raise SystemExit("new-subsystem needs >= 2 members")
        new.append({"name": command["name"], "members": command["members"], "why": "decided on the issue"})
        for m in command["members"]:
            assign[m] = f"new:{command['name']}"
    return assign, new, None


def structural_l2(sub_id, ents, edges, nav):
    """Structural fields of an L2 record: size, layer profile, entry points (top-5 external in-degree)."""
    members = {e["name"] for e in ents if (e.get("curated") or e.get("subsystem")) == str(sub_id)}
    ext_in = {}
    sub_of = {e["name"]: (e.get("curated") or e.get("subsystem") or "") for e in ents}
    for ed in edges:
        if ed["dst"] in members and sub_of.get(ed["src"]) not in ("", str(sub_id)):
            ext_in[ed["dst"]] = ext_in.get(ed["dst"], 0) + 1
    profile = {}
    for e in ents:
        if e["name"] in members:
            profile[e["entity_type"]] = profile.get(e["entity_type"], 0) + 1
    top = sorted(ext_in.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    nav = dict(nav)
    nav["size"] = len(members)
    nav["layer_profile"] = json.dumps(profile, sort_keys=True)
    nav["entry_points"] = json.dumps([f"{n} ({k} ext in-edges)" for n, k in top])
    nav["structure_updated_at"] = now_iso()
    return nav


ENUMERATING = {"overview", "cohort", "health", "boundary"}  # archetypes whose gold enumerates or counts members


def rel_path(file_path):
    """The repository-relative path of a canonical pack key (the pack keeps the authoring box's prefixes)."""
    fp = (file_path or "").replace("\\", "/")
    for r in REPOS.get("repos", {}).values():
        if fp.startswith(r["prefix"]):
            return fp[len(r["prefix"]):]
    return fp


def apply(proposal, delta, pack, command, by, version, ledger_dir, notes_path, mfq_inval=True):
    ents, edges, l2 = load_pack(pack)
    by_name = {e["name"]: e for e in ents}
    assign, new_subs, rejected = decide(proposal, command)
    at = now_iso()
    repo, head = delta.get("repo"), delta.get("head") or proposal.get("head") or ""
    row = {"schema": 1, "pack_version": version, "repo": repo, "head": head, "decided_by": by, "decided_at": at,
           "command": command, "t_valid": at, "t_invalid": None, "superseded_by": None,
           "assignments": [], "new_subsystems": [], "rejected": rejected, "changed_subsystems": []}
    if rejected:
        return row, []
    # new subsystems get fresh ids
    existing_ids = {int(n["sub_id"]) for n in l2 if str(n.get("sub_id", "")).lstrip("-").isdigit()}
    existing_ids |= {int(e["curated"] or e["subsystem"]) for e in ents if (e.get("curated") or e.get("subsystem") or "").isdigit()}
    next_id = max(existing_ids) + 1 if existing_ids else 1
    new_ids = {}
    for n in new_subs:
        sid = next_id
        next_id += 1
        new_ids[n["name"]] = sid
        members = n.get("members") or []
        parent = None
        for m in members:
            e = by_name.get(m)
            if e and (e.get("curated") or e.get("subsystem")):
                pn = next((x for x in l2 if str(x.get("sub_id")) == str(e.get("curated") or e.get("subsystem"))), None)
                if pn and pn.get("parent") not in (None, "", "None"):
                    parent = pn["parent"]
                    break
        nav = {"sub_id": sid, "name": n["name"], "role": "SLICE", "parent": parent, "children": [], "routable": True,
               "ai_summary": n.get("why") or f"{n['name']} (created by a delta decision on {at[:10]})",
               "responsibilities": [], "spines": [], "caveats": [], "contracts": [], "curated_by": by, "curated_at": at,
               "clue_version": "delta-structural", "clue_body_status": "STRUCTURE_ONLY", "clue_delta_batch": delta.get("date")}
        l2.append(nav)
        row["new_subsystems"].append({"id": sid, "name": n["name"], "members": members, "why": n.get("why")})
    changed, moved = set(), set()
    for entity, sub in assign.items():
        e = by_name.get(entity)
        if not e:
            raise SystemExit(f"apply: unknown entity {entity}")
        sid = str(new_ids[sub[4:]]) if sub.startswith("new:") else str(sub)
        if not sid.lstrip("-").isdigit() or (int(sid) not in existing_ids and int(sid) not in new_ids.values()):
            raise SystemExit(f"apply: unknown subsystem {sub} for {entity}")
        prev = e.get("curated") or e.get("subsystem") or ""
        e["subsystem"] = e["subsystem"] or sid
        e["curated"] = sid
        e["delta_batch"] = delta.get("date") or at[:10]
        row["assignments"].append({"entity": entity, "path": rel_path(e["file_path"]), "from": prev or None, "to": sid})
        changed.add(sid)
        if prev and prev != sid:
            changed.add(prev)
            moved |= {prev, sid}
    # previous ledger rows for these entities get t_invalid (bi-temporal)
    os.makedirs(ledger_dir, exist_ok=True)
    for name in sorted(os.listdir(ledger_dir)):
        if not name.endswith(".json"):
            continue
        p = os.path.join(ledger_dir, name)
        try:
            old = json.load(open(p, encoding="utf-8"))
        except ValueError:
            continue
        touched = False
        for a in old.get("assignments", []):
            if a["entity"] in assign and a.get("t_invalid") is None:
                a["t_invalid"] = at
                a["superseded_by"] = version
                touched = True
        if touched:
            json.dump(old, open(p, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    # structural L2 fields for changed subsystems
    l2_out = []
    for nav in l2:
        if str(nav.get("sub_id")) in changed:
            nav = structural_l2(nav["sub_id"], ents, edges, nav)
        l2_out.append(nav)
    row["changed_subsystems"] = sorted(changed, key=lambda s: int(s) if s.lstrip("-").isdigit() else 0)
    # write the pack files
    with open(os.path.join(pack, "entities.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ENT_COLS)
        w.writeheader()
        for e in ents:
            w.writerow({k: e.get(k, "") for k in ENT_COLS})
    with open(os.path.join(pack, "l2_navigators.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for nav in l2_out:
            f.write(json.dumps(nav, ensure_ascii=False) + "\n")
    # FAQ invalidation: entries depending on a changed subsystem
    invalidated = []
    mfq_path = os.path.join(pack, "mfq.jsonl")
    if mfq_inval and os.path.exists(mfq_path):
        for line in open(mfq_path, encoding="utf-8"):
            if not line.strip():
                continue
            q = json.loads(line)
            deps = {str(x) for x in (q.get("depends_on_subsystems") or [])}
            if deps & moved or (deps & changed and q.get("archetype") in ENUMERATING):
                invalidated.append(q["id"])
        inv_p = os.path.join(pack, "INVALIDATED_delta.json")
        old_inv = []
        if os.path.exists(inv_p):
            try:
                old_inv = json.load(open(inv_p, encoding="utf-8")).get("invalidated", [])
            except ValueError:
                old_inv = []
        json.dump({"invalidated": sorted(set(old_inv) | set(invalidated)), "by": "graph/delta/apply.py", "at": at,
                   "changed_subsystems": row["changed_subsystems"]}, open(inv_p, "w", encoding="utf-8", newline="\n"), indent=1)
    row["mfq_invalidated"] = invalidated
    # curation note (append-only) — the "response in the prompt"
    with open(notes_path, "a", encoding="utf-8", newline="\n") as f:
        for a in row["assignments"]:
            f.write(f"- {at[:10]} {repo}{'@' + head[:7] if head else ''} (pack {version}): {a['entity']} → subsystem {a['to']}"
                    + (f" (was {a['from']})" if a["from"] else "") + f" — decided by {by}"
                    + (" (timeout)" if command.get("timeout") else "") + "\n")
        for n in row["new_subsystems"]:
            f.write(f"- {at[:10]} {repo}@{head[:7]} (pack {version}): NEW subsystem [{n['id']}] {n['name']} = {', '.join(n['members'])} — decided by {by}\n")
    # ledger row + rebuilt lbdb + manifest
    with open(os.path.join(ledger_dir, f"{version}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(row, f, indent=1, sort_keys=True)
    sys.path.insert(0, HERE)
    import extract
    extract.build_lbdb(pack)
    import build_release
    man_prev = {}
    try:
        man_prev = json.load(open(os.path.join(pack, "manifest.json"), encoding="utf-8"))
    except (OSError, ValueError):
        pass
    indexed = dict(man_prev.get("indexed_sha") or {})
    indexed[repo] = head
    build_release.write_manifest(pack, version, indexed, note=f"delta {repo}@{head[:7]} decided by {by}: {command['kind']}")
    return row, invalidated


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", required=True)
    ap.add_argument("--delta", required=True)
    ap.add_argument("--pack-next", required=True)
    ap.add_argument("--command", required=True)
    ap.add_argument("--by", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--ledger", default=os.path.join(R, "graph", "ledger"))
    ap.add_argument("--notes", default=os.path.join(R, "prompts", "navigator", "curation_notes.md"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cmd = parse_command(a.command)
    if not cmd:
        raise SystemExit(f"not a decision: {a.command!r}")
    proposal = json.load(open(a.proposal, encoding="utf-8"))
    delta = json.load(open(a.delta, encoding="utf-8"))
    row, inval = apply(proposal, delta, a.pack_next, cmd, a.by, a.version, a.ledger, a.notes)
    if a.out:
        json.dump(row, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print(json.dumps({"version": a.version, "assignments": len(row["assignments"]), "new_subsystems": len(row["new_subsystems"]),
                      "changed_subsystems": row["changed_subsystems"], "mfq_invalidated": len(inval), "rejected": row["rejected"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
