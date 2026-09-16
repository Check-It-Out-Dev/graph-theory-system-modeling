"""The partition proposal: a deterministic candidate per unassigned entity, then (optionally) Grothendieck
on the subscription reviewing it, then a checker that refuses invented ids, then an issue on the graph
repository where a person decides.

    python graph/delta/propose.py --delta delta/<run>/delta.json --pack delta/<run>/pack.next --out delta/<run>/proposal.json
                                  [--backend claude|none] [--model claude-sonnet-5]
    python graph/delta/propose.py --check --proposal proposal.json --pack <pack.next>       # exit 2 on an invented id

Candidate rule (GrothendieckV5 delta, made mechanical): the subsystem shared by the most structural
neighbours (edges in and out, weighted: INJECTS/EXTENDS/IMPLEMENTS 2, IMPORTS 1, TESTED_BY 1), then the
folder's majority subsystem as the tiebreak; confidence = the winner's share. A margin below 0.6 is
flagged for the reviewer. The model may keep, move or propose a new subsystem, and must say why.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "app"))
sys.path.insert(0, HERE)

WEIGHT = {"INJECTS": 2, "EXTENDS": 2, "IMPLEMENTS": 2, "IMPORTS": 1, "TESTED_BY": 1, "CALLS": 2, "USES": 1, "MODIFIES": 1}
SCHEMA = {"type": "object", "required": ["assignments"], "properties": {
    "assignments": {"type": "array", "items": {"type": "object", "required": ["entity", "subsystem", "confidence", "why"],
                                               "properties": {"entity": {"type": "string"}, "subsystem": {"type": ["integer", "string"]},
                                                              "confidence": {"type": "number"}, "why": {"type": "string"},
                                                              "alternatives": {"type": "array", "items": {"type": ["integer", "string"]}}}}},
    "new_subsystems": {"type": "array", "items": {"type": "object", "required": ["name", "members", "why"]}},
    "unresolved": {"type": "array", "items": {"type": "string"}}}}


def candidates(pack, delta):
    from pack_mcp import Pack
    P = pack if hasattr(pack, "ents") else Pack(pack)
    sub_of = lambda n: (P.by_name.get(n) or {}).get("curated") or (P.by_name.get(n) or {}).get("subsystem") or ""
    out = []
    for cp in delta.get("unassigned", []):
        name = os.path.basename(cp)
        votes = {}
        for d, rel in P.out_e.get(name, []):
            s = sub_of(d)
            if s:
                votes[s] = votes.get(s, 0) + WEIGHT.get(rel, 1)
        for s_, rel in P.in_e.get(name, []):
            s = sub_of(s_)
            if s:
                votes[s] = votes.get(s, 0) + WEIGHT.get(rel, 1)
        folder = cp.rsplit("/", 1)[0]
        fmix = {}
        for e in P.ents:
            if e["file_path"].rsplit("/", 1)[0] == folder and e["name"] != name:
                s = e.get("curated") or e.get("subsystem")
                if s:
                    fmix[s] = fmix.get(s, 0) + 1
        total = sum(votes.values())
        if votes:
            best = max(votes, key=lambda k: (votes[k], fmix.get(k, 0)))
            conf = round(votes[best] / total, 3)
            rule = "structural-neighbours"
        elif fmix:
            best = max(fmix, key=fmix.get)
            conf = round(fmix[best] / sum(fmix.values()), 3)
            rule = "folder-majority"
        else:
            best, conf, rule = None, 0.0, "none"
        alts = sorted([k for k in set(votes) | set(fmix) if k != best], key=lambda k: -(votes.get(k, 0) + fmix.get(k, 0)))[:3]
        out.append({"entity": name, "file_path": cp, "subsystem": best, "confidence": conf, "rule": rule,
                    "votes": votes, "folder_mix": fmix, "alternatives": alts, "queued": conf < 0.6 or best is None,
                    "why": f"{rule}: {votes if votes else fmix}"})
    return out


def review_prompt(delta, cands, pack_dir):
    manual = open(os.path.join(R, "graph", "prompts", "GrothendieckV5.md"), encoding="utf-8").read()
    lines = ["You are GrothendieckV5 in MODE delta. The deterministic candidates below were computed from the structural edges "
             "and folders of pack.next; you have read-only tools over that pack (pack_subsystem, pack_entity, pack_folder, pack_cypher). "
             "For each unassigned entity: keep the candidate, move it to an alternative, or propose a new subsystem (only with >= 2 "
             "members and a one-line reason). Use only subsystem ids that exist (pack_subsystem answers) and entity names from the "
             "candidates. Reply ONLY with the JSON object described; every `why` is one sentence.",
             "", "## Delta", json.dumps({k: delta[k] for k in ("repo", "date", "mode", "churn", "counts")}),
             "", "## Candidates", json.dumps(cands, ensure_ascii=False, indent=1),
             "", "## Your operating manual (delta procedure)", manual[:6000]]
    return "\n".join(lines)


def review(delta, cands, pack_dir, backend="claude", model="claude-sonnet-5", runner=None):
    if backend == "none" or not cands:
        return {"assignments": [{"entity": c["entity"], "subsystem": c["subsystem"], "confidence": c["confidence"],
                                 "alternatives": c["alternatives"], "why": c["why"]} for c in cands],
                "new_subsystems": [], "unresolved": [c["entity"] for c in cands if c["subsystem"] is None],
                "reviewed_by": "deterministic"}
    import claude_cli
    cfg = {"mcpServers": {"pack": {"command": sys.executable, "args": [os.path.join(HERE, "pack_mcp.py")],
                                   "env": {"CODEMAP_PACK_DIR": os.path.abspath(pack_dir), "PYTHONUTF8": "1"}}}}
    res = claude_cli.run(review_prompt(delta, cands, pack_dir), model, role="grothendieck", mcp_config=cfg,
                         allowed_tools=("mcp__pack__*",), max_turns=20, json_schema=SCHEMA, timeout=600, runner=runner)
    doc = res.get("structured")
    if not doc and res.get("text"):
        import re
        m = re.search(r"\{.*\}", res["text"], re.S)
        if m:
            try:
                doc = json.loads(m.group(0))
            except ValueError:
                doc = None
    if not isinstance(doc, dict) or "assignments" not in doc:
        doc = review(delta, cands, pack_dir, backend="none")
        doc["reviewed_by"] = f"deterministic (model {'error: ' + str(res.get('error')) if res.get('is_error') else 'gave no parseable reply'})"
        return doc
    doc.setdefault("new_subsystems", [])
    doc.setdefault("unresolved", [])
    # a reviewer that answers for some entities and stays silent on the rest: the silent ones keep their
    # deterministic candidate (marked as such) or stay unresolved — nothing is silently dropped
    seen = {a.get("entity") for a in doc["assignments"]} | {m for n in doc["new_subsystems"] for m in (n.get("members") or [])}
    for c in cands:
        if c["entity"] in seen:
            continue
        if c["subsystem"] is not None:
            doc["assignments"].append({"entity": c["entity"], "subsystem": c["subsystem"], "confidence": c["confidence"],
                                       "alternatives": c["alternatives"], "why": "deterministic candidate (the reviewer was silent): " + c["why"]})
        elif c["entity"] not in doc["unresolved"]:
            doc["unresolved"].append(c["entity"])
    doc["reviewed_by"] = f"{backend}:{model}"
    doc["usage"] = res.get("usage")
    return doc


def check(proposal, pack):
    """-> list of problems; empty means every id and name exists."""
    from pack_mcp import Pack
    P = pack if hasattr(pack, "ents") else Pack(pack)
    subs = set(P.l2) | {str(e.get("curated") or e.get("subsystem")) for e in P.ents if (e.get("curated") or e.get("subsystem"))}
    problems = []
    for a in proposal.get("assignments", []):
        if a.get("entity") not in P.by_name:
            problems.append(f"unknown entity {a.get('entity')!r}")
        s = a.get("subsystem")
        if s is not None and str(s) not in subs and not any(n.get("name") == s for n in proposal.get("new_subsystems", [])):
            problems.append(f"unknown subsystem {s!r} for {a.get('entity')}")
    for n in proposal.get("new_subsystems", []):
        if len(n.get("members") or []) < 2:
            problems.append(f"new subsystem {n.get('name')!r} needs >= 2 members")
        for m in n.get("members") or []:
            if m not in P.by_name:
                problems.append(f"new subsystem {n.get('name')!r} names unknown entity {m!r}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out")
    ap.add_argument("--backend", default="claude", choices=["claude", "none"])
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--proposal")
    a = ap.parse_args(argv)
    if a.check:
        prop = json.load(open(a.proposal, encoding="utf-8"))
        probs = check(prop, a.pack)
        for p in probs:
            print("INVENTED:", p)
        print(f"check: {len(probs)} problems")
        return 2 if probs else 0
    delta = json.load(open(a.delta, encoding="utf-8"))
    cands = candidates(a.pack, delta)
    doc = review(delta, cands, a.pack, a.backend, a.model)
    doc["candidates"] = cands
    doc["repo"], doc["date"], doc["head"] = delta.get("repo"), delta.get("date"), delta.get("head")
    probs = check(doc, a.pack)
    doc["check"] = probs
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False, sort_keys=True)
    print(json.dumps({"assignments": len(doc["assignments"]), "new_subsystems": len(doc["new_subsystems"]),
                      "unresolved": len(doc["unresolved"]), "queued": sum(1 for c in cands if c["queued"]),
                      "reviewed_by": doc["reviewed_by"], "check_problems": len(probs)}, indent=1))
    return 2 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
