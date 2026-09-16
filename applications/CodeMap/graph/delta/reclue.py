"""Reclue: new L2 navigator prose for the subsystems a decision touched, and only those.

    python graph/delta/reclue.py --pack delta/pack.next --subsystems 0,6,205 --version 1.0.1 [--ledger graph/ledger]
                                 [--backend claude|none] [--model claude-sonnet-5] [--max 8]

The ErdosNavigatorV5 delta procedure, on the subscription: for each touched subsystem the pack yields
a dossier (the record's structural fields, its members with their external in-degree, its seams with
the neighbouring subsystems' names, the curation notes that mention it, the prose it has today), and
`claude -p` with a JSON schema returns the three prose fields — `ai_summary` (≤ 80 words),
`responsibilities` (3–5 bullets), `caveats`. Gates before anything is written: every number in the
prose is copied from the dossier (I3: numbers are never derived), every file name resolves to a member
or an entry point, lengths hold, nothing is empty. A record that passes gets the new prose and fresh
provenance (`clue_version=delta-<version>`, `generated_by`, `generated_at`, `clue_body_status=CURRENT`);
the previous prose is snapshotted into `graph/ledger/<version>.reclue.json` (supersede, never
overwrite). Every other line of `l2_navigators.jsonl` is written back byte-identical: a delta run
that rewrites everything is a failed delta run. Best-effort by contract — a failed or gated-out
reclue is recorded and the pack ships with the structure-only record it already has.
"""

import argparse
import json
import os
import re
import sys
import time

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(R, "app"))

SCHEMA = {
    "type": "object",
    "properties": {
        "ai_summary": {"type": "string", "description": "at most 80 words: what the subsystem does, who calls it, its shape"},
        "responsibilities": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 5},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ai_summary", "responsibilities", "caveats"],
}

SYSTEM = """You are ErdosNavigator, the understanding engine of CodeMap, running the delta procedure: one subsystem of a
code graph changed membership and you write its navigation clue again. You receive a dossier (JSON) with the
subsystem's structural facts, its members, its seams to other subsystems, the curation notes that mention it,
and the prose it has today. Return JSON only, matching the schema.

Rules (the acceptance battery checks them mechanically):
- ai_summary: at most 80 words. What the subsystem does, who invokes it (actor roots or callers), its dominant
  layer shape. Present tense, concrete file names where they help a reader navigate.
- responsibilities: 3 to 5 short bullets (each under 20 words), one duty per bullet, named by the members that
  carry it.
- caveats: what a newcomer would get wrong — coverage gaps, seams carried by one file, structure-only membership,
  test harness files living beside domain code. Cite the dossier field the caveat rests on. Empty list if none.
- Numbers are COPIED from the dossier, never derived or estimated. If you would need a number the dossier lacks,
  write the sentence without it.
- Name only files that appear in the dossier (members, entry points). Never invent a file or a subsystem.
- Untouched facts stay: keep what the current prose gets right; change what the new members changed."""

FILE_RX = re.compile(r"[A-Za-z0-9_\-]+\.(?:java|kt|ts|tsx|js|properties|ya?ml|md|json|xml|sql|gradle|feature)")
NUM_RX = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?(?![A-Za-z_])")


def load_pack(pack):
    import csv
    ents = list(csv.DictReader(open(os.path.join(pack, "entities.csv"), encoding="utf-8")))
    edges = list(csv.DictReader(open(os.path.join(pack, "edges.csv"), encoding="utf-8")))
    lines = [l for l in open(os.path.join(pack, "l2_navigators.jsonl"), encoding="utf-8").read().split("\n")]
    return ents, edges, lines


def dossier(pack, sid, ents, edges, l2, notes_path=None):
    sid = str(sid)
    nav = next((n for n in l2 if str(n.get("sub_id")) == sid), None)
    if nav is None:
        raise SystemExit(f"reclue: subsystem {sid} is not in the pack")
    sub_of = {e["name"]: (e.get("curated") or e.get("subsystem") or "") for e in ents}
    members = [e for e in ents if sub_of[e["name"]] == sid]
    names = {e["name"] for e in members}
    ext_in, ext_out, seams = {}, {}, {}
    p2n = {e["file_path"]: e["name"] for e in ents}
    for ed in edges:
        s, d = p2n.get(ed["src"], ed["src"]), p2n.get(ed["dst"], ed["dst"])
        if d in names and s not in names:
            ext_in[d] = ext_in.get(d, 0) + 1
            seams[sub_of.get(s, "?")] = seams.get(sub_of.get(s, "?"), 0) + 1
        elif s in names and d not in names:
            ext_out[s] = ext_out.get(s, 0) + 1
            seams[sub_of.get(d, "?")] = seams.get(sub_of.get(d, "?"), 0) + 1
    name_of = {str(n.get("sub_id")): n.get("name") for n in l2}
    notes = []
    if notes_path and os.path.exists(notes_path):
        for line in open(notes_path, encoding="utf-8"):
            if line.startswith("-") and (f"subsystem {sid} " in line or f"subsystem {sid}\n" in line or f"[{sid}]" in line):
                notes.append(line.strip())
    members_sorted = sorted(members, key=lambda e: (-ext_in.get(e["name"], 0), e["name"]))
    return {
        "sub_id": sid, "name": nav.get("name"), "role": nav.get("role"), "parent": nav.get("parent"),
        "size": len(members), "layer_profile": _j(nav.get("layer_profile")), "entry_points": _j(nav.get("entry_points")),
        "spines": _j(nav.get("spines")), "contracts": _j(nav.get("contracts")), "external_ratio": nav.get("external_ratio"),
        "seams": [{"subsystem": k, "name": name_of.get(k), "edges": v} for k, v in sorted(seams.items(), key=lambda kv: -kv[1])[:8]],
        "members": [{"name": e["name"], "type": e.get("entity_type"), "ext_in": ext_in.get(e["name"], 0),
                     "ext_out": ext_out.get(e["name"], 0), "added_by_delta": bool(e.get("delta_batch"))} for e in members_sorted[:60]],
        "members_truncated": max(0, len(members) - 60),
        "curation_notes": notes[-12:],
        "current_prose": {"ai_summary": nav.get("ai_summary"), "responsibilities": nav.get("responsibilities"),
                          "caveats": nav.get("caveats"), "clue_body_status": nav.get("clue_body_status")},
    }


def _j(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def gates(prose, doss):
    """-> list of problems (empty = pass). Mechanical: lengths, copied numbers, resolvable file names."""
    problems = []
    summary = (prose.get("ai_summary") or "").strip()
    resp = prose.get("responsibilities") or []
    cav = prose.get("caveats")
    if not summary:
        problems.append("ai_summary empty")
    if len(summary.split()) > 80:
        problems.append(f"ai_summary {len(summary.split())} words > 80")
    if not (3 <= len(resp) <= 5):
        problems.append(f"responsibilities {len(resp)} not in 3..5")
    for r in resp:
        if not str(r).strip():
            problems.append("empty responsibility")
        elif len(str(r).split()) > 20:
            problems.append(f"responsibility too long: {str(r)[:40]}…")
    if cav is None or not isinstance(cav, list):
        problems.append("caveats must be a list")
    text = " ".join([summary] + [str(x) for x in resp] + [str(x) for x in (cav or [])])
    allowed_files = {m["name"] for m in doss["members"]} | set(FILE_RX.findall(json.dumps(doss.get("entry_points") or "")))
    allowed_files |= set(FILE_RX.findall(json.dumps(doss.get("spines") or ""))) | set(FILE_RX.findall(" ".join(doss.get("curation_notes") or [])))
    for f in sorted(set(FILE_RX.findall(text))):
        if f not in allowed_files:
            problems.append(f"file not in the dossier: {f}")
    doss_numbers = set(NUM_RX.findall(json.dumps({k: v for k, v in doss.items() if k != "current_prose"}, ensure_ascii=False)))
    for n in sorted(set(NUM_RX.findall(text))):
        if n not in doss_numbers:
            problems.append(f"number not in the dossier: {n}")
    if "TODO" in text or "lorem" in text.lower():
        problems.append("placeholder text")
    return problems


def parse_json_text(text):
    """The first JSON object in a text answer (code fences tolerated)."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def ask(doss, model="claude-sonnet-5", runner=None, timeout=300):
    import claude_cli
    prompt = ("Dossier of the subsystem to reclue (JSON):\n" + json.dumps(doss, ensure_ascii=False, indent=1) +
              "\n\nAnswer with ONE JSON object only - keys ai_summary (string), responsibilities (array of strings), "
              "caveats (array of strings) - no prose around it, no code fence.")
    import tempfile
    fd, sys_path = tempfile.mkstemp(prefix="reclue-system-", suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(SYSTEM)
    # plain text JSON in one turn: --json-schema routes the answer through a tool turn that a long dossier
    # can exhaust (error_max_turns seen live on a 51-member subsystem); the parser above is the contract
    out = claude_cli.run(prompt, model=model, role="reclue", system_file=sys_path, max_turns=1, timeout=timeout, runner=runner, tools=[])
    try:
        os.unlink(sys_path)
    except OSError:
        pass
    prose = out.get("structured") if isinstance(out.get("structured"), dict) else parse_json_text(out.get("text"))
    return prose, out


def reclue(pack, subsystems, version, ledger_dir, backend="claude", model="claude-sonnet-5", runner=None,
           notes_path=None, by="graph-delta", max_subsystems=8):
    ents, edges, lines = load_pack(pack)
    l2 = [json.loads(l) for l in lines if l.strip()]
    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = {"schema": 1, "pack_version": version, "at": at, "backend": backend, "model": model if backend == "claude" else None,
              "results": []}
    new_prose = {}
    for sid in [str(s) for s in subsystems][:max_subsystems]:
        doss = dossier(pack, sid, ents, edges, l2, notes_path)
        res = {"sub_id": sid, "name": doss["name"], "status": "skipped", "problems": [], "usage": None,
               "before": doss["current_prose"], "after": None}
        if backend == "none":
            res["problems"] = ["backend none: nothing generated"]
            report["results"].append(res)
            continue
        prose, out = ask(doss, model=model, runner=runner)
        res["usage"] = out.get("usage")
        if out.get("is_error") or not isinstance(prose, dict):
            res["problems"] = [f"model: {out.get('error') or (out.get('text') or '')[:120] or 'no structured output'}",
                               f"subtype: {(out.get('raw') or {}).get('subtype')}"]
            report["results"].append(res)
            continue
        probs = gates(prose, doss)
        res["problems"] = probs
        if probs:
            res["status"] = "gated"
        else:
            res["status"] = "reclued"
            res["after"] = {"ai_summary": prose["ai_summary"].strip(), "responsibilities": [str(x).strip() for x in prose["responsibilities"]],
                            "caveats": [str(x).strip() for x in prose["caveats"]]}
            new_prose[sid] = res["after"]
        report["results"].append(res)
    # write back: only the reclued lines change, every other line byte-identical
    if new_prose:
        out_lines = []
        for line in lines:
            if not line.strip():
                out_lines.append(line)
                continue
            nav = json.loads(line)
            sid = str(nav.get("sub_id"))
            if sid in new_prose:
                nav.update(new_prose[sid])
                nav.update({"clue_version": f"delta-{version}", "generated_by": "ErdosNavigator/reclue-delta", "generated_at": at,
                            "clue_body_status": "CURRENT", "clue_delta_batch": version, "reclued_by": by})
                out_lines.append(json.dumps(nav, ensure_ascii=False))
            else:
                out_lines.append(line)
        with open(os.path.join(pack, "l2_navigators.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(out_lines))
    if ledger_dir:
        os.makedirs(ledger_dir, exist_ok=True)
        with open(os.path.join(ledger_dir, f"{version}.reclue.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=1, ensure_ascii=False, sort_keys=True)
    report["reclued"] = sorted(new_prose)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--subsystems", required=True, help="comma-separated ids (apply.json changed_subsystems)")
    ap.add_argument("--version", required=True)
    ap.add_argument("--ledger", default=os.path.join(R, "graph", "ledger"))
    ap.add_argument("--notes", default=os.path.join(R, "prompts", "navigator", "curation_notes.md"))
    ap.add_argument("--backend", default="claude", choices=["claude", "none"])
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--by", default=os.environ.get("GITHUB_ACTOR") or "graph-delta")
    ap.add_argument("--max", type=int, default=8)
    a = ap.parse_args(argv)
    subs = [s.strip() for s in a.subsystems.split(",") if s.strip()]
    rep = reclue(a.pack, subs, a.version, a.ledger, backend=a.backend, model=a.model, notes_path=a.notes, by=a.by, max_subsystems=a.max)
    for r in rep["results"]:
        print(f"reclue [{r['sub_id']}] {r['name']}: {r['status']}" + (f" — {'; '.join(r['problems'])[:200]}" if r["problems"] else ""))
    print(f"reclued {len(rep['reclued'])}/{len(rep['results'])} subsystems for pack {a.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
