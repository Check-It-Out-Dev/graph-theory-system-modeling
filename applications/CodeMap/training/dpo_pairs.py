# CodeMap DPO pair builder (R4) — preference pairs where REJECTED is a recorded model
# error (r2.2 batch misses, loop-tier premature answers / wrong verbs / wrong abstains,
# replayed deterministically) or a mechanical corruption of a measured failure mode
# (sibling-extension swap, find-when-verbatim, wrong-selection-from-context — the class
# the grammar cannot catch). CHOSEN always satisfies the open-book invariant.
# STEP ROWS ONLY: answer() prose is never preference-trained (degeneration + reward
# hacking guard, PIPELINE law).
#
# Usage: PYTHONUTF8=1 python dpo_pairs.py   -> data/dpo_pairs.jsonl + stats

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "app"))
from engine import Engine  # noqa: E402
from dsl import execute, parse, ParseError  # noqa: E402
from datagen import (CANONICAL_OVERRIDES, ENTITY_ARG, canonical_dsl,  # noqa: E402
                     digest_v2, mentions_of)

MASTER = open(os.path.join(HERE, "master_prompt_v1.txt"), encoding="utf-8").read().strip()
random.seed(42)


def conv(prompt_user, chosen, rejected, src):
    return dict(
        prompt=[dict(role="system", content=MASTER), dict(role="user", content=prompt_user)],
        chosen=[dict(role="assistant", content=chosen)],
        rejected=[dict(role="assistant", content=rejected)],
        meta=dict(src=src))


def open_book_ok(dsl_expr, prompt_user):
    try:
        verb, args = parse(dsl_expr)
    except ParseError:
        return False
    if verb in ENTITY_ARG and args and args[0] not in prompt_user:
        return False
    return True


def main():
    e = Engine(use_ladybug=False)
    names = sorted(e.by_name.keys())
    recs = {r["id"]: r for r in (json.loads(l) for l in open(
        os.path.join(ROOT, "eval", "q", "mfq_all.jsonl"), encoding="utf-8"))}
    pairs, skipped = [], []

    # ---- source 1: r2.2 batch step misses (prompt/ref straight from the eval rows) ----
    test_rows = {r["messages"][1]["content"]: r for r in
                 (json.loads(l) for l in open(os.path.join(HERE, "data", "test.jsonl"),
                                              encoding="utf-8"))
                 if r["meta"]["kind"] == "step"}
    for l in open(os.path.join(HERE, "data", "GEN_lora-r22_test.jsonl"), encoding="utf-8"):
        g = json.loads(l)
        row = test_rows.get(g["user"])
        if not row:
            continue
        ref, got = row["messages"][2]["content"], g["gen"].splitlines()[0].strip()
        try:
            same = parse(got) == parse(ref)
        except ParseError:
            same = False
        if not same and open_book_ok(ref, g["user"]):
            pairs.append(conv(g["user"], ref, got, "batch-miss"))

    # ---- source 2: loop-tier misses, replayed deterministically ----
    loop = json.load(open(os.path.join(HERE, "data", "LOOP_r22_cpu.json"), encoding="utf-8"))
    for row in loop["rows"]:
        rec = recs[row["id"]]
        if row["status"] != "EXECUTED" or row.get("success"):
            continue
        dsl1 = CANONICAL_OVERRIDES.get(row["id"]) or canonical_dsl(rec, names=names)
        if not dsl1:
            continue
        traj = row["traj"]
        u = f"QUESTION: {rec['q']}\n(cache: miss)"
        # wrong bare-start abstain: chosen = the canonical opener (must be open-book)
        if traj and traj[0].startswith("pass(") and open_book_ok(dsl1, u):
            pairs.append(conv(u, dsl1, traj[0], "loop-wrong-abstain"))
            continue
        # premature answer after find(): chosen = canonical descent, rejected = the answer
        if (len(traj) >= 2 and traj[0].startswith("find(")
                and traj[1].startswith("answer(")):
            try:
                res = execute(e, traj[0])
            except Exception:
                continue
            u2 = f"{u}\nRESULT of {traj[0]}: {digest_v2(res)}"
            if open_book_ok(dsl1, u2):
                pairs.append(conv(u2, dsl1, traj[1][:400], "loop-premature-answer"))
            else:
                skipped.append((row["id"], "canonical arg not in find digest"))

    # ---- source 2b: the MIRROR class (r3-dpo loop regression, 02.09 11:12) ----
    # One-sided wrong-abstain pairs suppressed bare-start passing on UNANSWERABLE
    # questions too (0.74 -> 0.58, 8/19 answered out-of-corpus). Balance: for every
    # unanswerable the model wrongly ANSWERED, chosen = the doctrine path (find at bare
    # start; canonical pass WITH the evidence in context), rejected = the recorded answer.
    for row in loop["rows"]:
        rec = recs[row["id"]]
        if row["status"] == "EXECUTED" or row.get("success"):
            continue
        traj = row["traj"]
        u = f"QUESTION: {rec['q']}\n(cache: miss)"
        canon_pass = ('pass("needs-content-read: the answer lives in file content; '
                      'the graph pins where — see the pointer in my reason")'
                      if row["status"] == "CONTENT_POINTER" else
                      'pass("out-of-corpus: the relevant file is not indexed in this pack")')
        wrong_answer = next((t for t in traj if t.startswith("answer(")), None)
        if not wrong_answer:
            continue
        if traj[0].startswith("find("):
            try:
                res = execute(e, traj[0])
                u2 = f"{u}\nRESULT of {traj[0]}: {digest_v2(res)}"
                pairs.append(conv(u2, canon_pass, wrong_answer[:400], "loop-mirror-abstain"))
            except Exception:
                pass
        else:
            toks = sorted(set(__import__("re").findall(r"[A-Za-z0-9-]{4,}", rec["q"])),
                          key=len, reverse=True)
            if toks:
                pairs.append(conv(u, f"find({toks[0]})", wrong_answer[:400],
                                  "loop-mirror-abstain"))

    # ---- source 3: mechanical corruptions of measured failure modes (train drills) ----
    train_steps = [r for r in (json.loads(l) for l in open(
        os.path.join(HERE, "data", "train.jsonl"), encoding="utf-8"))
        if r["meta"]["kind"] == "step" and (r["meta"].get("src") or "").startswith("drill-")
        and not (r["meta"].get("src") or "").startswith("drill-enter")]
    random.shuffle(train_steps)  # NOSONAR - seeded split, never a secret; see sonar-project.properties
    made = 0
    for r in train_steps:
        if made >= 240:
            break
        u, target = r["messages"][1]["content"], r["messages"][2]["content"]
        try:
            verb, args = parse(target)
        except ParseError:
            continue
        if verb not in ENTITY_ARG:
            continue
        name = args[0]
        rejected = None
        kind = made % 3
        if kind == 0:
            # sibling-extension swap (the r2 failure class)
            stem = name.rsplit(".", 1)[0]
            sib = next((n for n in names if n != name and n.rsplit(".", 1)[0] == stem), None)
            if sib:
                rejected = target.replace(name, sib)
        elif kind == 1:
            # wrong-selection-from-context: another REAL name present in the digest —
            # grammatical, wrong, exactly what GBNF cannot catch
            others = [m for m in mentions_of(u, names) if m != name]
            if others:
                rejected = target.replace(name, others[0])
        else:
            # find-when-the-name-is-visible (protocol slack, GR14 class)
            rejected = f"find({name.rsplit('.', 1)[0]})"
        if rejected and rejected != target:
            pairs.append(conv(u, target, rejected, f"corrupt-{('sibling','selection','findslack')[kind]}"))
            made += 1

    random.shuffle(pairs)  # NOSONAR - seeded split, never a secret; see sonar-project.properties
    out = os.path.join(HERE, "data", "dpo_pairs.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    stats = dict(total=len(pairs),
                 by_src={s: sum(1 for p in pairs if p["meta"]["src"] == s)
                         for s in sorted({p["meta"]["src"] for p in pairs})},
                 skipped=skipped[:10])
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
