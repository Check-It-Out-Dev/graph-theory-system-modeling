# CodeMap training data generator v2 — OPEN-BOOK SELECTION (PIPELINE.md round-2 lever 2).
# Round-1 lesson (0/20 exec on held-out): closed-book supervision taught RECALL of entity
# names; the model invented plausible ones. v2 makes every entity/sub argument a COPY:
# visible in the question or in a result the pair carries. THE OPEN-BOOK INVARIANT is
# asserted over every emitted pair — datagen fails loudly rather than emit recall pairs.
#
# digest_v2 rendering (per result kind, caps below) is part of the SERVE CONTRACT for
# r2 models: the runtime client must compress results the same way (train-serve law D2).
# Master prompt: v1 (adds THE SELECTION LAW) — frozen before this generation.
#
# Usage: PYTHONUTF8=1 python datagen.py   -> data/{train,val,test}.jsonl + STATS.json

import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
from engine import Engine  # noqa: E402
from dsl import execute, parse  # noqa: E402

QDIR = os.path.join(HERE, "..", "eval", "q")
OUT = os.path.join(HERE, "data")
MASTER = open(os.path.join(HERE, "master_prompt_v1.txt"), encoding="utf-8").read().strip()
random.seed(42)

# argument domains mirror graph/scripts/gbnf_vocab.py DOMAINS (the vocabulary truth);
# the invariant below is the training-side enforcement of the same law
ENTITY_ARG = {"impact": 1, "flow": 1, "cohort": 1, "read": 1}   # first arg is an entity
SUB_ARG = {"enter": (0,), "spine": (0,), "seam": (0, 1)}         # which args are sub ids

PASS_NEGATIVES = [
    ("How do I deploy this to Kubernetes?", "out-of-corpus: deployment infra is not in the code graph"),
    ("What is the product roadmap for next quarter?", "out-of-corpus: roadmap is not code"),
    ("Who committed the last change to the billing module?", "out-of-corpus: git history is not in the graph"),
    ("Why is production slow today?", "out-of-corpus: runtime telemetry is not in the graph"),
    ("What does the React hook in the dashboard do?", "out-of-corpus: no React in this codebase (Angular)"),
    ("Show me the database schema migrations for 2024", "needs-content-read: Liquibase changelogs are content, not structure"),
    ("Is the code good?", "ambiguous: name a subsystem or a quality dimension"),
    ("What changed last week?", "out-of-corpus: the graph holds structure, not history"),
    ("Translate the app to German", "out-of-corpus: a task, not a question about structure"),
    ("Which developer owns the auth module?", "out-of-corpus: ownership metadata is not indexed"),
    ("Jak wdrożyć to na Kubernetesa?", "out-of-corpus: infrastruktura wdrożeniowa nie jest w grafie kodu"),
    ("Kto ostatnio commitował w billingu?", "out-of-corpus: historia gita nie jest w grafie"),
    ("Czy ten kod jest dobry?", "ambiguous: wskaż podsystem albo wymiar jakości"),
    ("What is our AWS bill this month?", "out-of-corpus: infrastructure cost is not code"),
    ("Summarize yesterday's standup", "out-of-corpus: meetings are not in the code graph"),
    ("Zrób refaktor całego frontendu", "out-of-corpus: zadanie do wykonania, nie pytanie o strukturę"),
    ("Które testy są flaky?", "out-of-corpus: wyniki uruchomień testów nie są indeksowane"),
    ("What's the test coverage percentage?", "out-of-corpus: coverage numbers live in CI reports, not the structure graph"),
]

# in-graph-term negatives get the TWO-STEP canonical: an honest find() first, then pass()
# on seeing the result — the model's find-first instinct on unanswerables (measured in r2)
# becomes correct protocol instead of a scored failure. No question here may also appear
# in PASS_NEGATIVES (the injective law: one question surface, one defensible target).
NEG_TWO_STEP = [
    ("How much does Stripe charge us per transaction?", "Stripe",
     "out-of-corpus: Stripe integration code exists here, but commercial pricing is not code"),
    ("Ile Stripe bierze od transakcji?", "Stripe",
     "out-of-corpus: kod integracji Stripe jest w grafie, ale warunki handlowe to nie kod"),
    ("Why is production slow today?", "production",
     "out-of-corpus: the graph shows structure, not today's runtime latency"),
    ("Czemu produkcja dziś muli?", "produkcja",
     "out-of-corpus: graf pokazuje strukturę, nie dzisiejszą telemetrię produkcji"),
    ("Fix the bug in the payment flow", "payment",
     "ambiguous: which bug? name a symptom or a file — the graph maps payment structure, it cannot pick your bug"),
    ("Napraw błąd w płatnościach", "płatnościach",
     "ambiguous: który błąd? podaj objaw albo plik — graf mapuje strukturę płatności, nie wskaże buga"),
    ("Is our subscription code secure?", "subscription",
     "ambiguous: name a security dimension or a file — structure alone does not audit security"),
    ("Czy konsenty są zgodne z RODO?", "konsenty",
     "needs-content-read: compliance lives in the legal texts and code bodies, the graph pins WHERE they are"),
]

ENTITY_TEMPLATES = {  # intent -> (lang, template) — {q} is the stem the user would type
    "impact": [("en", "What breaks if I change {q}?"), ("pl", "Co się zepsuje, jak zmienię {q}?")],
    "read": [("en", "Show me where {q} lives — I want to open the file."), ("pl", "Pokaż, gdzie leży {q} — chcę otworzyć plik.")],
    "cohort": [("en", "What usually changes together with {q}?"), ("pl", "Co zwykle zmienia się razem z {q}?")],
    "flow": [("en", "Where do calls from {q} lead?"), ("pl", "Dokąd prowadzą wywołania z {q}?")],
}
# depth-2 must be SIGNALLED in the question (injective law — r2 forensics: identical
# phrasing supervised two different targets and the model was scored wrong for guessing 1)
DEPTH2_TEMPLATES = [("en", "What breaks if I change {q} — including second-order dependents, 2 hops out?"),
                    ("pl", "Co się zepsuje, jak zmienię {q} — łącznie z zależnościami drugiego rzędu, 2 poziomy?")]
# null-result answers (owner design 02.09): questions the graph CAN answer where the true
# answer is NOTHING — positive supervision that emptiness is a correct, stated answer
NULL_ANSWERS = {
    "impact": [("en", 'answer("Nothing depends on {n} — the graph records zero incoming dependencies; changing it is isolated.")'),
               ("pl", 'answer("Nic nie zależy od {n} — graf nie notuje żadnych zależności przychodzących; zmiana jest izolowana.")')],
    "cohort": [("en", 'answer("{n} participates in no recorded co-change cohorts — no hyperedge couples it to other files.")'),
               ("pl", 'answer("{n} nie występuje w żadnym kohorcie współzmian — żadna hiperkrawędź nie wiąże go z innymi plikami.")')],
    "seam": [("en", 'answer("No seam between subsystems {a} and {b} — zero direct edges cross that boundary.")'),
             ("pl", 'answer("Brak szwu między podsystemami {a} i {b} — żadna krawędź nie przecina tej granicy.")')],
}
SUB_TEMPLATES = [("en", 'Which subsystem covers "{label}"? Enter it.'),
                 ("pl", 'Wejdź w podsystem od "{label}".')]
# spine-intent drills (r2.1 forensics: GR08 picked seam over spine twice — the
# learning-path intent had almost no training mass outside two golds)
SPINE_TEMPLATES = [("en", 'What is the minimal reading path through "{label}"? Where do I start?'),
                   ("pl", 'Jaka jest minimalna ścieżka czytania przez "{label}"? Od czego zacząć?')]
# poisoned golds (r2.1 forensics): FE08's canonical map() cannot ground its six-interceptor
# answer — supervision taught recall and all three aliases generated invented names.
# find(interceptor) grounds every name the answer cites ("interceptor"/"interceptory" is
# verbatim in each question surface).
CANONICAL_OVERRIDES = {"FE08": "find(interceptor)"}


# the serve-contract digest lives in app/serve_digest.py (ONE implementation — L11);
# re-exported here so loop_runner and older imports keep working
from serve_digest import digest_v2  # noqa: F401,E402


def mentions_of(text, names):
    """Graph names appearing verbatim in text, left-boundary-checked (User.java must not
    fire inside SuperUser.java-style mentions)."""
    out = []
    for n in names:
        i = text.find(n)
        if i >= 0 and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] in "_-")):
            out.append(n)
    return out


def canonical_dsl(rec, names=()):
    """Archetype+params -> the canonical DSL step. None = no clean derivation (skip)."""
    a, p = rec.get("archetype"), rec.get("recipe_params", {}) or {}

    def resolve(term):
        if term in names:
            return term
        hits = sorted((n for n in names if term.lower() in n.lower()), key=len)
        return hits[0] if hits else None  # shortest full match, deterministic

    try:
        if a == "impact":
            return f"impact({p['name']})"
        if a == "flow":
            rx = p.get("rx", "")
            term = max((t for t in rx.replace("^", "").replace("$", "").split("|")), key=len)
            term = term.replace("\\.", ".").strip(".*()?")
            if not term or "(" in term:
                return None
            full = resolve(term)
            return f"flow({full})" if full else None
        if a == "boundary":
            return f"seam({p['a']}, {p['b']})"
        if a == "cohort":
            if p.get("name"):
                return f"cohort({p['name']})"
            # r2 forensics (GR14): a cohort question NAMING a file fell to the hubs
            # fallback and supervised against the model's better move — resolve the
            # entity from the question itself when it names exactly one
            m = mentions_of(rec.get("q") or "", names)
            return f"cohort({m[0]})" if len(m) == 1 else "health(hubs)"
        if a == "health":
            k = p.get("kind")
            return f"health({k})" if k in ("coupling", "hubs") else None
        if a == "onboarding":
            return f"spine({p['sub']})"
        if a == "onboarding_path":
            return "spine(11)"
        if a == "overview":
            return "map()"
        if a == "locate":
            rx = p.get("rx", "")
            term = max(rx.replace("^", "").replace("$", "").split("|"), key=len)
            term = term.replace("\\.", ".").strip(".*()?$^")
            bad = set("[](){}\\+?")
            return f"find({term})" if term and not (set(term) & bad) else None
        if a == "content":
            return None
    except (KeyError, ValueError):
        return None
    return None


def mk(user, assistant, kind, rec_id, stratum=None, **meta):
    return dict(messages=[
        dict(role="system", content=MASTER),
        dict(role="user", content=user),
        dict(role="assistant", content=assistant)],
        meta=dict(kind=kind, rec=rec_id, stratum=stratum, **meta))


def pick_find_term(nl, target_name, e):
    """A term COPIED from the question whose find() surfaces the target — or None."""
    toks = sorted(set(re.findall(r"[A-Za-z0-9_.]{4,}", nl)), key=len, reverse=True)
    for t in toks[:8]:
        try:
            hits = execute(e, f"find({t})").get("hits") or []
        except Exception:
            continue
        if any(h["name"] == target_name for h in hits[:8]):
            return t
    return None


def open_book_check(pairs):
    """THE INVARIANT: every entity/sub argument must be visible in the pair's user text."""
    bad = []
    for p in pairs:
        user, asst = p["messages"][1]["content"], p["messages"][2]["content"]
        try:
            verb, args = parse(asst)
        except Exception:
            bad.append((p["meta"], f"unparseable target: {asst[:60]}"))
            continue
        if verb in ENTITY_ARG and args and args[0] not in user:
            bad.append((p["meta"], f"{verb} arg '{args[0]}' not in context"))
        for i in SUB_ARG.get(verb, ()):
            if i < len(args) and str(args[i]) not in user:
                bad.append((p["meta"], f"{verb} sub '{args[i]}' not in context"))
    if bad:
        for m, why in bad[:15]:
            print("OPEN-BOOK VIOLATION:", m.get("rec"), why)
        print(f"total violations: {len(bad)} / {len(pairs)}")
        sys.exit(1)


def main():
    os.makedirs(OUT, exist_ok=True)
    e = Engine(use_ladybug=False)
    names = sorted(e.by_name.keys())
    recs = [json.loads(l) for l in open(os.path.join(QDIR, "mfq_all.jsonl"), encoding="utf-8")]
    by_id = {r["id"]: r for r in recs}
    pairs, skipped, oob_skipped = [], [], []
    alias_conflicts = 0
    mdig = digest_v2(execute(e, "map()"))

    # ---- A. gold trajectories (open-book rebuild) --------------------------------------
    for r in recs:
        base = by_id.get(r.get("dup_of"), r) if r.get("dup_of") else r
        dsl1 = CANONICAL_OVERRIDES.get(base["id"]) or canonical_dsl(base, names=names)
        nls = [r["q"]] + (r.get("aliases") or [])[:3]
        if not dsl1:
            skipped.append((r["id"], f"no canonical dsl ({r.get('archetype')})"))
        else:
            verb, args = parse(dsl1)
            if verb in ("seam", "spine", "enter"):
                # recipes may carry pre-curation sub ids; supervision teaches LIVE ids only
                # (the engine resolves stale ids for backward compat — the model never should)
                args = [str(e.successors.get(int(a), a)) if str(a).isdigit() else a
                        for a in args]
                dsl1 = f"{verb}({', '.join(args)})"
            for nl in nls:
                # skip-alias law (r2 forensics): an alias whose surface entities are not
                # exactly the canonical's args is conflicting supervision — drop and count
                men = mentions_of(nl, names)
                if men and not all(m in dsl1 for m in men):
                    alias_conflicts += 1
                    continue
                u1 = f"QUESTION: {nl}\n(cache: miss)"
                if verb in ENTITY_ARG:
                    name = args[0]
                    if name in nl:
                        pairs.append(mk(u1, dsl1, "step", r["id"], r.get("stratum"), src="gold-direct"))
                    else:
                        t = pick_find_term(nl, name, e)
                        if not t:
                            oob_skipped.append((r["id"], nl[:50]))
                            continue
                        pairs.append(mk(u1, f"find({t})", "step", r["id"], r.get("stratum"), src="gold-open"))
                        fd = digest_v2(execute(e, f"find({t})"))
                        pairs.append(mk(f"{u1}\nRESULT of find({t}): {fd}", dsl1,
                                        "step", r["id"], r.get("stratum"), src="gold-select"))
                elif verb in ("seam", "spine", "enter"):
                    # ids come from the map index — teach the descent, then the selection
                    pairs.append(mk(u1, "map()", "step", r["id"], r.get("stratum"), src="gold-descend"))
                    pairs.append(mk(f"{u1}\nRESULT of map(): {mdig}", dsl1,
                                    "step", r["id"], r.get("stratum"), src="gold-select"))
                else:  # map()/health()/find() — no vocabulary argument to ground
                    pairs.append(mk(u1, dsl1, "step", r["id"], r.get("stratum"), src="gold-direct"))
            # answer pairs, grounded in the executed result — primary q AND aliases
            # (r2 forensics: n=4 answer rows measured nothing; volume + termination here)
            try:
                res = execute(e, dsl1)
                ans = base.get("gold_answer")
                if ans and base.get("gold_status") == "EXECUTED":
                    final = ans
                    if res.get("mermaid") and base.get("archetype") == "flow":
                        final = ans + "\n\n```mermaid\n" + res["mermaid"] + "\n```"
                    for nl in nls[:3]:
                        if mentions_of(nl, names) and not all(
                                m in dsl1 for m in mentions_of(nl, names)):
                            continue  # same skip-alias law as steps
                        pairs.append(mk(f"QUESTION: {nl}\n(cache: miss)\nRESULT of {dsl1}: "
                                        f"{digest_v2(res)}", f'answer("{final[:900]}")',
                                        "answer", r["id"], r.get("stratum"), src="gold"))
                    if verb in ENTITY_ARG:
                        # two-result trajectory: teaches TERMINATION after a longer loop
                        # (r2: 2 of 4 answer rows kept navigating instead of answering)
                        fd2 = digest_v2(execute(e, f"find({args[0]})"))
                        pairs.append(mk(f"QUESTION: {r['q']}\n(cache: miss)"
                                        f"\nRESULT of find({args[0]}): {fd2}"
                                        f"\nRESULT of {dsl1}: {digest_v2(res)}",
                                        f'answer("{final[:900]}")',
                                        "answer", r["id"], r.get("stratum"), src="gold-2step"))
            except Exception as ex:
                skipped.append((r["id"], f"exec: {ex}"))
        if r.get("gold_status") == "CONTENT_POINTER" and not r.get("dup_of"):
            for nl in nls[:2]:
                pairs.append(mk(f"QUESTION: {nl}\n(cache: miss)",
                                'pass("needs-content-read: the answer lives in file content; '
                                'the graph pins where — see the pointer in my reason")',
                                "pass", r["id"], r.get("stratum"), src="gold"))
        if r.get("gold_status") == "COVERAGE_GAP" and not r.get("dup_of"):
            pairs.append(mk(f"QUESTION: {r['q']}\n(cache: miss)",
                            'pass("out-of-corpus: the relevant file is not indexed in this pack")',
                            "pass", r["id"], r.get("stratum"), src="gold"))

    # ---- B. full-coverage entity drills (the volume lever) -----------------------------
    # injective law: when two entities share a stem (component.ts/.html families), the
    # drill question must carry the FULL name — "X.component" supervised both siblings in
    # r2 and the model was scored wrong for a defensible pick
    stem_count = Counter(n.rsplit(".", 1)[0] for n in names)
    drill_fail = []
    for name in names:
        stem = name.rsplit(".", 1)[0]
        term, fdig = None, None
        for cand in (stem, name):
            hits = execute(e, f"find({cand})").get("hits") or []
            if any(h["name"] == name for h in hits[:8]):
                term, fdig = cand, digest_v2(execute(e, f"find({cand})"))
                break
        if not term:
            drill_fail.append(name)
            continue
        hsh = int(hashlib.sha256(name.encode()).hexdigest(), 16)
        qname = name if (stem_count[stem] > 1 or term == name) else stem
        for k, intent in enumerate(("impact", "read", "cohort", "flow")):
            if intent == "impact" and (hsh + k) % 10 == 0:
                lang, tmpl = DEPTH2_TEMPLATES[(hsh + k) % 2]
                q, target = tmpl.format(q=qname), f"impact({name}, 2)"
            else:
                lang, tmpl = ENTITY_TEMPLATES[intent][(hsh + k) % 2]
                q, target = tmpl.format(q=qname), f"{intent}({name})"
            pairs.append(mk(f"QUESTION: {q}\n(cache: miss)\nRESULT of find({term}): {fdig}",
                            target, "step", f"drill:{name}", "D", src=f"drill-{intent}",
                            entity=name, lang=lang))
        # step-1 opener: the find() itself, term copied from the question — ONLY when the
        # question carries the stem, not the full name (selection law: a verbatim full name
        # goes straight to the entity verb; teaching find() there contradicts gold-direct)
        if qname != name:
            lang, tmpl = ENTITY_TEMPLATES["impact"][hsh % 2]
            pairs.append(mk(f"QUESTION: {tmpl.format(q=qname)}\n(cache: miss)", f"find({qname})",
                            "step", f"drill:{name}", "D", src="drill-opener", entity=name, lang=lang))

    # ---- C. subsystem drills from the map index ----------------------------------------
    sub_labels = re.findall(r"\[(\d+)\]\s*(?:GROUP\s*)?([^(]+?)\s*\(", mdig)
    for sid, label in sub_labels:
        for lang, tmpl in SUB_TEMPLATES:
            pairs.append(mk(f"QUESTION: {tmpl.format(label=label.strip())}\n(cache: miss)\n"
                            f"RESULT of map(): {mdig}", f"enter({sid})",
                            "step", f"subdrill:{sid}", "D", src="drill-enter", lang=lang))
        if int(sid) in e.leaves or sid in e.leaves:  # spines live on leaves, not GROUPs
            for lang, tmpl in SPINE_TEMPLATES:
                pairs.append(mk(f"QUESTION: {tmpl.format(label=label.strip())}\n(cache: miss)\n"
                                f"RESULT of map(): {mdig}", f"spine({sid})",
                                "step", f"subdrill:{sid}", "D", src="drill-spine", lang=lang))

    # ---- C2. null-result answers (owner design 02.09): questions the graph CAN answer
    # where the true answer is NOTHING — positive supervision that emptiness is a correct,
    # stated answer; removes the "every question must have content" hallucination pressure
    null_counts = Counter()
    for name in names:
        if null_counts["impact"] >= 60 and null_counts["cohort"] >= 60:
            break
        nhsh = int(hashlib.sha256(name.encode()).hexdigest(), 16)
        nstem = name.rsplit(".", 1)[0]
        nqname = name if stem_count[nstem] > 1 else nstem
        for nverb in ("impact", "cohort"):
            if null_counts[nverb] >= 60:
                continue
            res = execute(e, f"{nverb}({name})")
            if any(isinstance(v, list) and v for k2, v in res.items()
                   if k2 not in ("affordances",)):
                continue  # non-empty — not a null case
            lang, tmpl = NULL_ANSWERS[nverb][nhsh % 2]
            _, qtmpl = ENTITY_TEMPLATES[nverb][nhsh % 2]
            pairs.append(mk(f"QUESTION: {qtmpl.format(q=nqname)}\n(cache: miss)"
                            f"\nRESULT of {nverb}({name}): {digest_v2(res)}",
                            tmpl.format(n=name), "answer", f"null:{name}", "D",
                            src=f"null-{nverb}", entity=name, lang=lang))
            null_counts[nverb] += 1
    sn = 0
    for a, b in combinations(sorted(e.leaves, key=str), 2):
        if sn >= 30:
            break
        res = execute(e, f"seam({a}, {b})")
        if any(isinstance(v, list) and v for k2, v in res.items()
               if k2 not in ("affordances",)):
            continue
        lang, tmpl = NULL_ANSWERS["seam"][sn % 2]
        qq = (f"Where is the seam between subsystems {a} and {b}?" if lang == "en"
              else f"Gdzie jest szew między podsystemami {a} a {b}?")
        bucket = ("synthetic-test", "synthetic-val", "nullseam", "nullseam", "nullseam")[sn % 5]
        pairs.append(mk(f"QUESTION: {qq}\n(cache: miss)\nRESULT of seam({a}, {b}): "
                        f"{digest_v2(res)}", tmpl.format(a=a, b=b),
                        "answer", bucket, "D", src="null-seam", lang=lang))
        sn += 1

    # UNIFORM ABSTENTION DOCTRINE (r2.1 forensics): immediate-pass and two-step negatives
    # were two behavior classes the model could not tell apart from the surface — 4 of 7
    # abstention "failures" were the model correctly gathering first. Now EVERY negative is
    # two-step: find(term-from-question) -> see the evidence -> pass(reason). The abstention
    # decision is always made WITH evidence in context.
    def _longest_term(q):
        toks = re.findall(r"[A-Za-zżźćńółęąśŻŹĆŃÓŁĘĄŚ0-9-]{4,}", q)
        return max(toks, key=len) if toks else q.split()[0]

    all_negs = [(q, _longest_term(q), reason) for q, reason in PASS_NEGATIVES] + NEG_TWO_STEP
    for i, (q, term, reason) in enumerate(all_negs):
        bucket = ("synthetic-test", "synthetic-test", "synthetic-val",
                  "synthetic", "synthetic")[i % 5]
        u1 = f"QUESTION: {q}\n(cache: miss)"
        pairs.append(mk(u1, f"find({term})", "step", bucket, "G0", src="neg-2step"))
        fd = digest_v2(execute(e, f"find({term})"))
        pairs.append(mk(f"{u1}\nRESULT of find({term}): {fd}", f'pass("{reason}")',
                        "pass", bucket, "G0", src="neg-2step"))

    # ---- THE GATE ----------------------------------------------------------------------
    open_book_check(pairs)

    # ---- splits: two-axis decontamination ----------------------------------------------
    # axis 1 (gold records): exemplar-linked records -> test; 10% of the rest -> val
    test_recs = {r["id"] for r in recs if r.get("dup_of")}
    rest = sorted({p["meta"]["rec"] for p in pairs
                   if not p["meta"]["rec"].startswith(("drill:", "subdrill:", "synthetic"))}
                  - test_recs)
    random.shuffle(rest)  # NOSONAR - seeded split, never a secret; see sonar-project.properties
    val_recs = set(rest[: max(6, len(rest) // 10)])
    # axis 2 (drill entities): 10% of entities ENTIRELY held out -> test, 3% -> val;
    # rung-2 then measures selection on names never seen in any training pair
    drilled = sorted({p["meta"].get("entity") for p in pairs if p["meta"].get("entity")})
    random.shuffle(drilled)  # NOSONAR - seeded split, never a secret; see sonar-project.properties
    n = len(drilled)
    test_ents = set(drilled[: n // 10])
    val_ents = set(drilled[n // 10: n // 10 + max(3, n * 3 // 100)])

    def split_of(p):
        ent = p["meta"].get("entity")
        if ent:
            return "test" if ent in test_ents else "val" if ent in val_ents else "train"
        rid = p["meta"]["rec"]
        if rid == "synthetic-test":
            return "test"
        if rid == "synthetic-val":
            return "val"
        return ("test" if rid in test_recs else "val" if rid in val_recs else "train")

    out = {"train": [], "val": [], "test": []}
    for p in pairs:
        out[split_of(p)].append(p)
    # abstention is 0.8% of pairs by construction — oversample pass rows x4 IN TRAIN ONLY
    # so the behavior class carries trainable mass (test/val stay unweighted measurements)
    out["train"] += [p for p in out["train"] if p["meta"]["kind"] == "pass"] * 3
    for nm, rows in out.items():
        with open(os.path.join(OUT, f"{nm}.jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # honesty count: train gold pairs whose target entity is held out (selection-skill
    # leak is acceptable — the SKILL generalizes, the NAME still never appears as a
    # train drill — but it gets measured, not hidden)
    leak = sum(1 for p in out["train"]
               if not p["meta"].get("entity")
               and any(t in p["messages"][2]["content"] for t in test_ents))
    stats = dict(total=len(pairs),
                 by_src={s: sum(1 for p in pairs if p["meta"].get("src") == s)
                         for s in sorted({p["meta"].get("src") for p in pairs})},
                 by_kind={k: sum(1 for p in pairs if p["meta"]["kind"] == k)
                          for k in ("step", "answer", "pass")},
                 by_lang={k: sum(1 for p in pairs if p["meta"].get("lang") == k)
                          for k in ("en", "pl")},
                 splits={k: len(v) for k, v in out.items()},
                 entities=dict(drilled=len(drilled), no_find_term=len(drill_fail),
                               held_test=len(test_ents), held_val=len(val_ents)),
                 gold_leak_pairs_into_train=leak,
                 alias_conflicts=alias_conflicts,
                 ambiguous_stems=sum(1 for c in stem_count.values() if c > 1),
                 nulls=dict(null_counts, seam=sn),
                 skipped=len(skipped), oob_skipped=len(oob_skipped),
                 skipped_detail=skipped[:10], drill_fail_detail=drill_fail[:10])
    json.dump(stats, open(os.path.join(OUT, "STATS.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(stats, ensure_ascii=False)[:900])


if __name__ == "__main__":
    main()
