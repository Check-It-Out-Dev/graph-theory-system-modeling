# CodeMap Erdős E2+E3 DELTA — bi-temporal supersession of the navigation layer.
#
# This is the delta counterpart to nav_writer.py. nav_writer.py opens with DETACH DELETE of
# the whole nav layer; that path is RETIRED (ErdosNavigator ledger L3) from the moment MFQ
# caches and curation history reference L2 sub_ids. Here the SubsystemNavigator nodes keep
# their identity (sub_id is a foreign key for mfq depends_on stamps) and the previous clue is
# preserved two ways: `superseded_summary_v1` on the node, and an immutable :ClueSnapshot
# carrying the full v1 payload with (t_valid, t_invalid) and a [:SUPERSEDED_BY] edge.
#
# Only subsystems whose facts actually changed are rewritten. Everything else stays
# byte-identical — a delta run that rewrites everything is a failed delta run.
#
# STORE PORT (2026-09-02): Neo4j -> Ladybug Store. Prose, INDEX_EDITS and caveats are
# byte-identical to the executed 2026-09-02 batch. In a full regen this runs BEFORE
# curation (its asserts are bound to the pre-curation 19-navigator tree and erdos-v2);
# against the already-curated store only --dry is meaningful.
#
# Usage: PYTHONUTF8=1 python nav_delta_writer.py [--dry]

import datetime as _dt
import hashlib, json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "authoring")))
from ladybug_store import Store

DOSS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dossiers"))
CLUE_VERSION, BATCH = "erdos-v2", "2026-09-02"
GENERATED_BY = "ErdosNavigator/delta"
NOW = _dt.datetime.now().isoformat()

# Subsystems whose MEMBERSHIP changed (GrothendieckV5 changed-subsystem list).
CHANGED = [3, 8, 15, 17]

# Authored prose. Every clause traces to a refreshed dossier field or an executed query;
# the query-grounded ones are named in `evidence` so the acceptance audit can re-run them.
CLUES = {
    3: dict(
        summary="Spring profiles (application-*.yml, incl. dev-lite), Cucumber/Spring test "
                "context, storage/upload and health config. Lowest purity in the system "
                "(Context 27%) — a mixed platform bag.",
        resp=["profile ymls for every run mode", "Cucumber test wiring",
              "storage, upload + health configuration"],
        caveats=["low purity (0.27) — expect heterogeneous members",
                 "holds StorageUrlValidator.java (storage URL validation), newly indexed this "
                 "batch: security-relevant validation sits in the config bag, not in billing "
                 "or tickets"],
        evidence="dossier purity 0.271, top_terms 'upload'/'storage'; query: the 7 batch-stamped "
                 "members are DevLiteUploadController, GoogleCredentialsProvider, LocalUploadSink, "
                 "StorageUrlValidator, application-dev-lite.yml, changelog.xml, logback-spring.xml"),
    8: dict(
        summary="Specification builders plus recaptcha, Vimeo-URL and social-post-URL "
                "validation rules (entry SpecificationBuilder, 48 external in-edges). "
                "Rule-dominant (67%).",
        resp=["specification/query builders", "recaptcha verification",
              "URL-format validators (Vimeo, social post)", "unit-tested validation rules"],
        caveats=["MERGE CHECK: external_ratio 0.944 (12 internal vs 202 external edges) — a rule "
                 "shelf consumed elsewhere rather than a self-contained module, and the +4 "
                 "validators this batch did not raise internal cohesion. Grothendieck curation "
                 "should judge merge vs keep."],
        evidence="dossier purity 0.673, edges_internal 12 / edges_external 202, external_ratio "
                 "0.944, top_terms 'vimeo'/'post'/'social'; query: batch members VimeoUrls.java, "
                 "VimeoUrlsValidator.java, SocialPostUrl.java, SocialPostUrlValidator.java"),
    15: dict(
        summary="Support ticket entities, attachments and flows — the public ticket "
                "create/status surface, plus TicketAccessTokenService (new this batch).",
        resp=["ticket entities + repos", "attachment handling", "ticket status flows",
              "ticket access tokens"],
        caveats=["attachment URL validation (StorageUrlValidator.java) is no longer a coverage "
                 "gap — it is indexed as of this batch but sits in sub-3, not here: attachment "
                 "answers span both subsystems",
                 "TicketAccessTokenService was a low-margin assignment (kNN top inconclusive at "
                 "0.333 over 6 distinct subsystems; folder prior support/ticket/services/ and a "
                 "1/1 in-edge decided sub-15)"],
        evidence="dossier size 35 / external_ratio 0.484 / Process 2->3; query: StorageUrlValidator "
                 "now in sub-3; node assignment_flag_reason on TicketAccessTokenService.java"),
    17: dict(
        summary="The entire Angular app (405 files): routes, standalone components, the full "
                "interceptor chain, i18n, SSR, sandbox fixtures and e2e tiers. No external "
                "in-edges — enter at the bootstrap (main.ts / main.server.ts, both indexed this "
                "batch, both INJECT app.component.ts), not via imports.",
        resp=["route table + guards", "feature components + layouts",
              "interceptor chain (8 of 8 now indexed)", "i18n (Transloco en/pl)",
              "sandbox fixtures + e2e (incl. the integration _trace recorder)"],
        caveats=["coverage gap narrowed by batch 2026-09-02: the set-to-array and "
                 "ssr-cookie-forward interceptors and seo-title.strategy are now indexed (the "
                 "interceptor chain is complete at 8/8), and 7 e2e-tests files entered the graph "
                 "(integration/_trace recorder+canonicalize+diff+expected-drift+types, "
                 "_framework/real-login, visual-parity/component-pairs). The residual FE gap is "
                 "concrete, not vague: 125 of 132 e2e-tests/*.ts are still unindexed (bdd 0/24, "
                 "integration 5/50, _framework 1/35, visual-parity 1/5, integration/_trace 5/8), "
                 "plus 27 files under src/testing (contract 18, builders 9) and 2 under "
                 "src/mocks — so e2e and contract-test answers are shape signals at best",
                 "SPLIT CANDIDATE (strengthened): MEGA flag, 405 files = 28.62% of the estate, "
                 "above the 20% split trigger; v3 overlap {0:180, 1:148} shows two latent halves; "
                 "and seven members now carry a measured cross-repo affinity naming a BE "
                 "counterpart — public-config.service.ts→11, subscription.client.ts→11, "
                 "real-login.ts→9, recorder.ts→7, rate-limit-state.service.ts→3, "
                 "social-connections.service.ts→2, social-connections-settings.component.ts→2 — "
                 "which are candidate seams for the split"],
        evidence="dossier size 405 / share 0.2862 / external_ratio 0.277 / seam 17->18 n=156; "
                 "queries: main.ts + main.server.ts INJECT app.component.ts (both delta_batch "
                 "2026-09-02, and app.component.ts is consequently no longer an actor root); 8 "
                 "distinct *.interceptor.ts in sub-17; 7 e2e-tests members all batch-stamped; "
                 "assignment_crossrepo_affinity on 7 nodes; V3Master.delta_provenance for the "
                 "20% split trigger"),
}

# Out-of-list correction. Sub-11's membership did NOT change, so its clue would normally stay
# byte-identical — but this batch made one of its caveats provably false: it claims
# StorageUrlValidator is an unindexed coverage gap, and the file is now indexed AND assigned to
# sub-3. The grounding law ("the runtime model cannot detect drift between clue and graph, so
# drift must not exist") outranks the byte-identity rule, which exists to forbid gratuitous
# rewrites, not necessary corrections. Minimal edit: one caveat element; summary untouched.
CAVEAT_FIX = {
    11: dict(
        old="StorageUrlValidator (pentest 3.1 fix) post-dates the scan — coverage gap",
        new="StorageUrlValidator is indexed as of batch 2026-09-02 and was assigned to sub-3 "
            "(storage/upload config), not here — storage-validation answers live there",
        reason="caveat asserted a coverage gap that batch 2026-09-02 closed; claim was false "
               "against the graph"),
}


# --- E3 / L1 -----------------------------------------------------------------------------
# The subsystem_index is edited by exact line replacement, not regenerated: that is what makes
# byte-identity for the other 15 entries a guarantee rather than a hope. Each replacement is
# asserted to hit exactly once. ai_instruction (the mandated checklist) is never touched.
INDEX_EDITS = [
    ("  - [3] Configuration & test context (123 files) - Spring profiles (application-*.",
     "  - [3] Configuration & test context (129 files) - Spring profiles (application-*.yml, "
     "incl. dev-lite), Cucumber/Spring test context, storage/upload and health config."),
    ("  - [8] Validation & recaptcha rules (45 files) - Specification builders and recaptcha "
     "validation rules (entry SpecificationBuilder, 48 external in-edges).",
     "  - [8] Validation & recaptcha rules (49 files) - Specification builders plus recaptcha, "
     "Vimeo-URL and social-post-URL validation rules (entry SpecificationBuilder, 48 external "
     "in-edges)."),
    ("  - [15] Support tickets (34 files) - Support ticket entities, attachments and flows — "
     "the public ticket create/status surface.",
     "  - [15] Support tickets (35 files) - Support ticket entities, attachments and flows — "
     "the public ticket create/status surface, plus TicketAccessTokenService."),
    ("  - [17] Greenfield frontend (375 files) - The entire Angular app (375 files): routes, "
     "standalone components, interceptor chain, i18n, SSR, sandbox fixtures and e2e tiers.",
     "  - [17] Greenfield frontend (405 files) - The entire Angular app (405 files): routes, "
     "standalone components, the full interceptor chain, i18n, SSR, sandbox fixtures and e2e "
     "tiers."),
]

L1_SUMMARY_EDIT = ("1374 indexed files in 19 candidate subsystems",
                   "1415 indexed files in 19 candidate subsystems")

# Global caveats: replaced wholesale because three of the four moved. The stale-scan and
# TESTED_BY numbers are executed-query results, not carried-over prose.
GLOBAL_CAVEATS_V2 = [
    "coverage gaps: 9 of the 11 documented COVERAGE_GAP records are closed by delta batch "
    "2026-09-02 - verified by re-running each record's own locator against the graph, not "
    "assumed (BE37, FE09, FE11, FE18, FE19, FE22, FE23, FE26, FE33). The remaining 2 are NOT "
    "backlog items: BE29 is a PERMANENT REFUSAL - the 4 credential-bearing files in BE "
    "src/main/resources (keystore.p12, service-account.json, service-accountProd.json, "
    "dashboard-config.json) are deliberately never indexed, because indexing them would send "
    "secrets to the remote embedding service; FE10 needs no new file, since ApiConfiguration "
    "lives inside the collapsed generated-client node (sub-18). Do not queue either as work",
    "remainder, measured post-batch and reproducible: 270 files are genuinely unindexed = 61 "
    "in-scope + 125 of 132 e2e-tests/*.ts + 84 of 104 BE src/main/resources. e2e-tests (bdd "
    "0/24, integration 5/50, _framework 1/35, visual-parity 1/5) and resources/ were never in "
    "the scan scope, so absence there is scope, not failure. The 271 generated-client files "
    "are collapsed into one node by design, not missing. SEPARATELY, and do not conflate the "
    "two: 321 already-indexed files are fingerprint-stale (318 content-changed + 3 mtime-only) "
    "- their nodes, edges and embeddings exist, only the source moved since the scan",
    "TRIGGERS (6) and TESTED_BY (113) edges are under-extracted - event-flow and test-coverage "
    "answers are shape signals",
    "subsystems are PRE-CURATION candidates: 16/2 and 8 merge checks, 13 retype, 17 split "
    "pending GrothendieckV5 - sub-17 is now 28.62% of the estate, above the 20% split trigger, "
    "with seven cross-repo affinities naming candidate seams",
    "behavioural-lens embeddings degenerate on quiet Resources (research F88)",
    "clue layer is erdos-v2 (delta batch 2026-09-02): subsystems 3, 8, 15, 17 were re-clued and "
    "sub-11 took a one-caveat correction; the other 14 carry unchanged erdos-v1 clues. Cached "
    "answers listed in eval/q/INVALIDATED_2026-09-02.json must be dropped",
]


def fingerprint(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def pull_nav(s, sub):
    """Full navigator state (typed cols + parsed props) as one flat dict."""
    cols = ["name", "role", "size", "external_ratio", "ai_summary", "clue_version",
            "dossier_fingerprint", "spines", "entry_points", "contracts", "caveats",
            "responsibilities"]
    r = s.one("MATCH (n:Nav) WHERE n.sub_id = $sub RETURN "
              + ", ".join(f"n.{c} AS {c}" for c in cols) + ", n.props AS props",
              dict(sub=sub))
    if r is None:
        return None
    p = json.loads(r.pop("props") or "{}")
    return {**p, **r}


def snap_create(s, sub, body, snap_id):
    s.create("ClueSnap", dict(
        snap_id=snap_id, sub_id=sub, taken_at=NOW, t_created=NOW, t_expired="",
        superseded_by=BATCH,
        body=json.dumps(body, ensure_ascii=False, default=str)), "snap_id")


def write_l1(s, dry):
    m = s.one("MATCH (nm:Master) RETURN nm.mid AS mid, nm.subsystem_index AS idx, "
              "nm.ai_summary AS summ, nm.global_caveats AS cav, nm.props AS props")
    mp = json.loads(m["props"] or "{}")
    idx, summ = m["idx"], m["summ"]
    for old, new in INDEX_EDITS:
        assert idx.count(old) == 1, f"index anchor not unique/found: {old[:60]!r} ({idx.count(old)})"
        idx = idx.replace(old, new)
    assert summ.count(L1_SUMMARY_EDIT[0]) == 1, "summary anchor not found"
    summ = summ.replace(*L1_SUMMARY_EDIT)
    untouched = [ln for ln in m["idx"].splitlines() if ln in idx.splitlines()]
    print(f"L1: {len(INDEX_EDITS)} index lines rewritten, "
          f"{len(untouched)} of {len(m['idx'].splitlines())} lines byte-identical")
    if dry:
        print("DRY: L1 not written")
        return
    snap_create(s, -1, dict(   # the master has no sub: sub_id=-1, level lives in body
        level="L1", clue_version=mp.get("clue_version"), ai_summary=m["summ"],
        subsystem_index=m["idx"], global_caveats=m["cav"], t_valid=mp.get("generated_at"),
        reason=f"delta batch {BATCH}: subsystem index sizes 3/8/15/17 and global "
               f"caveats changed"), f"delta-{BATCH}-L1")
    cav = json.dumps(GLOBAL_CAVEATS_V2, ensure_ascii=False)
    s.conn.execute(
        "MATCH (nm:Master) WHERE nm.mid = $mid SET nm.ai_summary = $summ, "
        f"nm.subsystem_index = $idx, nm.global_caveats = {Store.lit(cav)}",
        parameters=dict(mid=m["mid"], summ=summ, idx=idx))
    s.merge_props("Master", "mid", m["mid"], dict(
        superseded_summary_v1=m["summ"], superseded_index_v1=m["idx"],
        clue_version=CLUE_VERSION, generated_by=GENERATED_BY, generated_at=NOW,
        clue_delta_batch=BATCH,
        scanned_at="2026-09-02 scan + delta batch 2026-09-02 (see caveats)"))
    chk = s.one("MATCH (nm:Master) WHERE nm.mid = $mid RETURN nm.subsystem_index AS idx, "
                "nm.global_caveats AS cav, nm.props AS props", dict(mid=m["mid"]))
    cp = json.loads(chk["props"] or "{}")
    assert cp.get("ai_instruction") == mp.get("ai_instruction"), \
        "ai_instruction must stay verbatim"
    for _, new in INDEX_EDITS:
        assert new in chk["idx"], "index edit missing after write"
    print(f"VERIFIED BY READ: L1 clue_version={cp.get('clue_version')}, "
          f"{len(json.loads(chk['cav']))} global caveats, checklist verbatim=True")


def main():
    dry, l1_only = "--dry" in sys.argv, "--l1" in sys.argv
    s = Store(read_only=dry)
    doss = {}
    # L10 guard: the dossier dir legitimately holds superseded orphans (e.g. pre-split 17);
    # the id set comes from LIVE leaf navigators, and each dossier must fingerprint-match.
    _live = {r["sub"]: r["fp"] for r in s.q(
        "MATCH (sn:Nav) WHERE sn.role IS NULL OR NOT sn.role IN ['MERGED','GROUP'] "
        "RETURN sn.sub_id AS sub, sn.dossier_fingerprint AS fp")}
    skipped_orphans = []
    for f in sorted(os.listdir(DOSS)):
        if f.startswith("subsystem_") and f.endswith(".json"):
            d = json.loads(open(os.path.join(DOSS, f), encoding="utf-8").read())
            sub = d["subsystem"]
            if sub not in _live:
                skipped_orphans.append(f)  # superseded orphan (e.g. pre-split 17) — never load
                continue
            fp = fingerprint(d)
            if _live[sub] and _live[sub] != fp:
                raise SystemExit(f"L10 STALE DOSSIER: {f} fingerprint {fp} != graph "
                                 f"{_live[sub]} — regenerate dossiers before writing clues")
            doss[sub] = d
    if skipped_orphans:
        print(f"L10 guard: skipped {len(skipped_orphans)} orphaned dossier(s): {skipped_orphans}")

    if l1_only:
        write_l1(s, dry)
        return
    for sub in CHANGED:
        if sub not in doss:
            print(f"skip sub-{sub}: not a live leaf (GROUP/merged/orphaned) — "
                  "per-run CHANGED lists must be refreshed against the live tree")
            continue
        d, c = doss[sub], CLUES[sub]
        dfp = fingerprint(d)
        if dry:
            print(f"DRY sub-{sub}: size->{d['size']} ext->{d['external_ratio']} fp->{dfp}")
            continue
        # 1) snapshot the outgoing version, 2) update in place.
        prev = pull_nav(s, sub)
        snap_create(s, sub, dict(prev, level="L2",
                                 reason=f"delta batch {BATCH}: membership and dossier "
                                        f"facts changed"), f"delta-{BATCH}-{sub}")
        sets, params = [], dict(
            sub=sub, summary=c["summary"], evidence=c["evidence"],
            size=d["size"], ext=d["external_ratio"], dfp=dfp, cv=CLUE_VERSION)
        for col, val in (("responsibilities", c["resp"]), ("caveats", c["caveats"]),
                         ("layer_profile", json.dumps(d["layer_profile"])),
                         ("entry_points", json.dumps(
                             [e["name"] for e in d["entry_points"]] + d["actor_roots"])),
                         ("spines", json.dumps(d["top_hyperedges"])),
                         ("contracts", json.dumps(d["top_seams"]))):
            if isinstance(val, list):
                val = json.dumps(val, ensure_ascii=False)
            if col == "layer_profile":     # no typed column — rides in props below
                continue
            sets.append(f"n.{col} = {Store.lit(val)}")
        s.conn.execute(
            "MATCH (n:Nav) WHERE n.sub_id = $sub SET n.ai_summary = $summary, "
            "n.size = $size, n.external_ratio = $ext, n.dossier_fingerprint = $dfp, "
            "n.clue_version = $cv, " + ", ".join(sets), parameters=params)
        s.merge_props("Nav", "sub_id", sub, dict(
            superseded_summary_v1=prev.get("ai_summary"),
            clue_evidence=c["evidence"], layer_profile=json.dumps(d["layer_profile"]),
            generated_by=GENERATED_BY, generated_at=NOW, clue_delta_batch=BATCH))

    for sub, fix in CAVEAT_FIX.items():
        if dry:
            print(f"DRY sub-{sub}: caveat correction")
            continue
        row = pull_nav(s, sub)
        cavs = row.get("caveats")
        cavs = json.loads(cavs) if isinstance(cavs, str) else (cavs or [])
        if fix["old"] not in cavs:
            print(f"skip sub-{sub}: caveat anchor absent (already corrected)")
            continue
        snap_create(s, sub, dict(row, level="L2", correction_only=True,
                                 reason=fix["reason"]), f"delta-{BATCH}-fix-{sub}")
        new_cavs = json.dumps([fix["new"] if x == fix["old"] else x for x in cavs],
                              ensure_ascii=False)
        s.conn.execute(
            "MATCH (n:Nav) WHERE n.sub_id = $sub "
            f"SET n.caveats = {Store.lit(new_cavs)}, n.clue_version = $cv",
            parameters=dict(sub=sub, cv=CLUE_VERSION))
        s.merge_props("Nav", "sub_id", sub, dict(
            superseded_caveats_v1=cavs, clue_delta_batch=BATCH,
            clue_correction_note=fix["reason"], generated_at=NOW))

    if dry:
        return
    rows = s.q("MATCH (sn:Nav) RETURN sn.sub_id AS sub, sn.clue_version AS cv, "
               "sn.size AS size, sn.external_ratio AS ext, "
               "sn.dossier_fingerprint AS fp, sn.props AS props")
    rows.sort(key=lambda r: r["sub"])            # determinism at the boundary
    snapc = {}
    for r in s.q("MATCH (c:ClueSnap) RETURN c.sub_id AS sid"):
        snapc[r["sid"]] = snapc.get(r["sid"], 0) + 1
    print("VERIFIED BY READ:")
    for r in rows:
        p = json.loads(r["props"] or "{}")
        mark = "*" if r["cv"] == CLUE_VERSION else " "
        print(f" {mark} sub-{r['sub']:<2} {str(r['cv']):<9} size={str(r['size']):<4} "
              f"ext={str(r['ext']):<6} fp={r['fp']} snapshots={snapc.get(r['sub'], 0)} "
              f"v1_kept={p.get('superseded_summary_v1') is not None}")
    changed = [r["sub"] for r in rows if r["cv"] == CLUE_VERSION]
    assert sorted(changed) == sorted(CHANGED + list(CAVEAT_FIX)), \
        f"unexpected changed set {changed}"
    for sub in CHANGED:
        r = next(x for x in rows if x["sub"] == sub)
        assert r["size"] == doss[sub]["size"], f"sub-{sub} size mismatch"
        assert r["fp"] == fingerprint(doss[sub]), f"sub-{sub} fingerprint mismatch"
    print(f"OK: {len(CHANGED)} clue updates + {len(CAVEAT_FIX)} caveat correction; "
          f"{19 - len(changed)} subsystems untouched")


if __name__ == "__main__":
    main()
