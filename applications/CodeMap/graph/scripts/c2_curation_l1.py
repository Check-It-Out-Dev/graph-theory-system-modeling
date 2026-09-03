# GrothendieckV5 P4 (part 2) — L1 index update after the curation batch.
# Surgical: the six navigation groups and the "- [id] Name (n files) - text" line format are
# preserved; every changed line traces to a CurationDecision. Follows the erdos-delta pattern —
# snapshot the outgoing L1, update in place, link the snapshot.
#
# STORE PORT (2026-09-02): Neo4j -> Ladybug Store. Neo4j held TWO master nodes
# (NavigationMaster with the index + V3Master with provenance stamps); the store has one
# Master row, so both write blocks merge into its columns + props. Index and caveat text
# byte-identical to the executed 2026-09-02 batch.
#
# Usage: PYTHONUTF8=1 python c2_curation_l1.py [--dry]

import argparse
import datetime as _dt
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "authoring")))
from ladybug_store import Store

BATCH, BY = "2026-09-02-curation", "GrothendieckV5"

IDX = """billing-legal: payments, consent, user core, notifications
  - [11] Subscriptions, payments & consent (208 files) - The largest backend subsystem: Stripe payments, invoicing with retry, consent enforcement, legal documents, registry lookup, the payments boot guard and the lifecycle crons.
  - [10] User identity & token exchange (39 files) - The User entity (154 in-edges, the most imported node in the estate), Firebase and SMTP glue, token exchange and social-auth sessions.
  - [12] Notifications & domain events (36 files) - Notification entities, transactional event listeners, email frequency preferences.
opportunities: the campaign domain
  - [4] Partnership opportunity lifecycle (172 files) - The campaign domain: PartnershipOpportunity and applied-opportunity entities, flows and permissions, now including its own 49-file test suite (ex-sub-13, merged 2026-09-02: 60.3% of that suite's coupling pointed here).
auth-security: authentication, second factor, validation rules
  - [9] Auth journeys & BDD harness (103 files) - Auth flows, registration, the post-auth enforcement filters that return 403 with a valid token, and the Cucumber scenario/actor harness that exercises them.
  - [6] Two-factor auth & user cache (51 files) - TOTP and step-up second factor plus the Firestore-backed user cache (UserCacheService, 51 in-edges).
  - [8] Validation, crypto & query specs (49 files) - SpecificationBuilder (48 in-edges) plus HMAC/TOTP crypto, validation annotations, recaptcha, Vimeo and social-post URL rules, with their unit tests.
platform: configuration and the cross-cutting layers
  - [3] Rate limits & runtime config (129 files) - Spring profiles (application-*.yml, incl. dev-lite), the rate-limit family, Cucumber/Spring test context, storage and upload, health, geo-IP GDPR.
  - [7] LAYER Translatable exceptions & logging (52 files) - The exception hierarchy and translatable error messages the whole backend imports: 91.1% fan-in across 15 consumers, ResourceNotFoundException alone at 101 in-edges.
  - [16] LAYER Status enums & OpenAPI contract (12 files) - Status and compensation enums, their i18n message bundles and the OpenAPI spec config: 93.7% fan-in across 12 consumers, zero internal edges.
  - [2] LAYER Social connection data model (15 files) - UserSocialConnection and Platform entities, repositories, DTOs and mapper: 86.2% fan-in across 8 consumers.
  - [0] User preferences & geo distance (27 files) - Preference entities and rules, geo distance calculation, PII masking utilities.
frontend: the Angular app, one group split into nine domain children
  - [17] GROUP Greenfield Angular frontend (406 files) - The whole Angular greenfield app, 100% FE-repo pure, cohesion 0.723 (the highest of any subsystem). Answer through its children, not through the group.
    - [170] FE auth, 2FA & interceptors (69 files) - core/auth and feature/auth, step-up, two-factor and the interceptor chain. Holds the largest internal seam in the app: core/auth to feature/auth, 40 edges.
    - [177] FE shell, i18n & generated client (74 files) - Layout, shell, theme, notifications, i18n, shared, health, error page, landing, root build config, and the collapsed generated OpenAPI client (ex-sub-18; 156 in-edges arriving from all nine children).
    - [173] LAYER FE fixtures & E2E harness (67 files) - src/app/sandbox fixtures plus the e2e-tests harness. 85% Resource purity, 4 internal edges, 23 files of degree zero: a fixture-data layer, not a feature slice.
    - [176] FE profile, settings & admin (52 files) - Profile, user, upload, settings, social connections, preferences, addresses, company, team, admin and dictionary surfaces.
    - [171] FE opportunities & applications (43 files) - Opportunity browsing, applied opportunities, collaborations and grants.
    - [172] FE onboarding survey (42 files) - The onboarding survey. Fully self-contained: cohesion 1.000, zero edges to any sibling.
    - [174] FE plan, billing & consent (23 files) - Plan and billing screens, subscription client, frozen API client, consent and legal. Two of its files carry a measured cross-repo affinity to backend [11].
    - [175] FE support & help centre (20 files) - Support ticket and help-centre screens with their core service.
    - [178] FE demo mode (16 files) - The demo interceptor and demo-mode screens. Cohesion 0.824.
data-support: addresses, deletion, FAQ, tickets
  - [1] Address resolution & storage (18 files) - Address entities, repositories, primary/copy resolution and their integration tests.
  - [5] Account deletion & Instagram sync (48 files) - Admin cascade deletion (the preview/request/confirm family fanning over 21 files) plus the Instagram OAuth and data-deletion callback services.
  - [14] FAQ content & categories (15 files) - FAQ entities and categories backing the public support content.
  - [15] Support tickets & attachments (35 files) - Support ticket entities, attachments and flows, the public ticket create/status surface, plus TicketAccessTokenService."""

CURATION_CAVEAT = (
    "subsystems are CURATED (GrothendieckV5 batch 2026-09-02-curation), superseding the "
    "pre-curation candidate caveat: 17 top-level navigators - 13 slices, 3 layers (2, 7, 16) and "
    "1 group (17) with 9 children (170-178). Layers were promoted on a measured fan-in criterion "
    "applied uniformly to all 19 candidates (in-share >= 85%, >= 8 distinct consumers, no seam "
    "partner above 40%); it selects exactly 2, 7 and 16. sub-13 merged into sub-4 and sub-18 into "
    "child 177; both navigators are RETAINED with role MERGED and a [:SUPERSEDED_BY] edge to "
    "their absorber, never deleted. Curated membership lives in CONTAINS_MEMBER plus the new "
    "curated_subsystem property on the 455 moved or split nodes; v4_subsystem is the untouched "
    "measured layer, so the two disagree BY DESIGN - read CONTAINS_MEMBER for navigation and "
    "v4_subsystem only for provenance")

CLUE_CAVEAT = (
    "clue BODIES are stale as of the curation batch and are the next agent's work, not a defect: "
    "GrothendieckV5 changed names, roles and membership but did NOT regenerate ai_summary, "
    "responsibilities, caveats, spines or contracts. Every navigator carries "
    "clue_body_status='STALE-awaiting-erdos' and the 9 new children carry "
    "clue_body_status='MISSING' with no clue body at all. clue_version reads 'curated-v1' to mark "
    "the curation, NOT a regenerated clue. Erdos must re-clue all 26 navigators before the L2 "
    "text can be trusted; until then trust name, role, size and membership only")


def main(dry):
    s = Store(read_only=dry)
    now = _dt.datetime.now().isoformat()
    m = s.one("MATCH (nm:Master) RETURN nm.mid AS mid, nm.ai_summary AS summ, "
              "nm.subsystem_index AS idx, nm.global_caveats AS cav, nm.props AS props")
    mp = json.loads(m["props"] or "{}")
    cav = json.loads(m["cav"] or "[]")
    new_cav, replaced = [], False
    for c in cav:
        if c.startswith("subsystems are PRE-CURATION candidates"):
            new_cav.append(CURATION_CAVEAT); replaced = True
        else:
            new_cav.append(c)
    if not replaced:
        new_cav.append(CURATION_CAVEAT)
    new_cav.append(CLUE_CAVEAT)
    print(f"caveats {len(cav)} -> {len(new_cav)} (pre-curation caveat replaced: {replaced})")
    print(f"index lines {len(IDX.splitlines())}")
    if dry:
        print("DRY — no writes"); return

    body = dict(level="L1", clue_version=mp.get("clue_version"), ai_summary=m["summ"],
                subsystem_index=m["idx"], global_caveats=m["cav"],
                t_valid=mp.get("generated_at"),
                reason=f"curation batch {BATCH}: 17 renames, 3 layer retypes, 1 split "
                       f"into 9 children, 2 merges")
    s.create("ClueSnap", dict(   # the master has no sub: sub_id=-1, level lives in body
        snap_id=f"c2l1-{BATCH}", sub_id=-1, taken_at=now, t_created=now, t_expired="",
        superseded_by=BATCH, body=json.dumps(body, ensure_ascii=False, default=str)),
        "snap_id")
    cav_lit = json.dumps(new_cav, ensure_ascii=False)
    s.conn.execute(
        "MATCH (nm:Master) WHERE nm.mid = $mid "
        f"SET nm.subsystem_index = $idx, nm.global_caveats = {Store.lit(cav_lit)}",
        parameters=dict(mid=m["mid"], idx=IDX))
    s.merge_props("Master", "mid", m["mid"], dict(
        superseded_index_curation=m["idx"], curated_by=BY, curated_at=now,
        curation_batch=BATCH, subsystem_count=17, child_count=9,
        curated_subsystem_count=17, curated_child_count=9, curation_decisions=20))

    # honest staleness stamps — queryable, not just prose
    for r in s.q("MATCH (n:Nav) RETURN n.sub_id AS sid, n.role AS role, n.parent AS parent"):
        st = ("MISSING" if r["parent"] is not None
              else "SUPERSEDED" if r["role"] == "MERGED"
              else "STALE-awaiting-erdos")
        s.conn.execute("MATCH (n:Nav) WHERE n.sub_id = $sid SET n.clue_body_status = $st",
                       parameters=dict(sid=r["sid"], st=st))
    chk = s.one("MATCH (nm:Master) WHERE nm.mid = $mid RETURN nm.subsystem_index AS idx, "
                "nm.global_caveats AS cav, nm.props AS props", dict(mid=m["mid"]))
    cp = json.loads(chk["props"] or "{}")
    print(f"VERIFY L1: subsystem_count={cp.get('subsystem_count')} "
          f"caveats={len(json.loads(chk['cav']))} "
          f"index_lines={len(chk['idx'].splitlines())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    main(ap.parse_args().dry)
