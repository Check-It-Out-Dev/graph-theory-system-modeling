# Incidents — what broke, what it cost, what gate now catches it

Kept in the open on purpose. Dates are the box's clock (CEST). Last revised 2026-09-16.

| date | what | impact | fix | gate added |
|---|---|---|---|---|
| 2026-09-16 | `fetch_pack.py` extracted a Release over an open LadybugDB on Windows; `codemap.lbdb` came out zero bytes | local engine dead until restored from the tarball | stage in `.incoming`, verify every manifest hash, `os.replace` per file | `remote/tests/test_reload.py` (swap under a live engine) |
| 2026-09-16 | the VPS container's entrypoint failed: `git archive` on Windows produced CRLF scripts | deploy failed before the service started | `.gitattributes` LF pin, `git -c core.autocrtlf=false archive`, `sed -i 's/\r$//'` in the Dockerfile | deploy script checks the archive |
| 2026-09-16 | certbot DNS-01 hooks hit Cloudflare error 971 (throttle) from the VPS | no certificate for `codemap.checkitout.app` | HTTP-01 webroot (`/var/www/le`) | `deploy.sh` cert check (`sudo test -f`) |
| 2026-09-16 | Loki push returned 400 on a hand-built body | no log lines in Grafana Cloud | JSON body built in Python; env names aliased (`GRAFANA_CLOUD_*`) | `remote/tests/test_credits_metrics.py` (push shapes) |
| 2026-09-16 | judge κ 0.30 on the first calibration | judge not trustworthy | a `located` dimension; oracle restricted to where-archetypes and the first five pointers; disputes on located only → κ 0.84 | `calibrate.py` gate ≥ 0.6 |
| 2026-09-16 | a `/codemap accept` comment posted from Git Bash arrived as `C:/Program Files/Git/codemap accept` | the decision job ignored it | `MSYS_NO_PATHCONV=1` for `gh` calls with slash-led bodies | the decide job's `if:` (`startsWith '/codemap '`) refused the mangled body — by design |
| 2026-09-16 | the decision job ran the default branch's workflow with a script that existed only on the feature branch | first three decision runs failed | the issue marker names the proposing branch; the job checks it out before running the script | `test_delta_apply` (marker with `ref=`) |
| 2026-09-16 | `pack.next` shipped without `manifest.json`; the prompt builder ignored `--pack` and rendered the repo's pack; the engine ignored `CODEMAP_PACK_DIR` | a Release could have carried the old L1/L2 in its prompt | extract carries the manifest and earlier invalidations; `Engine(pack_dir=…)` honoured everywhere | `test_delta_drift` (two packs in one process), `test_delta_apply` |
| 2026-09-16 | after the first reload the VPS prompt hash differed from the pull request's (stale notes in the container) | the served prompt ≠ the reviewed prompt | curation notes ride the pack | `build_navigator --check` on `active.md` in CI |
| 2026-09-16 | `eval/humans/personas.json` was never committed (`*.json` ignored); CI's humans suite failed on a file only the box had | red CI after the persona slice was believed green | explicit `.gitignore` negation; the file committed | CI itself |
| 2026-09-16 | the FAQ invalidation rule invalidated 29 of 104 bank rows on a six-file addition | the quality bank would have shrunk silently | narrow rule: moves invalidate both subsystems, growth only the enumerating archetypes (10 rows) | `test_delta_apply` bounds |
| 2026-09-16 | a full digest from the pack's indexed base measured 24.9 % churn on the public backend main | the pipeline stops with `mode: full` by design; no delta possible from that base | live proof ran from a base fifteen commits back; owner decision recorded (THREAT_MODEL R4) | `extract.py` churn threshold |
