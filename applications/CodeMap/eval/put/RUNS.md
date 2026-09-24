# Campaign runs

Every prompt-under-test campaign that produced a committed number: its Actions run on the self-hosted runner, the artifact it uploaded (id and sha256 digest, 90-day retention; the full streams are also in the release asset named in the last column when one exists), and what it measured.

| date | label | mode | Actions run | artifact (id, digest) | prompt | runs | score | release asset |
|---|---|---|---|---|---|---|---|---|
| 2026-09-23 | box-smoke-1 | smoke | [35884706660](https://github.com/Check-It-Out-Dev/checkitout-backend/actions/runs/35884706660) | 10762652703, `f333340ca77053c8` | `put@0c3af0cdaacfe166` | 1 | 0.9125 | |
| 2026-09-23 | baseline-2026-09-23 | baseline | [35885709905](https://github.com/Check-It-Out-Dev/checkitout-backend/actions/runs/35885709905) | 10765315832, `e0a71cb31d72126c` | `put@0c3af0cdaacfe166` | 18 | 0.9109 | |
| 2026-09-23 | hosted-smoke-1 | smoke | [35886327924](https://github.com/Check-It-Out-Dev/checkitout-backend/actions/runs/35886327924) | 10764554776, `ca079d6121cf0aeb` | `put@0c3af0cdaacfe166` | 1 | 0.9875 | |
| 2026-09-24 | baseline-2026-09-23-r5 | rejudge | [35986676711](https://github.com/Check-It-Out-Dev/checkitout-backend/actions/runs/35986676711) | 10802815732, `11712ad7ba58b1fa` | `put@0c3af0cdaacfe166` | 18 | 0.9074 | |
| 2026-09-24 | gepa-2026-09-24 | gepa | [35987646721](https://github.com/Check-It-Out-Dev/checkitout-backend/actions/runs/35987646721) | 10805832219, `26967d2c278b8f42` | `put@68531a1b36fc9a72` | 0 | 0.9792 | |
| 2026-09-24 | certify-2026-09-24 | certify | [35995327027](https://github.com/Check-It-Out-Dev/checkitout-backend/actions/runs/35995327027) | 10811422522, `b102093d235b6085` | `put@68531a1b36fc9a72` | 40 | 0.9679 | |

Notes. `runs` counts the scored coder runs a summary rests on: a `gepa` summary records 0 there because its 54
cells are GEPA metric calls (one replicate per candidate and task, in `gepa.log.jsonl` and the candidate directories),
and its verdict is the certification's, not its own. `runs/s4-smoke/` is the first local run of S4, made before the
contract was frozen in S5 and outside the evidence plane; it is kept as the harness's first trace and carries no
published number, so it has no row here.
