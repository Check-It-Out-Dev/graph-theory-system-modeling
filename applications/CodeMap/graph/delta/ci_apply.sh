#!/usr/bin/env bash
# Apply a /codemap decision from an issue comment (or the 48 h timeout): download the proposing run's
# artifacts, apply, bump the pack version, publish the Release, open a pull request with the ledger,
# the curation note, the rebuilt navigator prompt and seen.json, then close the issue with a summary.
# Inputs (env): ISSUE, ISSUE_BODY, COMMENT, BY, GH_TOKEN. Runs inside the graph repository checkout (any ref).
set -euo pipefail
R=applications/CodeMap
MARKER=$(printf '%s\n' "$ISSUE_BODY" | grep -o '<!-- codemap-delta [^>]* -->' | head -1 || true)
[ -n "$MARKER" ] || { echo "no codemap-delta marker on issue #$ISSUE"; exit 0; }
KEY=${MARKER#<!-- codemap-delta }; KEY=${KEY% -->}
set -- $KEY
REPO=${1%@*}; SHA=${1#*@}; REF=${2:-}; REF=${REF#ref=}; REF=${REF:-main}
echo "[apply] issue #$ISSUE → $REPO@$SHA (proposed on $REF) by $BY: $COMMENT"
# issue_comment and schedule events run the workflow from the default branch; the decision applies on the
# branch that produced the proposal (after the merge that is main as well)
git fetch -q origin "$REF" && git checkout -q "$REF"

# the proposing run: the newest run of this workflow that uploaded proposal-<repo>-<sha>
RUN_ID=$(gh api "repos/$GITHUB_REPOSITORY/actions/artifacts?name=proposal-$REPO-$SHA&per_page=1" --jq '.artifacts[0].workflow_run.id' || true)
[ -n "$RUN_ID" ] && [ "$RUN_ID" != "null" ] || { gh issue comment "$ISSUE" --body "The proposal's artifacts have expired (30 days); re-run graph-delta for \`$REPO@$SHA\` and decide on the new issue."; exit 0; }
mkdir -p work && gh run download "$RUN_ID" -n "delta-$REPO-$SHA" -D work/delta && gh run download "$RUN_ID" -n "proposal-$REPO-$SHA" -D work/proposal
D="work/delta/$REPO-$SHA"
cp work/proposal/proposal.json "$D/proposal.json"

# next pack version: patch + 1 of the manifest inside pack.next (which came from the latest Release)
CUR=$(python -c "import json;print(json.load(open('$D/pack.next/manifest.json')).get('pack_version','1.0.0'))")
NEXT=$(python -c "v='$CUR'.split('.');v[-1]=str(int(v[-1])+1);print('.'.join(v))")
export DECISION="$COMMENT"
python "$R/graph/delta/apply.py" --proposal "$D/proposal.json" --delta "$D/delta.json" --pack-next "$D/pack.next" \
  --command "$DECISION" --by "$BY" --version "$NEXT" --ledger "$R/graph/ledger" --notes "$R/prompts/navigator/curation_notes.md" \
  --out "$D/apply.json" | tee -a "$GITHUB_STEP_SUMMARY"
REJECTED=$(python -c "import json;print(json.load(open('$D/apply.json')).get('rejected') or '')")

if [ -z "$REJECTED" ]; then
  # the prompt for the new pack (template + pack + notes), the seen list, the Release
  PYTHONUTF8=1 python "$R/tools/prompt/build_navigator.py" --pack "$D/pack.next" --out "$R/prompts/navigator/active.md"
  python - <<PY
import json, os
p = "$R/graph/delta/seen.json"
d = json.load(open(p)) if os.path.exists(p) else {"heads": []}
d["heads"] = sorted(set(d["heads"]) | {"$SHA"})[-500:]
json.dump(d, open(p, "w"), indent=1)
PY
  python "$R/tools/pack/build_release.py" --version "$NEXT" --pack "$D/pack.next" --out dist \
    --note "delta $REPO@${SHA:0:7} · decision by $BY: $DECISION" --publish
fi

git config user.name "graph-delta"; git config user.email "graph-delta@users.noreply.github.com"
BR="graph-delta/$REPO-${SHA:0:7}-$NEXT"
git checkout -b "$BR"
git add "$R/graph/ledger" "$R/prompts/navigator/curation_notes.md" "$R/prompts/navigator/active.md" "$R/graph/delta/seen.json" 2>/dev/null || true
if git diff --cached --quiet; then
  echo "nothing to commit (rejected or no change)"
else
  git commit -q -m "graph-delta: $REPO@${SHA:0:7} → pack $NEXT — $DECISION (decided by $BY on #$ISSUE)"
  git push -q origin "$BR"
  PR_URL=$(gh pr create --base "$REF" --head "$BR" --title "pack $NEXT: $REPO@${SHA:0:7} — $DECISION" \
    --body "Decision on #$ISSUE by @$BY: \`$DECISION\`. Ledger row \`graph/ledger/$NEXT.json\`, curation note appended, navigator prompt rebuilt, seen.json updated. Release: pack-$NEXT (the VPS reloads it within ten minutes).")
  echo "pull request: $PR_URL" | tee -a "$GITHUB_STEP_SUMMARY"
fi
SUMMARY=$(python -c "import json;d=json.load(open('$D/apply.json'));print(f\"assignments {len(d['assignments'])}, new subsystems {len(d['new_subsystems'])}, changed subsystems {d['changed_subsystems']}, FAQ invalidated {len(d.get('mfq_invalidated',[]))}\")")
gh issue comment "$ISSUE" --body "Applied \`$DECISION\` by @$BY → pack **$NEXT**: $SUMMARY. ${PR_URL:+Pull request: $PR_URL}"
gh issue close "$ISSUE" --reason completed
