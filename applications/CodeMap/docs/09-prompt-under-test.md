# 09 — Prompt under test: can we write a conventions prompt and measure that the agent follows it?

_Arc 5, set by the owner 2026-09-23. Design of record: `~/.claude/plans/GOAL-prompt-under-test.md` (not in this repository). Decision log: `docs/07-ai-quality-governance.md` D-R29 onwards. Code: `eval/put/`._

The question is one: a team writes a prompt that tells coding agents how the code base is built (its conventions and design patterns), hands it to every agent, and wants proof that the agents follow it. This document is the long form of that proof: the pipeline, its statistics, the runs and what they showed. The graph appears once, as one of the rules ("look in the code graph before you edit"); whether the graph pays for itself is not asked here.

## S0 — provisioning and probes (2026-09-23)

Every item below was run, not assumed. Raw transcripts: `eval/put/probes/s0/`.

| Item | Result |
|---|---|
| Models behind the aliases | `--model sonnet` → `claude-sonnet-5` (the coder); `--model opus` → `claude-opus-5-5` (judge, reflector). Read from the `system.init` event; pinned for the arc, stored per run. |
| Bash under `--restricted` | Present when `--tools` names it. **Not confined**: with Bash allowed wholesale, `echo probe > ../x` wrote outside the worktree. The Write tool is confined ("outside … `--restricted` confines the file tools to the working directory"). |
| A narrow Bash allowlist under `--permission-mode dontAsk` | `--allowedTools "Bash(./mvnw *),Bash(./mvnw.cmd *),Bash(git status*),Bash(git diff*)"` admitted `./mvnw -v`, `./mvnw.cmd -v`, `git status`; **denied** `echo … > ../x`, `rm`, `python -c`; read-only `ls` passed (Claude Code auto-allows read-only commands). So the coder runs with that allowlist, never with bare `Bash`. |
| JAVA_HOME in the tool's shell | visible (`C:\Users\Norbert\.jdks\corretto-21.0.7`, set for the user and the machine) |
| Worktree lifecycle (Windows) | `git worktree add --detach` 0.5 s; `remove --force` 0.2 s clean, 0.7 s with `target/` |
| Build and test in a fresh worktree | `./mvnw -q -o test-compile` 30 s cold (the repository's Maven build-cache extension restores unchanged modules); one unit test class with `-Ptest -Dtest=…` 23 s. The planned warm-`target/` copy is unnecessary. |
| Evidence plane | Runner group `put-box` (id 3): visibility selected (checkitout-backend only), public repositories allowed, **restricted to one workflow**: `checkitout-backend/.github/workflows/prompt-eval.yml@refs/heads/ci/prompt-eval`. Runner `norbert-box` (Windows, labels `self-hosted,X64,Windows,put`) online; started at logon from the user's Run key (`conhost.exe --headless C:\actions-runner\run.cmd`; a scheduled task needs administrator rights). |
| Campaign trigger | The owner keeps the work on feature branches (2026-09-23 16:28), so the workflow lives on the backend branch `ci/prompt-eval` and a campaign is a commit that edits `.github/prompt-eval/campaign.json` (push trigger with a path filter; no pull-request trigger). No other backend workflow runs on a push to that branch (checked: pushes trigger only on `main`, `greenfield`, `prod`). First run with `mode: none`: plan job green, both campaign jobs skipped. |
| Environment | `ai-review` holds `CLAUDE_CODE_OAUTH_TOKEN` and has no branch policy, so a job on `ci/prompt-eval` can use it. The executor never reads the secret. |
| Disk | 152 GB free on C:; a worktree is 26 MB checked out, ~55 MB with `target/`, removed after its artifacts are captured |
