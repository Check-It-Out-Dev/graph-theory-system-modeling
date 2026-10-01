"""Fixtures for the deterministic checks: every reference solution, and degraded variants that break exactly one rule,
captured from real worktrees and committed so `test_put_checks.py` runs anywhere without the repository under test.

    PYTHONUTF8=1 python eval/put/put_fixtures.py --source C:/Users/Norbert/erdos-ws/backend

A fixture directory holds what a run directory holds for the checks (diff.patch, after/, calls.json, tests.json,
meta.json, answer.md) plus `base/` (the base text of every changed or exemplar file) and `fixture.json` (the task
id and the rule the variant must fail). The trace of a fixture is synthetic: a graph query, a read of the task's
first exemplar, the edits, the completion line; process variants change only the trace.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import put_paths
import put_contract
import put_runner
import put_tasks
import put_worktree

OUT = os.path.join(put_paths.CODEMAP, "remote", "tests", "fixtures", "put")
PKG = "src/main/java/com/sm/instagram/platform/"


def _edit(wt, rel, old, new, count=1):
    p = os.path.join(wt, *rel.split("/"))
    with open(p, encoding="utf-8") as f:
        s = f.read()
    if old not in s:
        raise RuntimeError(f"{rel}: {old!r} not found")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s.replace(old, new, count))


def _sub(wt, rel, pattern, repl):
    p = os.path.join(wt, *rel.split("/"))
    with open(p, encoding="utf-8") as f:
        s = f.read()
    s2, n = re.subn(pattern, repl, s, flags=re.M)
    if not n:
        raise RuntimeError(f"{rel}: pattern {pattern!r} not found")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s2)


def _rm(wt, rel):
    os.remove(os.path.join(wt, *rel.split("/")))


def _mv(wt, src, dst, old_pkg=None, new_pkg=None):
    a, b = os.path.join(wt, *src.split("/")), os.path.join(wt, *dst.split("/"))
    os.makedirs(os.path.dirname(b), exist_ok=True)
    shutil.move(a, b)
    if old_pkg:
        _edit(wt, dst, f"package {old_pkg};", f"package {new_pkg};")


CRON = PKG + "support/ticket/services/StaleTicketCloseCronJob.java"
CRON_TEST = "src/test/java/com/sm/instagram/platform/unit/service/StaleTicketCloseCronJobUnitTest.java"
LISTENER = PKG + "support/ticket/event/SupportTicketCreatedEventListener.java"
CONVERSION = PKG + "currency/CurrencyConversionService.java"
FAQ_CTRL = PKG + "support/faq/FaqController.java"
FAQ_SVC = PKG + "support/faq/services/FaqService.java"
TICKET = PKG + "support/ticket/models/SupportTicket.java"
MASTER = "src/main/resources/db/changelog/changelog.xml"
PL = "src/main/resources/messages_pl.properties"

# (task, variant, rule that must fail, mutation(wt) or None, trace kind)
VARIANTS = [
    ("stale-ticket-close", "field-injection", "constructor_injection",
     lambda wt: _edit(wt, CRON, "    private final SupportTicketService supportTicketService;",
                      "    @Autowired\n    private SupportTicketService supportTicketService;") or
     _edit(wt, CRON, "import lombok.RequiredArgsConstructor;\n", "import org.springframework.beans.factory.annotation.Autowired;\n") or
     _edit(wt, CRON, "@RequiredArgsConstructor\n", ""), "normal"),
    ("stale-ticket-close", "no-lock", "scheduler_lock",
     lambda wt: _sub(wt, CRON, r"^\s*@SchedulerLock\(.*\)\n", ""), "normal"),
    ("stale-ticket-close", "technical-package", "feature_first",
     lambda wt: _mv(wt, CRON, PKG + "jobs/StaleTicketCloseCronJob.java", "com.sm.instagram.platform.support.ticket.services",
                    "com.sm.instagram.platform.jobs") or
     _edit(wt, PKG + "jobs/StaleTicketCloseCronJob.java", "import lombok.RequiredArgsConstructor;",
           "import com.sm.instagram.platform.support.ticket.services.SupportTicketService;\nimport lombok.RequiredArgsConstructor;") or
     _edit(wt, CRON_TEST, "import com.sm.instagram.platform.support.ticket.services.StaleTicketCloseCronJob;",
           "import com.sm.instagram.platform.jobs.StaleTicketCloseCronJob;"), "normal"),
    ("stale-ticket-close", "no-unit-test", "tests_written", lambda wt: _rm(wt, CRON_TEST), "normal"),
    ("stale-ticket-close", "deletes-a-test", "scope",
     lambda wt: _rm(wt, "src/test/java/com/sm/instagram/platform/unit/service/SupportTicketModelsUnitTest.java"), "normal"),
    ("stale-ticket-close", "edits-first", "graph_first", None, "graph_after_edit"),
    ("stale-ticket-close", "no-exemplar", "exemplar_read", None, "no_exemplar"),
    ("stale-ticket-close", "keeps-working", "marker", None, "call_after_marker"),
    ("ticket-created-after-commit", "plain-listener", "after_commit_listener",
     lambda wt: _edit(wt, LISTENER, "@TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)", "@EventListener") or
     _edit(wt, LISTENER, "import org.springframework.transaction.event.TransactionPhase;\nimport org.springframework.transaction.event.TransactionalEventListener;",
           "import org.springframework.context.event.EventListener;"), "normal"),
    ("exchange-rate-port", "plain-exception", "translatable_errors",
     lambda wt: _edit(wt, CONVERSION, 'new BusinessRuleTranslatableException(\n                        "error.business.exchange_rate_unavailable", isoCode)',
                      'new IllegalArgumentException("No rate for " + isoCode)'), "normal"),
    ("exchange-rate-port", "missing-polish-key", "translatable_errors",
     lambda wt: _sub(wt, PL, r"^error\.business\.exchange_rate_unavailable=.*\n", ""), "normal"),
    ("exchange-rate-port", "service-imports-adapter", "ports_adapters",
     lambda wt: _edit(wt, CONVERSION, "import com.sm.instagram.platform.currency.port.ExchangeRatePort;",
                      "import com.sm.instagram.platform.currency.adapter.FixedExchangeRateAdapter;\nimport com.sm.instagram.platform.currency.port.ExchangeRatePort;"), "normal"),
    ("faq-reorder", "no-preauthorize", "controller_guards",
     lambda wt: _edit(wt, FAQ_CTRL, '    @PutMapping("/categories/{categoryId}/order")\n    @PreAuthorize("hasAuthority(\'ADMIN\')")',
                      '    @PutMapping("/categories/{categoryId}/order")'), "normal"),
    ("faq-reorder", "request-type", "dto_naming",
     lambda wt: _edit(wt, FAQ_CTRL, "@Valid @RequestBody FaqReorderDtoIn dto", "@Valid @RequestBody List<Long> faqIds") or
     _edit(wt, FAQ_CTRL, "faqService.reorderCategory(categoryId, dto.getFaqIds());", "faqService.reorderCategory(categoryId, faqIds);"), "normal"),
    ("support-ticket-version", "no-version-annotation", "version_field",
     lambda wt: _sub(wt, TICKET, r"^\s*@Version\n", ""), "normal"),
    ("support-ticket-version", "edits-old-changeset", "liquibase_changeset",
     lambda wt: _rm(wt, "src/main/resources/db/changelog/2026/09/23-09-2026-add-version-to-support-ticket.sql") or
     _sub(wt, MASTER, r"^.*September 2026: Optimistic locking.*\n.*23-09-2026-add-version-to-support-ticket.sql.*\n", "") or
     _sub(wt, "src/main/resources/db/changelog/2026/09/02-09-2026-widen-ticket-reference-column.sql",
          r"^-- rollback", "ALTER TABLE support_ticket ADD COLUMN IF NOT EXISTS version BIGINT NOT NULL DEFAULT 0;\n\n-- rollback"), "normal"),
]


def synthetic_calls(task, changed, kind):
    """A plausible trace: graph first, one exemplar read, edits, the completion line."""
    ex = task["exemplar_files"][0]
    calls = [{"name": "mcp__graph__graph_query", "input": {"statement": "MATCH (e:Entity) RETURN e.file_path LIMIT 1"}}]
    read_ex = {"name": "Read", "input": {"file_path": "C:/put-ws/runs/x/" + ex}}
    edits = [{"name": "Write" if st == "??" else "Edit", "input": {"file_path": "C:/put-ws/runs/x/" + p}} for st, p in changed]
    if kind == "graph_after_edit":
        calls = [read_ex] + edits[:1] + calls + edits[1:]
    elif kind == "no_exemplar":
        calls = calls + edits
    else:
        calls = calls + [read_ex] + edits
    calls.append({"name": "Bash", "input": {"command": "./mvnw -q test -Ptest -Dtest=X"}})
    out = []
    for i, c in enumerate(calls, 1):
        out.append({"i": i, "name": c["name"], "input": c["input"], "after_marker": False, "is_error": False, "result_head": ""})
    if kind == "call_after_marker":
        out.append({"i": len(out) + 1, "name": "Read", "input": {"file_path": "C:/put-ws/runs/x/pom.xml"},
                    "after_marker": True, "is_error": False, "result_head": ""})
    return out


def exemplar_files(source, base, task):
    """Existing main files whose base text matches the task's first exemplar pattern (the file a trace reads)."""
    pat = task["exemplars"][0]
    p = subprocess.run(["git", "grep", "-l", "-P", pat, base, "--", "src/main"], cwd=source, capture_output=True, text=True)
    files = [line.split(":", 1)[1] for line in p.stdout.splitlines() if ":" in line]
    return sorted(files)[:1]


def build(source, instance, only=None):
    contract = put_contract.load(instance)
    base = contract["base_sha"]
    tasks = {t["id"]: t for t in put_contract.tasks(instance)}
    wanted = [(t, "reference", None, None, "normal") for t in tasks] + VARIANTS
    work = os.path.join(os.path.expanduser("~"), "put-ws", "fixtures")
    os.makedirs(OUT, exist_ok=True)
    for task_id, variant, rule, mutate, kind in wanted:
        name = f"{task_id}--{variant}"
        if only and name not in only:
            continue
        task = dict(tasks[task_id])
        task["exemplar_files"] = exemplar_files(source, base, task)
        wt = os.path.join(work, name)
        put_worktree.create(source, base, wt)
        try:
            put_worktree.apply(wt, put_tasks.reference_patch(instance, task_id))
            if mutate:
                mutate(wt)
            diff = put_worktree.capture(wt)
            changed = put_worktree.changed_paths(wt)
            d = os.path.join(OUT, name)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
            with open(os.path.join(d, "diff.patch"), "w", encoding="utf-8", newline="\n") as f:
                f.write(diff)
            put_runner.snapshot(wt, changed, d)
            for rel in [p for st, p in changed if not st.startswith("?") and st != "A"] + task["exemplar_files"] + \
                    ["src/main/resources/messages_en.properties", "src/main/resources/messages_pl.properties"]:
                txt = subprocess.run(["git", "show", f"{base}:{rel}"], cwd=source, capture_output=True)
                if txt.returncode == 0:
                    dst = os.path.join(OUT, "base", *rel.split("/"))            # one shared copy per base file
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with open(dst, "wb") as f:
                        f.write(txt.stdout.replace(b"\r\n", b"\n"))
            new_tests = put_tasks.new_unit_tests(diff)
            tests = {"build_green": True, "own_classes": new_tests,
                     "own_reports": {n: {"tests": 2, "failures": 0, "errors": 0, "skipped": 0} for n in new_tests},
                     "own": [2 * len(new_tests), 2 * len(new_tests)], "hidden": [4, 4], "pass_to_pass": [10, 10]}
            for fname, obj in (("tests.json", tests), ("calls.json", synthetic_calls(task, changed, kind)),
                               ("meta.json", {"task": task_id, "marker": True}),
                               ("fixture.json", {"task": task_id, "variant": variant, "must_fail": rule})):
                with open(os.path.join(d, fname), "w", encoding="utf-8", newline="\n") as f:
                    json.dump(obj, f, indent=1)
            with open(os.path.join(d, "answer.md"), "w", encoding="utf-8", newline="\n") as f:
                f.write("Changed the files; ran the unit tests.\n=== ANSWER COMPLETE ===\n")
            print("fixture", name, len(changed), "files", flush=True)
        finally:
            put_worktree.remove(source, wt)
    types = subprocess.run(["git", "grep", "-h", "-E", r"class\s+\w+\s+extends\s+\w+", base, "--", "src/main/java"],
                           cwd=source, capture_output=True).stdout.decode("utf-8", "replace")
    pairs = sorted(set(re.findall(r"class\s+(\w+)\s+extends\s+(\w*Exception)\b", types)))
    with open(os.path.join(OUT, "exception_hierarchy.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(pairs, f, indent=0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="C:/Users/Norbert/erdos-ws/backend")
    ap.add_argument("--instance", default="backend-conventions")
    ap.add_argument("--only", action="append")
    a = ap.parse_args(argv)
    build(a.source, a.instance, a.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
