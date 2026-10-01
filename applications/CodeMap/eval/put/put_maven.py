"""Maven in a worktree of the backend: compile, run named test classes, read the surefire reports.

The unit tier only (`-Ptest` on every invocation, law 14): integration tests need Docker and never run in the
harness. Windows runs `mvnw.cmd` through cmd; everything else runs `./mvnw`.
"""

import glob
import os
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET


def wrapper(cwd):
    if sys.platform == "win32":
        return ["cmd", "/c", os.path.join(cwd, "mvnw.cmd")]
    return [os.path.join(cwd, "mvnw")]


def _env():
    env = dict(os.environ)
    env.pop("MAVEN_OPTS", None)
    return env


# harness-side Maven processes at once (agent sessions run their own Maven inside their sessions)
SLOTS = threading.Semaphore(int(os.environ.get("PUT_MAVEN_SLOTS", "2")))


def run(cwd, args, timeout=1800, log_path=None):
    """-> (exit code, seconds, tail of the output)."""
    with SLOTS:
        return _run(cwd, args, timeout, log_path)


def _run(cwd, args, timeout, log_path):
    t0 = time.time()
    try:
        p = subprocess.run(wrapper(cwd) + ["-q", "-B"] + list(args), cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=_env())
        out, code = (p.stdout or "") + (p.stderr or ""), p.returncode
    except subprocess.TimeoutExpired as ex:
        out, code = f"timeout after {timeout}s: {ex}", 124
    if log_path:
        with open(log_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(out)
    return code, round(time.time() - t0, 1), out[-4000:]


def compile_tests(cwd, log_path=None):
    return run(cwd, ["test-compile", "-Ptest"], log_path=log_path)


def test(cwd, classes, log_path=None):
    """Run the named test classes (simple names) in the unit tier."""
    selector = ",".join(classes)
    shutil.rmtree(os.path.join(cwd, "target", "surefire-reports"), ignore_errors=True)
    return run(cwd, ["test", "-Ptest", f"-Dtest={selector}", "-Dsurefire.failIfNoSpecifiedTests=false",
                     "-Djacoco.skip=true"], log_path=log_path)


def reports(cwd, classes=None):
    """-> {simple class name: {"tests", "failures", "errors", "skipped", "cases": [{name, status}]}} from surefire XML.
    Nested classes (`Outer$Inner`) count toward their outer class."""
    out = {}
    for path in glob.glob(os.path.join(cwd, "target", "surefire-reports", "TEST-*.xml")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        full = root.get("name", "")
        simple = full.rsplit(".", 1)[-1].split("$", 1)[0]
        if classes and simple not in classes:
            continue
        row = out.setdefault(simple, {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "cases": []})
        for key in ("tests", "failures", "errors", "skipped"):
            row[key] += int(root.get(key, "0") or 0)
        for case in root.iter("testcase"):
            status = "passed"
            if case.find("failure") is not None:
                status = "failed"
            elif case.find("error") is not None:
                status = "error"
            elif case.find("skipped") is not None:
                status = "skipped"
            row["cases"].append({"name": f"{case.get('classname', '').rsplit('.', 1)[-1]}.{case.get('name')}",
                                 "status": status})
    return out


def pass_rate(report, classes):
    """-> (passed, total) over the named classes; a class with no report counts as one failed test."""
    passed = total = 0
    for c in classes:
        r = report.get(c)
        if not r or r["tests"] == 0:
            total += 1
            continue
        total += r["tests"] - r["skipped"]
        passed += r["tests"] - r["skipped"] - r["failures"] - r["errors"]
    return passed, total
