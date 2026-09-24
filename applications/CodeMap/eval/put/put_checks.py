"""The deterministic checks of the instance backend-conventions: one function per rule of `contract.json`, each reading
only the run's committed artifacts (diff, after/ files, calls.json, tests.json, meta.json) and the base commit.

    check_run(run_dir, task, contract, repo)  -> {"checks": {rule: {"value", "applicable", "passed", "seen"}}, ...}

Every check is scoped to the agent's diff (law 2): the repository follows its own written rules only partly
(7 of 40 entities with @Version, 9 of 23 @Scheduled with a lock, 69 @Autowired fields at the base), so a check
that read the whole tree would score the neighbours, not the agent. A value lies in [0, 1]; `passed` is value == 1
(compliance rates count only full passes; partial credit enters the score). `seen` says what was counted, in words
a reviewer can check against the diff, never a verdict.

Java is read line by line: annotations are collected into the stack of the declaration they precede (multi-line
annotation arguments balanced by parentheses), bodies are found by brace counting. No Java parser: javalang stops
at Java 8 (records and `var` are in this code base) and tree-sitter needs a native wheel; the fixtures in
`test_put_checks.py` hold every pattern the checks rely on.
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass, field

import put_diff

SOLVED = "=== ANSWER COMPLETE ==="
GRAPH_PREFIX = "mcp__graph__"
EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
MAIN = "src/main/java/"
TEST = "src/test/java/"
BUNDLES = ("src/main/resources/messages_en.properties", "src/main/resources/messages_pl.properties")
CHANGELOG_DIR = "src/main/resources/db/changelog/"
MASTER = CHANGELOG_DIR + "changelog.xml"
MAPPING = re.compile(r"@(Get|Post|Put|Patch|Delete|Request)Mapping\b")
MONTHS = "(January|February|March|April|May|June|July|August|September|October|November|December)"


# ----------------------------------------------------------------------------- Java, line by line

@dataclass
class Member:
    line: int                       # 1-based line of the declaration
    decl: str                       # the declaration text up to '{' or ';'
    annotations: list = field(default_factory=list)       # [(line, text)]
    body: str = ""                  # text between the declaration's braces ('' for fields and abstract methods)

    def has(self, name):
        return any(re.match(r"@" + name + r"\b", a.strip()) for _, a in self.annotations)

    def annotation(self, name):
        for _, a in self.annotations:
            if re.match(r"@" + name + r"\b", a.strip()):
                return a
        return None

    @property
    def name(self):
        m = re.search(r"(?:class|interface|record|enum)\s+(\w+)", self.decl)
        if m:
            return m.group(1)
        m = re.search(r"(\w+)\s*\(", self.decl)
        if m:
            return m.group(1)
        m = re.search(r"(\w+)\s*(?:=[^;]*)?;?\s*$", self.decl)          # a field: the last name before = or ;
        return m.group(1) if m else ""


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def members(text):
    """-> [Member] for every declaration (class, method, field) with the annotations directly above it."""
    lines = _strip_comments(text or "").split("\n")
    out, pending, i = [], [], 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("@") and not s.startswith("@interface"):
            start, buf, depth = i, s, s.count("(") - s.count(")")
            while depth > 0 and i + 1 < len(lines):
                i += 1
                buf += " " + lines[i].strip()
                depth += lines[i].count("(") - lines[i].count(")")
            # an annotation followed by the declaration on the same line: split them
            m = re.match(r"(@\w+(?:\.\w+)*(?:\((?:[^()]|\([^()]*\))*\))?)\s+(.+)$", buf)
            if m and not m.group(2).startswith("@"):
                pending.append((start + 1, m.group(1)))
                out.append(_member(lines, i, m.group(2), pending))
                pending = []
            else:
                for part in re.findall(r"@\w+(?:\.\w+)*(?:\((?:[^()]|\([^()]*\))*\))?", buf):
                    pending.append((start + 1, part))
            i += 1
            continue
        if s.startswith(("package ", "import ")) or s in ("{", "}", "};"):
            pending = []
            i += 1
            continue
        if re.match(r"(public|protected|private|static|final|abstract|class|interface|record|enum|default|synchronized|"
                    r"[A-Z][\w<>\[\], ?]*\s+\w+\s*[(=;]|void\s)", s) or pending:
            out.append(_member(lines, i, s, pending))
            pending = []
        i += 1
    return out


def _member(lines, i, first, annotations):
    decl, j = first, i
    while "{" not in decl and ";" not in decl and j + 1 < len(lines) and j - i < 8:
        j += 1
        decl += " " + lines[j].strip()
    body = ""
    if "{" in decl:
        depth, k, collected = 0, i, []
        started = False
        while k < len(lines):
            for ch in lines[k]:
                if ch == "{":
                    depth += 1
                    started = True
                elif ch == "}":
                    depth -= 1
            collected.append(lines[k])
            if started and depth <= 0:
                break
            k += 1
        body = "\n".join(collected)
    return Member(line=i + 1, decl=decl.split("{")[0].strip(), annotations=list(annotations), body=body)


def classes(text):
    return [m for m in members(text) if re.search(r"\b(class|interface|record|enum)\s+\w+", m.decl)
            and not re.search(r"\bnew\s", m.decl)]


def class_of(text, line):
    """-> the innermost class whose body holds the line (by position), or the first class."""
    best = None
    for c in classes(text):
        end = c.line + c.body.count("\n")
        if c.line <= line <= end:
            best = c
    return best or (classes(text)[0] if classes(text) else None)


def added(member, change):
    """A member counts as added when its declaration line or any of its annotations is on an added line."""
    lines = {member.line} | {ln for ln, _ in member.annotations}
    return bool(lines & change.added_numbers) or change.status == "A"


# ----------------------------------------------------------------------------- the run's record

@dataclass
class Run:
    run_dir: str
    task: dict
    contract: dict
    changes: dict
    calls: list
    tests: dict
    meta: dict
    answer: str
    repo: "Repo"

    def main_java(self, statuses=("A", "M")):
        return {p: c for p, c in self.changes.items() if p.startswith(MAIN) and p.endswith(".java")
                and c.status in statuses and c.after is not None}

    def after_or_base(self, path):
        c = self.changes.get(path)
        if c is not None:
            return c.after
        return self.repo.base_text(path)


class Repo:
    """The repository under test at the base commit, read through git (no checkout needed)."""

    def __init__(self, source, base_sha):
        self.source, self.base_sha = source, base_sha
        self.base_text = put_diff.git_base_reader(source, base_sha)
        self._translatable = None

    def translatable_types(self):
        """-> names of TranslatableException and every class that extends it, transitively, at the base."""
        if self._translatable is None:
            p = subprocess.run(["git", "grep", "-h", "-E", r"class\s+\w+\s+extends\s+\w+", self.base_sha, "--", "src/main/java"],
                               cwd=self.source, capture_output=True)
            pairs = re.findall(r"class\s+(\w+)\s+extends\s+(\w+)", p.stdout.decode("utf-8", "replace"))
            known = {"TranslatableException"}
            grew = True
            while grew:
                grew = False
                for child, parent in pairs:
                    if parent in known and child not in known:
                        known.add(child)
                        grew = True
            self._translatable = known
        return set(self._translatable)


def rel_path(path):
    """A tool's file path as a repository path: the part from src/ on (absolute worktree paths, backend/ prefixes)."""
    p = (path or "").replace("\\", "/")
    for marker in ("/src/", "src/"):
        k = p.find(marker)
        if k >= 0:
            return p[k + (1 if marker.startswith("/") else 0):]
    return p


def load_run(run_dir, task, contract, repo):
    changes = put_diff.changes(run_dir, repo.base_text)

    def read_json(name, default):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            return default
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    answer = ""
    p = os.path.join(run_dir, "answer.md")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            answer = f.read()
    return Run(run_dir, task, contract, changes, read_json("calls.json", []), read_json("tests.json", {}),
               read_json("meta.json", {}), answer, repo)


def _r(value, seen, applicable=True):
    value = max(0.0, min(1.0, float(value)))
    return {"value": round(value, 4), "applicable": applicable, "passed": applicable and value >= 1.0, "seen": seen}


def _share(ok, total):
    return 1.0 if total == 0 else ok / total


# ----------------------------------------------------------------------------- process rules

def first_index(calls, pred):
    return next((c["i"] for c in calls if pred(c)), None)


def graph_first(run):
    first_graph = first_index(run.calls, lambda c: c["name"].startswith(GRAPH_PREFIX))
    first_edit = first_index(run.calls, lambda c: c["name"] in EDIT_TOOLS)
    n_graph = sum(1 for c in run.calls if c["name"].startswith(GRAPH_PREFIX))
    if first_graph is None:
        return _r(0, "no graph query in the run")
    if first_edit is None or first_graph < first_edit:
        before = sum(1 for c in run.calls if c["name"].startswith(GRAPH_PREFIX) and (first_edit is None or c["i"] < first_edit))
        return _r(1, f"{before} graph queries before the first edit ({n_graph} in all)")
    return _r(0.5, f"the first graph query came after the first edit (call {first_graph} vs {first_edit})")


def exemplar_read(run):
    patterns = [re.compile(p, re.M) for p in run.task.get("exemplars", ())]
    first_edit = first_index(run.calls, lambda c: c["name"] in EDIT_TOOLS)
    changed = set(run.changes)
    read_ok = []
    for c in run.calls:
        if first_edit is not None and c["i"] >= first_edit:
            break
        if c["name"] != "Read" or c.get("is_error"):
            continue
        path = rel_path((c.get("input") or {}).get("file_path"))
        if path in changed or not (path.startswith("src/main/")):
            continue
        text = run.repo.base_text(path)
        if text and any(p.search(text) for p in patterns):
            read_ok.append(path.rsplit("/", 1)[-1])
    if read_ok:
        return _r(1, f"read before the first edit: {', '.join(sorted(set(read_ok))[:4])}")
    return _r(0, "no existing file of the task's kinds was read before the first edit")


# ----------------------------------------------------------------------------- convention rules

def feature_first(run):
    prefixes = run.task.get("package_prefix", ())
    new_main = [p for p, c in run.changes.items() if c.status == "A" and p.startswith(MAIN)]
    ok = [p for p in new_main if any(p.startswith(x) for x in prefixes)]
    return _r(_share(len(ok), len(new_main)), f"{len(ok)} of {len(new_main)} new main files under the feature package")


def constructor_injection(run):
    autowired = sum(1 for p, c in run.main_java().items() for line in c.added if re.search(r"@Autowired\b", line))
    bad_classes = []
    for p, c in run.main_java(("A",)).items():
        for cls in classes(c.after):
            if not any(cls.has(a) for a in ("Service", "Component", "RestController", "Controller", "Configuration", "Repository")):
                continue
            has_final_dep = re.search(r"^\s*private\s+final\s+(?!static)[\w<>, ?]+\s+\w+\s*;", cls.body, re.M)
            has_ctor = re.search(r"\b(public|protected)?\s*" + re.escape(cls.name) + r"\s*\(", cls.body)
            if has_final_dep and not (cls.has("RequiredArgsConstructor") or cls.has("AllArgsConstructor") or has_ctor):
                bad_classes.append(cls.name)
    ok = autowired == 0 and not bad_classes
    seen = f"{autowired} added @Autowired lines; {len(bad_classes)} new beans with final fields and no constructor"
    return _r(1 if ok else 0, seen)


def after_commit_listener(run):
    parts, seen = [], []
    for p, c in run.main_java().items():
        for m in members(c.after):
            if not (m.has("TransactionalEventListener") or m.has("EventListener")) or not added(m, c):
                continue
            tel = m.annotation("TransactionalEventListener")
            phase_ok = tel is not None and ("AFTER_COMMIT" in tel or "phase" not in tel) and not m.has("EventListener")
            catch_ok = bool(re.search(r"\bcatch\s*\(", m.body))
            parts.append((1.0 if phase_ok else 0.0) * 0.5 + (0.5 if catch_ok else 0.0))
            seen.append(f"{m.name}: {'after-commit' if phase_ok else 'not after-commit'}, {'catches' if catch_ok else 'no catch'}")
    if not parts:
        return _r(0, "no listener method added")
    return _r(sum(parts) / len(parts), "; ".join(seen))


def scheduler_lock(run):
    parts, seen = [], []
    for p, c in run.main_java().items():
        for m in members(c.after):
            if not m.has("Scheduled") or not added(m, c):
                continue
            cls = class_of(c.after, m.line)
            lock = m.annotation("SchedulerLock")
            sched = m.annotation("Scheduled") or ""
            checks = [
                lock is not None,
                bool(lock and re.search(r'name\s*=\s*"[A-Za-z][\w-]*:[A-Za-z][\w-]*"', lock)),
                bool(re.search(r'cron\s*=\s*"\$\{[^}]+\}"', sched)),
                bool(cls and re.search(r'@Value\("\$\{[^}]*enabled[^}]*\}"\)|@ConditionalOnProperty', cls.body + " ".join(a for _, a in cls.annotations))),
                bool(cls and cls.name.endswith("CronJob")),
            ]
            parts.append(sum(checks) / len(checks))
            seen.append(f"{(cls.name if cls else '?')}.{m.name}: lock={checks[0]}, area:job name={checks[1]}, "
                        f"cron property={checks[2]}, enabled flag={checks[3]}, *CronJob={checks[4]}")
    if not parts:
        return _r(0, "no @Scheduled method added")
    return _r(sum(parts) / len(parts), "; ".join(seen))


def ports_adapters(run):
    v = run.task.get("vendor") or {}
    port_dir, adapter_dir = MAIN + "com/sm/instagram/platform/" + v.get("port_dir", ""), MAIN + "com/sm/instagram/platform/" + v.get("adapter_dir", "")
    new = {p: c for p, c in run.main_java(("A",)).items()}
    ports = [p for p, c in new.items() if p.startswith(port_dir) and re.search(r"\binterface\s+\w+Port\b", c.after)]
    port_names = [re.search(r"\binterface\s+(\w+Port)\b", new[p].after).group(1) for p in ports]
    adapters = [p for p, c in new.items() if p.startswith(adapter_dir)
                and re.search(r"\bclass\s+\w+Adapter\b[^{]*\bimplements\b[^{]*\b(" + "|".join(port_names or ["NoPort"]) + r")\b", c.after)]
    leaks = []
    for p, c in run.main_java().items():
        if p.startswith(adapter_dir):
            continue
        for line in c.added:
            if re.match(r"\s*import\s+[\w.]*\.adapter\.\w+Adapter\s*;", line):
                leaks.append(p.rsplit("/", 1)[-1])
    checks = [bool(ports), bool(adapters), not leaks]
    return _r(sum(checks) / 3, f"port interface={bool(ports)}, adapter implementing it={bool(adapters)}, "
                               f"adapter imports outside the adapter package={len(leaks)}")


def _bundle_keys(run):
    keys = []
    for b in BUNDLES:
        text = run.after_or_base(b) or ""
        keys.append({line.split("=", 1)[0].strip() for line in text.splitlines() if "=" in line and not line.lstrip().startswith("#")})
    return keys


def translatable_errors(run):
    known = run.repo.translatable_types()
    for p, c in run.main_java(("A",)).items():                        # exception classes the diff adds
        for child, parent in re.findall(r"class\s+(\w+)\s+extends\s+(\w+)", c.after or ""):
            if parent in known:
                known.add(child)
    # every exception the added code creates, thrown directly or from a supplier (`orElseThrow(() -> new X(...))`),
    # and every message key literal it names
    throws, ok, literal_keys = 0, 0, set()
    for p, c in run.main_java().items():
        for line in c.added:
            for m in re.finditer(r"\bnew\s+(\w*(?:Exception|Error))\s*\(", line):
                throws += 1
                ok += m.group(1) in known
            literal_keys.update(re.findall(r'"((?:error|validation)\.[a-z][a-z0-9_]*\.[a-z0-9_.]+)"', line))
    keys = set(run.task.get("i18n_keys", ())) | literal_keys
    en, pl = _bundle_keys(run)
    present = sum(1 for k in keys if k in en and k in pl)
    part_throw = _share(ok, throws)
    part_keys = _share(present, len(keys))
    return _r((part_throw + part_keys) / 2, f"{ok} of {throws} added throws are translatable; "
                                            f"{present} of {len(keys)} message keys in both bundles")


def _changesets(run, statuses=("A",)):
    return {p: c for p, c in run.changes.items() if p.startswith(CHANGELOG_DIR) and p != MASTER
            and c.status in statuses and c.after is not None}


def _adds_column(text, table, column):
    sql = re.search(r"(?is)alter\s+table\s+(public\.)?\"?" + re.escape(table) + r"\"?\s+add\s+(column\s+)?(if\s+not\s+exists\s+)?\"?"
                    + re.escape(column) + r"\"?\b", text or "")
    xml = re.search(r'(?is)<addColumn[^>]*tableName="' + re.escape(table) + r'"[^>]*>.*?<column[^>]*name="' + re.escape(column) + '"', text or "")
    return bool(sql or xml)


def liquibase_changeset(run):
    table, column = run.task.get("schema_table"), run.task.get("schema_column")
    new = _changesets(run)
    modified_existing = [p for p, c in run.changes.items() if p.startswith(CHANGELOG_DIR) and p != MASTER and c.status in ("M", "D", "R")]
    master = run.changes.get(MASTER)
    target = next((p for p, c in new.items() if _adds_column(c.after, table, column)), None) or next(iter(new), None)
    if target is None:
        return _r(1 / 7 if not modified_existing else 0, "no new changeset file")
    text = new[target].after
    first = text.lstrip("﻿").split("\n", 1)[0].strip()
    name = target[len(CHANGELOG_DIR):]
    inc_ok = False
    if master and master.after:
        lines = master.after.split("\n")
        for k, line in enumerate(lines):
            if f'file="{name}"' in line and (k + 1) in master.added_numbers:
                above = "\n".join(lines[max(0, k - 2):k])
                inc_ok = bool(re.search(r"<!--\s*" + MONTHS + r"\s+\d{4}\s*:", above))
    checks = [
        bool(re.match(r"\d{4}/\d{2}/\d{2}-\d{2}-\d{4}-[a-z0-9]+(-[a-z0-9]+)*\.sql$", name)),
        bool(re.match(r"--\s?liquibase formatted sql", first)),
        bool(re.search(r"^--\s?changeset\s+[\w.-]+:[\w.-]+", text, re.M)),
        bool(re.search(r"^--\s?rollback\b", text, re.M | re.I)),
        inc_ok,
        not modified_existing,
        _adds_column(text, table, column),
    ]
    labels = ["path", "header", "changeset id", "rollback", "include with month comment", "no existing changeset edited",
              f"adds {table}.{column}"]
    return _r(sum(checks) / 7, ", ".join(f"{l}={v}" for l, v in zip(labels, checks)))


def version_field(run):
    entity_ok = []
    for path in run.task.get("contended_entities", ()):
        c = run.changes.get(path)
        text = c.after if c else None
        ok = False
        for m in members(text or ""):
            if m.has("Version") and re.search(r"\b(Long|long|Integer|int)\s+version\b", m.decl):
                ok = True
        entity_ok.append(ok)
    # the column may come from any changeset the diff adds or edits; editing an old one is liquibase_changeset's failure
    column_ok = any(_adds_column(c.after, run.task.get("schema_table"), "version") for c in _changesets(run, ("A", "M")).values())
    checks = [all(entity_ok) and bool(entity_ok), column_ok]
    return _r(sum(checks) / 2, f"@Version field on the contended entity={checks[0]}, changeset adds the column={checks[1]}")


def _mapping_methods(run):
    for p, c in run.main_java().items():
        for m in members(c.after):
            if any(MAPPING.match(a.strip()) for _, a in m.annotations) and "(" in m.decl and added(m, c) \
                    and not re.search(r"\b(class|interface)\s", m.decl):
                yield p, c, m


def controller_guards(run):
    total, ok, seen = 0, 0, []
    for p, c, m in _mapping_methods(run):
        cls = class_of(c.after, m.line)
        pre = m.has("PreAuthorize") or (cls is not None and cls.has("PreAuthorize"))
        rate = m.has("RateLimit") or (cls is not None and cls.has("RateLimit"))
        total += 1
        ok += pre and rate
        seen.append(f"{m.name}: PreAuthorize={pre}, RateLimit={rate}")
    if total == 0:
        return _r(0, "no endpoint added")
    return _r(ok / total, "; ".join(seen))


def _innermost(type_text):
    t = type_text.strip()
    while True:
        m = re.match(r"(?:ResponseEntity|List|Page|Set|Collection|Optional)<(.+)>$", t)
        if not m:
            return t
        t = m.group(1).strip()


def dto_naming(run):
    total, ok, seen = 0, 0, []
    for p, c, m in _mapping_methods(run):
        bodies = re.findall(r"@RequestBody\s+(?:@\w+(?:\([^)]*\))?\s+)*([\w<>, ?]+?)\s+\w+\s*[,)]", m.decl)
        ret = re.search(r"(?:public|protected|private)?\s*([\w<>, ?]+?)\s+" + re.escape(m.name) + r"\s*\(", m.decl)
        ret_type = _innermost(ret.group(1)) if ret else ""
        body_ok = all(_innermost(b).endswith("DtoIn") for b in bodies)
        ret_ok = ret_type.endswith("DtoOut") or ret_type in ("Void", "void", "?")
        total += 1
        ok += body_ok and ret_ok
        seen.append(f"{m.name}: body {', '.join(bodies) or 'none'}, returns {ret_type or '?'}")
    if total == 0:
        return _r(0, "no endpoint added")
    return _r(ok / total, "; ".join(seen))


# ----------------------------------------------------------------------------- tests, gates, correctness

def tests_written(run):
    new_tests = {p: c for p, c in run.changes.items() if p.startswith(TEST) and p.endswith("UnitTest.java") and c.status == "A"}
    if not new_tests:
        return _r(0, "no new *UnitTest class")
    style = [bool(re.search(r"MockitoExtension", c.after or "")) and not re.search(r"@(SpringBootTest|DataJpaTest|WebMvcTest)\b", c.after or "")
             for c in new_tests.values()]
    build = bool(run.tests.get("build_green"))
    reports = run.tests.get("own_reports", {})
    names = [p.rsplit("/", 1)[-1][:-5] for p in new_tests]
    passed = all(n in reports and reports[n]["tests"] > 0 and reports[n]["failures"] + reports[n]["errors"] == 0 for n in names)
    parts = [sum(style) / len(style), 1.0 if build else 0.0, 1.0 if (build and passed) else 0.0]
    return _r(sum(parts) / 3, f"{len(new_tests)} new unit test classes; Mockito without a Spring context={sum(style)}/{len(style)}; "
                              f"compiles={build}; pass={build and passed}")


def build_green(run):
    ok = bool(run.tests.get("build_green"))
    return _r(1 if ok else 0, "test-compile green" if ok else "test-compile red")


def scope(run):
    deleted_tests = [p for p, c in run.changes.items() if p.startswith("src/test/") and c.status == "D"]
    outside = [p for p in run.changes if not (p.startswith("src/main/") or p.startswith("src/test/"))]
    n = len(run.changes)
    ok = not deleted_tests and not outside and n <= 12
    return _r(1 if ok else 0, f"{n} files changed; {len(outside)} outside src/main and src/test; {len(deleted_tests)} tests deleted")


def marker(run):
    idx = next((c for c in run.calls if c.get("marker_seen_before")), None)
    present = SOLVED in (run.answer or "") or bool(run.meta.get("marker"))
    after = sum(1 for c in run.calls if c.get("after_marker"))
    ok = present and after == 0
    return _r(1 if ok else 0, f"marker={'present' if present else 'absent'}, {after} tool calls after it")


def hidden_pass(run):
    if not run.tests.get("build_green"):
        return _r(0, "build red: acceptance tests not run")
    h = run.tests.get("hidden", [0, 1])
    p = run.tests.get("pass_to_pass", [0, 0])
    total = h[1] + p[1]
    return _r((h[0] + p[0]) / total if total else 0, f"hidden {h[0]}/{h[1]}, pass-to-pass {p[0]}/{p[1]}")


CHECKS = {
    "graph_first": graph_first, "exemplar_read": exemplar_read, "feature_first": feature_first,
    "constructor_injection": constructor_injection, "after_commit_listener": after_commit_listener,
    "scheduler_lock": scheduler_lock, "ports_adapters": ports_adapters, "translatable_errors": translatable_errors,
    "liquibase_changeset": liquibase_changeset, "version_field": version_field, "controller_guards": controller_guards,
    "dto_naming": dto_naming, "tests_written": tests_written, "build_green": build_green, "scope": scope,
    "marker": marker, "hidden_pass": hidden_pass,
}


def applies(rule, task):
    a = rule.get("applies", "all")
    return a == "all" or bool(set(a) & set(task.get("tags", ())))


def check_run(run_dir, task, contract, repo, write=True):
    run = load_run(run_dir, task, contract, repo)
    out = {}
    for rule in contract["rules"]:
        if rule["kind"] != "deterministic":
            continue
        fn = CHECKS[rule["id"]]
        if not applies(rule, task):
            out[rule["id"]] = {"value": None, "applicable": False, "passed": None, "seen": "does not apply to this task"}
            continue
        try:
            out[rule["id"]] = fn(run)
        except Exception as ex:                                       # a crashing check is a bug to see, not a zero to hide
            out[rule["id"]] = {"value": None, "applicable": True, "passed": None, "seen": f"CHECK ERROR {type(ex).__name__}: {ex}"}
    record = {"schema": 1, "task": task["id"], "checks": out}
    if write:
        with open(os.path.join(run_dir, "checks.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(record, f, indent=1)
    return record
