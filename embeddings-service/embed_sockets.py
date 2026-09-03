"""
V3 triple-lens socket embedder (task V2 / #38).

One file becomes THREE texts, not one text under three instructions. F71 measured
instruction-only separation at 0.9451 cosine (5.5%), so the instruction cannot carry
the lenses -- the TEXT must. Each socket therefore reads a DISJOINT slice of the file:

    S  semantic / WHAT    declarations, docs, signatures, literals, domain nouns
    B  behavioural / HOW  control flow, effects, transactions, errors, scheduling
    T  structural / WHERE package, imports, inheritance, DI, graph neighbourhood

T is rendered AS PROSE from the graph, never as hand-crafted numeric features -- the
F47 mistake -- so all three land in the same embedding space.

The instruction is prepended CLIENT-SIDE as "Instruct: <task>\\nQuery:" (F72: 0.9451
vs 1.0000 identical), byte-equivalent to the server's `instruct` field but iterable
without redeploying the GPU service. The service is called with {"texts": [...]} only.

Shardable: --shard i --shard-count n partitions on id(n) % n, so several agents run
concurrently against one namespace. Write-back is atomic per node (all three vectors
in one SET), so an interrupted run resumes cleanly -- re-running only touches nodes
still missing a lens.

Usage:
    python embed_sockets.py --shard 0 --shard-count 3
    python embed_sockets.py --shard 0 --shard-count 3 --limit 8 --dry   # socket preview
    python embed_sockets.py --shard 0 --shard-count 3 --hyperedges      # meta-path emit
    python embed_sockets.py --shard 0 --shard-count 3 --report          # lens cosines
"""

import argparse
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from neo4j import GraphDatabase

# id(n) % shard is the agreed shard predicate across the parallel agents, so the driver's
# elementId deprecation notice fires once per query and drowns the progress log.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

EMBED_URL = os.environ.get("EMBED_URL", "https://ramzesx--v3-code-embeddings-serve.modal.run")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7611")
NEO4J_AUTH = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASS", "password"))
NAMESPACE = os.environ.get("EMBED_NAMESPACE", "CheckItOutV3")

SOCKET_VERSION = "v2-sockets-1"
EMBED_BATCH = 8            # texts per POST, per the service's own batch_size
NODE_GROUP = 24            # nodes embedded then written before the next group
WORKERS = 4                # concurrent POSTs; Modal autoscales containers
SOCKET_MAX_CHARS = 8_000   # per socket; service clips the whole payload at 24k
READ_MAX_CHARS = 200_000   # source read cap

# One line each, in Qwen3-Embedding's native instruction slot. These bias the encoder;
# the disjoint socket TEXT is what actually separates the lenses.
LENS_INSTRUCT = {
    "S": (
        "Represent the business meaning of this source file for retrieval: the domain "
        "concepts it names, what it does functionally, and its API vocabulary"
    ),
    "B": (
        "Represent the runtime behaviour of this source file for retrieval: its control "
        "flow, state transitions, transaction boundaries, error and retry paths, "
        "concurrency, scheduling and side effects"
    ),
    "T": (
        "Represent the position of this source file in the system dependency graph for "
        "retrieval: its package, imports, inheritance, injected dependencies and its "
        "neighbourhood of callers and callees"
    ),
}
LENS_PROP = {
    "S": "semantic_embedding",
    "B": "behavioral_embedding",
    "T": "structural_embedding",
}

# --------------------------------------------------------------------------------------
# Source reading and lexical helpers
# --------------------------------------------------------------------------------------

STRING_RE = re.compile(r'"(?:[^"\\\n]|\\.){3,}"')
BLOCK_COMMENT_RE = re.compile(r"/\*\*?(.*?)\*/", re.S)
LINE_COMMENT_RE = re.compile(r"^\s*//\s?(.*)$")


def read_source(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(READ_MAX_CHARS)
    except OSError:
        return ""


def _depth_scan(content):
    """Split brace-language lines by nesting depth.

    Returns (surface, bodies): `surface` are lines at declaration depth (a class body's
    fields, method signatures, annotations, the type declaration itself); `bodies` are
    lines inside method bodies. Approximate but stable -- string literals and comments
    are masked before counting braces so a `{` in a string cannot skew the depth.
    """
    surface, bodies = [], []
    depth = 0
    in_block = False
    for raw in content.splitlines():
        line = raw.rstrip()
        masked = line
        if in_block:
            end = masked.find("*/")
            if end < 0:
                continue
            masked = masked[end + 2:]
            in_block = False
        start = masked.find("/*")
        if start >= 0 and "*/" not in masked[start:]:
            masked = masked[:start]
            in_block = True
        masked = re.sub(r'"(?:[^"\\]|\\.)*"', '""', masked)
        masked = re.sub(r"'(?:[^'\\]|\\.)*'", "''", masked)
        masked = re.sub(r"`(?:[^`\\]|\\.)*`", "``", masked)
        masked = re.sub(r"//.*$", "", masked)
        if depth <= 1:
            surface.append(line)
        else:
            bodies.append(line)
        depth += masked.count("{") - masked.count("}")
        depth = max(depth, 0)
    return surface, bodies


def _comments(content):
    out = [re.sub(r"^\s*\*\s?", "", ln) for blk in BLOCK_COMMENT_RE.findall(content)
           for ln in blk.splitlines()]
    for ln in content.splitlines():
        m = LINE_COMMENT_RE.match(ln)
        if m:
            out.append(m.group(1))
    return [c.strip() for c in out if len(c.strip()) > 3]


def _literals(content, cap=80):
    seen, out = set(), []
    for lit in STRING_RE.findall(content):
        val = lit[1:-1].strip()
        if not val or val in seen or len(val) > 160:
            continue
        seen.add(val)
        out.append(val)
        if len(out) >= cap:
            break
    return out


def _uniq(seq, cap=None):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
            if cap and len(out) >= cap:
                break
    return out


def _count(content, *needles):
    return sum(content.count(n) for n in needles)


def _section(title, items, bullet="- "):
    items = [i for i in items if i]
    if not items:
        return ""
    return title + "\n" + "\n".join(bullet + str(i) for i in items) + "\n"


# --------------------------------------------------------------------------------------
# S socket -- semantic / WHAT
# --------------------------------------------------------------------------------------

JAVA_TYPE_DECL = re.compile(
    r"^\s*(?:public|protected|private|abstract|final|static|sealed|\s)*"
    r"(class|interface|enum|record|@interface)\s+(\w+)"
)
JAVA_METHOD = re.compile(
    r"^\s*(?:@\w+[^\n]*\s*)?(?:public|protected|private|static|final|abstract|synchronized|"
    r"default|native|\s)*[\w<>\[\],.?\s]+\s+(\w+)\s*\(([^;{]*)\)\s*(?:throws [\w,.\s]+)?\s*[{;]"
)
JAVA_FIELD = re.compile(
    r"^\s*(?:public|protected|private|static|final|transient|volatile|\s)+"
    r"([\w<>\[\],.?]+)\s+(\w+)\s*(?:=|;)"
)
ROUTE_ANN = re.compile(
    r'@(?:Request|Get|Post|Put|Delete|Patch)Mapping\s*\(\s*(?:value\s*=\s*)?"([^"]*)"'
)
DOMAIN_ANN = re.compile(r'@(?:Table|Column|Entity|Schema|Operation|Query)\s*\(([^)]{0,200})\)')


def semantic_java(content, name):
    surface, _ = _depth_scan(content)
    types, methods, fields, enums = [], [], [], []
    for ln in surface:
        s = ln.strip()
        if not s or s.startswith(("package ", "import ", "//", "*", "/*")):
            continue
        m = JAVA_TYPE_DECL.match(ln)
        if m:
            # Keep the declared name and kind; extends/implements belong to T.
            types.append(f"{m.group(1)} {m.group(2)}")
            continue
        m = JAVA_METHOD.match(ln)
        if m:
            sig = re.sub(r"\s*\{\s*$", "", s).rstrip(";")
            sig = re.sub(r"^\s*(?:public|protected|private|static|final|abstract|"
                         r"synchronized|default)\s+", "", sig)
            methods.append(sig)
            continue
        m = JAVA_FIELD.match(ln)
        if m:
            fields.append(f"{m.group(1)} {m.group(2)}")
            continue
        if re.match(r"^\s*[A-Z][A-Z0-9_]{2,}\s*(?:\(|,|;|$)", s):
            enums.append(s.rstrip(",;"))

    routes = ROUTE_ANN.findall(content)
    anns = [a.strip() for a in DOMAIN_ANN.findall(content)]

    parts = [f"\nSource unit {name} -- what it means.\n"]
    parts.append(_section("Declared types:", _uniq(types, 20)))
    parts.append(_section("Documentation:", _comments(content)[:60]))
    parts.append(_section("API surface (method signatures):", _uniq(methods, 70)))
    parts.append(_section("Domain state (fields):", _uniq(fields, 50)))
    parts.append(_section("Enumerated values:", _uniq(enums, 40)))
    parts.append(_section("HTTP routes exposed:", _uniq(routes, 25)))
    parts.append(_section("Persistence and contract annotations:", _uniq(anns, 20)))
    parts.append(_section("Literal vocabulary:", _literals(content)))
    return "".join(parts)


TS_CLASS = re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?(class|interface|enum|type)\s+(\w+)")
TS_METHOD = re.compile(
    r"^\s*(?:public|private|protected|readonly|static|async|override|\s)*"
    r"(\w+)\s*\(([^;{)]*)\)\s*(?::\s*[^{;]+)?\s*[{;]"
)
TS_PROP = re.compile(r"^\s*(?:public|private|protected|readonly|static|\s)*(\w+)\s*[:?]\s*([^=;{]+)")
TS_SELECTOR = re.compile(r"selector\s*:\s*['\"]([^'\"]+)['\"]")
TS_ROUTE_PATH = re.compile(r"path\s*:\s*['\"]([^'\"]*)['\"]")


def semantic_ts(content, name):
    surface, _ = _depth_scan(content)
    types, methods, props = [], [], []
    for ln in surface:
        s = ln.strip()
        if not s or s.startswith(("import ", "//", "*", "/*", "@")):
            continue
        m = TS_CLASS.match(ln)
        if m:
            types.append(f"{m.group(1)} {m.group(2)}")
            continue
        m = TS_METHOD.match(ln)
        if m and m.group(1) not in {"if", "for", "while", "switch", "catch", "return"}:
            methods.append(re.sub(r"\s*\{\s*$", "", s).rstrip(";"))
            continue
        m = TS_PROP.match(ln)
        if m:
            props.append(f"{m.group(1)}: {m.group(2).strip().rstrip(',')}")

    parts = [f"\nSource unit {name} -- what it means.\n"]
    parts.append(_section("Declared types:", _uniq(types, 20)))
    parts.append(_section("Documentation:", _comments(content)[:60]))
    parts.append(_section("Component selectors:", _uniq(TS_SELECTOR.findall(content), 10)))
    parts.append(_section("Route paths:", _uniq(TS_ROUTE_PATH.findall(content), 30)))
    parts.append(_section("API surface (methods):", _uniq(methods, 70)))
    parts.append(_section("Model state (properties):", _uniq(props, 50)))
    parts.append(_section("Literal vocabulary:", _literals(content)))
    return "".join(parts)


HTML_TEXT = re.compile(r">([^<>{}]{3,120})<")
HTML_I18N = re.compile(r"(?:transloco|translate|i18n)[^'\"]*['\"]([\w.\-]+)['\"]")
HTML_LABEL = re.compile(r'(?:label|placeholder|title|aria-label|alt)\s*=\s*"([^"]{2,120})"')


def semantic_html(content, name):
    texts = [t.strip() for t in HTML_TEXT.findall(content) if t.strip()]
    parts = [f"\nTemplate {name} -- what it presents.\n"]
    parts.append(_section("Visible copy:", _uniq(texts, 80)))
    parts.append(_section("Labels and accessible names:", _uniq(HTML_LABEL.findall(content), 40)))
    parts.append(_section("Translation keys:", _uniq(HTML_I18N.findall(content), 40)))
    parts.append(_section("Documentation:", _uniq(re.findall(r"<!--(.*?)-->", content, re.S), 20)))
    return "".join(parts)


def semantic_style(content, name):
    selectors = [ln.strip().rstrip("{").strip() for ln in content.splitlines()
                 if ln.strip().endswith("{") and not ln.strip().startswith("@")]
    variables = re.findall(r"(\$[\w-]+|--[\w-]+)\s*:", content)
    parts = [f"\nStylesheet {name} -- what it styles.\n"]
    parts.append(_section("Selectors:", _uniq(selectors, 60)))
    parts.append(_section("Design tokens:", _uniq(variables, 50)))
    parts.append(_section("Documentation:", _comments(content)[:30]))
    return "".join(parts)


def semantic_config(content, name):
    keys, values = [], []
    for ln in content.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r'^["\']?([\w.\-]+)["\']?\s*[:=]\s*(.*)$', s)
        if m:
            keys.append(m.group(1))
            v = m.group(2).strip().strip('",')
            if v and not v.startswith(("{", "[")):
                values.append(f"{m.group(1)} = {v[:120]}")
    parts = [f"\nConfiguration {name} -- what it declares.\n"]
    parts.append(_section("Settings:", _uniq(values, 90)))
    parts.append(_section("Keys:", _uniq(keys, 60)))
    parts.append(_section("Documentation:",
                          _uniq([ln.strip("# ") for ln in content.splitlines()
                                 if ln.strip().startswith("#")], 20)))
    return "".join(parts)


def semantic_feature(content, name):
    titles, steps, tags = [], [], []
    for ln in content.splitlines():
        s = ln.strip()
        if s.startswith(("Feature:", "Scenario:", "Scenario Outline:", "Rule:", "Background:")):
            titles.append(s)
        elif s.startswith(("Given ", "When ", "Then ", "And ", "But ")):
            steps.append(s)
        elif s.startswith("@"):
            tags.append(s)
    parts = [f"\nSpecification {name} -- the behaviour it requires.\n"]
    parts.append(_section("Features and scenarios:", _uniq(titles, 40)))
    parts.append(_section("Acceptance steps:", _uniq(steps, 90)))
    parts.append(_section("Tags:", _uniq(tags, 20)))
    return "".join(parts)


# --------------------------------------------------------------------------------------
# B socket -- behavioural / HOW
# --------------------------------------------------------------------------------------

# (probe, sentence) -- presence-driven runtime facts, phrased as prose so the behavioural
# vocabulary (not the domain vocabulary) dominates the vector.
JAVA_BEHAVIOUR_PROBES = [
    (("@Transactional",), "Runs inside a declarative transaction boundary"),
    (("readOnly = true", "readOnly=true"), "Declares a read-only transaction"),
    (("Propagation.REQUIRES_NEW",), "Opens a nested independent transaction"),
    (("TransactionTemplate", "PlatformTransactionManager"), "Manages transactions programmatically"),
    (("@TransactionalEventListener",), "Reacts after transaction commit"),
    (("@EventListener", "publishEvent"), "Publishes or consumes application events"),
    (("@Scheduled",), "Executes on a schedule"),
    (("@SchedulerLock", "ShedLock"), "Guards scheduled execution with a distributed lock"),
    (("@Async", "CompletableFuture", "ExecutorService", "ThreadPoolTaskExecutor"),
     "Executes asynchronously off the calling thread"),
    (("@Retryable", "RetryTemplate", "maxAttempts"), "Retries on failure with a retry policy"),
    (("@CircuitBreaker", "Resilience4j", "RateLimiter"), "Applies resilience or rate limiting"),
    (("synchronized", "ReentrantLock", "@Lock", "LockModeType"), "Serialises access with locking"),
    (("@Version",), "Uses optimistic locking on version conflicts"),
    (("@ExceptionHandler", "@ControllerAdvice"), "Translates thrown errors into responses"),
    ((".save(", ".saveAll(", ".persist(", ".merge("), "Writes rows to the database"),
    ((".delete(", ".deleteBy", ".deleteAll("), "Deletes rows from the database"),
    (("@Modifying", "createQuery", "jdbcTemplate", "entityManager"), "Issues direct data-store queries"),
    (("RestTemplate", "WebClient", "HttpClient", "FeignClient"), "Calls remote services over HTTP"),
    (("redisTemplate", "@Cacheable", "@CacheEvict"), "Reads or invalidates cached state"),
    (("mailSender", "JavaMailSender", "sendEmail"), "Sends outbound mail"),
    (("Stripe", "stripeClient"), "Drives an external payment provider"),
    (("Firebase", "FirebaseAuth"), "Delegates to the external identity provider"),
    (("@PreAuthorize", "@Secured", "hasRole", "hasAuthority"), "Denies execution on failed authorisation"),
    (("@Valid", "@Validated", "requireNonNull", "orElseThrow"), "Rejects invalid input before proceeding"),
    ((".stream()", ".map(", ".filter("), "Transforms collections through a stream pipeline"),
    (("Optional.", "isPresent(", "orElse("), "Branches on presence or absence of a value"),
    (("@PostConstruct", "@PreDestroy", "InitializingBean"), "Runs work at bean lifecycle boundaries"),
    (("Pageable", "Page<", "Slice<"), "Returns results one page at a time rather than all at once"),
    (("Thread.sleep", "TimeUnit.", "Duration.of", "awaitTermination"), "Waits on wall-clock time"),
    (("@Test", "assertThat", "assertEquals", "verify(", "when("),
     "Drives the unit under test and asserts its observable outcome"),
]

TS_BEHAVIOUR_PROBES = [
    (("ngOnInit",), "Initialises on component creation"),
    (("ngOnDestroy",), "Tears down on component destruction"),
    (("ngAfterViewInit", "ngAfterContentInit"), "Runs after the view is initialised"),
    (("ngOnChanges",), "Reacts to input property changes"),
    ((".subscribe(",), "Subscribes to an asynchronous stream"),
    (("takeUntil", "takeUntilDestroyed", "unsubscribe"), "Unsubscribes to avoid a leak"),
    (("switchMap", "mergeMap", "concatMap", "exhaustMap"), "Flattens nested async work"),
    (("catchError", "retry(", "retryWhen"), "Recovers from a failed stream"),
    (("debounceTime", "throttleTime", "distinctUntilChanged"), "Rate-limits emissions over time"),
    (("BehaviorSubject", "ReplaySubject", "Subject<"), "Holds and multicasts mutable stream state"),
    (("signal(", "computed(", "effect("), "Propagates state through reactive signals"),
    ((".set(", ".update(", ".next("), "Mutates observable or signal state"),
    (("http.get", "http.post", "http.put", "http.delete", "http.patch"),
     "Issues HTTP requests to the backend"),
    (("HttpInterceptor", "intercept("), "Intercepts every outgoing request"),
    (("canActivate", "canMatch", "CanActivateFn"), "Blocks or permits route activation"),
    (("try {", "catch (", "catch("), "Handles thrown errors locally"),
    (("throwError", "throw new"), "Raises an error to the caller"),
    (("async ", "await ", "Promise"), "Awaits asynchronous completion"),
    (("setTimeout", "setInterval", "timer("), "Defers or repeats work on a timer"),
    (("router.navigate", "routerLink"), "Navigates to another route"),
    (("Validators.", "setValidators", "updateValueAndValidity"),
     "Validates form state before submission"),
    (("localStorage", "sessionStorage", "cookie"), "Persists state in browser storage"),
    (("MatDialog", "open(", "snackBar"), "Opens a dialog or transient notification"),
    (("Observable<",), "Returns a cold stream that stays inert until a caller subscribes"),
    (("shareReplay",), "Caches the latest emission and replays it to late subscribers"),
    (("tap(", "finalize("), "Runs side effects inside the stream without changing its value"),
    (("firstValueFrom", "lastValueFrom", "toSignal", "toObservable"),
     "Bridges between stream, promise and signal execution models"),
]

EFFECT_LINE_RE = re.compile(
    r"\b(if|else|for|while|switch|case|try|catch|finally|throw|return|await|yield|break|"
    r"continue|synchronized)\b|\.(save|delete|update|insert|persist|merge|flush|publish|"
    r"send|execute|subscribe|next|emit|navigate|set[A-Z]\w*)\s*\("
)
THROW_RE = re.compile(r"throw\s+new\s+(\w+)")
CATCH_RE = re.compile(r"catch\s*\(\s*(?:final\s+)?([\w.|\s]+?)\s+\w+\s*\)")
STATE_SET_RE = re.compile(r"\.set(Status|State|Stage|Phase|Flag|Enabled|Active)\s*\(\s*([\w.]+)")
CRON_RE = re.compile(r'cron\s*=\s*"([^"]+)"')


def behavioural(content, name, probes):
    _, bodies = _depth_scan(content)
    body_text = "\n".join(bodies)
    scan = content  # annotations live on surface lines, effects in bodies

    facts = [txt for needles, txt in probes if any(n in scan for n in needles)]

    branches = _count(body_text, "if (", "if(", "else", "for (", "for(", "while (",
                      "while(", "case ", "?", "&&", "||")
    tries = _count(body_text, "try {", "try{")
    catches = len(CATCH_RE.findall(scan))
    throws = THROW_RE.findall(scan)
    returns = _count(body_text, "return ")
    logs = _count(scan, "log.error", "log.warn", "log.info", "log.debug",
                  "console.error", "console.warn", "logger.")
    depth = 0
    max_depth = 0
    for ch in body_text:
        if ch == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == "}":
            depth = max(depth - 1, 0)

    shape = [
        f"Branch points in executable code: {branches}",
        f"Maximum nesting depth reached at runtime: {max_depth}",
        f"Guarded blocks: {tries} try, {catches} catch",
        f"Exit points: {returns} returns",
        f"Diagnostic emissions: {logs} log statements",
    ]

    transitions = [f"transitions {m[0].lower()} to {m[1]}" for m in STATE_SET_RE.findall(scan)]
    effect_lines = [ln.strip() for ln in bodies if EFFECT_LINE_RE.search(ln)]

    if not facts and branches == 0 and not effect_lines:
        # A record, DTO or constant holder. Saying so beats an all-zero socket that would
        # collapse every such node onto the same point in the behavioural space.
        facts = ["Executes no logic of its own; it is inert data carried across boundaries",
                 "Its runtime cost is construction and serialisation only",
                 "Behaviour appears only where a collaborator reads or writes these values"]

    parts = [f"\nRuntime profile of {name} -- how it executes.\n"]
    parts.append(_section("Execution characteristics:", facts))
    parts.append(_section("Failure paths:",
                          [f"raises {t}" for t in _uniq(throws, 25)] +
                          [f"recovers from {c.strip()}" for c in _uniq(CATCH_RE.findall(scan), 20)]))
    parts.append(_section("Scheduling:", [f"cron expression {c}" for c in _uniq(CRON_RE.findall(scan), 10)]))
    parts.append(_section("State transitions:", _uniq(transitions, 25)))
    parts.append(_section("Control-flow shape:", shape))
    observed = _uniq(effect_lines, 70)
    if observed:
        blob = "\n".join(observed)[:2_600]
        parts.append("Observed statements:\n" + blob + "\n")
    return "".join(parts)


HTML_BEHAVIOUR = [
    (("(click)", "(submit)", "(change)", "(input)", "(blur)", "(keyup)"),
     "Binds DOM events to component handlers"),
    (("*ngIf", "@if", "[hidden]"), "Renders conditionally"),
    (("*ngFor", "@for", "trackBy"), "Repeats over a collection"),
    (("@switch", "*ngSwitch"), "Branches between alternative templates"),
    (("| async",), "Unwraps an asynchronous stream in the template"),
    (("[disabled]", "[readonly]"), "Disables controls on state"),
    (("formGroup", "formControlName", "ngModel"), "Two-way binds form state"),
    (("routerLink", "routerLinkActive"), "Navigates on interaction"),
    (("@defer", "@placeholder", "@loading"), "Defers rendering until a trigger"),
    (("@empty", "@error"), "Renders an empty or error state"),
]


def behavioural_html(content, name):
    facts = [txt for needles, txt in HTML_BEHAVIOUR if any(n in content for n in needles)]
    handlers = _uniq(re.findall(r"\((\w+)\)\s*=\s*\"([^\"]{0,80})\"", content), 40)
    conds = _uniq(re.findall(r"(?:\*ngIf|@if\s*\()\s*=?\s*\"?([^\"\n)]{0,80})", content), 30)
    loops = _uniq(re.findall(r"(?:\*ngFor|@for)\s*=?\s*\"?([^\"\n)]{0,80})", content), 25)
    parts = [f"\nRuntime profile of {name} -- how it reacts.\n"]
    parts.append(_section("Execution characteristics:", facts))
    parts.append(_section("Event handlers:", [f"on {e} calls {h}" for e, h in handlers]))
    parts.append(_section("Conditional rendering:", conds))
    parts.append(_section("Iteration:", loops))
    return "".join(parts)


def behavioural_style(content, name):
    facts = []
    for needles, txt in [
        (("transition", "animation", "@keyframes"), "Animates between visual states"),
        (("@media",), "Adapts to viewport breakpoints"),
        ((":hover", ":focus", ":active"), "Responds to pointer and focus state"),
        ((":disabled", "[disabled]", ".is-"), "Reflects disabled or toggled state"),
        (("@supports", "prefers-"), "Branches on capability or user preference"),
    ]:
        if any(n in content for n in needles):
            facts.append(txt)
    parts = [f"\nRuntime profile of {name} -- how it responds.\n"]
    parts.append(_section("Execution characteristics:", facts))
    parts.append(_section("Breakpoints:", _uniq(re.findall(r"@media([^{]{0,80})", content), 20)))
    parts.append(_section("State selectors:", _uniq(re.findall(r"(:[\w-]+)\s*[{,]", content), 30)))
    parts.append(_section("Transitions:", _uniq(re.findall(r"transition\s*:\s*([^;]{0,80})", content), 20)))
    return "".join(parts)


RUNTIME_KEY_RE = re.compile(
    r"^\s*([\w.\-]*(?:timeout|retry|retries|backoff|pool|max|min|ttl|cron|schedule|interval|"
    r"delay|limit|rate|threshold|enabled|batch|concurrency|size|window)[\w.\-]*)\s*[:=]\s*(.+)$",
    re.I,
)


def behavioural_config(content, name):
    tuning = [f"{m.group(1)} set to {m.group(2).strip().strip(',')[:80]}"
              for m in (RUNTIME_KEY_RE.match(ln) for ln in content.splitlines()) if m]
    parts = [f"\nRuntime profile of {name} -- how it tunes execution.\n"]
    parts.append(_section("Operational tuning:", _uniq(tuning, 70)))
    parts.append(_section("Profiles and activation:",
                          _uniq(re.findall(r"(?:profiles?|activate|on-profile)[:=\s]+([\w,\-!]+)",
                                           content, re.I), 20)))
    return "".join(parts)


def behavioural_feature(content, name):
    whens = [ln.strip() for ln in content.splitlines() if ln.strip().startswith(("When ", "Then "))]
    tags = _uniq([t for ln in content.splitlines() if ln.strip().startswith("@")
                  for t in ln.split()], 20)
    parts = [f"\nRuntime profile of {name} -- the execution it drives.\n"]
    parts.append(_section("Execution characteristics:",
                          ["Drives the system through an end-to-end scenario",
                           "Asserts observable outcomes after each action"]))
    parts.append(_section("Actions and assertions:", _uniq(whens, 80)))
    parts.append(_section("Execution tags:", tags))
    parts.append(_section("Data-driven runs:",
                          _uniq([ln.strip() for ln in content.splitlines()
                                 if ln.strip().startswith("|")], 30)))
    return "".join(parts)


# --------------------------------------------------------------------------------------
# T socket -- structural / WHERE (graph rendered as prose)
# --------------------------------------------------------------------------------------

JAVA_EXTENDS = re.compile(r"\b(?:class|interface)\s+\w+[^{]*?\bextends\s+([\w<>,.\s]+?)(?:\bimplements\b|\{)")
JAVA_IMPLEMENTS = re.compile(r"\bimplements\s+([\w<>,.\s]+?)\{")
TS_EXTENDS = re.compile(r"\b(?:class|interface)\s+\w+\s+extends\s+([\w<>,\s]+?)(?:\bimplements\b|\{)")
TS_IMPLEMENTS = re.compile(r"\bimplements\s+([\w<>,\s]+?)\{")
JAVA_INJECT = re.compile(r"^\s*(?:private|protected)\s+final\s+([\w<>\[\],.]+)\s+\w+\s*;")
TS_INJECT = re.compile(r"(?:inject\s*<?\s*\(\s*(\w+)|(?:private|public|protected|readonly)\s+\w+\s*:\s*(\w+))")

RELATION_PROSE = {
    "PERFORMS": ("performs", "is performed by"),
    "USES": ("uses", "is used by"),
    "MODIFIES": ("modifies", "is modified by"),
    "CALLS": ("calls", "is called by"),
    "ACCESSES": ("accesses", "is accessed by"),
    "INJECTS": ("injects", "is injected into"),
    "EXTENDS": ("extends", "is extended by"),
    "IMPLEMENTS": ("implements", "is implemented by"),
    "CONSTRAINS": ("constrains", "is constrained by"),
    "VALIDATES": ("validates", "is validated by"),
    "AFFECTS": ("affects", "is affected by"),
    "TRIGGERS": ("triggers", "is triggered by"),
    "TESTED_BY": ("is exercised by", "exercises"),
    "APPLIES_IN": ("applies in", "is a context for"),
    "CONFIGURED_BY": ("is configured by", "configures"),
    "INITIATES": ("initiates", "is initiated by"),
    "ALGEBRA_VIOLATION": ("violates the layering toward", "is the target of a layering violation from"),
}


def _imports(content, ext):
    if ext == "java":
        raw = re.findall(r"^\s*import\s+(?:static\s+)?([\w.*]+)\s*;", content, re.M)
    else:
        raw = re.findall(r"""from\s+['"]([^'"]+)['"]""", content)
        raw += re.findall(r"""import\s+['"]([^'"]+)['"]""", content)
    internal, external = [], []
    for imp in raw:
        if imp.startswith(("com.sm", ".", "src/", "@app", "@core", "@shared", "app/")):
            internal.append(imp)
        else:
            external.append(imp)
    return internal, external


def structural(node, content, ext):
    name = node["name"]
    path = (node["path"] or "").replace("\\", "/")
    segments = [s for s in path.split("/") if s][3:]  # drop the drive and user prefix
    pkg = re.search(r"^\s*package\s+([\w.]+)\s*;", content, re.M)
    internal, external = _imports(content, ext)

    parents, ifaces, injected = [], [], []
    if ext == "java":
        parents = [x.strip() for x in JAVA_EXTENDS.findall(content)]
        ifaces = [i.strip() for m in JAVA_IMPLEMENTS.findall(content) for i in m.split(",")]
        injected = JAVA_INJECT.findall(content)
    elif ext == "ts":
        parents = [x.strip() for x in TS_EXTENDS.findall(content)]
        ifaces = [i.strip() for m in TS_IMPLEMENTS.findall(content) for i in m.split(",")]
        injected = [a or b for a, b in TS_INJECT.findall(content)]

    out_edges, in_edges = [], []
    for e in node.get("outs") or []:
        verb = RELATION_PROSE.get(e["rel"], (e["rel"].lower(), ""))[0]
        out_edges.append(f"{name} {verb} {e['name']} (a {str(e['nt']).lower()})")
    for e in node.get("ins") or []:
        verb = RELATION_PROSE.get(e["rel"], ("", "is " + e["rel"].lower() + " of"))[1]
        in_edges.append(f"{name} {verb} {e['name']} (a {str(e['nt']).lower()})")

    place = [
        f"Lives at {'/'.join(segments)}",
        f"Java package {pkg.group(1)}" if pkg else None,
        f"Classified as a {node['node_type']} playing the {node['entity_type']} role "
        f"in the six-entity model",
        f"Sits at hierarchy level {node['level']}" if node.get("level") is not None else None,
        f"Belongs to subsystem {node['subsystem']}, module {node['module']}"
        if node.get("subsystem") is not None else None,
        "Sits on a subsystem boundary, so it mediates between communities"
        if node.get("boundary") else "Sits in the interior of its community",
        f"Spans {node['lines']} lines" if node.get("lines") else None,
    ]

    degree = [
        f"Imports {node.get('imports_out', 0)} indexed units and is imported by "
        f"{node.get('imports_in', 0)}",
        f"Emits {len(node.get('outs') or [])} typed outgoing relations and receives "
        f"{len(node.get('ins') or [])} incoming",
    ]

    parts = [f"\nGraph position of {name} -- where it sits.\n"]
    parts.append(_section("Placement:", [p for p in place if p]))
    parts.append(_section("Inherits from:", _uniq(parents, 10)))
    parts.append(_section("Fulfils contracts:", _uniq(ifaces, 15)))
    parts.append(_section("Injected collaborators:", _uniq(injected, 30)))
    parts.append(_section("Outgoing dependencies in the graph:", _uniq(out_edges, 40)))
    parts.append(_section("Incoming dependents in the graph:", _uniq(in_edges, 40)))
    parts.append(_section("Imports project units:", _uniq(node.get("imported") or [], 30)))
    parts.append(_section("Imported by project units:", _uniq(node.get("importers") or [], 30)))
    parts.append(_section("Internal module references:", _uniq(internal, 40)))
    parts.append(_section("Third-party frameworks:", _uniq(external, 40)))
    parts.append(_section("Connectivity:", degree))
    return "".join(parts)


# --------------------------------------------------------------------------------------
# Socket assembly
# --------------------------------------------------------------------------------------

def build_sockets(node):
    path = node["path"] or ""
    ext = path.rsplit(".", 1)[-1].lower() if "." in os.path.basename(path) else ""
    name = node["name"] or os.path.basename(path)
    content = read_source(path)
    if not content:
        stub = f"\nSource unit {name} at {path} could not be read.\n"
        return {"S": stub, "B": stub, "T": structural(node, "", ext)}

    if ext == "java":
        s, b = semantic_java(content, name), behavioural(content, name, JAVA_BEHAVIOUR_PROBES)
    elif ext in ("ts", "js", "mts"):
        s, b = semantic_ts(content, name), behavioural(content, name, TS_BEHAVIOUR_PROBES)
    elif ext == "html":
        s, b = semantic_html(content, name), behavioural_html(content, name)
    elif ext in ("scss", "css", "sass"):
        s, b = semantic_style(content, name), behavioural_style(content, name)
    elif ext == "feature":
        s, b = semantic_feature(content, name), behavioural_feature(content, name)
    else:
        s, b = semantic_config(content, name), behavioural_config(content, name)

    t = structural(node, content, ext)
    return {"S": s[:SOCKET_MAX_CHARS], "B": b[:SOCKET_MAX_CHARS], "T": t[:SOCKET_MAX_CHARS]}


def lensed(lens, socket):
    # Byte-equivalent to the service's own `instruct` handling, but client-side (F72).
    return f"Instruct: {LENS_INSTRUCT[lens]}\nQuery:" + socket


# --------------------------------------------------------------------------------------
# Embedding transport
# --------------------------------------------------------------------------------------

_model_seen = {"model": None}
_model_lock = threading.Lock()


def embed_chunk(texts, retries=6):
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(EMBED_URL, json={"texts": texts}, timeout=900)
            r.raise_for_status()
            data = r.json()
            if "embeddings" not in data:
                raise RuntimeError(f"service error: {str(data)[:300]}")
            vecs = data["embeddings"]
            if len(vecs) != len(texts) or any(v is None or not v for v in vecs):
                raise RuntimeError("service returned an incomplete batch")
            with _model_lock:
                _model_seen["model"] = data.get("model", "unknown")
            return vecs
        except Exception as exc:  # noqa: BLE001 -- network/HTTP/JSON/service, all retryable
            last = exc
            wait = min(45, 2 ** attempt)
            print(f"  [embed] attempt {attempt + 1}/{retries} failed: {str(exc)[:160]}; "
                  f"retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"chunk failed after {retries} retries: {last}")


FETCH_QUERY = """
MATCH (n:EntityDetail {namespace: $ns})
WHERE n.file_path IS NOT NULL AND id(n) % $shard_count = $shard
  AND ($force OR n.semantic_embedding IS NULL OR n.behavioral_embedding IS NULL
       OR n.structural_embedding IS NULL)
WITH n ORDER BY id(n)
CALL (n) {
  MATCH (n)-[r]->(m:EntityDetail {namespace: $ns}) WHERE type(r) <> 'IMPORTS'
  RETURN collect(DISTINCT {rel: type(r), name: m.name, nt: m.node_type})[0..40] AS outs
}
CALL (n) {
  MATCH (n)<-[r]-(m:EntityDetail {namespace: $ns}) WHERE type(r) <> 'IMPORTS'
  RETURN collect(DISTINCT {rel: type(r), name: m.name, nt: m.node_type})[0..40] AS ins
}
CALL (n) {
  MATCH (n)-[:IMPORTS]->(m:EntityDetail {namespace: $ns})
  RETURN count(m) AS imports_out, collect(DISTINCT m.name)[0..30] AS imported
}
CALL (n) {
  MATCH (n)<-[:IMPORTS]-(m:EntityDetail {namespace: $ns})
  RETURN count(m) AS imports_in, collect(DISTINCT m.name)[0..30] AS importers
}
RETURN elementId(n) AS eid, n.name AS name, n.file_path AS path,
       n.node_type AS node_type, n.entity_type AS entity_type,
       n.v3_subsystem AS subsystem, n.v3_module AS module,
       n.is_boundary AS boundary, n.hierarchy_level AS level, n.line_count AS lines,
       outs, ins, imports_out, imported, imports_in, importers
"""

WRITE_QUERY = """
UNWIND $rows AS row
MATCH (n) WHERE elementId(n) = row.eid
SET n.semantic_embedding   = row.s,
    n.behavioral_embedding = row.b,
    n.structural_embedding = row.t,
    n.lens_embedding_model = row.model,
    n.lens_socket_version  = $version,
    n.lens_embedded_at     = datetime()
"""


def run_embedding(session, args):
    nodes = session.run(FETCH_QUERY, ns=NAMESPACE, shard=args.shard,
                        shard_count=args.shard_count, force=args.force).data()
    if args.limit:
        nodes = nodes[: args.limit]
    print(f"[shard {args.shard}/{args.shard_count}] {len(nodes)} nodes to embed", flush=True)
    if not nodes:
        return 0, 0

    if args.dry:
        for node in nodes:
            sockets = build_sockets(node)
            print("=" * 100)
            print(f"{node['name']}  ({node['node_type']}/{node['entity_type']})")
            for lens in ("S", "B", "T"):
                print(f"\n----- {lens} ({len(sockets[lens])} chars) -----")
                print(sockets[lens][:1500])
        return len(nodes), 0

    written, errors = 0, 0
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    for start in range(0, len(nodes), NODE_GROUP):
        group = nodes[start: start + NODE_GROUP]
        jobs = []   # (node_index_in_group, lens, text)
        for i, node in enumerate(group):
            sockets = build_sockets(node)
            for lens in ("S", "B", "T"):
                jobs.append((i, lens, lensed(lens, sockets[lens])))

        chunks = [jobs[c: c + EMBED_BATCH] for c in range(0, len(jobs), EMBED_BATCH)]
        try:
            results = list(pool.map(lambda ch: embed_chunk([t for _, _, t in ch]), chunks))
        except RuntimeError as exc:
            print(f"  [shard {args.shard}] group at {start} permanently failed: {exc}", flush=True)
            errors += len(group)
            continue

        vectors = [{} for _ in group]
        for chunk, vecs in zip(chunks, results):
            for (idx, lens, _), vec in zip(chunk, vecs):
                vectors[idx][lens] = vec

        rows = [{"eid": node["eid"], "s": v["S"], "b": v["B"], "t": v["T"],
                 "model": _model_seen["model"]}
                for node, v in zip(group, vectors) if len(v) == 3]
        session.run(WRITE_QUERY, rows=rows, version=SOCKET_VERSION)
        written += len(rows)
        errors += len(group) - len(rows)
        rate = written / max(time.time() - t0, 1e-9)
        print(f"  [shard {args.shard}] {written}/{len(nodes)} written, {errors} errored, "
              f"{rate * 60:.0f} nodes/min", flush=True)
    pool.shutdown()
    return written, errors


# --------------------------------------------------------------------------------------
# Meta-path hyperedge candidates
# --------------------------------------------------------------------------------------

# Anchored on the CENTRE node of each meta-path, and the centre is what the shard
# predicate selects -- so the three shards form a partition of all instances (no gaps),
# while the member-set key still makes concurrent MERGE idempotent if a sibling shard
# anchors differently.
METAPATHS = {
    "P_R_P": """
        MATCH (c:EntityDetail {namespace:$ns, entity_type:'Resource'})
        WHERE id(c) % $shard_count = $shard
        MATCH (p1:EntityDetail {namespace:$ns, entity_type:'Process'})-[r1]->(c)
        MATCH (p2:EntityDetail {namespace:$ns, entity_type:'Process'})-[r2]->(c)
        WHERE type(r1) IN $fwd AND type(r2) IN $fwd AND id(p1) < id(p2)
        WITH c, p1, p2, collect(DISTINCT type(r1)) + collect(DISTINCT type(r2)) AS rels
        RETURN [id(p1), id(c), id(p2)] AS ids,
               [{eid: elementId(p1), role:'PROCESS',  dir:'SOURCE'},
                {eid: elementId(c),  role:'RESOURCE', dir:'TARGET'},
                {eid: elementId(p2), role:'PROCESS',  dir:'SOURCE'}] AS members,
               [p1.name, c.name, p2.name] AS names, rels
    """,
    "A_P_A": """
        MATCH (c:EntityDetail {namespace:$ns, entity_type:'Process'})
        WHERE id(c) % $shard_count = $shard
        MATCH (a1:EntityDetail {namespace:$ns, entity_type:'Actor'})-[r1]->(c)
        MATCH (a2:EntityDetail {namespace:$ns, entity_type:'Actor'})-[r2]->(c)
        WHERE type(r1) IN $act AND type(r2) IN $act AND id(a1) < id(a2)
        WITH c, a1, a2, collect(DISTINCT type(r1)) + collect(DISTINCT type(r2)) AS rels
        RETURN [id(a1), id(c), id(a2)] AS ids,
               [{eid: elementId(a1), role:'ACTOR',   dir:'SOURCE'},
                {eid: elementId(c),  role:'PROCESS', dir:'TARGET'},
                {eid: elementId(a2), role:'ACTOR',   dir:'SOURCE'}] AS members,
               [a1.name, c.name, a2.name] AS names, rels
    """,
    "A_P_R": """
        MATCH (c:EntityDetail {namespace:$ns, entity_type:'Process'})
        WHERE id(c) % $shard_count = $shard
        MATCH (a:EntityDetail {namespace:$ns, entity_type:'Actor'})-[r1]->(c)
        MATCH (c)-[r2]->(res:EntityDetail {namespace:$ns, entity_type:'Resource'})
        WHERE type(r1) IN $act AND type(r2) IN $fwd
        WITH c, a, res, collect(DISTINCT type(r1)) + collect(DISTINCT type(r2)) AS rels
        RETURN [id(a), id(c), id(res)] AS ids,
               [{eid: elementId(a),   role:'ACTOR',    dir:'SOURCE'},
                {eid: elementId(c),   role:'PROCESS',  dir:'MEDIATOR'},
                {eid: elementId(res), role:'RESOURCE', dir:'TARGET'}] AS members,
               [a.name, c.name, res.name] AS names, rels
    """,
}
FWD_RELS = ["USES", "MODIFIES", "ACCESSES", "EXTENDS", "IMPLEMENTS", "INJECTS"]
ACT_RELS = ["PERFORMS", "INJECTS", "IMPLEMENTS", "CALLS", "TRIGGERS"]

MERGE_HYPEREDGE = """
UNWIND $rows AS row
MERGE (h:HyperedgeCandidate {key: row.key})
ON CREATE SET h.namespace = $ns, h.metapath = row.metapath, h.arity = size(row.members),
              h.member_ids = row.ids, h.member_names = row.names,
              h.via_relations = row.rels, h.created_at = datetime(),
              h.first_emitted_by = $shard_tag, h.source = 'metapath-v2'
WITH h, row
UNWIND row.members AS member
MATCH (n) WHERE elementId(n) = member.eid
MERGE (n)-[im:IN_HYPEREDGE]->(h)
ON CREATE SET im.role = member.role, im.direction = member.dir, im.weight = 1.0
"""


def run_hyperedges(session, args):
    totals = {}
    for metapath, query in METAPATHS.items():
        instances = session.run(query, ns=NAMESPACE, shard=args.shard,
                                shard_count=args.shard_count,
                                fwd=FWD_RELS, act=ACT_RELS).data()
        rows = []
        for inst in instances:
            ids = sorted(inst["ids"])
            # Deterministic sorted-member-id key: identical for any shard that finds the
            # same member set, so concurrent MERGE converges instead of duplicating.
            rows.append({
                "key": f"{metapath}:" + "-".join(str(i) for i in ids),
                "metapath": metapath,
                "ids": ids,
                "names": inst["names"],
                "rels": sorted(set(inst["rels"])),
                "members": inst["members"],
            })
        # Dedupe within the shard before the write (a pair can be reached by two arrows).
        rows = list({r["key"]: r for r in rows}.values())
        if rows:
            session.run(MERGE_HYPEREDGE, rows=rows, ns=NAMESPACE,
                        shard_tag=f"shard-{args.shard}")
        totals[metapath] = len(rows)
        print(f"  [shard {args.shard}] {metapath}: {len(rows)} candidates merged", flush=True)
    return totals


# --------------------------------------------------------------------------------------
# Lens-independence report
# --------------------------------------------------------------------------------------

EMBEDDED_IDS_QUERY = """
MATCH (n:EntityDetail {namespace: $ns})
WHERE id(n) % $shard_count = $shard AND n.semantic_embedding IS NOT NULL
  AND n.behavioral_embedding IS NOT NULL AND n.structural_embedding IS NOT NULL
RETURN elementId(n) AS eid ORDER BY id(n)
"""

SAMPLE_QUERY = """
UNWIND $eids AS eid
MATCH (n) WHERE elementId(n) = eid
RETURN n.name AS name, n.node_type AS node_type,
       n.semantic_embedding AS s, n.behavioral_embedding AS b, n.structural_embedding AS t
"""


def run_report(session, args):
    import numpy as np

    eids = [r["eid"] for r in session.run(EMBEDDED_IDS_QUERY, ns=NAMESPACE, shard=args.shard,
                                          shard_count=args.shard_count).data()]
    if not eids:
        print("no embedded nodes in this shard yet")
        return None
    # Stride sample DISTINCT nodes -- a naive index formula repeats nodes whenever the
    # population is smaller than the sample, which silently flatters the cosine.
    total = len(eids)
    k = min(args.sample, total)
    picked = [eids[i * total // k] for i in range(k)]
    rows = session.run(SAMPLE_QUERY, eids=picked).data()
    if not rows:
        print("no embedded nodes in this shard yet")
        return None

    def norm(m):
        m = np.asarray(m, dtype=np.float64)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    S, B, T = (norm([r[k] for r in rows]) for k in ("s", "b", "t"))
    sb = float(np.mean(np.sum(S * B, axis=1)))
    st = float(np.mean(np.sum(S * T, axis=1)))
    bt = float(np.mean(np.sum(B * T, axis=1)))
    mean = (sb + st + bt) / 3

    print(f"\n[shard {args.shard}] lens independence over {len(rows)} sampled nodes "
          f"(dim {S.shape[1]})")
    print(f"  mean cosine S-B : {sb:.4f}")
    print(f"  mean cosine S-T : {st:.4f}")
    print(f"  mean cosine B-T : {bt:.4f}")
    print(f"  MEAN PAIRWISE   : {mean:.4f}")
    print(f"  (instruction-only baseline was 0.9451; lower is more independent)")
    return {"S_B": sb, "S_T": st, "B_T": bt, "mean": mean, "n": len(rows)}


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="cap nodes this run (0 = all)")
    ap.add_argument("--force", action="store_true", help="re-embed nodes that already have vectors")
    ap.add_argument("--dry", action="store_true", help="print sockets, embed nothing")
    ap.add_argument("--hyperedges", action="store_true", help="emit meta-path candidates only")
    ap.add_argument("--report", action="store_true", help="lens-cosine report only")
    ap.add_argument("--sample", type=int, default=50)
    args = ap.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    t0 = time.time()
    with driver.session() as session:
        if args.report:
            run_report(session, args)
        elif args.hyperedges:
            run_hyperedges(session, args)
        else:
            written, errors = run_embedding(session, args)
            print(f"\n[shard {args.shard}] embedding done: {written} written, {errors} errored, "
                  f"model={_model_seen['model']}, {time.time() - t0:.0f}s")
            if not args.dry:
                run_hyperedges(session, args)
                run_report(session, args)
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
