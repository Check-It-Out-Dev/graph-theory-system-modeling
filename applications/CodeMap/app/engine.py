# CodeMap engine v0 — the DSL's executor over the exported pack (docs/03-dsl-interface.md).
# One intelligence module, two transports (HTTP for the app/small model, MCP-stdio later):
# 13 verbs -> compiled queries/lookups; results carry `affordances` (executable next steps).
#
# Data: graph/pack/ (entities.csv, edges_lb.csv, hyperedges.csv, l2_navigators.jsonl,
# l1_master.json, mfq.jsonl) + the embedded Ladybug DB for dialect-real queries.
# find/impact run on Ladybug (the runtime DB, dialect-proven); the rest read the pack
# in memory (1.4k nodes — RAM beats round-trips; documented v0 choice).

import csv, json, os, re
from collections import Counter, defaultdict

PACK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "graph", "pack"))
QDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval", "q"))


def _tok(s):
    return set(re.findall(r"[a-z0-9]{3,}", s.lower()))


class Engine:
    def __init__(self, use_ladybug=True):
        self.ents = list(csv.DictReader(open(os.path.join(PACK, "entities.csv"), encoding="utf-8")))
        self.by_name = {}
        for e in self.ents:
            self.by_name.setdefault(e["name"], e)  # 2 dup basenames: first wins, documented
        self.edges = list(csv.DictReader(open(os.path.join(PACK, "edges_lb.csv"), encoding="utf-8")))
        self.p2n = {e["file_path"]: e["name"] for e in self.ents}
        self.in_e, self.out_e = defaultdict(list), defaultdict(list)
        for ed in self.edges:
            s, d = self.p2n.get(ed["src"]), self.p2n.get(ed["dst"])
            if s and d:
                self.out_e[s].append((d, ed["rel"]))
                self.in_e[d].append((s, ed["rel"]))
        self.hyper = list(csv.DictReader(open(os.path.join(PACK, "hyperedges.csv"), encoding="utf-8")))
        self.l2 = {json.loads(l)["sub_id"]: json.loads(l)
                   for l in open(os.path.join(PACK, "l2_navigators.jsonl"), encoding="utf-8")}
        # MERGED navigators ride the pack as successor records (ledger L9), never as routes
        self.successors = {k: (n.get("superseded_by") or [None])[0]
                           for k, n in self.l2.items() if n.get("routable") is False}
        self.l2 = {k: n for k, n in self.l2.items() if n.get("routable") is not False}
        self.roots = sorted(k for k, n in self.l2.items()
                            if n.get("parent") is None or n.get("parent") not in self.l2)
        self.leaves = {k: n for k, n in self.l2.items() if n.get("role") != "GROUP"}
        self.l1 = json.load(open(os.path.join(PACK, "l1_master.json"), encoding="utf-8"))
        self.mfq = [json.loads(l) for l in open(os.path.join(PACK, "mfq.jsonl"), encoding="utf-8")]
        inv_path = os.path.join(QDIR, "INVALIDATED_2026-09-02.json")
        self.invalidated = set()
        if os.path.exists(inv_path):
            j = json.load(open(inv_path, encoding="utf-8"))
            self.invalidated = {r["id"] if isinstance(r, dict) else r
                                for r in (j if isinstance(j, list) else j.get("invalidated", []))}
        self.lb = None
        if use_ladybug:
            try:
                import real_ladybug as lb
                # read_only: the pack is an EXPORT — nothing in the app may write it,
                # and read-only lets a bench and the app share the DB across processes
                self.lb = lb.Connection(lb.Database(os.path.join(PACK, "codemap.lbdb"),
                                                    read_only=True))
            except Exception as e:  # engine still works from memory; dialect path reported off
                self.lb_error = str(e)

    # ---------- verbs ----------
    def map(self):
        return dict(kind="l1", summary=self.l1.get("ai_summary"),
                    index=self.l1.get("subsystem_index"),
                    caveats=self.l1.get("global_caveats"),
                    affordances=[f"enter({k})" for k in self.roots] + ["health(coupling)"])

    def enter(self, sub):
        sub = self._sub(sub)
        n = self.l2[sub]
        if n.get("role") == "GROUP":
            kids = [self.l2[c] for c in n.get("children", []) if c in self.l2]
            return dict(kind="l2_group", sub=sub, name=n.get("name"),
                        summary=n.get("ai_summary"),
                        children=[dict(sub=k["sub_id"], name=k.get("name"),
                                       size=k.get("size"), role=k.get("role"))
                                  for k in kids],
                        affordances=[f"enter({k['sub_id']})" for k in kids][:9])
        aff = [f"spine({sub})"]
        for ep in (json.loads(n.get("entry_points", "[]")) if isinstance(n.get("entry_points"), str) else n.get("entry_points") or [])[:3]:
            aff += [f"impact({ep})", f"flow({ep},1)"]
        return dict(kind="l2", sub=sub, name=n.get("name"), summary=n.get("ai_summary"),
                    responsibilities=n.get("responsibilities"), caveats=n.get("caveats"),
                    size=n.get("size"), external_ratio=n.get("external_ratio"),
                    contracts=n.get("contracts"), affordances=aff[:9])

    def find(self, term):
        t = term.lower()
        if self.lb:
            res = self.lb.execute(
                f"MATCH (e:Entity) WHERE lower(e.name) CONTAINS '{self._safe(t)}' "
                "RETURN e.name, e.entity_type, e.subsystem, e.file_path LIMIT 12")
            rows = []
            while res.has_next():
                rows.append(res.get_next())
        else:
            rows = [[e["name"], e["entity_type"], e["subsystem"], e["file_path"]]
                    for e in self.ents if t in e["name"].lower()][:12]
        return dict(kind="hits", term=term,
                    hits=[dict(name=r[0], type=r[1], sub=r[2]) for r in rows],
                    affordances=[f"impact({r[0]})" for r in rows[:3]] + [f"read({r[0]})" for r in rows[:2]])

    def impact(self, name, depth=1):
        name = self._entity(name)
        seen, frontier, out = {name}, [name], []
        for d in range(int(depth)):
            nxt = []
            for f in frontier:
                for src, rel in self.in_e.get(f, []):
                    out.append(dict(file=src, rel=rel, of=f, hop=d + 1))
                    if src not in seen:
                        seen.add(src)
                        nxt.append(src)
            frontier = nxt
        subs = sorted({self.by_name[r["file"]].get("curated") or self.by_name[r["file"]]["subsystem"]
                       for r in out if r["file"] in self.by_name})
        return dict(kind="impact", target=name, dependents=len(seen) - 1, subsystems=subs,
                    top=[n for n, _ in Counter(r["file"] for r in out).most_common(8)],
                    rows=out[:40], affordances=[f"enter({s})" for s in subs[:4]] + [f"read({name})"])

    def flow(self, name, depth=1):
        name = self._entity(name)
        seen, frontier, out = {name}, [name], []
        for d in range(int(depth)):
            nxt = []
            for f in frontier:
                for dst, rel in self.out_e.get(f, []):
                    out.append(dict(file=dst, rel=rel, frm=f, hop=d + 1))
                    if dst not in seen:
                        seen.add(dst)
                        nxt.append(dst)
            frontier = nxt
        mm = ["sequenceDiagram"]
        for r in out[:12]:
            a = r["frm"].replace(".", "_").replace("-", "_")
            b = r["file"].replace(".", "_").replace("-", "_")
            mm.append(f"    {a}->>{b}: {r['rel']}")
        return dict(kind="flow", start=name, touches=len(seen) - 1, rows=out[:40],
                    mermaid="\n".join(mm) if out else None,
                    affordances=[f"read({r['file']})" for r in out[:3]])

    def seam(self, a, b):
        a, b = self._sub(a), self._sub(b)
        rows = []
        for ed in self.edges:
            s, d = self.p2n.get(ed["src"]), self.p2n.get(ed["dst"])
            if not s or not d:
                continue
            sa = self.by_name[s].get("curated") or self.by_name[s]["subsystem"]
            sb = self.by_name[d].get("curated") or self.by_name[d]["subsystem"]
            if {int(sa or -1), int(sb or -1)} == {a, b}:
                rows.append(dict(src=s, rel=ed["rel"], dst=d))
        kinds = Counter(r["rel"] for r in rows)
        mm = ["sequenceDiagram"]
        for r in ([x for x in rows if x["rel"] != "IMPORTS"][:8] or rows[:8]):
            sa_ = r["src"].replace(".", "_").replace("-", "_")
            sb_ = r["dst"].replace(".", "_").replace("-", "_")
            mm.append(f"    {sa_}->>{sb_}: {r['rel']}")
        return dict(kind="seam", subs=[a, b], edges=len(rows), by_type=dict(kinds),
                    notable=[r for r in rows if r["rel"] != "IMPORTS"][:6],
                    mermaid="\n".join(mm) if rows else None,
                    affordances=[f"enter({a})", f"enter({b})"])

    def cohort(self, name):
        name = self._entity(name)
        hs = [h for h in self.hyper if name == h["hub"] or name in json.loads(h["members"] or "[]")]
        return dict(kind="cohort", of=name,
                    cohorts=[dict(metapath=h["metapath"], hub=h["hub"], idf=h["idf"],
                                  members=json.loads(h["members"] or "[]")) for h in hs],
                    affordances=[f"impact({h['hub']})" for h in hs[:3]] or [f"impact({name})"])

    def spine(self, sub):
        sub = self._sub(sub)
        n = self.l2[sub]
        sp = n.get("spines")
        sp = json.loads(sp) if isinstance(sp, str) else (sp or [])
        eps = n.get("entry_points")
        eps = json.loads(eps) if isinstance(eps, str) else (eps or [])
        return dict(kind="spine", sub=sub, entry_points=eps[:5], spines=sp,
                    affordances=[f"read({e})" for e in eps[:3]])

    def health(self, kind):
        if kind == "coupling":
            rows = sorted(((s, n.get("size"), n.get("external_ratio")) for s, n in self.leaves.items()),
                          key=lambda r: -(r[2] or 0))
            return dict(kind="health", metric="coupling",
                        rows=[dict(sub=s, size=z, external_ratio=x) for s, z, x in rows],
                        affordances=[f"enter({rows[0][0]})", f"seam({rows[0][0]},{rows[1][0]})"])
        if kind == "hubs":
            cnt = Counter(d for outs in [self.in_e[n] for n in self.in_e] for d, _ in [])
            top = Counter({n: len(v) for n, v in self.in_e.items()}).most_common(10)
            return dict(kind="health", metric="hubs",
                        rows=[dict(file=n, in_degree=c) for n, c in top],
                        affordances=[f"impact({top[0][0]})"])
        raise DslError(f"health kind '{kind}' not in v0 (coupling|hubs)")

    def read(self, name):
        e = self.by_name.get(self._entity(name))
        return dict(kind="pointer", file=e["name"], path=e["file_path"],
                    note="content answer — open the file; the graph pins WHERE",
                    affordances=[f"impact({e['name']})"])

    def cache(self, q):
        qt = _tok(q)
        best, score = None, 0.0
        for r in self.mfq:
            if r["id"] in self.invalidated or not r.get("gold_answer"):
                continue
            for cand in [r["q"]] + (r.get("aliases") or []):
                ct = _tok(cand)
                j = len(qt & ct) / max(1, len(qt | ct))
                if j > score:
                    best, score = r, j
        if best and score >= 0.5:
            return dict(kind="cache_hit", id=best["id"], score=round(score, 2),
                        answer=best["gold_answer"], affordances=[])
        return dict(kind="cache_miss", score=round(score, 2),
                    nearest=best["id"] if best else None,
                    affordances=["map()"])

    # ---------- helpers ----------
    def _sub(self, s):
        s = str(s).strip()
        if s.isdigit() and int(s) in self.successors:
            succ = self.successors[int(s)]
            if succ in self.l2:
                return succ  # stale foreign key resolves to its absorber (L9)
        if s.isdigit() and int(s) in self.l2:
            return int(s)
        for k, n in self.l2.items():
            if s.lower() in (n.get("name") or "").lower():
                return k
        raise DslError(f"unknown subsystem '{s}' — try map()")

    def _entity(self, n):
        n = n.strip()
        if n in self.by_name:
            return n
        hits = [e["name"] for e in self.ents if n.lower() in e["name"].lower()]
        if len(hits) == 1:
            return hits[0]
        raise DslError(f"unknown entity '{n}'" + (f" — candidates: {hits[:5]}" if hits else " — try find()"))

    @staticmethod
    def _safe(t):
        return re.sub(r"[^a-z0-9 ._-]", "", t)


class DslError(Exception):
    pass
