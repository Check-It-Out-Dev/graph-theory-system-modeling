# Movie scene 2 generator — turns the REAL `codemap up` transcript (movie/out/wizard.log,
# ANSI codes intact) into a typed-replay HTML terminal. The replay is presentation only:
# every byte of output shown was produced by an actual wizard run on this machine.
# Usage: python movie/gen_terminal.py   ->  movie/out/scene_terminal.html

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "out", "wizard.log")
OUT = os.path.join(HERE, "out", "scene_terminal.html")

ANSI = {"92": "ok", "91": "bad", "96": "info"}


def to_spans(line):
    """ANSI SGR -> <span class=...>; only the three codes the wizard emits."""
    out, cls = [], None
    for tok in re.split(r"(\x1b\[\d+m)", line):
        m = re.fullmatch(r"\x1b\[(\d+)m", tok)
        if m:
            cls = ANSI.get(m.group(1))
            continue
        if not tok:
            continue
        esc = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        out.append(f'<span class="{cls}">{esc}</span>' if cls else esc)
        if tok.endswith("\x1b[0m"):
            cls = None
    return "".join(out).replace("\x1b[0m", "")


def main():
    raw = open(LOG, encoding="utf-8").read().replace("\x1b[0m", "\x1b[92m" * 0 + "\x1b[0m")
    lines = raw.rstrip("\n").split("\n")
    # per-line reveal delays (ms): banner slow, checks snappy, boot line breathes
    delay = []
    for ln in lines:
        if "starting server" in ln:
            delay.append(650)
        elif "CodeMap is up" in ln:
            delay.append(1300)
        elif "Ctrl+C" in ln:
            delay.append(500)
        else:
            delay.append(240)
    spans = [to_spans(re.sub(r"\x1b\[0m", "", ln)) for ln in lines]
    payload = json.dumps([{"h": s, "d": d} for s, d in zip(spans, delay)],
                         ensure_ascii=False)

    html = """<!doctype html><meta charset="utf-8"><title>codemap up</title>
<style>
  :root{--bg:#0e1116;--term:#0b0e12;--chrome:#1b212b;--line:#262d38;
        --text:#fbf9f4;--dim:#98a2b0;--accent:#ff5a36}
  *{margin:0;box-sizing:border-box}
  body{background:var(--bg);height:100vh;display:flex;align-items:center;
       justify-content:center;font-family:'Cascadia Mono','Consolas',monospace}
  .term{width:1460px;height:800px;background:var(--term);border:1px solid var(--line);
        border-radius:14px;overflow:hidden;box-shadow:0 30px 90px #000a}
  .bar{background:var(--chrome);padding:.85rem 1.2rem;color:var(--dim);
       font-size:1.05rem;display:flex;gap:.6rem;align-items:center}
  .dot{width:13px;height:13px;border-radius:50%;background:#2c333d}
  .dot.r{background:#e0605e}.dot.y{background:#e0b25e}.dot.g{background:#5ec06a}
  .body{padding:1.6rem 2rem;font-size:1.5rem;line-height:2.15;color:var(--text);
        white-space:pre-wrap}
  .ok{color:#6ee7b7}.bad{color:#f87171}.info{color:#c4b5fd}
  .ps{color:var(--dim)} .cmd{color:var(--accent);font-weight:600}
  .cur{display:inline-block;width:.62em;height:1.25em;background:var(--text);
       vertical-align:-0.2em;animation:bl 1s steps(2,start) infinite}
  @keyframes bl{to{visibility:hidden}}
  .cap{position:fixed;left:0;right:0;bottom:46px;text-align:center;color:var(--text);
       font-family:'Segoe UI',system-ui,sans-serif;font-size:1.65rem;opacity:0;
       transition:opacity .6s}
  .cap b{color:var(--accent)}
</style>
<div class="term">
  <div class="bar"><span class="dot r"></span><span class="dot y"></span>
    <span class="dot g"></span>&nbsp; PowerShell — codemap</div>
  <div class="body" id="b"><span class="ps">PS C:\\codemap&gt; </span><span
    class="cmd" id="typed"></span><span class="cur" id="cur"></span></div>
</div>
<div class="cap" id="cap">One installer — or one command after clone. <b>No cloud,
no accounts</b> — Python, the graph pack and the model ride inside.</div>
<script>
const LINES = __PAYLOAD__;
const CMD = "python codemap.py up";
const typed = document.getElementById('typed'), b = document.getElementById('b'),
      cur = document.getElementById('cur');
const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  await sleep(900);
  document.getElementById('cap').style.opacity = 1;
  for (const ch of CMD) { typed.textContent += ch; await sleep(52 + Math.sin(typed.textContent.length) * 14); }
  await sleep(600);
  cur.remove();
  for (const l of LINES) {
    await sleep(l.d);
    const d = document.createElement('div');
    d.innerHTML = l.h || '\\u00a0';
    b.appendChild(d);
  }
  await sleep(2400);
  window.__done = true;
})();
</script>
"""
    open(OUT, "w", encoding="utf-8").write(html.replace("__PAYLOAD__", payload))
    print(f"scene_terminal.html: {len(lines)} transcript lines")


if __name__ == "__main__":
    main()
