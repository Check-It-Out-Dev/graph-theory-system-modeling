// The quality dashboard: one static page per repository, published at the root of its GitHub Pages site.
// It reads two files next to itself, quality-metrics.json (the latest run) and metrics/history.jsonl
// (one line per run), both described in docs/ci/METRICS.md. No framework, no build step.
// Preview with fixture data through any static server: index.html?data=fixtures
(() => {
  'use strict';

  const params = new URLSearchParams(location.search);
  const BASE = (params.get('data') || '.').replace(/\/$/, '');
  const WINDOW = 30;
  const FLAKY_WINDOW = 10;

  const numberFormat = new Intl.NumberFormat('en-GB');
  const n = (x) => numberFormat.format(x);
  const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)} %`;
  const ms = (x) => (x >= 1000 ? `${(x / 1000).toFixed(2)} s` : `${x.toFixed(x < 10 ? 1 : 0)} ms`);
  const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'];
  const word = (k) => (k < WORDS.length ? WORDS[k] : n(k));
  const ORDINALS = ['', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth'];
  const ordinal = (k) => (k < ORDINALS.length ? ORDINALS[k] : `${k}th`);
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const when = (iso) =>
    `${new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'UTC' }).format(new Date(iso))} UTC`;
  const dur = (s) => {
    s = Math.round(s || 0);
    const m = Math.floor(s / 60);
    return m ? `${m} min ${s % 60} s` : `${s} s`;
  };
  const outcome = (r) => (r.failed > 0 ? 'fail' : r.flaky > 0 ? 'flaky' : 'pass');

  const TIER_LABELS = {
    jest: 'Unit (Jest)',
    sandbox: 'Sandbox (Playwright, mocked API)',
    msw: 'Mock-API (Playwright, MSW)',
    integration: 'Integration (Playwright, live backend)',
    bdd: 'Behaviour (Cucumber)',
    perf: 'Performance (Playwright)',
    visual: 'Visual regression',
    unit: 'Unit (JUnit)',
    e2e: 'End to end (Cucumber)',
    pytest: 'Python (pytest)',
  };

  const SVG = 'http://www.w3.org/2000/svg';

  function el(tag, attrs = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = v;
      else e.setAttribute(k, v);
    }
    for (const c of children.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(String(c)));
    return e;
  }
  const $ = (id) => document.getElementById(id);

  function parseJsonl(text) {
    return text
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean)
      .map((l) => {
        try {
          return JSON.parse(l);
        } catch {
          return null;
        }
      })
      .filter(Boolean)
      .sort((a, b) => a.run - b.run);
  }

  /* ---------- pieces ---------- */

  function sparkline(values, toneClass) {
    const v = values.filter((x) => typeof x === 'number' && Number.isFinite(x));
    if (v.length < 2) return el('span', { class: 'spark spark--none', text: 'one run so far' });
    const W = 120, H = 32, P = 3;
    const min = Math.min(...v), max = Math.max(...v), span = max - min || 1;
    const pts = v.map((y, i) => [P + (i * (W - 2 * P)) / (v.length - 1), H - P - ((y - min) / span) * (H - 2 * P)]);
    const svg = document.createElementNS(SVG, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('class', 'spark');
    svg.setAttribute('aria-hidden', 'true');
    const line = document.createElementNS(SVG, 'polyline');
    line.setAttribute('points', pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' '));
    line.setAttribute('fill', 'none');
    line.setAttribute('stroke', 'currentColor');
    line.setAttribute('stroke-width', '1.5');
    line.setAttribute('stroke-linejoin', 'round');
    svg.append(line);
    const [lx, ly] = pts[pts.length - 1];
    const dot = document.createElementNS(SVG, 'circle');
    dot.setAttribute('cx', lx.toFixed(1));
    dot.setAttribute('cy', ly.toFixed(1));
    dot.setAttribute('r', '3');
    dot.setAttribute('class', toneClass);
    dot.setAttribute('fill', 'currentColor');
    svg.append(dot);
    return svg;
  }

  function tile(value, label, tone, series) {
    return el('div', { class: 'tile' },
      el('div', { class: `tile__value tone-${tone}`, text: value }),
      el('div', { class: 'tile__label', text: label }),
      sparkline(series, `tone-${tone}`),
    );
  }

  function table(headers, rows, opts = {}) {
    const thead = el('thead', {}, el('tr', {}, headers.map((h) => el('th', { class: h.num ? 'num' : null, scope: 'col', text: h.text }))));
    const tbody = el('tbody', {}, rows.map((r) => el('tr', { class: r.total ? 'total' : null }, r.cells.map((c) => el('td', { class: c.num ? 'num' : null }, c.node ?? c.text)))));
    return el('div', { class: 'tablewrap' }, el('table', { 'aria-label': opts.label }, thead, tbody));
  }

  /* ---------- sections ---------- */

  function renderMast(m) {
    const repoUrl = `https://github.com/${m.repo}`;
    const link = $('repoLink');
    link.href = repoUrl;
    link.textContent = m.repo.replace('/', ' / ');
    $('actionsLink').href = `${repoUrl}/actions`;
    document.title = `Quality dashboard, ${m.repo.split('/').pop()}`;
    return repoUrl;
  }

  function renderVerdict(m, history) {
    const t = m.tests, r = m.run;
    const scored = t.passed + t.failed + t.flaky;
    let text;
    if (t.failed > 0) text = `${n(t.failed)} of ${n(scored)} tests failed on ${r.branch}, run ${r.number}.`;
    else if (t.flaky > 0) text = `${n(t.passed)} of ${n(scored)} tests passed on ${r.branch}, run ${r.number}: ${word(t.flaky)} flaky, none failed.`;
    else text = `All ${n(t.passed)} tests passed on ${r.branch}, run ${r.number}.`;
    $('verdict').textContent = text;

    const parts = [`Started ${when(r.startedAt)}, took ${dur(r.durationSec)}.`];
    if (t.skipped) parts.push(`${cap(word(t.skipped))} skipped.`);
    // Streak: this run first, then history newest-first, counting runs with nothing failed.
    const runs = [{ run: r.number, failed: t.failed }, ...history.filter((h) => h.run !== r.number).reverse()];
    if (t.failed > 0) {
      let green = 0;
      for (const h of runs.slice(1)) { if (h.failed > 0) break; green++; }
      if (green) parts.push(`The first red run after ${word(green)} green ${green === 1 ? 'one' : 'ones'}.`);
    } else {
      let streak = 0;
      for (const h of runs) { if (h.failed > 0) break; streak++; }
      if (streak >= 2) parts.push(`${cap(ordinal(streak))} green run in a row.`);
      else if (runs.length > 1) parts.push('First green run after a failure.');
    }
    $('verdictMeta').textContent = parts.join(' ');
  }

  function renderRibbon(history, repoUrl) {
    const box = $('ribbon');
    box.replaceChildren();
    const caption = $('ribbonCaption');
    const idle = caption.textContent;
    const runs = history.slice(-WINDOW);
    for (let i = runs.length; i < WINDOW; i++) box.append(el('span', { class: 'bar bar--empty', 'aria-hidden': 'true' }));
    // Height is the run's duration against the slowest run in the window: pass rates in a healthy project
    // never leave 99–100 %, so they would all look the same; a slow run is worth seeing.
    const slowest = Math.max(...runs.map((h) => h.durationSec || 0), 1);
    for (const h of runs) {
      const height = 25 + 75 * Math.max(0, Math.min(1, (h.durationSec || 0) / slowest));
      const label = `Run ${h.run}, ${when(h.at)}: ${n(h.passed)} passed, ${h.flaky} flaky, ${h.failed} failed, ${dur(h.durationSec)}.`;
      const bar = el(h.id ? 'a' : 'span', {
        class: `bar bar--${outcome(h)}`,
        style: `height:${height.toFixed(0)}%`,
        href: h.id ? `${repoUrl}/actions/runs/${h.id}` : null,
        tabindex: h.id ? null : '0',
        role: 'listitem',
        'aria-label': label,
      });
      const show = () => { caption.textContent = label; };
      const hide = () => { caption.textContent = idle; };
      bar.addEventListener('mouseenter', show);
      bar.addEventListener('focus', show);
      bar.addEventListener('mouseleave', hide);
      bar.addEventListener('blur', hide);
      box.append(bar);
    }
  }

  function renderTiles(m, history) {
    const t = m.tests;
    const tiles = $('tiles');
    tiles.replaceChildren();
    const series = (key) => history.map((h) => h[key]);

    tiles.append(tile(pct(t.passRate, 2), `Pass rate, run ${m.run.number}`, outcome(t), series('passRate')));

    const flakyCount = (m.flaky || []).length;
    tiles.append(tile(
      `${flakyCount} ${flakyCount === 1 ? 'test' : 'tests'}`,
      `Flaky or failing, last ${word(FLAKY_WINDOW)} runs`,
      flakyCount === 0 ? 'pass' : flakyCount <= 3 ? 'flaky' : 'fail',
      series('flaky'),
    ));

    if (m.coverage) {
      const c = m.coverage.lines;
      tiles.append(tile(`${c.toFixed(1)} %`, 'Line coverage, unit tests', c >= 75 ? 'pass' : c >= 60 ? 'flaky' : 'fail', series('coverageLines')));
    }
    if (m.perf && m.perf.k6) {
      const k = m.perf.k6;
      tiles.append(tile(ms(k.p95Ms), `k6 p95, ${k.profile} profile`, k.thresholdsOk ? 'pass' : 'fail', series('k6P95Ms')));
    }
    if (m.lighthouse) {
      const l = m.lighthouse;
      const low = Math.min(l.performance, l.accessibility, l.bestPractices, l.seo);
      tiles.append(tile(String(l.performance), 'Lighthouse performance', low >= 90 ? 'pass' : low >= 75 ? 'flaky' : 'fail', series('lhPerformance')));
    }
  }

  function renderTiers(m) {
    const t = m.tests;
    const box = $('tiers');
    box.replaceChildren(el('h2', { id: 'tiers-title', text: `Run ${m.run.number} by tier` }));
    const rows = Object.entries(t.tiers || {}).map(([key, v]) => ({
      cells: [
        { text: TIER_LABELS[key] || key },
        { num: true, text: n(v.total) },
        { num: true, text: n(v.passed) },
        { num: true, node: el('span', { class: v.failed ? 'tone-fail' : null, text: n(v.failed) }) },
        { num: true, node: el('span', { class: v.flaky ? 'tone-flaky' : null, text: n(v.flaky) }) },
        { num: true, node: el('span', { class: 'tone-skip', text: n(v.skipped) }) },
        { num: true, text: dur(v.durationSec) },
      ],
    }));
    rows.push({
      total: true,
      cells: [
        { text: 'All tiers' },
        { num: true, text: n(t.total) },
        { num: true, text: n(t.passed) },
        { num: true, text: n(t.failed) },
        { num: true, text: n(t.flaky) },
        { num: true, text: n(t.skipped) },
        { num: true, text: dur(m.run.durationSec) },
      ],
    });
    box.append(table(
      [{ text: 'Tier' }, { text: 'Tests', num: true }, { text: 'Passed', num: true }, { text: 'Failed', num: true }, { text: 'Flaky', num: true }, { text: 'Skipped', num: true }, { text: 'Took', num: true }],
      rows,
      { label: 'Tests by tier' },
    ));
    const notes = [];
    if (typeof t.durationP95Sec === 'number') notes.push(`The slowest 5 % of tests take ${t.durationP95Sec.toFixed(1)} s or more; the mean is ${t.durationMeanSec.toFixed(1)} s.`);
    if (m.kubernetes) {
      const k = m.kubernetes;
      notes.push(`Browser tests ran as ${word(k.shards)} Playwright shards on a kind cluster inside the runner, ${dur(k.wallSec)} wall clock${k.k6Runners ? `, with ${word(k.k6Runners)} k6 runners generating load from inside the cluster` : ''}.`);
    }
    if (notes.length) box.append(el('p', { class: 'note', text: notes.join(' ') }));
  }

  function renderK6(m) {
    const box = $('k6');
    box.replaceChildren();
    if (!m.perf || !m.perf.k6) return;
    const k = m.perf.k6;
    box.append(el('h2', { text: 'k6 journeys' }));
    box.append(el('p', { class: 'lede', text: `${cap(k.profile)} profile: ${n(k.requests)} requests, ${pct(k.failedRate, 2)} failed, thresholds ${k.thresholdsOk ? 'within budget' : 'crossed'}.` }));
    const rows = Object.entries(k.journeys || {}).map(([name, j]) => ({
      cells: [
        { text: name },
        { num: true, text: ms(j.p95Ms) },
        { num: true, text: ms(j.p99Ms) },
        { num: true, node: el('span', { class: j.ok ? 'tone-pass' : 'tone-fail', text: `${j.ok ? 'under' : 'over'} ${ms(j.budgetMs)}` }) },
      ],
    }));
    box.append(table([{ text: 'Journey' }, { text: 'p95', num: true }, { text: 'p99', num: true }, { text: 'Budget', num: true }], rows, { label: 'k6 journeys' }));
  }

  function renderFlaky(m) {
    const box = $('flaky');
    const list = m.flaky || [];
    box.replaceChildren(el('h2', { text: `Flaky and failing tests, last ${word(FLAKY_WINDOW)} runs` }));
    box.append(el('p', { class: 'lede', text: 'A test is listed when it flaked or failed in any of the last ten runs. It leaves the list after ten green runs in a row, never by being removed from the report.' }));
    if (!list.length) {
      box.append(el('p', { class: 'note', text: 'Nothing flaked or failed in the last ten runs.' }));
      return;
    }
    const allure = (run) => `${(m.reports && m.reports.allure ? m.reports.allure.replace(/\d+\/?$/, '') : 'allure/')}${run}/`;
    const rows = list.map((f) => {
      const strip = Array.isArray(f.history)
        ? el('span', { class: 'strip', 'aria-label': f.history.map((s, i) => `run ${f.lastSeen - (f.history.length - 1 - i)} ${s}`).join(', ') },
            f.history.map((s, i) => el('i', { class: s, title: `run ${f.lastSeen - (f.history.length - 1 - i)}: ${s}` })))
        : el('span', { class: 'tone-skip', text: 'no per-run detail' });
      return {
        cells: [
          { node: el('span', {}, el('strong', { text: f.title }), el('span', { class: 'file', text: f.file })) },
          { node: strip },
          { num: true, node: el('span', { class: f.runsFlaky ? 'tone-flaky' : null, text: n(f.runsFlaky || 0) }) },
          { num: true, node: el('span', { class: f.runsFailed ? 'tone-fail' : null, text: n(f.runsFailed || 0) }) },
          { num: true, node: el('a', { href: allure(f.lastSeen), text: `run ${f.lastSeen}` }) },
        ],
      };
    });
    box.append(table([{ text: 'Test' }, { text: 'Last ten runs' }, { text: 'Flaky', num: true }, { text: 'Failed', num: true }, { text: 'Last seen', num: true }], rows, { label: 'Flaky and failing tests' }));
  }

  function renderReports(m, repoUrl) {
    const box = $('reports');
    box.replaceChildren(el('h2', { text: `Reports of run ${m.run.number}` }));
    const r = m.reports || {};
    const items = [];
    if (r.allure) items.push(['Allure report', 'Every tier, with the history of each test across runs', r.allure]);
    if (r.playwright) items.push(['Playwright report', 'The merged browser tiers, traces on failure', r.playwright]);
    if (r.k6) items.push(['k6 summaries', 'One per runner, HTML and JSON', r.k6]);
    if (r.lighthouse) items.push(['Lighthouse report', 'Performance, accessibility, best practices, SEO', r.lighthouse]);
    items.push(['Workflow run on GitHub', `${m.run.workflow}, run ${m.run.number}`, m.run.url]);
    items.push(['quality-metrics.json', 'The data behind this page', `${BASE}/quality-metrics.json`]);
    items.push(['metrics/history.jsonl', 'One line per run, never pruned', `${BASE}/metrics/history.jsonl`]);
    box.append(el('ul', { class: 'reports' }, items.map(([title, sub, href]) => el('li', {}, el('a', { href, text: title }), el('small', { text: sub })))));
    if (r.allure) box.append(el('p', { class: 'note' }, 'The newest Allure report is always at ', el('a', { href: r.allure.replace(/\d+\/?$/, 'latest/'), text: 'allure/latest' }), '.'));
    return repoUrl;
  }

  function renderFoot(m, repoUrl) {
    $('foot').replaceChildren(
      'Every number on this page is measured from the run’s own artifacts, never typed by hand. Schema ',
      String(m.schema),
      ', described in ',
      el('a', { href: `${repoUrl}/blob/main/docs/ci/METRICS.md`, text: 'docs/ci/METRICS.md' }),
      `. Latest run started ${when(m.run.startedAt)}.`,
    );
  }

  function renderEmpty(err) {
    const v = $('verdict');
    v.textContent = 'No run has been published yet.';
    v.classList.add('is-empty');
    $('verdictMeta').textContent = 'The first workflow run writes quality-metrics.json next to this page; until then there is nothing to measure.';
    $('ribbonSection').hidden = true;
    $('tiles').hidden = true;
    $('foot').textContent = `Could not read ${BASE}/quality-metrics.json (${err && err.message ? err.message : err}).`;
  }

  async function main() {
    try {
      const [m, history] = await Promise.all([
        fetch(`${BASE}/quality-metrics.json`, { cache: 'no-cache' }).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))),
        fetch(`${BASE}/metrics/history.jsonl`, { cache: 'no-cache' }).then((r) => (r.ok ? r.text() : '')).then(parseJsonl).catch(() => []),
      ]);
      const repoUrl = renderMast(m);
      renderVerdict(m, history);
      renderRibbon(history, repoUrl);
      renderTiles(m, history);
      renderTiers(m);
      renderK6(m);
      renderFlaky(m);
      renderReports(m, repoUrl);
      renderFoot(m, repoUrl);
    } catch (err) {
      renderEmpty(err);
    }
  }

  main();
})();
