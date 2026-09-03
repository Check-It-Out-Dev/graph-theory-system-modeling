// CodeMap demo movie — Playwright recorder. Five scenes, each a separate context so
// each lands in its own .webm. Every UI frame is a REAL run against the live server
// on :7345 (local 4B model + consent-gated API tier). Node + playwright-core from the
// fe-greenfield node_modules; browsers from the standard ms-playwright cache.
// Usage: node movie/record.js   (server must be up: python codemap.py up --no-browser)

const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/node_modules/playwright-core');

const HERE = __dirname;
const OUT = path.join(HERE, 'out');
const SIZE = { width: 1920, height: 1080 };
const APP = 'http://localhost:7345';
const Q_LOCAL = 'what breaks if I change UserRepository.java?';
const Q_NATIVE = 'how many files import ResourceNotFoundException, and which subsystem do most of them live in?';
const Q_ESC = 'why was HMAC chosen over JWT for the consent cookies?';

const CHROME = [
  'C:/Users/Norbert/AppData/Local/ms-playwright/chromium-1217/chrome-win64/chrome.exe',
  'C:/Users/Norbert/AppData/Local/ms-playwright/chromium-1217/chrome-win/chrome.exe',
].find(fs.existsSync);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const marks = {};

async function scene(browser, name, fn) {
  const ctx = await browser.newContext({
    viewport: SIZE, deviceScaleFactor: 1,
    recordVideo: { dir: OUT, size: SIZE },
  });
  const page = await ctx.newPage();
  const t0 = Date.now();
  const mark = (k) => { (marks[name] = marks[name] || {})[k] = Date.now() - t0; };
  await fn(page, mark);
  const video = page.video();
  await ctx.close();
  const p = await video.path();
  const dst = path.join(OUT, `${name}.webm`);
  if (fs.existsSync(dst)) fs.unlinkSync(dst);
  fs.renameSync(p, dst);
  console.log(`${name}: ${path.basename(dst)} ${(fs.statSync(dst).size / 1e6).toFixed(1)} MB`);
}

async function caption(page, html) {
  await page.evaluate((h) => {
    let c = document.getElementById('__cap');
    if (!c) {
      c = document.createElement('div');
      c.id = '__cap';
      c.style.cssText = 'position:fixed;left:0;right:0;bottom:158px;text-align:center;' +
        'font:600 1.55rem "Segoe UI",system-ui,sans-serif;color:#d7dde4;z-index:9999;' +
        'text-shadow:0 2px 12px #000c;transition:opacity .45s;opacity:0;pointer-events:none';
      document.body.appendChild(c);
    }
    c.style.opacity = 0;
    setTimeout(() => { c.innerHTML = h; c.style.opacity = 1; }, 460);
  }, html);
}

const ACCENT = (t) => `<span style="color:#4fd6c0">${t}</span>`;

const ONLY = process.argv.slice(2);          // e.g. `node record.js C_local D_escalate`
const wants = (name) => !ONLY.length || ONLY.includes(name);

(async () => {
  if (!CHROME) throw new Error('chromium not found in ms-playwright cache');
  const browser = await chromium.launch({ executablePath: CHROME });

  // A — title card (self-timed HTML)
  if (wants('A_title')) await scene(browser, 'A_title', async (page) => {
    await page.goto('file:///' + path.join(HERE, 'scene_title.html').replace(/\\/g, '/'));
    await page.waitForFunction('window.__done === true', null, { timeout: 20000 });
  });

  // B — terminal replay of the real wizard transcript
  if (wants("B_terminal")) await scene(browser, "B_terminal", async (page) => {
    await page.goto('file:///' + path.join(OUT, 'scene_terminal.html').replace(/\\/g, '/'));
    await page.waitForFunction('window.__done === true', null, { timeout: 60000 });
  });

  // C — browser UI, live local model answering
  if (wants("C_local")) await scene(browser, "C_local", async (page, mark) => {
    await page.goto(APP);
    await page.waitForSelector('#chips .chip.on', { timeout: 15000 });
    await caption(page, `Ask about a ${ACCENT('430k-LOC production codebase')} — offline, on a plain CPU`);
    await sleep(2100);
    await page.click('#q');
    await page.locator('#q').pressSequentially(Q_LOCAL, { delay: 36 });
    await sleep(650);
    mark('ask');
    await page.click('#go');
    await page.waitForSelector('.card .tag.model', { timeout: 120000 });
    mark('answer');
    await caption(page, `The ${ACCENT('4B local navigator')} walks the graph — every hop is an inspectable DSL step`);
    await sleep(8200);
  });

  // F — the 80B tier: a big LOCAL model speaks the graph's native Cypher
  if (wants("F_native")) await scene(browser, "F_native", async (page, mark) => {
    await page.goto(APP);
    await page.waitForSelector('#chips .chip.big', { timeout: 15000 });
    await page.waitForSelector('#nav option[value="big"]', { state: 'attached', timeout: 15000 });
    await caption(page, `Counting questions want a ${ACCENT('real query language')} — switch to the 80B tier`);
    await sleep(1900);
    await page.selectOption('#nav', 'big');
    await sleep(700);
    await page.click('#q');
    await page.locator('#q').pressSequentially(Q_NATIVE, { delay: 30 });
    await sleep(600);
    mark('ask');
    await page.click('#go');
    await caption(page, `An untrained ${ACCENT('80B MoE')} — still local, still plain CPU — walks the same graph`);
    await page.waitForSelector('.tag.big', { timeout: 300000 });
    mark('answer');
    await page.evaluate(() => { const s = document.getElementById('stream'); s.scrollTop = s.scrollHeight; });
    await caption(page, `It queries the graph in ${ACCENT('native Cypher')} — typed edges, curated subsystems, grounded counts`);
    await sleep(9000);
    mark('end');
  });

  // D — honest abstention -> consent card -> Claude API navigates the same graph
  if (wants("D_escalate")) await scene(browser, "D_escalate", async (page, mark) => {
    await page.goto(APP);
    await page.waitForSelector('#chips .chip.on', { timeout: 15000 });
    await caption(page, `Some answers live in ${ACCENT('file content')}, not the graph…`);
    await sleep(1400);
    await page.click('#q');
    await page.locator('#q').pressSequentially(Q_ESC, { delay: 36 });
    await sleep(550);
    await page.click('#go');
    await page.waitForSelector('.offer', { timeout: 120000 });
    mark('offer');
    await caption(page, `…so the local model ${ACCENT('says so honestly')} — and asks for YOUR consent`);
    await sleep(4200);
    mark('click');
    await page.click('.offer button.use');
    await page.waitForSelector('.tag.api', { timeout: 240000 });
    mark('api');
    await page.evaluate(() => { const s = document.getElementById('stream'); s.scrollTop = s.scrollHeight; });
    await caption(page, `${ACCENT('Claude Sonnet')} navigates the same graph by tool calls — you saw every step it took`);
    await sleep(9500);
    mark('end');
  });

  // E — closing card
  if (wants("E_closing")) await scene(browser, "E_closing", async (page) => {
    await page.goto('file:///' + path.join(HERE, 'scene_closing.html').replace(/\\/g, '/'));
    await page.waitForFunction('window.__done === true', null, { timeout: 20000 });
  });

  await browser.close();
  fs.writeFileSync(path.join(OUT, 'marks.json'), JSON.stringify(marks, null, 1));
  console.log('marks:', JSON.stringify(marks));
})();
