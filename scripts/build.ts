#!/usr/bin/env bun
/**
 * build-site.ts – Parse res/ and tsk/ to generate docs/index.html
 */

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { join, basename } from "path";

const ROOT = join(import.meta.dir, "..");
const RES_DIR = join(ROOT, "res");
const TSK_DIR = join(ROOT, "tsk");
const DOCS_DIR = join(ROOT, "docs");

// ── Parse result files ──────────────────────────────────────────────

type TaskResult = {
  id: string;
  score: number;
  pass: boolean;
  time: number;
  bits?: number;
  ref?: number;
};

type RunResult = {
  filename: string;
  timestamp: string;
  model: string;
  score: number;
  right: number;
  total: number;
  tasks: TaskResult[];
};

function parse_result_file(path: string): RunResult {
  const text = readFileSync(path, "utf-8");
  const lines = text.split("\n");

  const model_line = lines.find(l => l.startsWith("model:"));
  const model = model_line ? model_line.slice("model:".length).trim() : "unknown";

  const right_line = lines.find(l => l.startsWith("right:"));
  const right_match = right_line?.match(/right:\s*(\d+)\/(\d+)/);
  const right = right_match ? parseInt(right_match[1]) : 0;
  const total = right_match ? parseInt(right_match[2]) : 120;

  const score_line = lines.find(l => l.startsWith("score:"));
  const score = score_line ? parseFloat(score_line.slice("score:".length).trim()) : 0;

  const tasks: TaskResult[] = [];
  for (const line of lines) {
    const m = line.match(/^- (\w+):\s+([\d.]+)\s+(pass|fail)\s+time=([\d.]+)s(?:\s+bits=(\d+))?(?:\s+ref=(\d+))?/);
    if (m) {
      tasks.push({
        id: m[1],
        score: parseFloat(m[2]),
        pass: m[3] === "pass",
        time: parseFloat(m[4]),
        bits: m[5] ? parseInt(m[5]) : undefined,
        ref: m[6] ? parseInt(m[6]) : undefined,
      });
    }
  }

  const fn = basename(path);
  // filename: 2026y04m23d.20h11m53s.openai_gpt-5.5.txt
  const ts_match = fn.match(/^(\d{4}y\d{2}m\d{2}d\.\d{2}h\d{2}m\d{2}s)/);
  const timestamp = ts_match ? ts_match[1] : fn;

  return { filename: fn, timestamp, model, score, right, total, tasks };
}

function load_all_results(): RunResult[] {
  const files = readdirSync(RES_DIR)
    .filter(f => f.endsWith(".txt"))
    .sort();
  return files.map(f => parse_result_file(join(RES_DIR, f)));
}

// Keep only the latest run per model (by filename sort = chronological)
function latest_per_model(runs: RunResult[]): RunResult[] {
  const map = new Map<string, RunResult>();
  // files are sorted chronologically, so last wins
  for (const run of runs) {
    map.set(run.model, run);
  }
  // Drop runs with zero passes, then sort by right desc, then by model name
  return [...map.values()]
    .filter(r => r.right > 0)
    .sort((a, b) => b.right - a.right || a.model.localeCompare(b.model));
}

// ── Parse task files ────────────────────────────────────────────────

type Task = {
  id: string;
  category: string;
  description: string;
  tests: { input: string; expected: string }[];
};

const CATEGORY_NAMES: Record<string, string> = {
  algo: "Algorithms",
  cnat: "Church Naturals",
  cbin: "Church Binaries",
  clst: "Church Lists",
  ctre: "Church Trees",
  cadt: "Church ADTs",
  snat: "Scott Naturals",
  sbin: "Scott Binaries",
  slst: "Scott Lists",
  stre: "Scott Trees",
  sadt: "Scott ADTs",
  ntup: "N-Tuples",
};

function parse_task_file(path: string): Task {
  const id = basename(path, ".tsk");
  const text = readFileSync(path, "utf-8").trim();
  const category = id.split("_")[0];

  const separator = text.indexOf("\n---\n");
  let description: string;
  let test_section: string;

  if (separator >= 0) {
    description = text.slice(0, separator).trim();
    test_section = text.slice(separator + 5).trim();
  } else {
    description = text;
    test_section = "";
  }

  const tests: { input: string; expected: string }[] = [];
  const test_lines = test_section.split("\n");
  for (let i = 0; i < test_lines.length; i++) {
    const line = test_lines[i].trim();
    if (line.startsWith("@main") || (line.startsWith("λ") && line.includes("@main"))) {
      const next = test_lines[i + 1]?.trim();
      if (next && next.startsWith("=")) {
        tests.push({ input: line, expected: next.slice(1).trim() });
        i++;
      }
    }
  }

  return { id, category, description, tests };
}

function load_all_tasks(): Task[] {
  const files = readdirSync(TSK_DIR)
    .filter(f => f.endsWith(".tsk"))
    .sort();
  return files.map(f => parse_task_file(join(TSK_DIR, f)));
}

// ── Build per-task model results map ────────────────────────────────

function build_task_model_map(rankings: RunResult[]): Record<string, Record<string, boolean>> {
  const map: Record<string, Record<string, boolean>> = {};
  for (const run of rankings) {
    for (const t of run.tasks) {
      if (!map[t.id]) map[t.id] = {};
      map[t.id][run.model] = t.pass;
    }
  }
  return map;
}

// ── Generate HTML ───────────────────────────────────────────────────

function generate_html(rankings: RunResult[], tasks: Task[], taskModelMap: Record<string, Record<string, boolean>>): string {
  const data = JSON.stringify({
    rankings: rankings.map(r => {
      const passing = r.tasks.filter(t => t.pass);
      const avgTime = passing.length
        ? passing.reduce((s, t) => s + t.time, 0) / passing.length
        : 0;
      return {
        model: r.model,
        right: r.right,
        total: r.total,
        pct: ((r.right / r.total) * 100).toFixed(1),
        avgTime: Number(avgTime.toFixed(1)),
        timestamp: r.timestamp,
        tasks: Object.fromEntries(r.tasks.map(t => [t.id, t.pass])),
        taskBits: Object.fromEntries(
          r.tasks
            .filter(t => t.pass && t.bits !== undefined)
            .map(t => [t.id, t.bits])
        ),
        taskRefs: Object.fromEntries(
          r.tasks
            .filter(t => t.ref !== undefined)
            .map(t => [t.id, t.ref])
        ),
      };
    }),
    tasks: tasks.map(t => ({
      id: t.id,
      category: t.category,
      categoryName: CATEGORY_NAMES[t.category] || t.category,
      description: t.description,
      testCount: t.tests.length,
      tests: t.tests.slice(0, 3),
    })),
    categories: Object.entries(CATEGORY_NAMES).map(([k, v]) => ({ id: k, name: v })),
  });

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LamBench</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }

:root {
  --base03:  #002b36;
  --base02:  #073642;
  --base01:  #586e75;
  --base00:  #657b83;
  --base0:   #839496;
  --base1:   #93a1a1;
  --base2:   #eee8d5;
  --base3:   #fdf6e3;
  --yellow:  #b58900;
  --orange:  #cb4b16;
  --red:     #dc322f;
  --magenta: #d33682;
  --violet:  #6c71c4;
  --blue:    #268bd2;
  --cyan:    #2aa198;
  --green:   #859900;
}

body {
  font-family: 'SF Mono', 'Cascadia Code', 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
  font-size: 14px;
  line-height: 1.5;
  background: var(--base3);
  color: var(--base00);
  min-height: 100vh;
}

/* ── vim chrome ──────────────────────────────────────── */

.vim {
  max-width: 820px;
  margin: 0 auto;
  border-left: 1px solid var(--base2);
  border-right: 1px solid var(--base2);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.buf {
  flex: 1;
  padding: 0;
}

.line {
  display: flex;
  min-height: 1.5em;
}

.ln {
  width: 4ch;
  text-align: right;
  padding-right: 1ch;
  color: var(--base1);
  user-select: none;
  flex-shrink: 0;
}

.lc {
  flex: 1;
  padding-right: 1ch;
  white-space: pre-wrap;
  word-break: break-word;
}

.tilde .ln { color: var(--blue); }
.tilde .lc { color: var(--blue); }

/* ── statusline ──────────────────────────────────────── */

.statusline {
  background: var(--base2);
  color: var(--base01);
  padding: 0 1ch;
  display: flex;
  justify-content: space-between;
  line-height: 1.8;
  border-top: 1px solid var(--base1);
  position: sticky;
  bottom: 0;
}

.statusline a {
  color: var(--blue);
  text-decoration: none;
}
.statusline a:hover { text-decoration: underline; }

/* ── tabline ─────────────────────────────────────────── */

.tabline {
  background: var(--base2);
  display: flex;
  border-bottom: 1px solid var(--base1);
  position: sticky;
  top: 0;
  z-index: 10;
}

.tabline button {
  font-family: inherit;
  font-size: inherit;
  background: var(--base2);
  color: var(--base01);
  border: none;
  padding: 0 2ch;
  line-height: 1.8;
  cursor: pointer;
}

.tabline button:hover { background: var(--base3); }
.tabline button.active {
  background: var(--base3);
  color: var(--base00);
  font-weight: bold;
}

.panel { display: none; }
.panel.active { display: block; }

/* ── colors ──────────────────────────────────────────── */

.h1      { color: var(--yellow); font-weight: bold; }
.h2      { color: var(--orange); font-weight: bold; }
.comment { color: var(--base1); }
.keyword { color: var(--green); }
.string  { color: var(--cyan); }
.type    { color: var(--violet); }
.func    { color: var(--blue); }
.err     { color: var(--red); }
.ok      { color: var(--green); }
.num     { color: var(--magenta); }
.bold    { font-weight: bold; }
.dim     { color: var(--base1); }

/* ── bar chart ───────────────────────────────────────── */

.bar {
  display: inline-block;
  height: 1em;
  vertical-align: middle;
  border-radius: 1px;
}

.bar-green  { background: var(--green); }
.bar-blue   { background: var(--blue); }
.bar-yellow { background: var(--yellow); }
.bar-red    { background: var(--red); }
.bar-bg     { background: var(--base2); display: inline-block; height: 1em; vertical-align: middle; border-radius: 1px; }

/* ── filter row ──────────────────────────────────────── */

.filter-row { padding: 0; }

.fbtn {
  font-family: inherit;
  font-size: inherit;
  background: none;
  border: none;
  color: var(--base1);
  cursor: pointer;
  padding: 0;
  margin-right: 1ch;
}
.fbtn:hover { color: var(--base00); }
.fbtn.active { color: var(--yellow); font-weight: bold; }

/* ── task row hover ──────────────────────────────────── */

.task-line { cursor: pointer; }
.task-line:hover .lc { background: var(--base2); }
.task-line .lc {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ── matrix table ──────────────────────────────────── */

.matrix-wrap {
  overflow-x: auto;
  padding: 0.5em 0;
}

table.matrix {
  border-collapse: collapse;
  font-family: inherit;
  font-size: 12px;
  margin: 0 auto;
}

table.matrix th, table.matrix td {
  padding: 2px 6px;
  border: 1px solid var(--base2);
  text-align: center;
  white-space: nowrap;
}

table.matrix thead th {
  position: sticky;
  top: 0;
  background: var(--base3);
  border-bottom: 1px solid var(--base1);
  color: var(--base01);
  font-weight: normal;
}

table.matrix th.mcol {
  height: 8em;
  vertical-align: bottom;
  padding: 4px 2px;
  min-width: 1.8em;
  max-width: 1.8em;
  width: 1.8em;
}

table.matrix th.mcol > div {
  writing-mode: vertical-rl;
  transform: rotate(180deg);
  white-space: nowrap;
  color: var(--blue);
  font-family: inherit;
}

table.matrix td.pid {
  text-align: left;
  color: var(--keyword, var(--green));
  font-weight: bold;
  padding-right: 10px;
}

table.matrix .ok  { color: var(--green); }
table.matrix .err { color: var(--red); }

table.matrix tbody tr:hover td { background: var(--base2); }

/* ── modal ────────────────────────────────────────────── */

.modal-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,43,54,0.35);
  z-index: 100;
  justify-content: center;
  align-items: flex-start;
  padding: 4em 1em;
  overflow-y: auto;
}
.modal-overlay.open { display: flex; }

.modal {
  background: var(--base3);
  border: 1px solid var(--base1);
  max-width: 760px;
  width: 100%;
  padding: 0;
  font-size: 13px;
}

.modal-tabline {
  background: var(--base2);
  border-bottom: 1px solid var(--base1);
  display: flex;
  justify-content: space-between;
  padding: 0 1ch;
  line-height: 1.8;
}

.modal-tabline .title { color: var(--base00); font-weight: bold; }
.modal-close {
  font-family: inherit;
  background: none;
  border: none;
  color: var(--base01);
  cursor: pointer;
  font-size: inherit;
}
.modal-close:hover { color: var(--red); }

.modal-buf {
  padding: 0;
  max-height: 65vh;
  overflow-y: auto;
}

.modal-status {
  background: var(--base2);
  border-top: 1px solid var(--base1);
  padding: 0 1ch;
  line-height: 1.8;
  color: var(--base01);
}

@media (max-width: 640px) {
  body { font-size: 12px; }
  .ln { width: 3ch; }
}
</style>
</head>
<body>
<div class="vim">

<div class="tabline">
  <button class="active" data-panel="intelligence">:intelligence</button>
  <button data-panel="speed">:speed</button>
  <button data-panel="elegance">:elegance</button>
  <button data-panel="problems">:problems</button>
  <button data-panel="matrix">:matrix</button>
</div>

<div class="buf">

<!-- ═══ INTELLIGENCE PANEL ═══ -->
<div id="intelligence" class="panel active">
  <div id="intelligence-buf"></div>
</div>

<!-- ═══ SPEED PANEL ═══ -->
<div id="speed" class="panel">
  <div id="speed-buf"></div>
</div>

<!-- ═══ ELEGANCE PANEL ═══ -->
<div id="elegance" class="panel">
  <div id="elegance-buf"></div>
</div>

<!-- ═══ PROBLEMS PANEL ═══ -->
<div id="problems" class="panel">
  <div id="filter-buf"></div>
  <div id="tasks-buf"></div>
</div>

<!-- ═══ MATRIX PANEL ═══ -->
<div id="matrix" class="panel">
  <div id="matrix-buf"></div>
</div>

</div><!-- /buf -->

<div class="statusline">
  <span><a href="https://github.com/VictorTaelin/LamBench">github.com/VictorTaelin/LamBench</a></span>
  <span id="status-right">v1</span>
</div>

</div><!-- /vim -->

<!-- Modal -->
<div class="modal-overlay" id="modal-overlay">
  <div class="modal" id="modal"></div>
</div>

<script>
const D = ${data};

// ── helpers ─────────────────────────────────────────────────────────

function esc(s) {
  var d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function L(n, c, cls) {
  return '<div class="line'+(cls?' '+cls:'')+'">' +
    '<span class="ln">'+(n===''?'':n)+'</span>' +
    '<span class="lc">'+c+'</span></div>';
}

function tilde() { return L('~', '', 'tilde'); }

function fmtModel(m) {
  // Strip provider prefix, keep the rest as-is.
  var p = m.split('/');
  return p.slice(1).join('/');
}

function barHtml(pct, w) {
  var filled = Math.round(pct/100*w);
  var empty = w - filled;
  var cls = pct>=70?'bar-green':pct>=45?'bar-blue':pct>=20?'bar-yellow':'bar-red';
  var s = '';
  for(var i=0;i<filled;i++) s+='█';
  for(var i=0;i<empty;i++) s+='░';
  return '<span class="'+cls+'">'+s.slice(0,filled)+'</span><span class="dim">'+s.slice(filled)+'</span>';
}

function pad(s,n) { while(s.length<n) s+=' '; return s; }
function rpad(s,n) { while(s.length<n) s=' '+s; return s; }

// ── intelligence + speed ────────────────────────────────────────────

function maxNameLen() {
  var m = 0;
  D.rankings.forEach(function(r) {
    var name = fmtModel(r.model);
    if (name.length > m) m = name.length;
  });
  return m;
}

function renderHeader(lines, n) {
  lines.push(L(n++, ''));
  lines.push(L(n++, '<span class="h1">LamBench</span>  <span class="comment">-- Lambda Calculus Benchmark for AI</span>'));
  lines.push(L(n++, ''));
  return n;
}

function renderIntelligence() {
  var el = document.getElementById('intelligence-buf');
  var lines = [];
  var n = 1;
  n = renderHeader(lines, n);

  lines.push(L(n++, '<span class="h2">Intelligence</span>  <span class="comment">-- by problems solved</span>'));
  lines.push(L(n++, ''));

  var maxName = maxNameLen();
  var bw = 28;
  var sorted = D.rankings.slice().sort(function(a, b){ return b.right - a.right; });

  sorted.forEach(function(r) {
    var name = fmtModel(r.model);
    var pct = parseFloat(r.pct);
    var score = r.right + '/' + r.total;
    var pctStr = rpad(pct.toFixed(1) + '%', 6);
    var line =
      '<span class="func">' + esc(pad(name, maxName + 1)) + '</span>' +
      barHtml(pct, bw) + ' ' +
      '<span class="num">' + esc(rpad(score, 7)) + '</span> ' +
      '<span class="dim">' + esc(pctStr) + '</span>';
    lines.push(L(n++, line));
  });

  lines.push(L(n++, ''));
  for (var i=0; i<8; i++) lines.push(tilde());

  el.innerHTML = lines.join('');
  document.getElementById('status-right').textContent =
    D.rankings.length + ' models · 120 tasks';
}

// ── elegance ─────────────────────────────────────────────────────────
//
// Per-task "shorter":  s = (1 - bits/ref) * 100      (higher = shorter)
//   s =   0%  -> matches reference
//   s = +16%  -> 16% shorter than reference
//   s = -30%  -> 30% longer than reference
//
// Model elegance = mean of per-task shorter% over tasks the model solved.
// Failed/missing tasks are excluded, so this is independent of intelligence.

function computeElegance() {
  var refs = {};
  D.rankings.forEach(function(r) {
    Object.keys(r.taskRefs || {}).forEach(function(tid) {
      if (refs[tid] == null) refs[tid] = r.taskRefs[tid];
    });
  });

  var entries = D.rankings.map(function(r) {
    var sum = 0;
    var passing = 0;
    D.tasks.forEach(function(t) {
      var ref = refs[t.id];
      var bits = (r.taskBits || {})[t.id];
      if (r.tasks[t.id] && ref && bits > 0) {
        sum += (1 - bits / ref);   // positive = shorter, negative = longer
        passing += 1;
      }
    });
    return {
      model: r.model,
      shorter: passing ? (sum / passing) * 100 : 0,  // as percentage
      passing: passing,
    };
  });

  var scored = entries.filter(function(e) { return e.passing > 0; });
  var mean = scored.reduce(function(s, e){ return s + e.shorter; }, 0)
    / Math.max(scored.length, 1);
  entries.forEach(function(e) {
    e.delta = e.passing > 0 ? e.shorter - mean : 0;
  });
  return { entries: entries, mean: mean };
}

function renderElegance() {
  var el = document.getElementById('elegance-buf');
  var lines = [];
  var n = 1;
  n = renderHeader(lines, n);

  lines.push(L(n++, '<span class="h2">Elegance</span>  <span class="comment">-- how much shorter than reference (higher = more elegant)</span>'));
  lines.push(L(n++, ''));

  var t = computeElegance();
  var sorted = t.entries.slice().sort(function(a, b){
    if (a.passing === 0 && b.passing === 0) return 0;
    if (a.passing === 0) return 1;
    if (b.passing === 0) return -1;
    return b.shorter - a.shorter;  // highest first
  });

  var maxName = maxNameLen();
  var bw = 28;

  // Bar: -40% => empty, +30% => full (clamped).
  var barLo = -40, barHi = 30;

  function fmtShorter(v) {
    if (v >= 0) return '+' + v.toFixed(1) + '%';
    return '−' + Math.abs(v).toFixed(1) + '%';
  }

  sorted.forEach(function(e) {
    var name = fmtModel(e.model);
    var valStr = e.passing
      ? rpad(fmtShorter(e.shorter), 7)
      : rpad('—', 7);
    var barPct = e.passing
      ? Math.max(0, Math.min(100, (e.shorter - barLo) * 100 / (barHi - barLo)))
      : 0;
    var deltaCls = e.passing
      ? (e.delta >= 0 ? 'ok' : 'err')
      : 'dim';
    var deltaStr = e.passing
      ? fmtShorter(e.delta)
      : '  —';
    var line =
      '<span class="func">' + esc(pad(name, maxName + 1)) + '</span>' +
      barHtml(barPct, bw) + ' ' +
      '<span class="num">' + esc(valStr) + '</span> ' +
      '<span class="dim">(' + rpad(String(e.passing), 3) + '/' + D.tasks.length + ')</span> ' +
      '<span class="' + deltaCls + '">Δ ' + esc(rpad(deltaStr, 7)) + '</span>';
    lines.push(L(n++, line));
  });

  lines.push(L(n++, ''));
  lines.push(L(n++,
    '<span class="comment">" mean: </span>' +
    '<span class="num">' + fmtShorter(t.mean) + '</span>' +
    '<span class="comment">  shorter than reference · Δ = distance from mean</span>'
  ));

  for (var i=0; i<8; i++) lines.push(tilde());

  el.innerHTML = lines.join('');
  document.getElementById('status-right').textContent =
    sorted.length + ' models · mean ' + fmtShorter(t.mean) + ' shorter';
}

function renderSpeed() {
  var el = document.getElementById('speed-buf');
  var lines = [];
  var n = 1;
  n = renderHeader(lines, n);

  lines.push(L(n++, '<span class="h2">Speed</span>  <span class="comment">-- tasks solved per minute (higher = faster)</span>'));
  lines.push(L(n++, ''));

  var valid = D.rankings.filter(function(r){ return r.avgTime > 0; });
  // Compute tasks/min for each model.
  var entries = valid.map(function(r) {
    return { model: r.model, tpm: 60 / r.avgTime, avgTime: r.avgTime };
  });
  var sorted = entries.slice().sort(function(a, b){ return b.tpm - a.tpm; });
  var maxTPM = sorted.length ? sorted[0].tpm : 1;

  var maxName = maxNameLen();
  var bw = 28;

  sorted.forEach(function(e) {
    var name = fmtModel(e.model);
    var pct = (e.tpm / maxTPM) * 100;
    var line =
      '<span class="func">' + esc(pad(name, maxName + 1)) + '</span>' +
      barHtml(pct, bw) + ' ' +
      '<span class="num">' + esc(rpad(e.tpm.toFixed(2) + '/min', 10)) + '</span> ' +
      '<span class="dim">(' + esc(e.avgTime.toFixed(0) + 's avg') + ')</span>';
    lines.push(L(n++, line));
  });

  lines.push(L(n++, ''));
  lines.push(L(n++, '<span class="comment">" Wall-clock time per passing task (model + interpreter)</span>'));

  for (var i=0; i<8; i++) lines.push(tilde());

  el.innerHTML = lines.join('');
  document.getElementById('status-right').textContent =
    sorted.length + ' models · tasks/min';
}

// ── problems ────────────────────────────────────────────────────────

var activeFilter = 'all';

function renderProblems() {
  renderFilters();
  renderTasks();
}

function renderFilters() {
  var el = document.getElementById('filter-buf');
  var cats = [{id:'all',name:'all'}].concat(D.categories.map(function(c){return {id:c.id,name:c.id};} ));
  var btns = cats.map(function(c) {
    return '<button class="fbtn'+(activeFilter===c.id?' active':'')+'" data-cat="'+c.id+'">'+esc(c.name)+'</button>';
  }).join('');
  var lines = [];
  lines.push(L('', ''));
  lines.push(L('', '<span class="comment">" filter:</span> ' + btns));
  lines.push(L('', ''));
  el.innerHTML = lines.join('');

  el.querySelectorAll('.fbtn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      activeFilter = btn.dataset.cat;
      renderProblems();
    });
  });
}

function renderTasks() {
  var el = document.getElementById('tasks-buf');
  var tasks = activeFilter==='all' ? D.tasks : D.tasks.filter(function(t){return t.category===activeFilter;});
  var lines = [];
  var n = 1;

  tasks.forEach(function(t) {
    var dots = D.rankings.map(function(r) {
      return r.tasks[t.id] ? '<span class="ok">●</span>' : '<span class="err">●</span>';
    }).join('');

    var desc = t.description.split('\\n')[0];
    if (desc.length > 60) desc = desc.slice(0, 57) + '...';

    var line =
      '<span class="keyword">' + esc(pad(t.id, 10)) + '</span>' +
      '  ' + dots +
      '  <span class="comment">' + esc(desc) + '</span>';

    lines.push('<div class="task-line" data-task="'+t.id+'">' + L(rpad(String(n),3), line).slice(0,-6) + '</div></div>');
    n++;
  });

  for (var i=0; i<5; i++) lines.push(tilde());

  el.innerHTML = lines.join('');

  el.querySelectorAll('.task-line').forEach(function(row) {
    row.addEventListener('click', function() { openTask(row.dataset.task); });
  });

  document.getElementById('status-right').textContent =
    tasks.length + '/' + D.tasks.length + ' tasks';
}

// ── matrix ─────────────────────────────────────────────────────────

function renderMatrix() {
  var el = document.getElementById('matrix-buf');
  var sorted = D.rankings.slice().sort(function(a, b){ return b.right - a.right; });

  var head = '<tr><th>problem</th>';
  sorted.forEach(function(r) {
    head += '<th class="mcol"><div>' + esc(fmtModel(r.model)) + '</div></th>';
  });
  head += '</tr>';

  var rows = D.tasks.map(function(t) {
    var cells = '<td class="pid">' + esc(t.id) + '</td>';
    sorted.forEach(function(r) {
      var passed = r.tasks[t.id];
      cells += '<td class="' + (passed ? 'ok' : 'err') + '">' +
        (passed ? '✓' : '✗') + '</td>';
    });
    return '<tr data-task="' + t.id + '">' + cells + '</tr>';
  }).join('');

  el.innerHTML =
    '<div class="matrix-wrap">' +
    '<table class="matrix"><thead>' + head + '</thead>' +
    '<tbody>' + rows + '</tbody></table></div>';

  el.querySelectorAll('tbody tr').forEach(function(row) {
    row.style.cursor = 'pointer';
    row.addEventListener('click', function() { openTask(row.dataset.task); });
  });

  document.getElementById('status-right').textContent =
    D.tasks.length + ' tasks × ' + sorted.length + ' models';
}

// ── modal ───────────────────────────────────────────────────────────

function openTask(id) {
  var task = D.tasks.find(function(t){return t.id===id;});
  if (!task) return;

  var lines = [];
  var n = 1;

  lines.push(L(n++, ''));
  lines.push(L(n++, '<span class="h2">' + esc(task.id) + '</span>  <span class="dim">' + esc(task.categoryName) + '</span>'));
  lines.push(L(n++, ''));

  task.description.split('\\n').forEach(function(ln) {
    lines.push(L(n++, '<span class="comment">" ' + esc(ln) + '</span>'));
  });

  lines.push(L(n++, ''));
  lines.push(L(n++, '<span class="type">Tests</span> <span class="dim">(' + task.tests.length + ' of ' + task.testCount + ')</span>'));
  lines.push(L(n++, ''));

  task.tests.forEach(function(t, i) {
    lines.push(L(n++, '<span class="string">' + esc(t.input) + '</span>'));
    lines.push(L(n++, '<span class="ok">= ' + esc(t.expected) + '</span>'));
    lines.push(L(n++, ''));
  });

  lines.push(L(n++, '<span class="type">Model Results</span>'));
  lines.push(L(n++, ''));

  D.rankings.forEach(function(r) {
    var passed = r.tasks[id];
    var icon = passed ? '<span class="ok">✓</span>' : '<span class="err">✗</span>';
    lines.push(L(n++, icon + ' ' + esc(fmtModel(r.model))));
  });

  lines.push(L(n++, ''));
  for (var i=0; i<4; i++) lines.push(tilde());

  document.getElementById('modal').innerHTML =
    '<div class="modal-tabline"><span class="title">' + esc(task.id) + '.tsk</span>' +
    '<button class="modal-close" id="modal-close">:q</button></div>' +
    '<div class="modal-buf">' + lines.join('') + '</div>' +
    '<div class="modal-status">' + esc(task.id) + ' [RO] ' + n + 'L</div>';

  document.getElementById('modal-overlay').classList.add('open');
  document.getElementById('modal-close').addEventListener('click', closeModal);
}

function closeModal() {
  document.getElementById('modal-overlay').classList.remove('open');
}

// ── tabs ────────────────────────────────────────────────────────────

var PANEL_RENDERERS = {
  intelligence: renderIntelligence,
  speed:        renderSpeed,
  elegance:      renderElegance,
  problems:     renderProblems,
  matrix:       renderMatrix,
};

document.querySelectorAll('.tabline button').forEach(function(tab) {
  tab.addEventListener('click', function() {
    document.querySelectorAll('.tabline button').forEach(function(t){t.classList.remove('active');});
    document.querySelectorAll('.panel').forEach(function(p){p.classList.remove('active');});
    tab.classList.add('active');
    var p = document.getElementById(tab.dataset.panel);
    p.classList.add('active');
    var fn = PANEL_RENDERERS[tab.dataset.panel];
    if (fn) fn();
  });
});

document.getElementById('modal-overlay').addEventListener('click', function(e) {
  if (e.target === e.currentTarget) closeModal();
});

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeModal();
});

// ── init ────────────────────────────────────────────────────────────

renderIntelligence();
</script>
</body>
</html>`;
}

// ── Main ────────────────────────────────────────────────────────────

function main() {
  const runs = load_all_results();
  const rankings = latest_per_model(runs);
  const tasks = load_all_tasks();
  const taskModelMap = build_task_model_map(rankings);

  mkdirSync(DOCS_DIR, { recursive: true });
  const html = generate_html(rankings, tasks, taskModelMap);
  writeFileSync(join(DOCS_DIR, "index.html"), html);

  console.log("Generated docs/index.html");
  console.log(`  ${rankings.length} models (latest runs)`);
  console.log(`  ${tasks.length} tasks`);
  for (const r of rankings) {
    console.log(`  ${r.model}: ${r.right}/${r.total} (${((r.right / r.total) * 100).toFixed(1)}%)`);
  }
}

main();
