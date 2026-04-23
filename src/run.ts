import { spawnSync } from "child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "fs";
import { basename, join } from "path";
import type { Task, Result } from "./types";
import { parse_task, load_tasks } from "./parse";

var TMP = join(import.meta.dir, "..", ".tmp");
var TMP_ID = 0;

type RunTaskOptions = {
  deadline_ms?: number;
};

function tmp_file(name: string): string {
  TMP_ID += 1;
  return join(TMP, `${process.pid}-${TMP_ID}-${name}.lam`);
}

export function lam_run(src: string, timeout = 10_000): string {
  mkdirSync(TMP, { recursive: true });
  var file = tmp_file("run");
  writeFileSync(file, src);
  return run_lam([file], timeout).trim();
}

function normalize(term: string, timeout = 10_000): string {
  return lam_run("@main = " + term, timeout);
}

export function bin_size(src: string, timeout = 10_000): number {
  mkdirSync(TMP, { recursive: true });
  var file = tmp_file("size");
  writeFileSync(file, src);
  return run_lam([file, "--to-bin"], timeout).trim().length;
}

function run_lam(args: string[], timeout = 10_000): string {
  var res = spawnSync("lam", args, {
    encoding: "utf-8",
    timeout,
    stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      NO_COLOR: "1",
      FORCE_COLOR: "0",
      TERM: "dumb",
    },
  });

  if (res.error) {
    throw new Error(clean_lam_error(res.error.message));
  }
  if (res.status !== 0) {
    var msg = res.stderr || res.stdout || `lam exited with ${res.status}`;
    throw new Error(clean_lam_error(msg));
  }

  return res.stdout;
}

function clean_lam_error(msg: string): string {
  var msg = msg.replace(/\x1b\[[0-9;?]*[ -/]*[@-~]/g, "");
  var lines = msg.split("\n").map(line => line.trim()).filter(Boolean);
  var hit =
    lines.find(line =>
      /^(RangeError|SyntaxError|TypeError|Error):/.test(line)
    ) ??
    lines.find(line => /^Expected /.test(line)) ??
    lines.find(line => /^error:/.test(line)) ??
    lines[0] ??
    "lam failed";

  return hit;
}

// Per-task score:
// reference bits = 0.5, each halving -> +0.25, each doubling -> x0.5
export function task_score(bits: number, reference_bits: number): number {
  return bits <= reference_bits
    ? 1 - bits / (2 * reference_bits)
    : reference_bits / (2 * bits);
}

export function reference_bits(
  task_id: string,
  timeout = 10_000,
): number | undefined {
  var path = join(import.meta.dir, "..", "sol", task_id + ".lam");
  if (!existsSync(path)) return undefined;
  return bin_size(readFileSync(path, "utf-8").trim(), timeout);
}

function remaining_timeout(deadline_ms?: number): number {
  if (deadline_ms === undefined) return 10_000;
  var remaining = deadline_ms - Date.now();
  if (remaining <= 0) throw new Error("task timed out");
  return Math.max(1, Math.min(10_000, remaining));
}

function is_timeout_error(error: any): boolean {
  var msg = error?.message ?? String(error);
  return msg.includes("timed out") || msg.includes("ETIMEDOUT");
}

export function run_task(
  task: Task,
  submission: string,
  ref_bits?: number,
  options: RunTaskOptions = {},
): Result {
  var errors: string[] = [];

  for (var t of task.tests) {
    try {
      var timeout = remaining_timeout(options.deadline_ms);
      var src  = submission + "\n@_ = " + t.expr;
      var got  = lam_run(src, timeout);
      var want = normalize(t.want, remaining_timeout(options.deadline_ms));
      if (got !== want) {
        errors.push(`${t.expr}\nwant: ${want}\n got: ${got}`);
      }
    } catch (e: any) {
      errors.push(`${t.expr}\nerror: ${e.message}`);
      if (is_timeout_error(e)) break;
    }
  }

  var pass  = errors.length === 0;
  var bits  = 0;
  var score = 0;

  if (pass) {
    try {
      bits  = bin_size(submission, remaining_timeout(options.deadline_ms));
      score = task_score(bits, ref_bits ?? bits);
    } catch (e: any) {
      pass = false;
      errors.push(
        is_timeout_error(e) ?
          "task timed out" :
          "failed to compute binary size"
      );
    }
  }

  return { id: task.id, pass, bits, score, errors };
}

export function show_result(r: Result): string {
  var status = r.pass ? "✓" : "✗";
  var detail = r.pass ? `${r.bits} bits, score: ${r.score.toFixed(3)}` : "FAIL";
  var lines  = [`${status} ${r.id}: ${detail}`];
  for (var e of r.errors) {
    lines.push("  " + e.split("\n").join("\n  "));
  }
  return lines.join("\n");
}

function run_file(path: string): Result {
  var task_id = basename(path, ".lam");
  var tsk_dir = join(import.meta.dir, "..", "tsk");
  var task = parse_task(join(tsk_dir, task_id + ".tsk"));
  var sub = readFileSync(path, "utf-8").trim();
  return run_task(task, sub, reference_bits(task_id));
}

function run_dir(path: string): Result[] {
  var tsk_dir = join(import.meta.dir, "..", "tsk");
  var tasks   = load_tasks(tsk_dir);
  var results: Result[] = [];

  for (var task of tasks) {
    var sub_path = join(path, task.id + ".lam");
    try {
      var sub = readFileSync(sub_path, "utf-8").trim();
    } catch {
      console.log(`- ${task.id}: no submission`);
      results.push({
        id: task.id,
        pass: false,
        bits: 0,
        score: 0,
        errors: ["no submission"],
      });
      continue;
    }
    var result = run_task(task, sub, reference_bits(task.id));
    results.push(result);
    console.log(show_result(result));
  }

  return results;
}

// CLI: bun src/run.ts <submission>
// submission can be a directory of task-named .lam files,
// or one task-named .lam file.
async function main() {
  var args = process.argv.slice(2);
  if (args.length === 0) {
    console.error("usage: bun src/run.ts <submissions_dir|submission.lam>");
    process.exit(1);
  }

  var submission = args[0];
  if (statSync(submission).isFile()) {
    console.log(show_result(run_file(submission)));
    return;
  }

  var results = run_dir(submission);
  var avg = results.reduce((s, r) => s + r.score, 0) / results.length;
  var passed = results.filter(r => r.pass).length;
  console.log(`\n${passed}/${results.length} passed`);
  console.log(`score: ${(avg * 100).toFixed(1)}`);
}

if (import.meta.main) main();
