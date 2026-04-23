import { execSync } from "child_process";
import { writeFileSync, readFileSync, mkdirSync } from "fs";
import { join } from "path";
import type { Task, Result } from "./types";
import { parse_task, load_tasks } from "./parse";

var TMP = join(import.meta.dir, "..", ".tmp");

function lam_run(src: string, timeout = 10_000): string {
  mkdirSync(TMP, { recursive: true });
  var file = join(TMP, "run.lam");
  writeFileSync(file, src);
  try {
    return execSync(`lam ${file}`, { timeout, encoding: "utf-8" }).trim();
  } catch (e: any) {
    throw new Error(e.stderr || e.message);
  }
}

function normalize(term: string): string {
  return lam_run("@main = " + term);
}

function bin_size(src: string): number {
  mkdirSync(TMP, { recursive: true });
  var file = join(TMP, "size.lam");
  writeFileSync(file, src);
  return execSync(`lam ${file} --to-bin`, { encoding: "utf-8" }).trim().length;
}

// Per-task score: 256 bits = 0.5, each halving → +0.25, each doubling → ×0.5
function task_score(bits: number): number {
  return bits <= 256 ? 1 - bits / 512 : 128 / bits;
}

export function run_task(task: Task, submission: string): Result {
  var errors: string[] = [];

  for (var t of task.tests) {
    try {
      var src  = submission + "\n@_ = " + t.expr;
      var got  = lam_run(src);
      var want = normalize(t.want);
      if (got !== want) {
        errors.push(`${t.expr}\nwant: ${want}\n got: ${got}`);
      }
    } catch (e: any) {
      errors.push(`${t.expr}\nerror: ${e.message}`);
    }
  }

  var pass  = errors.length === 0;
  var bits  = 0;
  var score = 0;

  if (pass) {
    try {
      bits  = bin_size(submission);
      score = task_score(bits);
    } catch {
      pass = false;
      errors.push("failed to compute binary size");
    }
  }

  return { id: task.id, pass, bits, score, errors };
}

function show_result(r: Result): string {
  var status = r.pass ? "✓" : "✗";
  var detail = r.pass ? `${r.bits} bits, score: ${r.score.toFixed(3)}` : "FAIL";
  var lines  = [`${status} ${r.id}: ${detail}`];
  for (var e of r.errors) {
    lines.push("  " + e.split("\n").join("\n  "));
  }
  return lines.join("\n");
}

// CLI: bun src/run.ts <submissions_dir>
// submissions_dir contains .lam files named by task id
async function main() {
  var args = process.argv.slice(2);
  if (args.length === 0) {
    console.error("usage: bun src/run.ts <submissions_dir>");
    process.exit(1);
  }

  var sub_dir = args[0];
  var tsk_dir = join(import.meta.dir, "..", "tsk");
  var tasks   = load_tasks(tsk_dir);
  var results: Result[] = [];

  for (var task of tasks) {
    var sub_path = join(sub_dir, task.id + ".lam");
    try {
      var sub = readFileSync(sub_path, "utf-8").trim();
    } catch {
      console.log(`- ${task.id}: no submission`);
      results.push({ id: task.id, pass: false, bits: 0, score: 0, errors: ["no submission"] });
      continue;
    }
    var result = run_task(task, sub);
    results.push(result);
    console.log(show_result(result));
  }

  var avg = results.reduce((s, r) => s + r.score, 0) / results.length;
  console.log(`\n${results.filter(r => r.pass).length}/${results.length} passed`);
  console.log(`score: ${(avg * 100).toFixed(1)}`);
}

main();
