import { readFileSync, readdirSync } from "fs";
import { join, basename } from "path";
import type { Task, Test } from "./types";

// Parses a .tsk file into a Task.
// Format: 2 sections separated by "---" on its own line
//   Section 1: description text
//   Section 2: test cases, each is two lines:
//              expression using @main
//              = expected_output
export function parse_task(path: string): Task {
  var id   = basename(path, ".tsk");
  var text = readFileSync(path, "utf-8");
  var secs = text.split(/\n---\n/);
  if (secs.length !== 2) throw `${id}: expected 2 sections, got ${secs.length}`;

  var desc = secs[0].trim();

  // Section 2: test pairs (expr line + "= expected" line)
  var lines = secs[1].trim().split("\n").filter(l => l.trim() !== "");
  var tests: Test[] = [];
  for (var i = 0; i < lines.length; i += 2) {
    var expr = lines[i].trim();
    var want_line = lines[i + 1];
    if (!want_line || !want_line.startsWith("= ")) {
      throw `${id}: line ${i + 2}: expected "= ..." after expression`;
    }
    tests.push({ expr, want: want_line.slice(2).trim() });
  }

  return { id, desc, tests };
}

// Load all tasks from tsk/ directory
export function load_tasks(dir: string): Task[] {
  var files = readdirSync(dir).filter(f => f.endsWith(".tsk")).sort();
  return files.map(f => parse_task(join(dir, f)));
}
