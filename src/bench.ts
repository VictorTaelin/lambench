import { spawn } from "child_process";
import { generateText, streamText, type LanguageModel } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { google } from "@ai-sdk/google";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { xai } from "@ai-sdk/xai";
import { mkdirSync, readFileSync, writeFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";
import type { Task } from "./check";
import { load_tasks } from "./check";
import { REF_DIR, reference_bits, run_task, task_score } from "./check";

const DEFAULT_TASK_TIMEOUT_MS = 3600 * 1000;

type EvalResult = {
  id: string;
  pass: boolean;
  bits: number;
  ref_bits?: number;
  score: number;
  seconds: number;
  created_reference: boolean;
  solution?: string;
  output_path?: string;
  error?: string;
  usage?: unknown;
};

type EvalModel = {
  spec:     string;
  provider: string;
  model_id: string;
  sdk?:     LanguageModel;
};

type Args = {
  model: string;
  filter?: string;
  concurrency: number;
  timeout_ms: number;
  no_reasoning: boolean;
};

function task_prompt(task: Task): string {
  var task_path = join(import.meta.dir, "..", "tsk", task.id + ".tsk");
  var task_text = readFileSync(task_path, "utf-8").trim();
  return `You are solving one problem from Lambench, a benchmark of pure
lambda-calculus programming tasks.

Your goal is to produce the smallest correct Lamb program you can.
Correctness is mandatory.

The evaluator will append test expressions that call @main, normalize them
with the Lamb interpreter, and compare the normalized output with the
expected result.

Task
id: ${task.id}
The task file below contains a natural-language specification followed by
test cases after the --- separator.

Each test case is two lines: an expression that uses @main, then the
expected normalized output prefixed with =.

\`\`\`text
${task_text}
\`\`\`

Output Requirement
Return exactly one .lam program and nothing else.
Do not use Markdown fences.
Do not explain your solution.

The program must define @main.
You may define helper functions with top-level @definitions.
The last top-level definition is the entry point.
Make @main the last definition.

Lamb Grammar
A .lam file is a book of top-level definitions:
@name = term

Terms use this grammar:
- variable: name
- reference: @name
- lambda: λname.term
- application: term(arg1,arg2,...,argN)
- grouping: (term)

Important syntax details:
- Names may contain only ASCII letters, digits, and underscore.
- Valid name characters are [0-9A-Za-z_].
- Do not use apostrophes, hyphens, Unicode subscripts, or punctuation.
- Lambda abstraction must use the λ character, for example λx.λy.x.
- Function application uses parentheses and comma-separated arguments.
- f(x,y,z) means (((f x) y) z).
- Whitespace application is invalid. Never write f x, @foo x y, or s n.
- Use f(x), @foo(x,y), or s(n) instead.
- Comments begin with //, but avoid comments in the final answer.
- Top-level definitions may refer to each other and may be recursive.

Valid Examples
@true  = λt.λf.t
@false = λt.λf.f
@not   = λb.λt.λf.b(f,t)
@main  = @not(@false)

@zero = λf.λx.x
@succ = λn.λf.λx.f(n(f,x))
@add  = λm.λn.λf.λx.m(f,n(f,x))
@main = @add(@succ(@zero),@succ(@succ(@zero)))

Now produce only the .lam source for the task above.`;
}

function token_path(name: string): string {
  return join(homedir(), ".config", name);
}

function read_token(name: string): string | undefined {
  try {
    var token = readFileSync(token_path(name), "utf-8").trim();
    return token.length === 0 ? undefined : token;
  } catch {
    return undefined;
  }
}

function set_env(name: string, files: string[]) {
  if (process.env[name]) return;
  for (var file of files) {
    var token = read_token(file);
    if (token) {
      process.env[name] = token;
      return;
    }
  }
}

function load_keys() {
  set_env("OPENAI_API_KEY", ["openai.token"]);
  set_env("ANTHROPIC_API_KEY", ["anthropic.token", "anthropic_vic.token"]);
  set_env("GOOGLE_GENERATIVE_AI_API_KEY", ["gemini.token", "google.token"]);
  set_env("XAI_API_KEY", ["xai.token", "xai_normal.token"]);
  set_env("OPENROUTER_API_KEY", ["openrouter.token"]);
  set_env("MOONSHOT_API_KEY", ["moonshot.token", "kimi.token"]);
}

function parse_args(): Args {
  var args = process.argv.slice(2);
  if (args.length === 0) {
    console.error(
      "usage: bun eval <provider/model> [--filter prefix] " +
      "[--concurrency n] [--timeout seconds]"
    );
    process.exit(1);
  }

  var parsed: Args = {
    model: args[0],
    concurrency: 40,
    timeout_ms: DEFAULT_TASK_TIMEOUT_MS,
    no_reasoning: false,
  };
  for (var i = 1; i < args.length; i++) {
    var arg = args[i];
    if (arg === "--filter") parsed.filter = args[++i];
    else if (arg.startsWith("--filter=")) {
      parsed.filter = arg.slice("--filter=".length);
    }
    else if (arg === "--concurrency") parsed.concurrency = Number(args[++i]);
    else if (arg.startsWith("--concurrency=")) {
      parsed.concurrency = Number(arg.slice("--concurrency=".length));
    }
    else if (arg === "--timeout") {
      parsed.timeout_ms = Number(args[++i]) * 1000;
    }
    else if (arg.startsWith("--timeout=")) {
      parsed.timeout_ms = Number(arg.slice("--timeout=".length)) * 1000;
    }
    else if (arg === "--no-reasoning") parsed.no_reasoning = true;
    else throw new Error(`unknown argument: ${arg}`);
  }

  if (!Number.isFinite(parsed.concurrency) || parsed.concurrency < 1) {
    throw new Error("--concurrency must be a positive number");
  }
  parsed.concurrency = Math.floor(parsed.concurrency);

  if (!Number.isFinite(parsed.timeout_ms) || parsed.timeout_ms < 1000) {
    throw new Error("--timeout must be at least 1 second");
  }
  parsed.timeout_ms = Math.floor(parsed.timeout_ms);
  return parsed;
}

function safe_name(name: string): string {
  return name.replace(/[^a-zA-Z0-9._-]+/g, "_");
}

function pad2(n: number): string {
  return n.toString().padStart(2, "0");
}

function report_stamp(date: Date): string {
  var yyyy = date.getFullYear();
  var mm = pad2(date.getMonth() + 1);
  var dd = pad2(date.getDate());
  var hh = pad2(date.getHours());
  var min = pad2(date.getMinutes());
  var ss = pad2(date.getSeconds());
  return `${yyyy}y${mm}m${dd}d.${hh}h${min}m${ss}s`;
}

function matches_filter(id: string, filter?: string): boolean {
  if (!filter) return true;
  if (filter.includes("*")) {
    var re = new RegExp(
      "^" + filter.split("*").map(escape_re).join(".*") + "$"
    );
    return re.test(id);
  }
  return id === filter || id.startsWith(filter) || id.includes(filter);
}

function escape_re(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function get_model(spec: string): EvalModel {
  var [provider, ...rest] = spec.split("/");
  var model_id = normalize_model_id(provider, rest.join("/"));
  if (!provider || !model_id) {
    throw new Error(
      "model must look like <provider>/<model>, for example openai/gpt-5.5"
    );
  }
  validate_model_provider(spec, provider, model_id);

  if (provider === "openai") {
    return { spec, provider, model_id };
  }
  if (provider === "anthropic") {
    return { spec, provider, model_id, sdk: anthropic(model_id) };
  }
  if (provider === "google") {
    return { spec, provider, model_id, sdk: google(model_id) };
  }
  if (provider === "xai") {
    return { spec, provider, model_id, sdk: xai(model_id) };
  }
  if (provider === "openrouter") {
    var openrouter = createOpenAICompatible({
      name: "openrouter",
      apiKey: process.env.OPENROUTER_API_KEY,
      baseURL: "https://openrouter.ai/api/v1",
    });
    return { spec, provider, model_id, sdk: openrouter(model_id) };
  }
  if (
    provider === "moonshotai" ||
    provider === "moonshot" ||
    provider === "kimi"
  ) {
    var moonshot = createOpenAICompatible({
      name: "moonshotai",
      apiKey: process.env.MOONSHOT_API_KEY,
      baseURL: "https://api.moonshot.ai/v1",
    });
    return { spec, provider, model_id, sdk: moonshot(model_id) };
  }
  return { spec, provider, model_id, sdk: spec };
}

function validate_model_provider(
  spec: string,
  provider: string,
  model_id: string,
) {
  if (provider === "openai" && looks_like_anthropic_model(model_id)) {
    throw new Error(
      `model "${spec}" looks like an Anthropic model; ` +
      `use "anthropic/${model_id}"`
    );
  }

  if (provider === "anthropic" && looks_like_openai_model(model_id)) {
    throw new Error(
      `model "${spec}" looks like an OpenAI model; use "openai/${model_id}"`
    );
  }
}

function looks_like_anthropic_model(model_id: string): boolean {
  return (
    model_id.startsWith("claude-") ||
    model_id.startsWith("opus-") ||
    model_id.startsWith("sonnet-") ||
    model_id.startsWith("haiku-")
  );
}

function looks_like_openai_model(model_id: string): boolean {
  return (
    model_id.startsWith("gpt-") ||
    model_id.startsWith("o1") ||
    model_id.startsWith("o3") ||
    model_id.startsWith("o4")
  );
}

function normalize_model_id(provider: string, model_id: string): string {
  if (provider === "anthropic") {
    return normalize_anthropic_model(model_id);
  }

  return model_id;
}

function normalize_anthropic_model(model_id: string): string {
  var aliases: Record<string, string> = {
    "haiku-4.5":  "claude-haiku-4-5",
    "opus-4":     "claude-opus-4-0",
    "opus-4.0":   "claude-opus-4-0",
    "opus-4.1":   "claude-opus-4-1",
    "opus-4.5":   "claude-opus-4-5",
    "opus-4.6":   "claude-opus-4-6",
    "opus-4.7":   "claude-opus-4-7",
    "sonnet-4":   "claude-sonnet-4-0",
    "sonnet-4.0": "claude-sonnet-4-0",
    "sonnet-4.5": "claude-sonnet-4-5",
    "sonnet-4.6": "claude-sonnet-4-6",
  };

  return aliases[model_id] ?? model_id;
}

function high_thinking_options() {
  return {
    openai: {
      reasoningEffort: "high",
      forceReasoning: true,
    },
    anthropic: {
      effort: "high",
      thinking: { type: "adaptive", display: "omitted" },
    },
    google: {
      thinkingConfig: { thinkingLevel: "high", includeThoughts: false },
    },
    xai: {
      reasoningEffort: "high",
    },
    openrouter: {
      reasoningEffort: "high",
    },
    moonshotai: {
      thinking: { type: "enabled" },
    },
    openaiCompatible: {
      reasoningEffort: "high",
    },
  };
}

function extract_submission(text: string): string {
  var fence = text.match(/```[a-zA-Z0-9_-]*\n([\s\S]*?)```/);
  var src = (fence ? fence[1] : text).trim();

  var lines = src.split("\n");
  var first_def = lines.findIndex(line => line.trim().startsWith("@"));
  if (first_def >= 0) src = lines.slice(first_def).join("\n").trim();

  return src;
}

function format_line(result: EvalResult): string {
  var time = `${result.seconds.toFixed(1)}s`;
  if (!result.pass) {
    var error = result.error ? ` ${summarize_error(result.error)}` : "";
    return `✗ ${result.id.padEnd(18)} ${time}${error}`;
  }

  var ref =
    result.ref_bits === undefined ? "new-ref" : `${result.ref_bits} ref`;
  var saved = result.created_reference ? " saved-ref" : "";
  var score = (result.score * 100).toFixed(1);
  return [
    `✓ ${result.id.padEnd(18)} ${time}`,
    `${result.bits} bits, ${ref}, score ${score}${saved}`,
  ].join(" ");
}

function summarize_error(error: string): string {
  var lines = error.split("\n").map(line => line.trim()).filter(Boolean);
  return (
    lines.find(line => line.startsWith("error:")) ??
    lines.find(line => line.startsWith("want:")) ??
    lines[0] ??
    "failed"
  );
}

function clean_process_error(
  cmd: string,
  stdout: string,
  stderr: string,
): string {
  var text = strip_ansi([stderr, stdout].filter(Boolean).join("\n"));
  var json_error = extract_json_error(text);
  if (json_error) return json_error;

  var lines = text.split("\n").map(line => line.trim()).filter(Boolean);
  var useful = lines.find(line => line.includes("invalid_request_error"));
  useful ??= lines.find(line => line.includes("not supported"));
  useful ??= lines.find(line => line.includes("Forbidden"));
  useful ??= lines.find(line => line.startsWith("ERROR:"));
  return useful ?? `${cmd} failed`;
}

function strip_ansi(text: string): string {
  return text.replace(/\x1b\[[0-9;?]*[ -/]*[@-~]/g, "");
}

function extract_json_error(text: string): string | undefined {
  var lines = text.split("\n");
  for (var i = lines.length - 1; i >= 0; i--) {
    var line = lines[i];
    var index = line.indexOf("ERROR:");
    if (index < 0) continue;

    var payload = line.slice(index + "ERROR:".length).trim();
    if (!payload.startsWith("{")) continue;

    try {
      var parsed = JSON.parse(payload);
      var message = json_error_message(parsed);
      if (message) return message;
    } catch {
      // Fall through to the non-JSON cleanup path.
    }
  }
}

function json_error_message(value: any): string | undefined {
  if (typeof value?.detail === "string") return value.detail;
  if (typeof value?.error?.message === "string") {
    return value.error.message;
  }
  if (typeof value?.message === "string") return value.message;
  if (typeof value?.error === "string") return value.error;
}

function run_process(
  cmd: string,
  args: string[],
  input: string,
  timeout_ms: number,
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    var child = spawn(cmd, args, {
      stdio: ["pipe", "pipe", "pipe"],
    });
    var stdout = "";
    var stderr = "";
    var timed_out = false;
    var timer = setTimeout(() => {
      timed_out = true;
      child.kill("SIGKILL");
    }, timeout_ms);

    child.stdout.on("data", data => {
      stdout += data.toString();
    });
    child.stderr.on("data", data => {
      stderr += data.toString();
    });
    child.on("error", error => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", code => {
      clearTimeout(timer);
      if (timed_out) {
        reject(new Error(`${cmd} timed out after ${timeout_ms}ms`));
      } else if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(clean_process_error(cmd, stdout, stderr)));
      }
    });

    child.stdin.write(input);
    child.stdin.end();
  });
}

function codex_args(
  model_id: string,
  work_dir: string,
  out_file: string,
): string[] {
  return [
    "exec",
    "--ephemeral",
    "--ignore-user-config",
    "--ignore-rules",
    "--skip-git-repo-check",
    "-C",
    work_dir,
    "--sandbox",
    "read-only",
    "-c",
    'approval_policy="never"',
    "-c",
    'model_reasoning_effort="high"',
    "--disable",
    "shell_tool",
    "--disable",
    "unified_exec",
    "--disable",
    "apps",
    "--disable",
    "browser_use",
    "--disable",
    "computer_use",
    "--disable",
    "plugins",
    "--disable",
    "multi_agent",
    "--disable",
    "tool_search",
    "--model",
    model_id,
    "--output-last-message",
    out_file,
    "-",
  ];
}

async function generate_with_codex(
  model: EvalModel,
  task: Task,
  prompt: string,
  out_dir: string,
  timeout_ms: number,
): Promise<string> {
  var work_dir = join(out_dir, "codex", task.id);
  var out_file = join(work_dir, "last.txt");
  mkdirSync(work_dir, { recursive: true });

  var args = codex_args(model.model_id, work_dir, out_file);
  await run_process("codex", args, prompt, timeout_ms);
  return readFileSync(out_file, "utf-8");
}

async function generate_solution(
  model: EvalModel,
  task: Task,
  out_dir: string,
  signal: AbortSignal,
  timeout_ms: number,
  no_reasoning: boolean,
): Promise<{ text: string; usage?: unknown }> {
  var prompt = task_prompt(task);

  if (model.provider === "openai") {
    var text = await generate_with_codex(
      model,
      task,
      prompt,
      out_dir,
      timeout_ms,
    );
    return { text };
  }

  if (!model.sdk) {
    throw new Error(`missing SDK model for ${model.spec}`);
  }

  // Use streaming to avoid Node/Bun's 300s default fetch headers/body timeout:
  // a non-streaming generateText() waits for the full response in one HTTP call,
  // so reasoning traces longer than 5 minutes die with "The operation timed out.".
  // Streaming resets the idle timer on each chunk, so long reasoning works.
  var stream = streamText({
    model: model.sdk,
    prompt,
    abortSignal: signal,
    timeout: { totalMs: timeout_ms },
    providerOptions: no_reasoning ? {} : high_thinking_options(),
  });

  var text = await stream.text;
  var usage = await stream.usage;

  return { text, usage };
}

function timeout_result(
  ms: number,
  abort: AbortController,
): { promise: Promise<never>; cancel: () => void } {
  var timer: ReturnType<typeof setTimeout>;
  var promise = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      abort.abort();
      reject(new Error(`task timed out after ${Math.floor(ms / 1000)}s`));
    }, ms);
  });
  return {
    promise,
    cancel: () => clearTimeout(timer),
  };
}

function throw_if_aborted(signal: AbortSignal) {
  if (signal.aborted) throw new Error("task timed out");
}

function remaining_task_ms(deadline_ms: number): number {
  var remaining_ms = deadline_ms - Date.now();
  if (remaining_ms <= 0) throw new Error("task timed out");
  return remaining_ms;
}

async function eval_task_body(
  task: Task,
  model: EvalModel,
  out_dir: string,
  started: number,
  deadline_ms: number,
  signal: AbortSignal,
  no_reasoning: boolean,
): Promise<EvalResult> {
  var raw_path = join(out_dir, task.id + ".txt");
  var lam_path = join(out_dir, task.id + ".lam");

  var response = await generate_solution(
    model,
    task,
    out_dir,
    signal,
    remaining_task_ms(deadline_ms),
    no_reasoning,
  );
  throw_if_aborted(signal);

  writeFileSync(raw_path, response.text);
  var submission = extract_submission(response.text);
  writeFileSync(lam_path, submission);

  var ref = reference_bits(task.id, remaining_task_ms(deadline_ms));
  var check = run_task(task, submission, ref, { deadline_ms });
  var created_reference = false;

  if (check.pass && ref === undefined) {
    mkdirSync(REF_DIR, { recursive: true });
    var ref_path = join(REF_DIR, task.id + ".lam");
    writeFileSync(ref_path, submission.trim() + "\n");
    ref = check.bits;
    check.score = task_score(check.bits, ref);
    created_reference = true;
  }

  return {
    id: task.id,
    pass: check.pass,
    bits: check.bits,
    ref_bits: ref,
    score: check.score,
    seconds: (Date.now() - started) / 1000,
    created_reference,
    solution: submission,
    output_path: lam_path,
    error: check.errors[0],
    usage: response.usage,
  };
}

async function eval_task(
  task: Task,
  model: EvalModel,
  out_dir: string,
  timeout_ms: number,
  no_reasoning: boolean,
): Promise<EvalResult> {
  var started = Date.now();
  var abort = new AbortController();

  try {
    var deadline_ms = started + timeout_ms;
    var timeout = timeout_result(timeout_ms, abort);
    return await Promise.race([
      eval_task_body(task, model, out_dir, started, deadline_ms, abort.signal, no_reasoning),
      timeout.promise,
    ]);
  } catch (e: any) {
    return {
      id: task.id,
      pass: false,
      bits: 0,
      score: 0,
      seconds: (Date.now() - started) / 1000,
      created_reference: false,
      error: e?.message ?? String(e),
    };
  } finally {
    timeout?.cancel();
  }
}

function build_text_report(
  model: string,
  results: EvalResult[],
  score: number,
  total_tasks: number,
): string {
  var lines: string[] = [];
  var right = results.filter(result => result.pass).length;

  lines.push(`score: ${score.toFixed(1)}`);
  lines.push(`evals: ${results.length}/${total_tasks}`);
  lines.push(`right: ${right}/${total_tasks}`);
  lines.push("");
  lines.push("task scores:");

  for (var result of results) {
    var task_score = (result.score * 100).toFixed(1);
    var status = result.pass ? "pass" : "fail";
    var time = ` time=${result.seconds.toFixed(1)}s`;
    var bits = result.pass ? ` bits=${result.bits}` : "";
    var ref = result.ref_bits === undefined ? "" : ` ref=${result.ref_bits}`;
    lines.push(`- ${result.id}: ${task_score} ${status}${time}${bits}${ref}`);
  }

  lines.push("");
  lines.push("solutions:");

  for (var result of results) {
    lines.push("");
    lines.push(`--- ${result.id} ---`);
    if (result.solution && result.solution.trim() !== "") {
      lines.push(result.solution.trim());
    } else {
      lines.push("(no solution)");
    }
  }

  lines.push("");
  lines.push(`model: ${model}`);
  return lines.join("\n") + "\n";
}

async function run_pool<T, R>(
  items: T[],
  concurrency: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> {
  var results: R[] = new Array(items.length);
  var next = 0;

  async function worker() {
    while (next < items.length) {
      var index = next++;
      results[index] = await fn(items[index]);
    }
  }

  var workers = [];
  for (var i = 0; i < Math.min(concurrency, items.length); i++) {
    workers.push(worker());
  }
  await Promise.all(workers);
  return results;
}

async function main() {
  load_keys();
  var args = parse_args();
  var model = get_model(args.model);
  var all_tasks = load_tasks(join(import.meta.dir, "..", "tsk"));
  var tasks = all_tasks.filter(task => matches_filter(task.id, args.filter));
  var started_at = new Date();
  var stamp = started_at.toISOString().replace(/[:.]/g, "-");
  var out_dir = join(
    import.meta.dir,
    "..",
    ".eval",
    safe_name(args.model),
    stamp,
  );
  mkdirSync(out_dir, { recursive: true });

  console.log(`model: ${args.model}`);
  var filter = args.filter ? ` filter=${args.filter}` : "";
  console.log(`tasks: ${tasks.length}/${all_tasks.length}${filter}`);
  console.log(`concurrency: ${args.concurrency}`);
  console.log(`timeout: ${Math.floor(args.timeout_ms / 1000)}s/task`);
  if (args.no_reasoning) console.log(`reasoning: disabled`);
  console.log(`output: ${out_dir}`);
  console.log("");

  var completed = 0;
  var results = await run_pool(tasks, args.concurrency, async task => {
    console.log(`→ ${task.id}`);
    var result = await eval_task(task, model, out_dir, args.timeout_ms, args.no_reasoning);
    completed += 1;
    console.log(`${format_line(result)} (${completed}/${tasks.length})`);
    return result;
  });

  var pass = results.filter(r => r.pass).length;
  var created_refs = results.filter(r => r.created_reference).length;
  var score =
    results.reduce((sum, r) => sum + r.score, 0) /
    Math.max(all_tasks.length, 1) *
    100;
  var report = {
    model: args.model,
    filter: args.filter,
    concurrency: args.concurrency,
    tasks: all_tasks.length,
    evaluated_tasks: results.length,
    pass,
    created_refs,
    score,
    results,
  };

  var report_path = join(out_dir, "report.json");
  writeFileSync(report_path, JSON.stringify(report, null, 2));
  var res_dir = join(import.meta.dir, "..", "res");
  mkdirSync(res_dir, { recursive: true });
  var text_report_path = join(
    res_dir,
    `${report_stamp(started_at)}.${safe_name(args.model)}.txt`,
  );
  var text_report = build_text_report(
    args.model,
    results,
    score,
    all_tasks.length,
  );
  writeFileSync(text_report_path, text_report);

  console.log("");
  console.log(`${pass}/${results.length} passed`);
  console.log(`score: ${score.toFixed(1)}`);
  console.log(`references created: ${created_refs}`);
  console.log(`report: ${report_path}`);
  console.log(`results: ${text_report_path}`);
}

if (import.meta.main) {
  main().catch((error: any) => {
    console.error(error?.message ?? String(error));
    process.exit(1);
  });
}
