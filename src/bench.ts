import { streamText, type LanguageModel } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { google } from "@ai-sdk/google";
import { openai } from "@ai-sdk/openai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { xai } from "@ai-sdk/xai";
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";
import type { Task } from "./check";
import { load_tasks } from "./check";
import {
  LAM_TIMEOUT_MS,
  REF_DIR,
  reference_bits,
  run_task,
  task_score,
} from "./check";

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
  finish_reason?: string;
};

type EvalModel = {
  spec:     string;
  provider: string;
  model_id: string;
  sdk?:     LanguageModel;
};

type ReasoningEffort = "low" | "medium" | "high";

type Args = {
  model: string;
  filter?: string;
  concurrency: number;
  timeout_ms: number;
  no_reasoning: boolean;
  reasoning_effort: ReasoningEffort;
};

type TaskProgressUpdate = {
  attempt?: number;
  phase?: string;
  last_error?: string;
};

type TaskProgress = (update: TaskProgressUpdate) => void;

type ActiveTask = {
  id: string;
  started_ms: number;
  attempt: number;
  phase: string;
  last_update_ms: number;
  last_error?: string;
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
  set_env("DEEPSEEK_API_KEY", ["deepseek.token"]);
}

function parse_args(): Args {
  var args = process.argv.slice(2);
  if (args.length === 0) {
    console.error(
      "usage: bun eval <provider/model> [--filter prefix] " +
      "[--concurrency n] [--timeout seconds] [--low|--medium|--high]"
    );
    process.exit(1);
  }

  var parsed: Args = {
    model: args[0],
    concurrency: 40,
    timeout_ms: DEFAULT_TASK_TIMEOUT_MS,
    no_reasoning: false,
    reasoning_effort: "high",
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
    else if (arg === "--low") parsed.reasoning_effort = "low";
    else if (arg === "--medium") parsed.reasoning_effort = "medium";
    else if (arg === "--high") parsed.reasoning_effort = "high";
    else if (arg === "--effort") {
      parsed.reasoning_effort = parse_reasoning_effort(args[++i]);
    }
    else if (arg.startsWith("--effort=")) {
      parsed.reasoning_effort = parse_reasoning_effort(
        arg.slice("--effort=".length),
      );
    }
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

function parse_reasoning_effort(value: string | undefined): ReasoningEffort {
  if (value === "low" || value === "medium" || value === "high") return value;
  throw new Error("reasoning effort must be low, medium, or high");
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
    return { spec, provider, model_id, sdk: openai(model_id) };
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
  if (provider === "deepseek") {
    var deepseek = createOpenAICompatible({
      name: "deepseek",
      apiKey: process.env.DEEPSEEK_API_KEY,
      baseURL: "https://api.deepseek.com",
    });
    return { spec, provider, model_id, sdk: deepseek(model_id) };
  }
  if (provider === "lmstudio") {
    var lmstudio = createOpenAICompatible({
      name: "lmstudio",
      apiKey: process.env.LMSTUDIO_API_KEY ?? "not-needed",
      baseURL: process.env.LMSTUDIO_API_BASE_URL ?? "http://localhost:1234/v1",
    });
    return { spec, provider, model_id, sdk: lmstudio(model_id) };
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

type AnthropicThinkingMode = "adaptive" | "enabled";

function anthropic_supports_adaptive_thinking(model_id: string): boolean {
  // @ai-sdk/anthropic documents adaptive thinking as supported by Sonnet 4.6,
  // Opus 4.6, and newer models. 4.5 and earlier need the older
  // { type: "enabled", budgetTokens } mode.
  var match = model_id.match(/^claude-(sonnet|opus|haiku)-4-(\d+)/);
  return match ? Number(match[2]) >= 6 : false;
}

function anthropic_initial_thinking_mode(
  model: EvalModel,
): AnthropicThinkingMode {
  return anthropic_supports_adaptive_thinking(model.model_id)
    ? "adaptive"
    : "enabled";
}

function anthropic_high_legacy_thinking_budget(model_id: string): number {
  // Keep enough room for the final answer while still giving pre-adaptive
  // Claude models a substantial extended-thinking budget.
  if (model_id.includes("claude-3-haiku")) return 2048;
  if (model_id.includes("claude-opus-4-0")) return 16000;
  if (model_id.includes("claude-opus-4-1")) return 16000;
  if (model_id.includes("claude-opus-4-")) return 32000;
  if (model_id.includes("claude-sonnet-4-")) return 32000;
  if (model_id.includes("claude-haiku-4-")) return 16000;
  if (model_id.includes("claude-3-7")) return 32000;
  return 2048;
}

function anthropic_legacy_thinking_budget(
  model_id: string,
  effort: ReasoningEffort,
): number {
  var high = anthropic_high_legacy_thinking_budget(model_id);
  if (effort === "high") return high;
  if (effort === "medium") return Math.max(2048, Math.floor(high / 2));
  return Math.max(1024, Math.floor(high / 4));
}

function thinking_options(
  model: EvalModel,
  effort: ReasoningEffort,
  anthropic_mode?: AnthropicThinkingMode,
) {
  var anthropic_options = anthropic_mode === "enabled"
    ? {
      // Legacy extended thinking for Claude 4.5 and earlier. Do not send
      // output_config.effort here; older models reject adaptive/output-config
      // thinking controls but accept an explicit budget.
      thinking: {
        type: "enabled",
        budgetTokens: anthropic_legacy_thinking_budget(model.model_id, effort),
      },
    }
    : {
      effort,
      thinking: { type: "adaptive", display: "omitted" },
    };

  return {
    openai: {
      reasoningEffort: effort,
      forceReasoning: true,
    },
    anthropic: anthropic_options,
    google: {
      thinkingConfig: { thinkingLevel: effort, includeThoughts: false },
    },
    xai: {
      reasoningEffort: effort,
    },
    openrouter: {
      reasoningEffort: effort,
    },
    moonshotai: {
      thinking: { type: "enabled" },
    },
    deepseek: {
      reasoningEffort: effort,
      thinking: { type: "enabled" },
    },
    lmstudio: {
      reasoningEffort: effort,
    },
    openaiCompatible: {
      reasoningEffort: effort,
    },
  };
}

function max_output_tokens(
  model: EvalModel,
  anthropic_mode?: AnthropicThinkingMode,
  effort: ReasoningEffort = "high",
): number | undefined {
  if (model.provider === "anthropic" && anthropic_mode === "enabled") {
    return anthropic_legacy_thinking_budget(model.model_id, effort) + 4096;
  }

  // maxOutputTokens: The dedicated SDKs (@ai-sdk/anthropic, @ai-sdk/google,
  // @ai-sdk/openai, @ai-sdk/xai) set model-aware defaults when maxOutputTokens
  // is omitted. But @ai-sdk/openai-compatible just sends max_tokens: undefined,
  // is omitted and the server applies its own default —
  // often far too low for reasoning models. We set 131072 for any provider that
  // uses createOpenAICompatible.
  var uses_openai_compatible = [
    "openrouter", "moonshotai", "moonshot", "kimi", "deepseek", "lmstudio",
  ].includes(model.provider);
  var has_dedicated_sdk =

    model.provider === "openai" ||
    model.provider === "anthropic" ||
    model.provider === "google" ||
    model.provider === "xai";
  return (!has_dedicated_sdk || uses_openai_compatible) ? 131072 : undefined;
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

var MAX_RETRIES = 3;
var RETRY_BACKOFF_MS = 10_000; // 10s, 20s, 30s between retries
var MIN_RETRY_REMAINING_MS = 30_000;
var HEARTBEAT_MS = 60_000;

// Transient finish reasons that warrant a retry (network drops, provider
// errors, etc.) as opposed to "stop" (success) or "length" (token budget
// exhausted — retrying won't help).
function is_retryable_finish(reason: string | undefined): boolean {
  return reason === "error" || reason === "other" || reason === "unknown";
}

function format_unknown_error(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}

function is_timeout_like_error(error: unknown): boolean {
  var msg = format_unknown_error(error).toLowerCase();
  return (
    msg.includes("timed out") ||
    msg.includes("timeout") ||
    msg.includes("abort") ||
    msg.includes("interrupted")
  );
}

function is_transient_error(error: unknown): boolean {
  var msg = format_unknown_error(error).toLowerCase();
  return (
    msg.includes("server_error") ||
    msg.includes("internal") ||
    msg.includes("overload") ||
    msg.includes("unavailable") ||
    msg.includes("rate limit") ||
    msg.includes("429") ||
    msg.includes("502") ||
    msg.includes("503") ||
    msg.includes("504") ||
    msg.includes("econnreset") ||
    msg.includes("econnrefused") ||
    msg.includes("socket") ||
    msg.includes("network") ||
    msg.includes("fetch failed")
  );
}

function retry_budget_remaining(deadline_ms: number): boolean {
  return deadline_ms - Date.now() > MIN_RETRY_REMAINING_MS;
}

function should_retry_error(
  error: unknown,
  stream_error: unknown,
  deadline_ms: number,
): boolean {
  if (!retry_budget_remaining(deadline_ms)) return false;
  if (is_timeout_like_error(error)) return false;
  if (stream_error !== undefined) return is_transient_error(stream_error);
  if (format_unknown_error(error).includes("No output generated")) return false;
  return is_transient_error(error);
}

function should_retry_empty_response(
  finish_reason: string | undefined,
  stream_error: unknown,
  deadline_ms: number,
): boolean {
  if (!retry_budget_remaining(deadline_ms)) return false;
  if (!is_retryable_finish(finish_reason)) return false;
  return stream_error === undefined || is_transient_error(stream_error);
}

async function sleep_or_abort(
  ms: number,
  signal: AbortSignal,
): Promise<boolean> {
  if (ms <= 0) return !signal.aborted;
  if (signal.aborted) return false;
  return await new Promise(resolve => {
    var timer: ReturnType<typeof setTimeout>;
    var done = (ok: boolean) => {
      clearTimeout(timer);
      signal.removeEventListener("abort", on_abort);
      resolve(ok);
    };
    var on_abort = () => done(false);
    timer = setTimeout(() => done(true), ms);
    signal.addEventListener("abort", on_abort, { once: true });
  });
}

function write_file_atomic(path: string, text: string) {
  var tmp = `${path}.${process.pid}.tmp`;
  writeFileSync(tmp, text);
  renameSync(tmp, path);
}

function looks_like_adaptive_thinking_error(error: unknown): boolean {
  var msg = format_unknown_error(error).toLowerCase();
  return (
    msg.includes("adaptive") ||
    (
      msg.includes("thinking") &&
      (
        msg.includes("not supported") ||
        msg.includes("unsupported") ||
        msg.includes("invalid") ||
        msg.includes("unrecognized") ||
        msg.includes("unknown")
      )
    ) ||
    (
      msg.includes("output_config") &&
      (msg.includes("not supported") || msg.includes("unsupported"))
    )
  );
}

function should_fallback_anthropic_thinking(
  model: EvalModel,
  mode: AnthropicThinkingMode | undefined,
  error: unknown,
): boolean {
  return (
    model.provider === "anthropic" &&
    mode === "adaptive" &&
    looks_like_adaptive_thinking_error(error)
  );
}

async function generate_solution(
  model: EvalModel,
  task: Task,
  out_dir: string,
  signal: AbortSignal,
  deadline_ms: number,
  no_reasoning: boolean,
  reasoning_effort: ReasoningEffort,
  progress?: TaskProgress,
): Promise<{ text: string; usage?: unknown; finish_reason?: string }> {
  var prompt = task_prompt(task);

  if (!model.sdk) {
    throw new Error(`missing SDK model for ${model.spec}`);
  }

  var anthropic_mode: AnthropicThinkingMode | undefined =
    !no_reasoning && model.provider === "anthropic"
      ? anthropic_initial_thinking_mode(model)
      : undefined;

  var last_error: unknown;
  var last_finish_reason: string | undefined;
  var last_stream_error: unknown;

  for (var attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    if (signal.aborted) break;

    // Backoff before retries (not before the first attempt), but never sleep
    // past the hard per-task deadline.
    if (attempt > 1) {
      if (!retry_budget_remaining(deadline_ms)) break;
      var remaining_before_sleep = remaining_task_ms(deadline_ms);
      var delay_ms = Math.min(
        RETRY_BACKOFF_MS * (attempt - 1),
        Math.max(0, remaining_before_sleep - MIN_RETRY_REMAINING_MS),
      );
      console.log(
        `  ↻ ${task.id} retry ${attempt}/${MAX_RETRIES} in ${delay_ms / 1000}s…`,
      );
      var slept = await sleep_or_abort(delay_ms, signal);
      if (!slept || signal.aborted) break;
    }

    try {
      var remaining_ms = remaining_task_ms(deadline_ms);
      progress?.({ attempt, phase: "generating" });
      var attempt_stream_error: unknown;
      var stream = streamText({
        model: model.sdk,
        prompt,
        abortSignal: signal,
        timeout: { totalMs: remaining_ms },
        maxOutputTokens: max_output_tokens(
          model,
          anthropic_mode,
          reasoning_effort,
        ),
        providerOptions: no_reasoning
          ? {}
          : thinking_options(model, reasoning_effort, anthropic_mode),
        // The AI SDK's default onError is console.error(error), which dumps
        // raw provider JSON mid-progress with no task context. Capture it
        // instead; retry/final error handling below will report the task id.
        onError: ({ error }) => {
          attempt_stream_error = error;
          last_stream_error = error;
          progress?.({
            attempt,
            phase: "stream-error",
            last_error: format_unknown_error(error),
          });
        },
        onChunk: ({ chunk }) => {
          var type = typeof (chunk as any)?.type === "string"
            ? (chunk as any).type
            : "chunk";
          progress?.({ attempt, phase: `stream:${type}` });
        },
      });

      var text = await stream.text;
      var usage = await stream.usage;
      var finish_reason = await stream.finishReason;
      last_finish_reason = finish_reason;

      // Got content — return it regardless of finish_reason.
      if (text.trim() !== "") {
        return { text, usage, finish_reason };
      }

      // Empty text: retry only if the finish reason / stream error looks
      // transient and there is still meaningful time left before the hard
      // per-task deadline. Never retry length/stop; that is model behavior.
      if (!is_retryable_finish(finish_reason)) {
        return { text, usage, finish_reason };
      }

      if (
        attempt_stream_error &&
        should_fallback_anthropic_thinking(
          model,
          anthropic_mode,
          attempt_stream_error,
        ) &&
        attempt < MAX_RETRIES &&
        retry_budget_remaining(deadline_ms)
      ) {
        anthropic_mode = "enabled";
        last_stream_error = undefined;
        console.log(
          `  ↻ ${task.id} adaptive thinking unsupported; ` +
          `retrying with legacy thinking budget ` +
          `${anthropic_legacy_thinking_budget(model.model_id, reasoning_effort)}…`,
        );
        continue;
      }

      if (
        attempt < MAX_RETRIES &&
        should_retry_empty_response(
          finish_reason,
          attempt_stream_error,
          deadline_ms,
        )
      ) {
        var detail = attempt_stream_error
          ? `: ${format_unknown_error(attempt_stream_error)}`
          : "";
        console.log(
          `  ↻ ${task.id} attempt ${attempt}/${MAX_RETRIES} ` +
          `empty response (finish_reason=${finish_reason ?? "unknown"})` +
          `${detail}, retrying…`,
        );
        continue;
      }
      break;
    } catch (e: any) {
      last_error = e;
      progress?.({
        attempt,
        phase: "error",
        last_error: format_unknown_error(e),
      });
      if (signal.aborted) break;
      if (
        should_fallback_anthropic_thinking(model, anthropic_mode, e) &&
        attempt < MAX_RETRIES &&
        retry_budget_remaining(deadline_ms)
      ) {
        anthropic_mode = "enabled";
        last_error = undefined;
        last_stream_error = undefined;
        console.log(
          `  ↻ ${task.id} adaptive thinking unsupported; ` +
          `retrying with legacy thinking budget ` +
          `${anthropic_legacy_thinking_budget(model.model_id, reasoning_effort)}…`,
        );
        continue;
      }

      if (
        attempt < MAX_RETRIES &&
        should_retry_error(e, attempt_stream_error, deadline_ms)
      ) {
        var detail = attempt_stream_error && attempt_stream_error !== e
          ? `${format_unknown_error(e)}; stream error: ` +
            `${format_unknown_error(attempt_stream_error)}`
          : format_unknown_error(e);
        console.log(
          `  ↻ ${task.id} attempt ${attempt}/${MAX_RETRIES} ` +
          `${detail}, retrying…`,
        );
        continue;
      }
      break;
    }
  }

  // All retries exhausted.
  if (last_error) throw last_error;
  if (last_stream_error) {
    throw new Error(`stream error: ${format_unknown_error(last_stream_error)}`);
  }
  return { text: "", finish_reason: last_finish_reason };
}

function timeout_result(
  ms: number,
  abort: AbortController,
): { promise: Promise<never>; cancel: () => void } {
  var timer: ReturnType<typeof setTimeout>;
  var promise = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      var error = new Error(`task timed out after ${Math.floor(ms / 1000)}s`);
      abort.abort(error);
      reject(error);
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
  reasoning_effort: ReasoningEffort,
  progress?: TaskProgress,
): Promise<EvalResult> {
  var raw_path = join(out_dir, task.id + ".txt");
  var lam_path = join(out_dir, task.id + ".lam");

  var response = await generate_solution(
    model,
    task,
    out_dir,
    signal,
    deadline_ms,
    no_reasoning,
    reasoning_effort,
    progress,
  );
  throw_if_aborted(signal);

  progress?.({ phase: "extracting" });
  writeFileSync(raw_path, response.text);
  var submission = extract_submission(response.text);

  // Guard against providers that return empty content with no error. The
  // Moonshot Kimi run on 2026-04-24 exposed this: with no maxOutputTokens
  // set, the model burned its whole budget on reasoning and returned an
  // empty string + finish_reason="length". Without this check, the harness
  // would write a 0-byte .lam, "run" it, and report misleading want/got
  // diffs as if the model had submitted a wrong answer.
  if (submission.trim() === "") {
    var fr = response.finish_reason;
    throw new Error(
      `no submission extracted from model output ` +
      `(raw_len=${response.text.length}, finish_reason=${fr ?? "unknown"})`,
    );
  }

  writeFileSync(lam_path, submission);

  progress?.({ phase: "checking" });
  var check_deadline_ms = Math.min(deadline_ms, Date.now() + LAM_TIMEOUT_MS);
  var ref = reference_bits(task.id, remaining_task_ms(check_deadline_ms));
  var check = run_task(task, submission, ref, { deadline_ms: check_deadline_ms });
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
    finish_reason: response.finish_reason,
  };
}

async function eval_task(
  task: Task,
  model: EvalModel,
  out_dir: string,
  timeout_ms: number,
  no_reasoning: boolean,
  reasoning_effort: ReasoningEffort,
  parent_signal?: AbortSignal,
  progress?: TaskProgress,
): Promise<EvalResult> {
  var started = Date.now();
  progress?.({ attempt: 0, phase: "queued" });
  var abort = new AbortController();
  var abort_from_parent = () => abort.abort(parent_signal?.reason);
  if (parent_signal?.aborted) abort_from_parent();
  parent_signal?.addEventListener("abort", abort_from_parent, { once: true });

  try {
    // --timeout is a hard wall-clock cap for the whole task: API calls,
    // retries, backoff, extraction, and local checking. Retries borrow from
    // this budget; they never multiply it.
    var deadline_ms = started + timeout_ms;
    var timeout = timeout_result(timeout_ms, abort);
    return await Promise.race([
      eval_task_body(
        task,
        model,
        out_dir,
        started,
        deadline_ms,
        abort.signal,
        no_reasoning,
        reasoning_effort,
        progress,
      ),
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
    parent_signal?.removeEventListener("abort", abort_from_parent);
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
  fn: (item: T, index: number) => Promise<R>,
  should_stop: () => boolean = () => false,
): Promise<(R | undefined)[]> {
  var results: (R | undefined)[] = new Array(items.length);
  var next = 0;

  async function worker() {
    while (next < items.length && !should_stop()) {
      var index = next++;
      results[index] = await fn(items[index], index);
    }
  }

  var workers = [];
  for (var i = 0; i < Math.min(concurrency, items.length); i++) {
    workers.push(worker());
  }
  await Promise.all(workers);
  return results;
}

function write_eval_reports(
  model: string,
  filter: string | undefined,
  concurrency: number,
  total_tasks: number,
  scheduled_tasks: number,
  reasoning_effort: ReasoningEffort,
  started_at: Date,
  out_dir: string,
  results: EvalResult[],
  interrupted: boolean,
): { report_path: string; text_report_path: string; score: number; pass: number; created_refs: number } {
  var pass = results.filter(r => r.pass).length;
  var created_refs = results.filter(r => r.created_reference).length;
  var score =
    results.reduce((sum, r) => sum + r.score, 0) /
    Math.max(total_tasks, 1) *
    100;
  var complete = results.length === scheduled_tasks;
  var report = {
    model,
    filter,
    concurrency,
    reasoning_effort,
    tasks: total_tasks,
    scheduled_tasks,
    evaluated_tasks: results.length,
    complete,
    interrupted,
    pass,
    created_refs,
    score,
    results,
  };

  var report_path = join(out_dir, "report.json");
  write_file_atomic(report_path, JSON.stringify(report, null, 2));
  var res_dir = join(import.meta.dir, "..", "res");
  mkdirSync(res_dir, { recursive: true });
  var text_report_path = join(
    res_dir,
    `${report_stamp(started_at)}.${safe_name(model)}.txt`,
  );
  var text_report = build_text_report(model, results, score, total_tasks);
  write_file_atomic(text_report_path, text_report);

  return { report_path, text_report_path, score, pass, created_refs };
}

function write_state(
  out_dir: string,
  total_tasks: number,
  completed: number,
  active: Map<string, ActiveTask>,
) {
  var now = Date.now();
  var active_tasks = [...active.values()]
    .sort((a, b) => a.started_ms - b.started_ms)
    .map(task => ({
      id: task.id,
      attempt: task.attempt,
      phase: task.phase,
      elapsed_s: Number(((now - task.started_ms) / 1000).toFixed(1)),
      idle_s: Number(((now - task.last_update_ms) / 1000).toFixed(1)),
      last_error: task.last_error,
    }));
  write_file_atomic(
    join(out_dir, "state.json"),
    JSON.stringify({
      completed,
      total_tasks,
      active: active_tasks.length,
      queued: Math.max(0, total_tasks - completed - active_tasks.length),
      active_tasks,
      updated_at: new Date(now).toISOString(),
    }, null, 2),
  );
}

function print_heartbeat(
  total_tasks: number,
  completed: number,
  active: Map<string, ActiveTask>,
) {
  if (active.size === 0) return;
  var now = Date.now();
  console.log(
    `… progress ${completed}/${total_tasks}, active ${active.size}`,
  );
  for (var task of [...active.values()].sort((a, b) => b.started_ms - a.started_ms)) {
    var elapsed = ((now - task.started_ms) / 1000).toFixed(0);
    var idle = ((now - task.last_update_ms) / 1000).toFixed(0);
    var error = task.last_error ? ` error=${task.last_error.slice(0, 80)}` : "";
    console.log(
      `  · ${task.id} ${elapsed}s attempt=${task.attempt} ` +
      `phase=${task.phase} idle=${idle}s${error}`,
    );
  }
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
  if (args.no_reasoning) {
    console.log(`reasoning: disabled`);
  } else {
    console.log(`reasoning effort: ${args.reasoning_effort}`);
  }
  console.log(`output: ${out_dir}`);
  console.log("");

  var completed = 0;
  var result_slots: (EvalResult | undefined)[] = new Array(tasks.length);
  var active = new Map<string, ActiveTask>();
  var abort_all = new AbortController();
  var interrupted = false;

  function current_results(): EvalResult[] {
    return result_slots.filter((r): r is EvalResult => r !== undefined);
  }

  function save(interrupted_now = interrupted) {
    var results = current_results();
    write_state(out_dir, tasks.length, results.length, active);
    return write_eval_reports(
      args.model,
      args.filter,
      args.concurrency,
      all_tasks.length,
      tasks.length,
      args.reasoning_effort,
      started_at,
      out_dir,
      results,
      interrupted_now,
    );
  }

  function interrupt(reason: string) {
    if (interrupted) return;
    interrupted = true;
    abort_all.abort(new Error(reason));
    var paths = save(true);
    console.log("");
    console.log(`${reason}; stopping new tasks and saving partial report…`);
    console.log(`report: ${paths.report_path}`);
    console.log(`results: ${paths.text_report_path}`);
    console.log("press Ctrl-C again to exit immediately");
    process.once("SIGINT", () => process.exit(130));
  }

  process.once("SIGINT", () => interrupt("interrupt received"));
  process.once("SIGTERM", () => interrupt("termination received"));

  var heartbeat = setInterval(() => {
    print_heartbeat(tasks.length, completed, active);
    write_state(out_dir, tasks.length, completed, active);
  }, HEARTBEAT_MS);

  await run_pool(
    tasks,
    args.concurrency,
    async (task, index) => {
      console.log(`→ ${task.id}`);
      var started_ms = Date.now();
      active.set(task.id, {
        id: task.id,
        started_ms,
        attempt: 0,
        phase: "starting",
        last_update_ms: started_ms,
      });
      var progress: TaskProgress = update => {
        var state = active.get(task.id);
        if (!state) return;
        if (update.attempt !== undefined) state.attempt = update.attempt;
        if (update.phase !== undefined) state.phase = update.phase;
        if (update.last_error !== undefined) state.last_error = update.last_error;
        state.last_update_ms = Date.now();
      };
      var result = await eval_task(
        task,
        model,
        out_dir,
        args.timeout_ms,
        args.no_reasoning,
        args.reasoning_effort,
        abort_all.signal,
        progress,
      );
      active.delete(task.id);
      completed += 1;
      result_slots[index] = result;
      console.log(`${format_line(result)} (${completed}/${tasks.length})`);
      save();
      return result;
    },
    () => abort_all.signal.aborted,
  );

  clearInterval(heartbeat);
  var results = current_results();
  var paths = save(interrupted);

  console.log("");
  console.log(`${paths.pass}/${results.length} passed`);
  console.log(`score: ${paths.score.toFixed(1)}`);
  console.log(`references created: ${paths.created_refs}`);
  console.log(`report: ${paths.report_path}`);
  console.log(`results: ${paths.text_report_path}`);
  if (interrupted) process.exitCode = 130;
}

if (import.meta.main) {
  main().catch((error: any) => {
    console.error(error?.message ?? String(error));
    process.exit(1);
  });
}
