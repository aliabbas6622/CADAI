# Actionable tool errors — handoff

Updated: 21 October 2026.

This is a handoff of **work in the working tree (uncommitted)**: it records what was changed and
why, so another agent can continue, verify or review it. Read `AGENTS.md` first.

## 1. The problem this work fixes

One agent turn in the AI sidebar produced a wall of repeated, unactionable errors:

```
Edit Feature: invalid feature: invalid type: map, expected an array of length 3   (×5)
Delete: `features` must list features                                              (×17)
Center Rectangle: `corner` must be a point [x, y]                                  (×2)
list_commands: unknown command list_commands
body_topology: unknown command body_topology
Inspect Feature: `feature` must be an id or name
```

Each message named what was wrong but not what the *accepted shape* was, so the model guessed
again and again: the identical failing call was re-run, tokens and time were spent, and the
user watched the same red line repeat. The design goal is: **a tool error must let the model
self-correct on the first retry, and a call that cannot succeed must not be run again.**

## 2. The rule the changes follow (researched)

| Source | Guidance |
|---|---|
| [Anthropic, *Writing effective tools for agents*](https://www.anthropic.com/research/writing-tools-for-agents) | "clearly communicate specific and actionable improvements, rather than opaque error codes or tracebacks"; an error may "give examples of correctly formatted tool inputs". |
| [Anthropic, *Handle tool calls*](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls) | "Instead of generic errors like 'failed', include what went wrong and what Claude should try next"; return it as a tool result with `is_error`; use `strict` schemas for type/required errors. |
| [MCP spec, *Tools — Error Handling*](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) | "Tool Execution Errors contain actionable feedback that language models can use to self-correct and retry with adjusted parameters"; "Clients SHOULD provide tool execution errors to language models to enable self-correction". Unknown-tool/protocol errors are the class models are *least* able to fix, so they must name the attempted tool. A JSON-RPC message "SHOULD be limited to a concise single sentence". |
| [OpenAI, *Function calling*](https://developers.openai.com/api/docs/guides/function-calling) | "Explicitly describe the purpose of the function and each parameter (and its format)"; "Include examples and edge cases, especially to rectify any recurring failures". |
| [OpenAI Agents SDK, *Running agents*](https://openai.github.io/openai-agents-python/running_agents/) | `tool_not_found_behavior="return_error_to_model"` — an unknown tool returns a model-visible error instead of aborting; `max_turns` bounds the loop. |
| [LangGraph `ToolNode`](https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/tool_node.py) | "Error: {requested_tool} is not a valid tool, try one of [{available_tools}]." — return the available names with an unknown tool. |
| [Community: tool-failure circuit breaker](https://github.com/icyaaaww/dsh-tool-failure-circuit-breaker) | Key failures by `(tool name, canonical arguments)`; after N identical failures block the unchanged call: "Change the arguments or approach … instead of retrying unchanged." Community implementation, not vendor guidance. |
| [Unity MCP server releases](https://github.com/AnkleBreaker-Studio/unity-mcp-server/releases) | Added "did-you-mean suggestions on unknown tool names"; restoring discarded exception text stopped agents retrying identical failing calls. Domain-server example. |

Consequences adopted here: (a) every error says *what arrived*, *what is accepted* and *one
example*; (b) a failed command carries its own parameter docs the first time it repeats;
(c) identical retries are answered, not run; (d) unknown tool/command names get the real names
back, nearest first.

## 3. What changed, file by file

### 3.1 Engine — the messages themselves

**`crates/engine/src/params.rs`**
- New `shape(&Value)` helper: describes what actually arrived ("an object with x, y, z",
  "an array of 2", "a string", "nothing").
- `req_vec2` / `req_vec3` now say both sides: *"`center` must be a point [x, y], e.g. [10, 20];
  received an object with x, y"* — this is the `Center Rectangle: corner must be a point` class.

**`crates/engine/src/cmd/edit.rs`**
- New `feature_key(&Value) -> Option<String>` accepts an id number, a name string **or an
  object** `{"id": …}` / `{"name": …}` — models copy timeline rows, which are objects.
- `feature_id` error: *"`feature` must be an id or name, e.g. \"Spire\" (or {\"name\": \"Spire\"})"*.
- `timeline.delete` and `feature_ids` accept a flat list of ids/names *or* a single name, and
  their rejection message carries a literal example
  (`{"features":["Spire"]}`) instead of *"`features` must list features"*.
- New `points_as_arrays(&mut Value)` rewrites every all-number `{x,y}` / `{x,y,z}` map to
  `[x,y]` / `[x,y,z]` through a feature definition *before* serde runs, so
  **`invalid type: map, expected an array of length 3` can no longer happen**; the remaining
  serde error now adds where points belong.

**`crates/engine/src/cmd/inspect.rs`**
- `engine.commands` takes an optional `filter`, so an error message can quote the exact
  parameter docs of the one command that failed instead of a 200 KB catalog.

### 3.2 MCP tools — shape acceptance and explanation

**`crates/mcp/src/tools.rs`**
- `TOOL_NAMES`: the ids of this server's tools. `execute {"command":"list_commands"}` is
  answered with *"`list_commands` is a tool, not an engine command: call it directly instead of
  through `execute`"*, which kills the `unknown command list_commands` /
  `body_topology: unknown command` class.
- `params_of()` — **the flat-vs-nested fix**: `execute` merges `params` *and* any other
  top-level keys beside `command`/`verbose`, and accepts `params` as JSON text. Nothing is
  silently dropped, so a misplaced `features` no longer becomes a misleading
  "`features` must list features".
- `steps_of()` — batch steps normalised: every key beside `command`/`id` becomes that step's
  params; errors name the step index and command.
- `explain()` + `command_params()` — every failed command now appends one of:
  the tool-vs-command distinction; the command's own parameter docs (looked up from
  `engine.commands {filter}`, bounded to 1500 chars) plus *"Correct the arguments and send the
  call again; do not repeat the same arguments"*; or, for an unknown id, a pointer to
  `list_commands {"filter":…}`.
- New read-only **`guide`** tool (`overview` | `model` | `edit` | `verify`): a short documented
  workflow, so a model that does not know what to do next has something to read instead of
  guessing.

**`crates/mcp/src/server.rs`** — the prompt mentions `guide`.
**`docs/mcp.md`** — `guide` documented (annotation list + tool table).

### 3.3 The API chat harness — not running what cannot succeed

**`crates/ui-egui/src/agent_chat/api_client.rs`** (this is where the screenshot's errors came
from — the OpenAI-compatible endpoint backend):
- The command catalog is loaded once and injected into the system prompt, and `list_commands`
  is answered locally from it (bounded), so common shapes are known without a round trip.
- **Per-command failure counting (new)**, `failed_commands: Vec<(id, count)>, scoped to a turn:
  - after **3** failures of one command the tool answer is *prefixed* with that command's own
    parameter docs: *"`timeline.delete` has now failed 3 times in this turn. Its parameters: …
    Use exactly that shape, or change the approach … Do not send the same call again."*
  - after **5** attempts the call is **refused**: *"Not run: `x` failed 5 times in this turn and
    was not tried again. … Change the approach (a different command, or ask the user)"*.
  - a success clears that command's count; New chat clears all counts.
  - counts only grow for calls that actually ran (`execute` → 1 command, `batch` → each step's
    command via the new `command_keys()`); replays and locally answered lookups are not counted.
- **Unknown names get suggestions (new)**: `unknown_tool` answers with the available tools and
  the nearest ones first (Levenshtein, `distance()` / `closest_names()`); `unknown_command_call`
  answers from the held catalog — a mistyped id gets *"Did you mean: timeline.delete?"*, and a
  tool name passed to `execute` is redirected to that tool. Nothing far away is offered as a
  suggestion.
- Already present and kept: an identical failing call is answered from memory instead of being
  re-run; unparseable arguments are answered, never run as `{}`; a turn that makes no progress
  for three rounds ends.

### 3.4 Other files touched earlier in the same branch (context for the next agent)

`crates/ui-egui/src/agent_chat/` also carries the earlier latency/streaming/cache work
(`api_cache.rs`, `api_stream.rs`, `creativity.rs` new; panel/settings/prefs changes),
`docs/ai-sidebar-performance.md` and `docs/performance-analysis.md` are analysis notes, and
`docs/agent-improvement-todo.md` is the wider handoff. This document covers only the error work.

## 4. Tests

- `crates/ui-egui/src/agent_chat/api_client.rs` (new):
  - `a_command_that_keeps_failing_gets_its_parameter_docs_then_is_refused`
  - `unknown_tools_and_commands_are_answered_with_the_close_ones`
  - `a_call_names_the_commands_whose_failures_are_counted`
- `crates/mcp/src/tests.rs`: `guide_reaches_every_client_without_changing_the_design`, plus
  `guide` in the tools-list assertion.

## 5. Verification status (be honest with the next agent)

- `cargo check -p solidcraft-ui-egui -p solidcraft-mcp -p solidcraft-engine --all-targets`
  — **passed, 0 warnings** (21 Oct).
- `cargo test -p solidcraft-ui-egui -p solidcraft-mcp -p solidcraft-engine` — run in progress
  at the time of writing; **re-run it before trusting this section**.
- **Not run:** `cargo xtask ci` (fmt, `clippy -D warnings`, full tests, assets, layers) and
  `cargo xtask wasm`.
- **Not done:** the desktop app has not been rebuilt, so the new messages have not been seen
  live. No screenshot evidence yet — see "Showing and following through on changes" in
  `AGENTS.md`: a UI-visible change needs a before/after screenshot or recording.
- **Not done:** regression tests for the engine-side changes (`feature_key`, `points_as_arrays`,
  the new `params` messages) — no test file for those helpers exists yet; per `AGENTS.md` a
  regression test belongs in the existing test file for its area, never a new file named after
  the bug.

## 6. Left to do

1. Re-run the three-crate test suite, then `cargo xtask ci` and `cargo xtask wasm`.
2. Add engine regression tests for: an object `feature` reference, a `{x,y}` point map, a flat
   and a nested `features` list, and the "received an object with x, y" wording.
3. Build the app, drive it with the control channel (or use an API-backend chat against a
   scripted endpoint) and capture before/after evidence of the repeated-error cases above.
4. Consider lowering `MAX_COMMAND_ATTEMPTS` if live turns still show refusals, and consider
   applying the same per-command counting to the ACP backends (`acp.rs`), which do not have it.
5. Decide whether `docs/agent-chat.md` should describe the new error behaviour for users.
