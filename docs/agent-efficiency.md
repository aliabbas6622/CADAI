# Agent efficiency: research and implementation

Research date: 28 October 2026. Scope: SolidCraft's MCP tools and built-in API agent
loop. This is an incremental change on top of the existing uncommitted harness work.

## What current systems do

| System / source | Published approach | Application to SolidCraft |
| --- | --- | --- |
| [Anthropic: effective tools](https://www.anthropic.com/research/writing-tools-for-agents) | Clear tool boundaries, meaningful results, bounded responses and actionable errors; evaluate tools on representative tasks. | Keep identifiers, dimensions and failures; remove repeated document state from edit replies. |
| [Anthropic: code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) | Compose operations and process intermediate data outside the model context, rather than repeatedly passing it through model turns. | Evaluate related arithmetic locally in one call with the existing dimension-aware Rust evaluator. |
| [Cursor: dynamic context discovery](https://cursor.com/blog/dynamic-context-discovery) | Retrieve relevant context when needed; preserve access to large tool outputs instead of silently discarding information. | Retain the existing local command lookup and let callers explicitly request verbose edit replies. |
| [Cloudflare: Code Mode](https://developers.cloudflare.com/agents/tools/codemode/) | Let models discover and compose tools programmatically, using an isolated execution environment. | Apply bounded composition to calculations and use the existing command batch for edits. A general code sandbox would be a separate capability. |

These are documented techniques, not evidence that every product uses the same internals.
Their published performance figures do not predict SolidCraft's speedup.

## Open-source prompt and protocol review

The following sources were inspected after checking their repository licenses. SolidCraft's
new wording and Rust implementation are original; no source or prompt text was copied.

| Source | Specific observation | Implemented decision |
| --- | --- | --- |
| [OpenCode runner prompt](https://github.com/anomalyco/opencode/blob/dev/packages/core/src/session/runner/prompt/system.txt), [MIT license](https://github.com/anomalyco/opencode/blob/dev/LICENSE) | A brief harness section identifies the execution environment and distinguishes independent calls; communication instructions are separate. | Organize the shared prompt into workflow, argument format, state/recovery, scope and response sections. Explicitly distinguish grouping independent calls from waiting for returned data. |
| [Aider edit prompts](https://github.com/Aider-AI/aider/blob/main/aider/coders/editblock_prompts.py), [Apache-2.0 license](https://github.com/Aider-AI/aider/blob/main/LICENSE.txt) | Concrete request/response examples accompany exact output-format rules. | Supply a complete native `execute` call example, state that arguments are JSON objects, and distinguish those arguments from compact result encoding. |
| [Cline response formatting](https://github.com/cline/cline/blob/main/apps/vscode/src/core/prompts/responses.ts), [Apache-2.0 license](https://github.com/cline/cline/blob/main/LICENSE) | Explicit failure feedback, repeat-call notices and multimodal result blocks carry protocol context back to the model. | Prefix model-facing tool replies with explicit success/error status while retaining call IDs. Consume structured MCP object data when supplied, preserving numeric precision. Keep existing image forwarding and bounded error recovery. |

These are task-specific adaptations of general patterns. SolidCraft keeps native JSON tool
calls and its existing JSON/TOON result encoder; it does not adopt a coding agent's patch or
XML protocol. It also avoids requiring a tool call for questions that need no design data.
The final-answer contract asks for the actual change, measured checks with units, and remaining
uncertainty. It does not require the model to expose private reasoning or emit a rigid report
for every simple operation.

## Findings in this checkout

The API harness already supplies design snapshots, serves command lookups from a local
catalog, caches repeated reads, limits repeated failures, streams replies and verifies
successful edits. Those were present before this change.

Three remaining gaps were actionable:

1. Read caches were cleared at the end of a model response. In a sequence such as
   `measure`, edit, `measure`, the second measurement could replay the old volume before
   that clearing occurred. Failed edits also skipped invalidation, despite partial batches
   or a lost reply potentially changing the design.
2. The prompt requested compact replies, but the harness did not default `verbose` to
   false for `execute` and `batch`. A model omitting that option received repeated state.
3. `calculate` supported independent strings only. Dependent computations required repeating
   arithmetic in longer expressions or passing intermediate numbers back through the model.

## Changes

- Invalidate cached reads and state-dependent failures immediately after each sent edit,
  including failed edits. The next call in the same response must fetch fresh state.
- After a failed edit, discard the local active-sketch assumption and request measured host
  verification after the response's tool calls. This covers partial batches and uncertain
  outcomes without retrying the mutation.
- Default API agent `execute` and `batch` calls to compact replies. Explicit `verbose:true`
  is respected. This changes the built-in API harness; external MCP clients keep their defaults.
- Allow `calculate` entries of the form `{name, expression}`. Later entries reuse earlier
  dimensioned results in the same call. Existing expression strings remain supported.
- Explain when to use expressions directly in supported command fields and when to group
  derived coordinates or independent numerical checks in `calculate`.
- Replace the mandatory guide/read startup ritual with a conditional workflow: use supplied
  observations, retrieve unfamiliar command docs only when needed, and avoid tools for ordinary
  questions. Make native JSON arguments, dependency order, recovery and final evidence explicit.
- Give API model tool replies an explicit `status: ok` / `status: error` header. Prefer MCP
  `structuredContent` objects over display text when present, keeping JSON/TOON precision and
  explicit truncation notices.
- Reject missing, duplicate or oversized tool call IDs and missing tool names before running
  a response's calls. Every recorded tool result must pair with a valid call identity.

For example, span, half-span and area can now be computed in one call:

```json
{"expressions":[{"name":"span","expression":"2 ft"},{"name":"half","expression":"span / 2"},"half * span"]}
```

The results are 609.6 mm, 304.8 mm and 185806.08 mm². The application preserves the
unit powers between calculations. Names are temporary, ordered and limited; forward
references, duplicates, reserved names and incompatible units are rejected. No design
parameter or undo entry is created. The calculator executes locally in the MCP process.

For three computations that previously depended on three separate model/tool rounds, this
representation permits one calculation round. Actual savings depend on the model using it;
three already-independent expressions could already be grouped before this change.

## Verification and limits

Regression tests cover named dimensional chains, unchanged design state, call-local names,
invalid identifiers, dependency order, incompatible units and invalid arithmetic. Harness
tests cover cache invalidation after successful and failed edits, local rejections, read-only
calls, and explicit versus default verbosity.

Local results: all 56 MCP tests, 31 API harness tests and eight shared ACP tests passed.
The API suite was rerun after the protocol changes. Formatting, asset attribution,
icon checks (including 15 Python tests), catalog ordering and dependency layering passed.
The `python3` Windows Store alias was worked around only for the check process using an
existing Python installation and a temporary helper under `target/`; no toolchain was installed.
Full `cargo xtask ci` stopped at Clippy: three existing harness style findings were corrected,
but `clippy::large_enum_variant` remains in the untouched `crates/ui-egui/src/dialogs_motion.rs`.
Focused Clippy passed for `solidcraft-ui-egui` and `solidcraft-mcp`, including all targets,
with `-D warnings -A clippy::large_enum_variant`; this explicitly excludes that existing
warning and is not a full-CI pass. The log is in `target/agent-check-clippy.log`.
The WebAssembly target is also absent locally. Full CI is therefore not certified, and the
this development snapshot still needs those checks before release. The desktop app has not
been rebuilt or relaunched.

This does not change model-provider inference latency, guarantee model correctness or make
geometry kernels faster. A dimensional calculation is planning evidence; it cannot prove that
the produced geometry matches it. The measured host observation remains necessary.

For provider acceptance, repeat the same small CAD task with the same model and starting
design, recording model requests, input/output tokens, total time, measured dimensions,
timeline errors and repair attempts. Use repeated runs rather than claiming a speedup from
one response. No live-provider speedup is claimed by this document.
