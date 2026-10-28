# AI sidebar and agent harness — latency, token and CPU deep dive

*Audit date: 2026-10-21. Third document in the series: [performance-analysis.md](performance-analysis.md)
covers the application breadth-first, [performance-deep-dive.md](performance-deep-dive.md) the
document evaluator, kernel and viewport, and this one the sidebar assistant: the OpenAI-compatible
API backend, its MCP tool loop, and what the panel does with a streamed answer.*

**Why this file exists.** A live session showed a 183-second turn that changed nothing:

| Observed in the panel | Value |
|---|---|
| Turn elapsed | 183 s, `0/24 changes` |
| Tool calls before any edit | 5 × `list_commands` with identical arguments |
| First response (HTTP headers) | 1.59 s |
| First text token | 91.93 s |
| Prompt tokens | 10,356 |
| Cached prompt tokens | 9,719 |
| Output tokens | 11,768 |
| Approximate input (bytes ÷ 4) | 10,468 |
| Model | `minimax/minimax-m3.1-flash:free` through OpenRouter |

The harness was not slow at the things it measures; it was slow because it made the model ask the
same question five times and answered it uselessly each time. This document traces that, measures
the payloads that cause it, records the fixes, and lists what is left.

**Evidence classes** (same convention as the deep dive):

| Label | Meaning |
|---|---|
| **[C]** | Read from the code; the cited line shows it. |
| **[M]** | Measured on this machine during this audit, with the debug build already in `target/`. Reproduce with §9. |
| **[O]** | Observed in the reported session (the table above); not reproducible without the same provider account and model. |
| **[E]** | Estimate from a stated model (payload size ÷ 4 bytes per token, request count × latency). The model is given so it can be falsified. |

Timings marked **[M]** come from a `cargo build` with debug assertions (`target/debug`), so they
are upper bounds for a release build; they are used for *sizes* and *ratios*, not as release
timings.

---

## 1. What one message actually does

```text
user presses Send
        │
        ▼
ApiBackend::prompt ─────────────────────────────── crates/ui-egui/src/agent_chat/api_client.rs:118
        │  (channel to the worker thread)
        ▼
Worker::turn                                       api_client.rs:287
  ├─ first turn only: MCP tools/call list_commands  api_client.rs:300
  │    └─ 203,555 chars ── 19,122 chars kept as the system note
  ├─ MCP tools/call inspect_design (measure:false)  api_client.rs:318
  └─ loop, at most 24 model requests                api_client.rs:22,327
        ├─ Worker::request ──────────────────────── api_client.rs:505
        │    body = {model, messages, tools, stream:true}
        │    ├─ tools: 15 MCP tools as JSON schemas
        │    ├─ messages: system note + command index + history
        │    └─ prefix cache (Gemini only): api_cache.rs
        ├─ SSE chunks → Event::Text/Thought/Tool ── api_stream.rs:33
        ├─ tool_calls → gate → MCP tools/call ───── gate.rs:12
        └─ after edits: inspect_design (measure:true)  api_client.rs:461
```

The 24-request loop is the unit that matters: **every round trip resends the whole prompt**, so
anything that adds a round trip multiplies the two costs that dominate — the model's time to first
token and the prompt size.

---

## 2. Where the prompt bytes go

Measured on the debug build with the sample command registry:

| Part of every request | Size | ≈ tokens | Source |
|---|---:|---:|---|
| System note and harness rules | 1,954 chars | 489 | [C] `api_client.rs:188` + `acp.rs:74` |
| Command index injected on the first turn | 19,122 chars | 4,780 | [M] §9.1, built by `command_context` (`api_client.rs:830`) |
| Tool schemas (15 after hidden ones) | 15,696 chars | 3,924 | [M] `tools/list` 16,609 chars minus `crash_reports` (913) |
| Design snapshot (empty design) | 286 chars | 72 | [M] `inspect_design {measure:false,sketches:false}` |
| User message and history | varies | — | [C] `MAX_HISTORY_BYTES = 192_000` (`api_client.rs:31`) |
| **Total for the reported session** | | **10,356** | [O] provider usage |

The measured static prefix is 36,772 chars ≈ 9,193 tokens (37,058 chars with the snapshot),
and the provider reported 10,356 prompt tokens with 9,719 cached [O]. That is a coherent
accounting: the harness's own overhead is ~89 % of the first request in a conversation, and it is
re-sent on every round trip.

Two details matter for cost:

- The index and the tool schemas are **stable within a session**, which is why 9,719 of 10,356
  tokens were served from the provider's prefix cache. Keeping them byte-identical (no timestamps,
  no reordered maps) is what makes the cache work; changing them per turn would cost ~4× on input.- `input_estimate` (`api_client.rs:542`) and `trim_history` (`api_client.rs:742`) both serialise
the whole history with `Value::to_string()` once per round trip. At the 192 KB history cap that is
  under a millisecond per round trip — noise next to the model, and left alone deliberately.

### 2.1 The catalog is 27× larger than what the harness keeps

| Form of the catalog | Chars | ≈ tokens | Evidence |
|---|---:|---:|---|
| `list_commands` MCP result, unfiltered | 203,555 | 50,889 | [M] |
| Command index kept in the system note | 19,122 | 4,780 | [M] |
| New local answer, unfiltered | 7,602 | 1,900 | [M] |
| `list_commands {filter:"fillet"}` through MCP | 3,210 | 802 | [M] |
| New local answer, `filter:"fillet"` | 2,346 | 586 | [M] |

The registry has **401 implemented commands, 391 of them with parameter docs** [M]. Sending all of
it on every lookup is what the old path did.

---

## 3. The five `list_commands` calls, explained

This is the failure that produced the 183-second turn, and it is a chain of three independent
decisions:

1. **The MCP tool returns everything.** `list_commands` without `filter` returns
   `engine.commands` unfiltered (`crates/mcp/src/tools.rs:396-404`), which is 203,555 chars
   (~200 KB, 34 ms) [M].
2. **The chat harness caps every tool result at 20,000 chars** (`MAX_TOOL_TEXT`,
   `api_client.rs:24`) and replaces an oversized result with an envelope containing only the first
   2,500 characters (`api_client.rs:1000`).
3. **The catalog starts with `sketch.*`.** In its JSON form, the 2,500-character preview contains
   **5 of the 401 ids**, all sketch commands:
   `sketch.create, sketch.edit, sketch.finish, sketch.line, sketch.draw.dimensioned` [M].

So the model asked "what commands exist?", received five sketch commands and a note saying data was
omitted, and asked again. Five times. Each round trip cost tens of seconds of model latency; the
turn reached 183 s with zero edits. The tool description even tells it to pass a filter
(`crates/mcp/src/tools.rs:75-84`), but nothing in the truncated answer explained *what* would have
helped, and the harness had already injected a command index the model could not reconcile with the
five ids it got back.

**The irony:** the harness had already paid for the answer. The system note contains 19,122 chars
of ids and labels [M], and the lookup that got truncated returned the same catalog in a form the
model could not read.

### 3.1 Why this was invisible in the timing card

The card reports first response, first token, input estimate, cache reference and provider usage
(`api_client.rs:674-700`). It does not report **tool result size before/after the cap**, so a
203 KB result silently became a 2.6 KB answer and the UI showed a completed tool in 0.03 s. The
cost appears only as repeated model requests, minutes later.

---

## 4. The latency arithmetic

Each round trip in the reported session had `first text token = 91.93 s` [O] — the provider spent
that long reasoning before emitting text, and streaming cannot show anything earlier. With that
number, the cost of a wasted round trip is easy to state:

| Round trips in a turn | Model wait alone | What the user sees |
|---:|---:|---|
| 1 | 1.5 min | answer |
| 3 | 4.6 min | answer |
| 5 (the observed lookups) | 7.7 min | still nothing changed |
| 24 (the cap) | 36.7 min | the request limit message |

This is why the harness fixes matter more than local CPU: **one avoided round trip is worth more
than every micro-optimisation in this document combined.** It also explains why the first-token
number belongs in the UI: with a fast model (2 s) the five lookups would have cost 10 s, not 8
minutes, and nobody would have filed a performance complaint.

The 24-request cap (`api_client.rs:22`) and the 24-change cap (`gate.rs:12`) are separate limits
that happen to share the number 24; the header's `0/24` is the change cap, not the request cap.

---

## 5. What was changed

Three code changes, all in the sidebar backend, plus the English string for the new stop reason.
They are deliberately local to the harness: the MCP tool schema other ACP agents see is unchanged
(and they do not apply the 20 KB chat cap, so the raw catalog still reaches them).

### 5.1 Command lookups are answered from the catalog already in the turn

`Worker` now keeps the rows returned by the first `list_commands` call
(`self.commands = command_rows(&result)`, `api_client.rs:301`) instead of discarding everything but
the rendered index text.

A later `list_commands` call is answered by `local_command_list` (`api_client.rs:789`) with no MCP
round trip and no bridge process involvement:

| Call | Before | After |
|---|---|---|
| `list_commands {}` | 2,500-char preview covering 5 of 401 ids, 34 ms bridge call, then truncation | 7,602 chars with **all 401 ids**, 0 ms |
| `list_commands {"filter":"fillet"}` | 3,210 chars of full rows | 2,346 chars of ids, labels and parameter docs |
| `list_commands {"filter":"sketch"}` | 20 KB+ result truncated again | 4,431 chars, 118 matches, ids and labels |
| Repeated identical call | another 200 KB transfer and another truncation | 2 KB replay (`replay`, `api_client.rs:272`) |

The local answer is bounded by `LOCAL_COMMANDS_CHARS = 16_000` (`api_client.rs:49`), comfortably
under the 20,000-char cap, so the model sees the whole answer. It also matches the tool description's
own advice — filtered lookup for parameter docs — instead of contradicting it with a truncated blob.

### 5.2 A repeated read-only call is replayed, not re-run

`Worker::remember` / `Worker::replay` (`api_client.rs:272-285`) keep the first successful answer of
`guide`, `list_commands`, `inspect_design`, `measure` and `body_topology` for the turn, keyed by
tool name plus canonical arguments. A second identical call gets a 2,000-character replay prefixed
with "Same call as earlier in this turn, whose answer still holds". The map is cleared when a
command actually ran (`self.seen.clear()`, `api_client.rs:460`), because then a read could genuinely
return something new, and on a new chat.

This removes the bridge transfer, the TOON re-encoding and the prompt growth for a duplicate. It
does **not** remove the model round trip — only the model can decide to stop asking — which is what
the next change is for.

### 5.3 A turn stops after three rounds without progress

The observed loop would now end at the third repeated round instead of waiting out the 24-request
cap. `stall` (`api_client.rs:732`) counts consecutive rounds in which every tool call was a replay
and no command ran; the third ends the turn with the localized `ui.ai_no_progress` note
(`api_client.rs:471`) and `TurnEnded { reason: "no_progress" }` (`api_client.rs:472`).

At 92 s per round trip this turns a worst-case 36-minute hang into a ~5-minute stop with an
explanation. A fresh lookup (a new filter, a new read) counts as progress and resets the counter, so
legitimate exploration is unaffected.

### 5.4 What the changes are worth, stated honestly

- **Measured, deterministic:** the lookup answer grows from 2.5 KB covering 5 ids to 7.6 KB covering
  401; the bridge call and its 200 KB payload disappear; a duplicate read costs 2 KB of prompt
  instead of up to 20 KB.
- **Expected, not measured live:** the five-call loop becomes one call, saving four round trips.
  At the session's 92 s first-token latency that is ~6 minutes; at a fast provider it is seconds.
- **Not changed:** first-token latency, output tokens, prompt caching, panel rendering. Those are
  §6.

Verification: `cargo test -p solidcraft-ui-egui --lib -- agent_chat` → **132 passed, 0 failed**
(four new tests: local lookup bounds, catalog row extraction, replay behaviour, stall counting).
`cargo clippy -p solidcraft-ui-egui --lib -- -D warnings` still reports two lints that predate
this change and are outside it: `api_client.rs:1033` (`% 4` in the existing base64 check, part of
the uncommitted chat work) and `dialogs_motion.rs:35` (a large `Mo` variant, untouched here).
No live provider session was run for this document.

---

## 6. What is left, ranked

### L1 — First-token latency is the wall (High, provider choice)

91.93 s to the first text token is a property of the model and the free endpoint, not of
SolidCraft. Nothing in the harness can stream text the provider has not produced. Options, in
order of leverage:

1. Make the first-token number impossible to miss when choosing a model (it is already measured;
   surface it in the model list, not only inside a finished request card).
2. Offer a per-connection *reasoning effort* setting for endpoints that support it. The Gemini 3/4
   path already sends `thinking_level: "low"` (`api_client.rs:525`); OpenRouter-compatible
   endpoints get nothing. This is a preference, not a default.
3. Keep a local-model option visible (Ollama/LM Studio presets exist) for users who prefer 2 s first
   tokens over a free remote model at 90 s.

### L2 — Prompt size per round trip (Medium, low risk)

36 KB of static prefix is re-sent every round trip. The provider cache absorbs most of it (9,719 of
10,356 tokens [O]), so the cost is mostly cache-read pricing, not wall time. If it needs to shrink:

- The command index's **labels** are 5,744 chars of the 19 KB [M]. An ids-only index is about
  8 KB (~2,000 tokens) and still lets the model name a command; labels are available through the
  new filtered lookup. Trade-off: the model recognizes commands by label more reliably.
- Do not remove the index outright: a missing index moves one lookup into every turn.

### L3 — Tool-result size in history (Medium, needs care)

Every result stays in history and is re-sent in later rounds, up to 20,000 chars each
(`MAX_TOOL_TEXT`), and history keeps up to 192,000 chars (`api_client.rs:30-31`). The measured
`inspect_design` snapshot is small on an empty design (286 chars) but grows with bodies, sketches
and timeline length. The verification snapshot asks for `measure:true` after every editing round,
which re-measures every body. For a 1,395-body design that is the expensive inspection in the
application, repeated per round.

A targeted improvement: let `inspect_design` take the body names that changed (the command result
already lists `changed_bodies`) and measure only those, or accept `bodies: [...]`. That is an MCP
schema change with tests in `crates/mcp/src/tests.rs`, so it is a deliberate follow-up rather than
part of this fix.

### L4 — Panel work while a long answer streams (Low–Medium, measurable)

`markdown_view::blocks` (`crates/ui-egui/src/agent_chat/markdown_view.rs:20-33`) hashes the whole
answer every frame and re-parses it whenever the text changed. While streaming, the text changes
every frame, so an 11,768-token answer (~50 KB [E]) is re-parsed at frame rate. At an assumed
1–3 ms per parse of a 50 KB answer, 30 s of streaming is 0.5–3 s of one core — real, but two orders
of magnitude below the model wait. A prefix-aware or time-throttled parse would remove it; it is not
worth the staleness risk until profiling shows it next to L1.

### L5 — Double reserialisation per round trip (Low)

`trim_history` and `input_estimate` serialise the whole history to measure it, once per round trip
(`api_client.rs:542`, `api_client.rs:742`). At the 192 KB cap that is ~0.2 ms each; leave it unless
the cap grows.

### L6 — Startup and connection (Low, already fine)

MCP server spawn → `initialize` measured **15–26 ms**, `tools/list` a further **5–6 ms** [M]. Model
discovery already runs on its own thread (`api_client.rs:195`). Nothing to do.

### L7 — Autosave during a long agent turn (Medium, cross-reference)

An editing turn can last minutes, which makes the five-minute autosave fire mid-turn. That path is
the window-thread deep copy described in the deep dive (D7): the UI stalls while the agent is
already slow. Moving autosave to the existing off-thread save job is the same fix and benefits the
sidebar most, because agent turns are the longest-running edits.

---

## 7. A budget the harness could assert

| Metric | Today | Proposed budget |
|---|---:|---|
| Static prefix per request | 36,772 chars [M] | ≤ 40,000 chars |
| Largest tool result stored | 20,000 chars cap | unchanged, but never truncate a *lookup* |
| Duplicate read-only calls that reach the bridge | 1 per repeat | 0 |
| Rounds a turn may repeat itself | 24 | 3 (implemented) |
| Turn wall clock at 90 s first token | up to 36 min | ≤ 5 min before the stall stop |
| Prompt tokens first request | 10,356 [O] | ≤ 12,000 |

These are cheap to test at the unit level (payload sizes, replay and stall counters) and are now
covered except for the prefix size, which `input_estimate` already computes per request.

---

## 8. What would have caught this earlier

1. **Log the tool result size before and after the cap** next to the tool card, and the truncation
   count for the turn. A single line — `203,555 → 2,500 chars, truncated` — would have pointed at
   the defect immediately.
2. **Assert that a lookup tool's result fits its consumer's cap.** The MCP server cannot know about
   the chat's 20 KB cap, but the chat backend can notice `truncated: true` in its own envelope and
   answer differently instead of handing the model a dead end (now it never produces the envelope
   for `list_commands`).
3. **A no-progress counter in the timing card.** The panel already counts changes and requests;
   counting consecutive no-change rounds would have shown `stalled 5` while the wall clock grew.

---

## 9. Reproducing the measurements

All commands run from the repository root, with the debug binaries already built. The scratch
outputs live under `target/ai-perf/` (ignored by git).

### 9.1 Catalog size and the index the harness builds

```bash
./target/debug/solidcraft-cli.exe commands > target/ai-perf/cmds.json
node -e '
const fs=require("fs");
const cmds=JSON.parse(fs.readFileSync("target/ai-perf/cmds.json","utf8"));
const common=["solid.box","solid.cylinder","solid.sphere","solid.extrude","solid.revolve",
  "sketch.create","sketch.finish","sketch.circle.center","sketch.rectangle.two_point","parameters.change"];
let text="Command index (id: label; parameter docs where included). For other parameters use list_commands filtered to the exact id.\n";
for(const c of [...cmds].sort((a,b)=>a.id<b.id?-1:1)){
  const params=common.includes(c.id)?(c.params||""):"";
  const line=`${c.id}: ${c.label||""}${params?"; ":""}${params}\n`;
  if(text.length+line.length>96000) break;
  text+=line;
}
console.log(cmds.length,"commands; index",text.length,"chars");
'
```

### 9.2 The MCP catalog answer, its timing and the truncation preview

```bash
node -e '
const {spawn}=require("child_process");
const p=spawn("./target/debug/solidcraft-cli.exe",["mcp"],{stdio:["pipe","pipe","ignore"]});
let out="";
p.stdout.on("data",d=>out+=d);
const send=o=>p.stdin.write(JSON.stringify(o)+"\n");
send({jsonrpc:"2.0",id:1,method:"initialize",params:{protocolVersion:"2025-06-18",capabilities:{},clientInfo:{name:"perf",version:"0"}}});
send({jsonrpc:"2.0",method:"notifications/initialized"});
setTimeout(()=>{
  const t=Date.now();
  send({jsonrpc:"2.0",id:2,method:"tools/call",params:{name:"list_commands",arguments:{}}});
  const iv=setInterval(()=>{
    const line=out.split(/\r?\n/).find(l=>l.includes("\"id\":2"));
    if(line){clearInterval(iv);
      const text=JSON.parse(line).result.content[0].text;
      console.log("list_commands",Date.now()-t,"ms",text.length,"chars");
      console.log("preview ids:",(text.slice(0,2500).match(/"id":"([^"]+)"/g)||[]).length);
      p.kill();}
  },10);
},1000);
'
```

Expected: ~200,000 chars in ~35 ms, and 5 ids in the first 2,500 chars.

### 9.3 Tool schema payload

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"perf","version":"0"}}}' > /tmp/mcp-init.txt
# then tools/list through the same stdio loop as above and sum JSON.stringify(tool).length
```

### 9.4 A live turn

Use the control channel (`docs/control-protocol.md`) against a running app with the sidebar open,
send one message, and read the request card: first response, first token, input estimate, cache
reference, provider tokens. Compare the tool card's result size with §2 to see whether the local
lookup path was used (a local lookup has no MCP duration and returns the full id list).

---

## 10. Bottom line

The sidebar's remaining slowness is not a rendering or serialisation problem. It is a **tool-loop
problem**: a 200 KB catalog was cut to a five-command preview by a 20 KB cap, the model retried, and
each retry costs the provider's first-token latency (92 s in the reported session). The harness now
answers catalog lookups from the catalog it already holds (7.6 KB, all 401 ids), replays duplicate
reads, and stops a turn after three rounds without progress. That removes the observed five-call
loop and bounds the worst case at roughly five minutes instead of thirty-seven.

The largest remaining lever is not in this repository: choose an endpoint whose first token arrives
in seconds, and the same workflow becomes interactive without touching a line of the harness.
