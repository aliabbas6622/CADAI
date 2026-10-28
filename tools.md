# AI assistant tools

The desktop AI sidebar gives its agent SolidCraft's MCP server connected to the open design. These are the tools the agent can use. `execute` and `batch` also reach the full engine command registry; call `list_commands` to discover command IDs and parameters.

| Tool | Arguments | Purpose |
|---|---|---|
| `guide` | `topic?`: `overview`, `model`, `edit`, `verify` | Read a short modeling, editing, or verification workflow. |
| `calculate` | `expressions`: 1–20 arithmetic or unit expressions | Get local results in mm, radians, or derived units without changing the design. |
| `list_commands` | `filter?` | Find command IDs, parameter docs, and whether each command is enabled. |
| `execute` | `command`, `params?`, `verbose?` | Run one engine command. Use `verbose: false` for a compact reply. |
| `batch` | `commands`, `rollback?`, `verbose?`, `timings?`, `profile?` | Run commands in order. By default, a failed step rolls back earlier steps in the batch. |
| `inspect_design` | `measure?`, `sketches?` | Inspect parameters, timeline, bodies, sketches, selection, and undo depth. For a compact overview, use `measure: false, sketches: false`. |
| `measure` | `body?` or `items?` | Measure bodies or selected geometry: volume, area, distances, centres, and bounds. |
| `body_topology` | `body` | List a body's edges and faces with points for selecting them. |
| `set_parameter` | `name`, `value` or `expression`, `unit?`, `comment?` | Create or change a parameter and recompute the design. |
| `screenshot` | `view?`, `camera?`, `path?`, `width?`, `height?`, `source?`, `grid?`, `axes?` | Get a PNG of the app window or a model render. `path` also writes the image to disk. |
| `export` | `path`, `format?`, `bodies?`, `mesh_prefix?`, `effort?`, `velocity?` | Export STEP, 3MF, STL, OBJ, or URDF. |
| `undo` | none | Undo one design change. |
| `redo` | none | Redo one design change. |
| `new_design` | `name?` | Start an empty design, replacing the current one. |
| `open` | `path`, `up?`, `names?` | Open a design or supported CAD/mesh file, replacing the current design. |
| `save` | `path?`, `format?` | Save the open design. |

The sidebar's permission gate can ask before file actions, design replacement, and edits, depending on the user's settings. `crash_reports` is not usable from the sidebar: the API backend hides it and the ACP bridge refuses it. A separately started `solidcraft-cli mcp` server offers it for local diagnostics (`id?`, `limit?`).

The engine has many more commands than the tool table. For example, `execute` can run `sketch.create`, `solid.extrude`, `timeline.inspect`, and `parameters.change`. The registry returned by `list_commands` is the current source for all IDs and JSON parameters. See [MCP documentation](docs/mcp.md) and [Agent chat](docs/agent-chat.md) for examples and access rules.
