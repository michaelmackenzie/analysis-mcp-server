# analysis-mcp-server

An example MCP server that exposes Mu2e analysis jobs as agent tools. An agent
asks which analyses exist, then runs one over art file(s):

```bash
mu2e -c <the analysis' fcl> -s <data file>     # one file
mu2e -c <the analysis' fcl> -S <file list>     # several, one path per line
```

Energy deposition (`EdepAna`) is the first analysis; the server is structured
so more are one small module each.

Built to the same pattern as
[`spectra-mcp-server`](https://github.com/HEP-KE/spectra-mcp-server), so the
`multiagent-client-demo` client in `../multiagent-client-demo` (or Claude
Code, Claude desktop, Codex, Cursor) can drive it unchanged.

## The one idea

> **The science code stays in usual Python. The MCP wrapper only publishes it.**

- `tools/` is an ordinary package that never imports MCP.
- `analysis_mcp_server/` is a generic wrapper: it reads one line of config from
  `pyproject.toml`, imports the science package, and registers every function
  in its `__all__` as an MCP tool.

```toml
[tool.mcp-server]
tool_modules = ["tools"]
```

Type hints, Pydantic `Field` constraints, and docstrings become the tool
schema agents see.

## Layout

```
tools/
  spec.py             AnalysisSpec (what an analysis provides) + ArtifactResult
  mu2e_job.py         running mu2e: env setup, -s/-S inputs, logs, timeouts
  registry.py         the catalogue: name -> AnalysisSpec
  analyses/
    edep.py           energy deposition: fcl + summary parser
  analysis_tools.py   the MCP tools: list_analyses, run_analysis
  __init__.py         __all__ — ONLY these names become tools
analysis_mcp_server/  generic drop-in wrapper (FastMCP): server.py, cli.py
tests/test_tools.py   parsers, registry, input handling — no mu2e, no MCP
```

The split that matters: **`mu2e_job.py` knows how to run a job but nothing
about physics; `analyses/*.py` knows the physics but never runs anything.**
The registry joins them, and the two tools are generic over it.

## Tools

| tool | what it does |
|---|---|
| `list_analyses()` | catalogue of analyses: description, fcl, metric names + units, what the input must contain |
| `run_analysis(analysis, output_dir, data_file=None, data_files=None, max_events=None, timeout_s=900)` | run one analysis, return its parsed metrics |

Both return the same `{status, files, message, metadata}` contract, so a
workflow can chain several runs and collect `metadata` uniformly.

### Inputs

Pass **exactly one** of:

- `data_file` — a single absolute art file path → `mu2e -s <file>`
- `data_files` — a list of absolute art file paths → written one per line to
  `filelist.txt` in `output_dir` → `mu2e -S <filelist.txt>`

With `data_files` the whole set runs as **one** job, so the metrics cover all
the inputs together, not one file each — call the tool once per file for
per-file numbers. Raise `timeout_s` (max 7200) when passing many files.

`max_events` (mu2e `--nevts`) caps events for a quick check before a full run.
Beware: generated-event counts come from the input's subrun bookkeeping and
cover the whole file regardless, so **any "per gen event" metric is
meaningless when `max_events` is set** — use it to confirm a job runs, not for
physics numbers.

### Results

Every result carries `analysis`, `fcl`, `data_files`, `n_input_files`,
`log_path`, and — for the `-S` case — `file_list_path`. On success the
analysis' metrics are merged into `metadata` under the names `list_analyses`
advertises. `files` lists any ROOT file the job's TFileService wrote into
`output_dir`. On failure `status="error"` and `metadata.stdout_tail` holds the
last 20 log lines, so an agent can diagnose without re-running.

`edep` reports:

| metric | unit | from the module's print |
|---|---|---|
| `n_events` | | `Saw <N> events` (weighted, so a float) |
| `n_gen_events` | | `(<N> gen events)` |
| `event_rate` | events / gen event | `output rate = <N>` |
| `avg_calo_edep_per_event_mev` | MeV | `Average calo energy deposition per event` |
| `avg_calo_edep_per_gen_event_mev` | MeV | `... per gen event` |
| `n_events_calo_edep_above_50mev` | | `Events with calo Edep > 50 MeV` |
| `avg_trk_edep_per_event_mev` | MeV | `Average tracker energy deposition per event` |
| `avg_trk_edep_per_gen_event_mev` | MeV | `... per gen event` |

## Adding an analysis

1. Write `tools/analyses/<name>.py` defining a `SPEC`:

```python
SPEC = AnalysisSpec(
    name="stops",
    fcl=MUSE_WORKAREA / "Mu2eOptAna" / "fcl" / "stops.fcl",
    description="Muon stops per POT.",
    metrics=("n_stops", "stops_per_pot"),
    units={"stops_per_pot": "stops / POT"},
    parse=parse_stops_summary,     # stdout -> {metric: value} or None
    summarize=lambda m: f"{m['n_stops']:g} stops",
    input_hint="art file(s) with ...",
)
```

2. Add `"stops"` to `_ANALYSIS_MODULES` in `tools/registry.py`.
3. Add a sample stdout to `tests/test_tools.py`.

That's it — the `analysis` enum, `list_analyses`, and the registry tests pick
it up. Step 3 is not optional in practice: the registry tests loop over every
analysis and fail with *"add a sample stdout to this test"* until you do.

If an analysis needs a knob the current signature can't express (a threshold,
a different collection), add it to `AnalysisSpec` and thread it through
`run_mu2e_job` — that is the seam meant for it.

## Environment

No installs needed on the mu2e machines — the `ana` python already has the
`mcp` SDK:

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana            # Python 3.12 with mcp + pydantic
```

The tool sets up mu2e **itself**, once per job, in a fresh bash subprocess:

```bash
cd /exp/mu2e/app/users/mmackenz/mu2eopt/   # the muse work area
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
muse setup                                  # backing -> Musings/SimJob/Run1Baq
mu2e -c <fcl> -s <data file>
```

so the server does not care whether the shell that launched it had the
Offline environment. `MUSE_WORKAREA` in `tools/mu2e_job.py` is the one path
that encodes this.

> **Why `muse setup` must run in the work area**: `EdepAna` is a locally built
> module (not in the Offline release yet). Its library comes from
> `build/al9-prof-e29-p103/Mu2eOptAna/lib/`, which only lands on
> `CET_PLUGIN_PATH` when `muse setup` runs in the area holding `backing`.
> Run it anywhere else and art dies with
> `Library specification "EdepAna" does not correspond to any library`.

## Test

```bash
python3 tests/test_tools.py
```

18 tests, none of which start a mu2e job. (The `ana` env has no pytest, so
these are bare asserts.)

## Run the server

**stdio** — the client spawns the server; nothing to start by hand:

```bash
python3 -m analysis_mcp_server --transport stdio
```

**Streamable HTTP** — the server is a visible process with a URL:

```bash
python3 -m analysis_mcp_server --transport streamable-http --port 8000
```

Clients connect to `http://127.0.0.1:8000/mcp`. Stop it with **Ctrl+C**
(Ctrl+Z only suspends it and keeps the port; just start again — a leftover
`analysis_mcp_server` holding the port is cleared automatically).

## Use it from a client

**Claude Code** — the checked-in `.mcp.json` already wires it up; or:

```bash
claude mcp add mu2e-analysis -- python3 -m analysis_mcp_server --transport stdio
```

**`multiagent-client-demo`** (`../multiagent-client-demo`), stdio — the
client spawns the server:

```python
CONFIG = {
    "mu2e-analysis": {
        "transport": "stdio",
        "command": "python3",
        "args": ["-m", "analysis_mcp_server"],
        "cwd": "/exp/mu2e/app/users/mmackenz/mu2eopt/analysis-mcp-server",
    }
}
tools = await load_tools(CONFIG)
```

or over HTTP, against a server started as above:

```python
CONFIG = {"mu2e-analysis": {"transport": "streamable_http",
                            "url": "http://127.0.0.1:8000/mcp"}}
```

(Note the spelling: the adapter config says `streamable_http`, the server CLI
says `--transport streamable-http`.)

A task for the agent, once connected:

```
List the available analyses, then evaluate the average energy deposition in
the detectors for
/exp/mu2e/app/users/mmackenz/mu2eopt/dts.mmackenz.EarlyMuBeamFlash.Run1Bak_local0818120248.001800_00000000.art
writing outputs to /exp/mu2e/data/users/mmackenz/localtest/agent-output.
Report the average calo and tracker Edep per event and per generated event.
```
