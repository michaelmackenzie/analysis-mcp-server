# analysis-mcp-server

An example MCP server that exposes Mu2e analyses as agent tools. An agent asks
which analyses exist, then runs one — either a mu2e job over art file(s):

```bash
mu2e -c <the analysis' fcl> -s <data file>     # one file
mu2e -c <the analysis' fcl> -S <file list>     # several, one path per line
```

or a Python computation over a ROOT file an earlier analysis produced. Two
analyses ship: energy deposition (`edep`) and approximate CE sensitivity
(`approx_ce_sensitivity`). Adding more is one small module each.

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
  spec.py             AnalysisSpec, ParamSpec, RunContext/RunOutcome, ArtifactResult
  mu2e_job.py         running mu2e: env setup, -s/-S inputs, logs, timeouts
  root_hist.py        a small TH1-like histogram for converted ROOT macros
  registry.py         the catalogue: name -> AnalysisSpec
  analyses/
    edep.py                    energy deposition: fcl + summary parser
    approx_ce_sensitivity.py   CE sensitivity from EdepAna histograms
  analysis_tools.py   the MCP tools: list_analyses, run_analysis
  __init__.py         __all__ — ONLY these names become tools
analysis_mcp_server/  generic drop-in wrapper (FastMCP): server.py, cli.py
tests/test_tools.py   parsers, registry, input handling — no mu2e, no MCP
```

The split that matters: **the shared machinery (`mu2e_job.py`, `root_hist.py`)
knows how to run jobs and read histograms but nothing about physics;
`analyses/*.py` knows the physics.** The registry joins them, and the two tools
are generic over it.

Each analysis declares an `input_kind`, which is the only thing the generic
layer needs to know about the difference between them:

| input_kind | what it consumes | what its runner does |
|---|---|---|
| `art_files` | mu2e art file(s) | runs an fcl with `mu2e`, parses the job's stdout |
| `root_file` | a ROOT file from an earlier analysis | Python computation over its histograms |

## Tools

| tool | what it does |
|---|---|
| `list_analyses()` | catalogue: description, `input_kind`, metric names + units, parameters and their defaults, what the input must contain |
| `run_analysis(analysis, output_dir, data_file=None, data_files=None, parameters=None, max_events=None, timeout_s=900)` | run one analysis, return its metrics |

Both return the same `{status, files, message, metadata}` contract, so a
workflow can chain several runs and collect `metadata` uniformly.

### The analyses

| analysis | input | reports |
|---|---|---|
| `edep` | art file(s) | average calo/tracker energy deposition per event and per generated event |
| `approx_ce_sensitivity` | `nts.*.root` from `edep` | `S/sqrt(B)` for the best momentum window, with the window and its signal/DIO/cosmic counts |

`approx_ce_sensitivity` declares `produced_by = ["edep"]`, so chaining is
discoverable: run `edep`, then pass the `nts.*.root` from its `files` to the
sensitivity.

### Parameters

Analyses declare their own physics knobs, passed as `parameters`:

```python
run_analysis(analysis="approx_ce_sensitivity",
             data_file=".../nts.owner.edep.Run1B.001800_00000000.root",
             output_dir=".../sens",
             parameters={"sig_eff": 0.1})     # npot defaults to 1e18
```

`list_analyses` reports each parameter's description, default, range, and
whether it is required; unknown, missing, or out-of-range values come back as
a plain error naming the offender.

### Inputs

Pass **exactly one** of:

- `data_file` — a single absolute art file path → `mu2e -s <file>`
- `data_files` — a list of absolute art file paths → written one per line to
  `filelist.txt` in `output_dir` → `mu2e -S <filelist.txt>`

With `data_files` the whole set runs as **one** job, so the metrics cover all
the inputs together, not one file each — call the tool once per file for
per-file numbers. Raise `timeout_s` (max 7200) when passing many files.
`data_files` and `max_events` apply only to `art_files` analyses.

`max_events` (mu2e `--nevts`) caps events for a quick check before a full run.
Beware: generated-event counts come from the input's subrun bookkeeping and
cover the whole file regardless, so **any "per gen event" metric is
meaningless when `max_events` is set** — use it to confirm a job runs, not for
physics numbers.

### Results

Every result carries `analysis`, `input_kind`, `data_files`, `n_input_files`,
`log_path`, the resolved `parameters`, and — for the `-S` case —
`file_list_path`. On success the analysis' metrics are merged into `metadata`
under the names `list_analyses` advertises. `files` lists what the run wrote:
the job's ROOT output for `edep`, the figures for `approx_ce_sensitivity`. On
failure `status="error"`, and for mu2e jobs `metadata.stdout_tail` holds the
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

`approx_ce_sensitivity` reports `sensitivity` (S/sqrt(B)),
`signal_box_low_mev` / `signal_box_high_mev`, `signal_rate`,
`dio_background`, `cosmic_background`, `total_background`, and the signal
peak's `signal_mpv_mev` / `signal_fwhm_mev`. Its `metadata` also records the
assumptions used (`npot`, `sig_eff`, `signal_br`, `cosmic_rate_per_mev`,
`onspill_seconds`), and it writes the macro's figures — `sig_vs_bkg.png`,
`dio.png`, `response.png`, `res.png`, `ce_z.png`, `ce_r.png` — into
`<output_dir>/figures`.

## approx_ce_sensitivity

A Python conversion of `Mu2eOptAna/scripts/rough_run1a_sensitivity.C`. Signal
is CE, background is DIO plus cosmics, and it estimates S/sqrt(B) for the best
momentum window:

1. **Signal shape** — `EDepAna/hist_2/trk_front_energy` (energy at the front of
   the tracker for events leaving >10 MeV in the calorimeter), rebinned x2 and
   scaled to a rate for `npot` protons at `SIGNAL_BR` (R_mue = 1e-9) and
   `sig_eff`, then smeared by a Gaussian tracker resolution (sigma = 0.2 MeV).
2. **DIO** — the Heeck/Szafron theoretical spectrum, scaled to a rate, then
   smeared by the *measured* energy-loss response
   (`hist_2/trk_front_energy_diff`) and the same resolution.
3. **Cosmics** — flat in momentum at a rough rate per MeV/c, scaled by the
   on-spill live time implied by `npot`.
4. **Window scan** — every `[x1, x2]` with `x1 >= 50 MeV`, keeping the best
   S/sqrt(B). The top 10 windows go to the log.

The numbers are rough by construction: this is a figure of merit for comparing
beamline configurations, not a sensitivity calculation. Note it needs a **CE
signal** sample — given a beam file where nothing leaves >10 MeV in the
calorimeter it reports that plainly instead of dividing by zero.

Two deviations from the macro, both deliberate:

- It prints the 10 best windows to the log rather than all ~100k scanned.
- Window sums are accumulated from each window's own edge, never as
  differences of whole-spectrum prefix sums. The DIO spectrum spans ~18 orders
  of magnitude, and differencing totals of ~4e17 to get a count of order 1
  loses it entirely to float cancellation (it silently reported
  `dio_background = 0`). `tests/test_tools.py` locks this in.

## Adding an analysis

1. Write `tools/analyses/<name>.py` with a `run(context) -> RunOutcome` and a
   `SPEC`. For a mu2e job, hand `run_mu2e_job` your fcl and parse its stdout:

```python
def run(context: RunContext) -> RunOutcome:
    outcome = run_mu2e_job(fcl=FCL, input_paths=context.input_paths,
                           outdir=context.outdir, timeout_s=context.timeout_s,
                           single=len(context.input_paths) == 1
                                  and not context.wants_file_list,
                           max_events=context.max_events)
    if outcome.failed:
        return RunOutcome(error=f"mu2e exited {outcome.returncode}",
                          log_path=outcome.log_path)
    return RunOutcome(metrics=parse_stops_summary(outcome.stdout),
                      files=outcome.new_root_files, log_path=outcome.log_path)

SPEC = AnalysisSpec(
    name="stops",
    input_kind="art_files",
    fcl=MUSE_WORKAREA / "Mu2eOptAna" / "fcl" / "stops.fcl",
    description="Muon stops per POT.",
    metrics=("n_stops", "stops_per_pot"),
    units={"stops_per_pot": "stops / POT"},
    parameters=(ParamSpec(name="pot_per_event", description="...",
                          default=1.6e7, minimum=0.0),),
    run=run, summarize=lambda m: f"{m['n_stops']:g} stops",
    input_hint="art file(s) with ...",
)
```

   For a ROOT-file analysis use `input_kind="root_file"`, name the analysis
   that produces the input in `produced_by`, read `context.input_path`, and
   leave `fcl` unset — `approx_ce_sensitivity.py` is the worked example.

2. Add `"stops"` to `_ANALYSIS_MODULES` in `tools/registry.py`.
3. Extend `tests/test_tools.py` (a stdout sample in `SAMPLE_STDOUT` for an
   `art_files` analysis; direct tests of the computation for a `root_file` one).

That's it — the `analysis` enum, `list_analyses`, and the registry tests pick
it up. Step 3 is not optional in practice: the registry tests loop over every
analysis and fail until each one is covered.

Physics knobs go in `parameters` as `ParamSpec`s (validated, with defaults and
ranges, and reported by `list_analyses`) rather than into the tool signature,
which stays the same however many analyses exist.

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

36 tests, none of which start a mu2e job. (The `ana` env has no pytest, so
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

Chaining the two analyses, which is what `produced_by` is for:

```
Run the edep analysis on <a CE signal art file>, then feed the nts.*.root it
produces to approx_ce_sensitivity with sig_eff = 0.1, and tell me the best
momentum window and its S/sqrt(B).
```
