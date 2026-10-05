# AGENTS.md

An MCP server that exposes Mu2e analyses as agent tools: `list_analyses` and
`run_analysis`. An analysis is either a `mu2e` job over art file(s), with its
stdout parsed, or a Python computation over a ROOT file. README.md is the
full reference. This file holds what you need before changing code.

## Setup, test, run

Use the `ana` python. No installs are needed, and there is no pytest:

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana                       # Python 3.12 with mcp, pydantic, uproot, awkward
python3 tests/test_tools.py     # bare asserts; prints ok/FAIL per test, "0 failure(s)"
python3 -m analysis_mcp_server --transport stdio [--work-area DIR | --musing 'SimJob MDC2025ay' | --code-tarball FILE]
```

- The tests never start a mu2e job. Keep it that way: test parsers and
  computations on sample stdout or synthetic ROOT files written with
  `uproot`.
- Write a synthetic tree with `f.mktree(...).extend(...)`. Plain
  `f["x"] = dict` writes an RNTuple with this uproot.
- To check a change against real data, run `run_analysis` with `max_events`
  into a scratch `output_dir`. A real job runs `mu2e` in that directory, and
  `output/` is gitignored.
- These are shared interactive nodes. Set `OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1` for anything you run, and never fan out many jobs.
- `.mcp.json` and `DEFAULT_WORK_AREA` in `tools/mu2e_env.py` point at the
  original author's work area. Pass your own `--work-area` or `--musing`
  rather than editing them for local use.

## Architecture rules

- **`tools/` never imports MCP.** It is ordinary science code.
  `analysis_mcp_server/` is a generic wrapper that registers every name in
  `tools/__init__.py`'s `__all__` as a tool. Do not add names to `__all__`:
  new analyses go through `run_analysis`, so the tool schema stays the same
  however many analyses exist.
- **One analysis = one module** in `tools/analyses/` that defines `SPEC` (an
  `AnalysisSpec`, see `tools/spec.py`), plus an entry in `_ANALYSIS_MODULES`
  in `tools/registry.py`. The registry tests loop over every analysis and
  fail until it is covered: add a `SAMPLE_STDOUT` entry for an `art_files`
  analysis, or direct tests for a `root_file` one.
- **Physics knobs are `ParamSpec`s**, not tool arguments. They are validated,
  have defaults and ranges, and `list_analyses` reports them. Their
  descriptions are what agents read, so write them for an agent.
- **Shared machinery knows nothing about physics.** That covers
  `mu2e_job.py` (running jobs), `mu2e_env.py` (where Offline comes from),
  `spectrum.py` (uniform-binned histograms: rebin, regrid, smear) and
  `selection.py` (cut expressions). Put physics in the analysis modules.
- **Errors are worded for the caller.** A runner returns
  `RunOutcome(error=...)` with a message an agent can act on, such as what
  is missing and what produces it. It never raises a traceback.
  `analysis_tools.run_analysis` wraps the result in the uniform
  `ArtifactResult`.

## Things that are easy to get wrong

- **Metric names are an API.** Downstream code relies on them, for example
  the Mu2eBO workflow uses `edep`'s `n_events_calo_edep_above_50mev`. Add
  new metrics rather than renaming or removing existing ones, and keep
  `metrics`, `units` and the summarizer in step.
- **The fcl ships here, in `fcl/`.** Analyses name it as `FCL_DIR / "x.fcl"`
  (absolute), so it is the same file under every environment. Its
  `#include`s resolve through art's `FHICL_FILE_PATH`.
- **EdepAna is in Offline from v13_39_00** (`SimJob MDC2025ay` and later).
  An environment on an older Offline cannot run `edep`.
- **Selections over the EdepAna tree** (`EDepAna/tree`) go through
  `read_edep_tree` / `select_events` / `EDEP_VARIABLES` in
  `tools/analyses/edep.py`, so every analysis of EdepAna output uses the same
  variables and semantics:
  - Per-primary variables are the first primary's.
  - A missing value is NaN, so any cut on it fails.
  - A new variable belongs in `EDEP_VARIABLES` and `edep_variables`.
- **A changed selection default must still reproduce the old fixed cut.**
  Today `edep` defaults to `event_calo_edep_vis > 50` and
  `approx_ce_sensitivity` to `event_calo_edep_vis > 10` (the module's
  `hist_2` set). If you change how a result is computed, compare it on real
  data with the old code at the default settings.
- **Selection expressions come from agents.** `tools/selection.py` parses
  them with `ast` and walks a whitelist. Never `eval` them, and extend the
  whitelist only with pure numpy functions.
- **The trigger analyses share one job** (`tools/analyses/trigger.py`): it
  copies `fcl/trigger.fcl` into a per-run fcl, because FHiCL cannot
  `#include` an absolute path. Prescales go into each path's
  `PrescaleEvent` filter, whose label is derived from the path name by the
  menu generator's convention (`prescale_module`). Timing comes from the
  TimeTracker sqlite database, not stdout: an event's time is the sum of its
  module times without the data fetch (`FETCH_MODULE_TYPES`, i.e. Prefetch;
  the source is timed apart), and the first event (DB setup) is skipped.
- **Default inputs.** An `AnalysisSpec` with `default_inputs` runs without
  `data_file(s)`; the trigger rate and timing default to
  `mu2e-trig-config/ci/data_files.txt`, found with `Mu2eEnv.find_code_file`
  down the configured code's `backing` chain.
- **`approx_ce_sensitivity` has a numerics trap.** Window sums run outward
  from each window's own edge, never as differences of prefix sums: the DIO
  spectrum spans ~18 orders of magnitude, and cancellation silently zeroes
  the background. A test locks this in.

## Style

Match the surrounding code: docstrings and comments explain *why*, constants
are named at the top of a module with a short note on where they come from,
and messages are plain sentences. When behaviour changes, update the
README.md section for that analysis, including the test count under "Test".
