# analysis-mcp-server

An example MCP server that exposes Mu2e analyses as agent tools. An agent asks
which analyses exist, then runs one — either a mu2e job over art file(s):

```bash
mu2e -c <the analysis' fcl> -s <data file>     # one file
mu2e -c <the analysis' fcl> -S <file list>     # several, one path per line
```

or a Python computation over a ROOT file. Nine analyses ship: energy
deposition (`edep`), event counts (`count`), muon stopping rate
(`muon_stop_rate`), approximate CE sensitivity (`approx_ce_sensitivity`),
muon stops per material (`stop_materials`), and the online trigger's
efficiency, rate and processing time (`trigger_efficiency`, `trigger_rate`,
`trigger_timing`), and the trigger efficiency read from EventNtuples
(`trigger_efficiency_ntuple`). Adding more is one small module each.

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
  mu2e_env.py         where Offline comes from: work area, Musing, or tarball
  mu2e_job.py         running mu2e: -s/-S inputs, logs, timeouts
  spectrum.py         bin contents on a uniform grid: rebin, regrid, smear
  selection.py        event selections: a safe cut expression over arrays
  registry.py         the catalogue: name -> AnalysisSpec
  analyses/
    edep.py                    energy deposition: fcl + summary parser
    count.py                   events, gen events, optional prescale + the
                               parsing/job muon_stop_rate builds on
    muon_stop_rate.py          stopping rate: count.py + POT scaling
    approx_ce_sensitivity.py   CE sensitivity from the EdepAna tree
    stop_materials.py          stops per material from <module>/stopmat
    trigger.py                 the trigger job (paths + prescales) and its
                               parsing, shared by the three below
    trigger_efficiency.py      fraction of signal events triggered
    trigger_rate.py            trigger rate on pileup, overall and per path
    trigger_timing.py          trigger processing time per event on pileup
    trigger_efficiency_ntuple.py  trigger efficiency from EventNtuple
                               trig_<path> branches, for a track selection
    fullsim/                   full-simulation analyses of EventNtuple
      eventntuple.py           reading EventNtuple/ntuple; MC origin of a track
      cuts.py                  the CE-like track cuts (pyfitter cut-set 80)
      sensitivity.py           fullsim_sensitivity: S/sqrt(B) in the best window
  analysis_tools.py   the MCP tools: list_analyses, run_analysis
  __init__.py         __all__ — ONLY these names become tools
analysis_mcp_server/  generic drop-in wrapper (FastMCP): server.py, cli.py
fcl/                  the fcl the art_files analyses run (spec.FCL_DIR)
tests/test_tools.py   parsers, registry, input handling — no mu2e, no MCP
```

The split that matters: **the shared machinery (`mu2e_job.py`, `spectrum.py`)
knows how to run jobs and handle binned spectra but nothing about physics;
`analyses/*.py` knows the physics.** The registry joins them, and the two tools
are generic over it.

Each analysis declares an `input_kind`, which is the only thing the generic
layer needs to know about the difference between them:

| input_kind | what it consumes | what its runner does |
|---|---|---|
| `art_files` | mu2e art file(s) | runs an fcl with `mu2e`, parses the job's stdout |
| `root_file` | a ROOT file from an earlier analysis | Python computation over its histograms or tree |

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
| `edep` | art file(s) | average calo/tracker energy deposition per event and per generated event, and the events passing a `selection` |
| `count` | any art file with subrun bookkeeping | events kept per generated event, dividing out a prescale only if you name the filter |
| `muon_stop_rate` | `sim.*.TargetStops.*.art` | stopped muons per generated event and per POT, from the file's event count, generated-event count and output prescale |
| `approx_ce_sensitivity` | `nts.*.root` from `edep` | `S/sqrt(B)` for the best momentum window, with the window and its signal/DIO/cosmic counts |
| `stop_materials` | `nts.*.root` file(s) from the stop-finding job (e.g. MuBeam) | muon stops per material, and per generated event for the `n_gen_events` you supply |
| `fullsim_sensitivity` | reconstructed CE-mixed-with-pileup EventNtuple `nts.*.root` file(s), e.g. CeMLeadingLogMix1BB | `S/sqrt(B)` for the best momentum window, as `approx_ce_sensitivity`, with the signal efficiency and detector response measured after the CE-like cuts |
| `trigger_efficiency` | signal digi art file(s) | fraction of events passing any of the given trigger paths, and per path |
| `trigger_rate` | pileup digi art file(s); default: mu2e-trig-config's CI sample | trigger rate in Hz averaged over the cycle (duty factor from `batch_mode`) and on spill, overall and per path |
| `trigger_timing` | pileup digi art file(s); default: mu2e-trig-config's CI sample | trigger processing time per event: mean, median, tail; per path and per module |
| `trigger_efficiency_ntuple` | EventNtuple file(s) with `trig_<path>` branches | of events with a track passing a selection, the fraction any given path accepted, and per path |

`approx_ce_sensitivity` declares `produced_by = ["edep"]`, so chaining is
discoverable: run `edep`, then pass the `nts.*.root` from its `files` to the
sensitivity.

### Parameters

Analyses declare their own physics knobs, passed as `parameters`:

```python
run_analysis(analysis="approx_ce_sensitivity",
             data_file=".../nts.owner.edep.Run1B.001800_00000000.root",
             output_dir=".../sens",
             parameters={"sig_eff": 0.1})     # npot and the cosmic rate
                                              # fall back to their defaults
```

`list_analyses` reports each parameter's description, default, range, and
whether it is required; unknown, missing, or out-of-range values come back as
a plain error naming the offender. A parameter is a number unless its `kind`
is `"text"`, which is for the ones that name something in the job's output —
like `muon_stop_rate`'s `prescale_filter` — or, as with `selection`, are an
expression.

### Selections

`edep` and `approx_ce_sensitivity` take a `selection`: a cut over the
per-event tree EdepAna writes (`EDepAna/tree`), applied by `tools/selection.py`
and read by `read_edep_tree` in `tools/analyses/edep.py`, which also lists the
variables (`EDEP_VARIABLES`, repeated in each parameter's description):

```python
parameters={"selection": "event_calo_edep_vis > 10 && primary_start_z > 5400"}
```

Comparisons, arithmetic, `and`/`or`/`not` (or `&&`/`||`/`!`) and
`abs`/`sqrt`/`hypot`/`min`/`max`/`log`/`exp` are allowed. The expression is
parsed with `ast` and walked against a whitelist, never `eval`ed. Per-primary
variables are the first primary's, as EdepAna's own histograms use; derived
ones (`primary_start_r` in the detector frame, `primary_trk_front_energy_diff`,
`has_trk_front`, ...) are computed in Python. A missing value (no primary, or
no tracker-front step) is NaN, so any cut on it fails. `""` selects every event.

The defaults reproduce the fixed cuts the module's histograms and summary had:
`event_calo_edep_vis > 50` for `edep` (so `n_events_selected` equals
`n_events_calo_edep_above_50mev`) and `event_calo_edep_vis > 10` (the
`hist_2` set) for `approx_ce_sensitivity`. On the Run-1B CE sample the
histograms rebuilt from the tree match `hist_0`..`hist_3` bin for bin, and the
sensitivity is unchanged.

### Inputs

Pass **exactly one** of the following — or neither, for an analysis that
`list_analyses` shows with `default_inputs` (`trigger_rate` and
`trigger_timing`, which then run over the files in the configured code's
`mu2e-trig-config/ci/data_files.txt`, found by following its `backing`
links):

- `data_file` — a single absolute art file path → `mu2e -s <file>`
- `data_files` — a list of absolute art file paths → written one per line to
  `filelist.txt` in `output_dir` → `mu2e -S <filelist.txt>`

With `data_files` the whole set runs as **one** job, so the metrics cover all
the inputs together, not one file each — call the tool once per file for
per-file numbers. Raise `timeout_s` (max 7200) when passing many files.
`max_events` applies only to `art_files` analyses. A `root_file` analysis
takes `data_files` only if `list_analyses` reports `takes_data_files: true`
for it (so far just `stop_materials`, which combines the files' histograms);
the others take a single `data_file`.

`max_events` (mu2e `--nevts`) caps events for a quick check before a full run.
Beware: generated-event counts come from the input's subrun bookkeeping and
cover the whole file regardless, so **any "per gen event" metric is
meaningless when `max_events` is set** — use it to confirm a job runs, not for
physics numbers. `edep` records the `max_events` it ran with next to each file
it writes (`<file>.edep.json`, not listed in `files`), and
`approx_ce_sensitivity` refuses such a file when given `stops_per_pot`.

### Results

Every result carries `analysis`, `input_kind`, `data_files`, `n_input_files`,
`log_path`, the resolved `parameters`, and — for the `-S` case —
`file_list_path`. `output_dir` can be reused: a rerun overwrites the previous
run's ROOT output, log, file list and figures, and `files` reports what this
run wrote whether or not the name was there before. On success the analysis' metrics are merged into `metadata`
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
| `n_events_selected` | | weighted count of tree events passing `selection` |
| `selected_per_gen_event` | events / gen event | `n_events_selected / n_gen_events` |

`count` runs `print_counts.fcl` and reports what any art file with subrun
bookkeeping can say about itself:

| metric | unit | from the job's print |
|---|---|---|
| `n_events` | | `<N> Event records found` — events kept in the file |
| `n_gen_events` | | `GenEventCount total: <N> events in <M> SubRuns` |
| `prescale` | | `with prescale fraction <P>`, for the `prescale_filter`, or 1 |
| `saved_per_gen_event` | events / generated event | `n_events / (n_gen_events * prescale)` |

Its one parameter, `prescale_filter`, is **optional and has no assumed name**:
most files were never prescaled, and a job that did prescale one chooses its
own module labels. Left unset (the default, `""`) the prescale is 1 and the
answer is the file's own events per generated event; name a filter and that
filter's prescale is divided out, giving the rate before the prescale threw
events away. A filter that is named but not in the job's output is an error
rather than a silent fallback to 1 — that fallback would report a prescaled
file's rate short by exactly the prescale, with nothing in the output to show
for it. Either way `metadata.prescale_filters` lists every block the job
printed, so an unset run is also how you discover what there was to name.

`muon_stop_rate` is `count` with the target-stop filter and one more factor.
It shares `count.py`'s parsing, input check and job run — `count.py` is the
only place that knows how to read `print_counts.fcl` — and adds what is about
muon stops rather than about counting. It reports:

| metric | unit | from the job's print |
|---|---|---|
| `n_events` | | `<N> Event records found` — stopped muons kept in the file |
| `n_gen_events` | | `GenEventCount total: <N> events in <M> SubRuns` |
| `prescale` | | `with prescale fraction <P>`, for the `prescale_filter` |
| `stops_per_gen_event` | stops / generated event | `n_events / (n_gen_events * prescale)` |
| `stops_per_pot` | stops / POT | the above times the required `upstream_eff` parameter |

It takes two parameters:

- `upstream_eff` (required) — the efficiency of everything upstream, i.e.
  generated events of this file's stage per POT, POT -> MuBeam for a Run-1B
  TargetStops file. No default: the rate per POT is only as meaningful as the
  number you supply.
- `prescale_filter` (optional, default `TargetStopPrescaleFilter`) — the
  module label of the filter whose stream the file belongs to. Pass
  `PolyStopPrescaleFilter` to read a poly-stop file's own rate, or another
  label for a job that named its filters differently. Unlike `count`'s
  same-named knob it will not take `""`: a target-stop file is always
  prescaled, so "no filter" is not an answer here — use `count` for a file
  that was not.

`metadata` carries every `PrescaleFilterFraction` block the job printed
(`prescale_filters`), not just the one used, so the other streams' fractions
are there to read off, alongside the `prescale_filter` that was applied.

The production job writes *all* of its filters' products into *every* output
stream, so a poly-stop file parses fine against the target filter and would
silently be divided by the wrong prescale. The input's Mu2e file name is
therefore checked before the job is started, against the stream named by
`prescale_filter` (`TargetStopPrescaleFilter` -> the description must contain
"targetstop"), and anything else comes back as an error naming what was
passed. A file whose name is not in Mu2e's
`<tier>.<owner>.<description>.<config>.<sequencer>.<format>` form is left
alone and run.

`approx_ce_sensitivity` reports `sensitivity` (S/sqrt(B)),
`signal_box_low_mev` / `signal_box_high_mev`, `signal_rate`,
`dio_background`, `cosmic_background`, `total_background`, the signal peak's
`signal_mpv_mev` / `signal_fwhm_mev`, and the two assumptions those rates are
built on — `npot` and `cosmic_rate_per_s_per_mev` — so a number never travels
without its normalization. The summary line carries them too. Its `metadata`
also records the rest of the assumptions (`sig_eff`, `signal_br`, the
`cosmic_rate_per_mev` the cosmic rate works out to over the live time, and
`onspill_seconds`; with `stops_per_pot`, also `stops_per_pot`,
`ce_acceptance`, `n_gen_events` and `edep_run_recorded`), and it writes the macro's figures — `sig_vs_bkg.png`,
`dio.png`, `response.png`, `res.png`, `ce_z.png`, `ce_r.png` — into
`<output_dir>/figures`.

`stop_materials` reads the TH1 `<stop_module>/stopmat` that a stop finder
books in its job's TFileService output — one alphanumeric bin per stopping
material, labelled with the material name, filled once per stopped muon. It
takes two parameters:

- `n_gen_events` (required) — the generated events the input is equivalent
  to, in total over all files when several are passed. The ntuple has no generated-event bookkeeping, so every rate is the
  stop count divided by this number.
- `stop_module` (optional, default `TargetMuonFinder`) — which finder's
  histogram to read, e.g. `PolyMuonFinder` or `IPAMuonFinder`. If you name
  one that has no `stopmat`, the error lists the modules that do.

It reports the totals as metrics:

| metric | unit | |
|---|---|---|
| `n_stops` | | stops summed over the labelled bins |
| `n_gen_events` | | the parameter, repeated so the rate carries its normalization |
| `stops_per_gen_event` | stops / generated event | `n_stops / n_gen_events` |
| `stops_per_gen_event_err` | stops / generated event | from the histogram's bin errors |
| `n_materials` | | materials with at least one stop |

Several files (`data_files`) are combined into one table, as `hadd` would.
Bins are matched by material *name*: each file's axis is labelled in the
order its own job met the materials, so the same bin number can be a
different material in two files. A material that one file never saw counts
as zero there. `metadata.per_file` gives each file's own stop count,
histogram entries and anything outside its labelled bins, and an error in any
one file names that file.

The breakdown is in `metadata.materials`, most stops first: one row per
material with `material`, `stops`, `stops_err`, `stops_per_gen_event`,
`stops_per_gen_event_err` and `fraction`. The same table is written to
`<output_dir>/stop_materials.log`. ROOT extends an alphanumeric axis by
doubling it, so unlabelled empty bins are normal and are dropped. Content in
an unlabelled bin or in the under/overflow is kept out of the total and
reported as `unnamed_stops`, with the details per file in `per_file`
(`unlabelled_bins`, `underflow`, `overflow`).

## The trigger analyses

`trigger_efficiency`, `trigger_rate` and `trigger_timing` run the online
trigger menu's paths over art files, as one job each, built in
`tools/analyses/trigger.py`. All three take `trigger_paths`, the paths and
their prescales:

```python
parameters={"trigger_paths": "apr_TrkDe_80m70p:1, cpr_TrkDe_80m70p:10, calo_photon"}
```

A path without `:N` has prescale 1. The job is `fcl/trigger.fcl`, which
builds on `mu2e-trig-config/test/timingTest.fcl` (the physics menu with a
Prefetch module, so reading the data is not charged to the first
reconstruction module). Each run writes `trigger_job.fcl` into `output_dir`:
that text with `physics.trigger_paths` and each path's prescale filter set.
FHiCL can only `#include` through `FHICL_FILE_PATH`, not an absolute path,
so the base is copied in rather than included.

- **Prescales are applied in the job**, by each path's `PrescaleEvent` filter
  (`event % prescale == 0`), as online. The overall count therefore carries
  the real overlap between prescaled paths, and a prescaled-away path costs
  no time. The filter's label follows the menu generator's convention
  (`apr_TrkDe_80m70p` → `AprTrkDe80m70pPS`); a test checks it against every
  path in the published menu except the `tpr_`/`mpr_` ones, which are not
  meant for the real trigger.
- **Counts** come from art's TrigReport: events, events passing any path,
  and each path's passes. Fractions carry the width of a uniform-prior
  posterior as their uncertainty, so a path that never fired on a short
  sample is not reported as exactly zero ± 0.
- **`trigger_efficiency`** is relative to the events in the input: fold in
  any filtering the sample already had (e.g. a "Triggerable" sample) yourself.
- **`trigger_rate`** turns the accept fraction into Hz with the online event
  rate: the on-spill microbunch rate (`microbunch_rate_hz`, one per 1695 ns)
  times the spill duty factor. The duty factor follows `batch_mode` —
  `1BB` (default) 0.322, `2BB` 0.246 — unless `duty_factor` is given. `rate_hz`
  is averaged over the accelerator cycle, what the online system sees;
  `onspill_rate_hz` is the rate during the spill. The event rate, duty factor
  and microbunch rate are returned with every result.
- **`trigger_timing`** reads the per-event times from the TimeTracker
  database the job writes (`triggerTiming.db`, returned in `files`). An
  event's time is the **sum of its modules' times without fetching the
  data**: the Prefetch module (`PrefetchDAQData`) and the input source, which
  TimeTracker times apart, are left out, as `mu2eTimingPlotsMaker` totals an
  event. TimeTracker's own whole-event time includes the fetch, about 1 ms of
  ~6 ms per event on the CI pileup sample, so it is reported only for
  comparison (`metadata.fetch_time_mean_ms`, `metadata.full_event_time_mean_ms`).
  The first `skip_events` (default 1) are left out too: the first event
  carries the database and geometry initialization the running trigger
  does not pay, ~1 s against a few ms per event, and would double the mean
  of a few hundred events.
  `metadata.modules` lists every module in the order it ran, with **N(seen)**
  (`n_seen`, the timed events it ran on after the filters and prescale
  upstream of it in its path, and `seen_fraction`), its time per run (mean,
  median, rms, max) and per timed event (`ms_per_event`). The data fetch is
  listed with `counted: false`. A module shared by several paths runs once
  per event, charged to whichever path reached it first, so its runs can be
  split over rows (e.g. `CaloHitMakerFast` under `calo_photon` on the events
  that path's prescale let through, under `apr_` on the rest);
  `n_seen_all_paths` and `ms_per_event_all_paths` add those up. `metadata.path_time_ms` is
  what each path adds: TimeTracker charges a module shared by several paths
  to the first that ran it. The times are wall-clock on whatever node runs
  the job, so they compare configurations run on the same node, not
  absolute online budgets.

Per-path results are in `metadata.paths`: prescale, passes, and the
efficiency or rate with its uncertainty. An unknown path name fails the job
with art's `Unknown path ... has been specified in 'trigger_paths'`, carried
in `metadata.stdout_tail`.

## trigger_efficiency_ntuple

The trigger efficiency without running anything: an EventNtuple records each
path's decision per event in a `trig_<path>` branch. Of the events with at
least one track passing `selection`, it reports the fraction any of
`trigger_paths` accepted, and each path's own (`metadata.paths`):

```python
run_analysis(analysis="trigger_efficiency_ntuple",
             data_file=".../nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root",
             output_dir=".../ntuple_eff",
             parameters={"trigger_paths": "apr_TrkDe_80m70p, cpr_TrkDe_80m70p",
                         "selection": "status == 1 and pdg == 11 and downstream "
                                      "and 100 < p_front < 110"})
```

`trigger_paths` are names separated by commas, with or without the `trig_`
prefix; a name with no branch is reported along with the ones the file has.
`selection` is a per-track cut in the `tools/selection.py` syntax over the
variables in `TRACK_VARIABLES` (repeated in the parameter's description):
the `trk` fields by name (`status`, `pdg`, `nactive`, `chisq`, `ndof`, ...),
`trkqual`, `pid`, momentum, angle and time at the tracker front and middle,
the helix fit parameters at TT_Mid (`d0`, `maxr`, `tandip`, `t0err`, ...),
and the matched calorimeter cluster. Two details:

- `*_front` quantities are at the **downstream-going** TT_Front crossing: a
  track can cross TT_Front twice, and the first crossing is usually the
  upstream leg. `downstream` means pz > 0 at TT_Mid.
- The fit parameters (`trksegpars_lh`) share the segments' indexing, so
  they are taken at the track's TT_Mid segment.

The default selection is a converged (`status == 1`) downstream electron fit
with p > 80 MeV/c at the tracker front, at least 15 active hits and
chi^2/dof < 5. On the CeMLeadingLogMix1BB example file it selects 7178 of
8212 events, of which 96.8% pass apr or cpr `TrkDe_80m70p`. The
track variables were checked against a per-track loop over the raw branches
(the same selected events, and identical TT_Mid fit parameters). Only the
leaves the variables need are read, about 2 s per file; `data_files`
combines several files into one result, with per-file counts in
`metadata.per_file`.

## approx_ce_sensitivity

A Python conversion of `Mu2eOptAna/scripts/rough_run1a_sensitivity.C`. Signal
is CE, background is DIO plus cosmics, and it estimates S/sqrt(B) for the best
momentum window:

1. **Signal shape** — `primary_trk_front_energy` (energy at the front of the
   tracker) for the events passing `selection` (default: >10 MeV visible in
   the calorimeter, the macro's `hist_2` cut), filled from `EDepAna/tree` with
   the binning of EdepAna's `trk_front_energy` histogram, rebinned x2 and
   scaled to a rate for `npot` protons at `SIGNAL_BR` (R_mue = 1e-13 per
   capture times 0.609 captures per stop on aluminum) and
   `sig_eff`, then smeared by a Gaussian tracker resolution (sigma = 0.2 MeV).
   Pass `stops_per_pot` instead of `sig_eff` to have it worked out:
   `sig_eff = stops_per_pot * acceptance`, the acceptance being the file's
   events passing `selection` (weighted) per generated CE event, from
   EdepAna's running `ngen` — so the selection sets the signal's size as well
   as its shape. Pass exactly one of the two. The file must come from one
   full `edep` run: one cut short by `max_events` undercounts the acceptance
   and is refused when `edep`'s record says so (a file without the record,
   e.g. from a production job, runs with a warning in the log), and a merged
   (hadd'ed) one is refused when `ngen` falls — not every merge shows that
   way, so do not merge `edep` outputs.
2. **DIO** — the Heeck/Szafron theoretical spectrum, scaled to a rate, then
   smeared by the *measured* energy-loss response
   of the same events (`primary_trk_front_energy_diff`) and the same
   resolution.
3. **Cosmics** — flat in momentum at `cosmic_rate_per_s_per_mev` (default:
   the rough Run-1A mu- -> e- rate), scaled by the on-spill live time implied
   by `npot`.
4. **Window scan** — every `[x1, x2]` with `x1 >= 50 MeV`, keeping the best
   S/sqrt(B). The top 10 windows go to the log.

The numbers are rough by construction: this is a figure of merit for comparing
beamline configurations, not a sensitivity calculation. Note it needs a **CE
signal** sample — given a file where no event passes the selection and
reaches the tracker it reports that plainly instead of dividing by zero. It
also needs an ntuple from EdepAna in Offline v13_39_00 or later: older ones
have only the histograms, and are refused with a message saying to rerun
`edep`.

Three deviations from the macro, all deliberate:

- It logs the best window for each of the 10 best lower edges rather than all
  ~100k scanned — the top 10 overall only ever differ by a bin.
- Window sums are accumulated from each window's own edge, never as
  differences of whole-spectrum prefix sums. The DIO spectrum spans ~18 orders
  of magnitude, and differencing totals of ~4e17 to get a count of order 1
  loses it entirely to float cancellation (it silently reported
  `dio_background = 0`). `tests/test_tools.py` locks this in.
- Smearing is `np.convolve` against a kernel resampled onto the smeared
  spectrum's own binning (`spectrum.py`), not a bin-by-bin redistribution: the
  Gaussian resolution is integrated over each offset bin and the measured
  response is interpolated rather than snapped to the nearest bin. On the
  Run-1B CE sample that moves `dio_background` by ~2% and `sensitivity` by
  0.01%.

## fullsim_sensitivity

The full-simulation counterpart of `approx_ce_sensitivity`, built the same
way: `sensitivity` = S/sqrt(B) in the best momentum window, with a DIO and
a cosmic background. Where `approx_ce_sensitivity` takes the signal shape
and energy loss from EdepAna's truth-level tree and assumes an efficiency
and a Gaussian resolution, this measures all of them from reconstructed CE
mixed with pileup: CeMLeadingLogMix1BB EventNtuple files, e.g.
`/pnfs/mu2e/tape/phy-nts/nts/mu2e/CeMLeadingLogMix1BB/MDC2025au_best_v1_1-001/`.
`fullsim/README.md` says how to run it.

1. **Cuts**: pyfitter's cut-set 80 (`80_1d`), track by track, as its
   `Analyze.define_cuts` applies them and in its order (`fullsim/cuts.py`).
   On by default: every cut pyfitter switches on except its fit ranges
   (`in_mom_range`, `within_t0`), which the momentum scan and `time_window`
   stand in for. Add or drop cuts by name with `enable_cuts` /
   `disable_cuts`, and move the `trkqual_min` / `trkpid_min` thresholds
   (default 0.155 and 0.54). `or_trigger` passes an event that any of
   `trigger_paths` accepted, by default the production tracker paths
   `apr_TrkDe_80m70p` or `cpr_TrkDe_80m70p`, as in pyfitter. With every cut
   on, the cut flow on a CE mix file matches pyfitter's cut for cut. The cut
   flow goes to `cut_flow.csv` and to `metadata.cut_flow`.
2. **Signal**: each event with a selected track is represented by its
   first selected track: momentum and time at the tracker front, MC origin,
   and true momentum at birth. An event whose first track has more than one
   front segment is not counted, as in pyCount. The signal is the true CE
   among these with a time inside `time_window` (default 640-1650 ns, open),
   so a pileup track passing the cuts is not signal. Their reconstructed
   momentum, binned as `approx_ce_sensitivity`'s signal (0.2 MeV/c), is
   scaled as Production's `normalizations.py` has it: NPOT x
   `stopped_muons_per_pot` x 0.609 captures per stopped mu- x `rmue`
   (default 1e-13) x efficiency. `stopped_muons_per_pot` is
   MuBeamCat x MuminusStopsCat efficiency x 1000, computed by default from
   the stop chain's SAM datasets (7.67e-4 for the MDC2025 chain) as
   Production's `CreateSimEfficiency.sh` does with
   `mu2eGenFilterEff`: events in each dataset over events generated
   (`fullsim/normalization.py`). The chain is traced from the input files'
   SAM ancestry (`stop_datasets` `auto`): the MuminusStopsCat dataset the
   CE were generated from and the MuBeamCat dataset behind it. No SimEfficiencies2 table is needed, so a
   new iteration of the simulation works before it is in the database;
   pass a number to skip SAM. The
   efficiency is measured: CE counted over the events generated to make the
   input files, so it includes the digitization filter's acceptance. The
   generated count is each file's `dh.gencount` in SAM, from its nearest
   ancestor that has one (the parent mcs file; `fullsim/provenance.py`),
   or the `n_generated` parameter where SAM cannot be reached.
   `upstream_eff` (default 1) multiplies in any further loss.
   No resolution is added; the reconstruction already has it.
3. **DIO**: `approx_ce_sensitivity`'s Heeck/Szafron spectrum, as a count per
   0.01 MeV bin for NPOT x `stopped_muons_per_pot` x 0.391 DIO per stopped
   mu-, folded with the measured response of the signal
   events (reconstructed momentum at the tracker front minus true momentum
   at birth: energy loss and resolution together). The response integrates
   to the measured efficiency, so DIO electrons pass the selection as CE do.
4. **Cosmics**: flat in momentum at `cosmic_rate_per_s_per_mev` over the
   on-spill time NPOT implies, exactly as in `approx_ce_sensitivity`, whose
   defaults (`npot`, `mean_pot_per_event`, the rate) it shares.
5. **Window**: `approx_ce_sensitivity`'s `scan_signal_box` over momentum. The
   time window is fixed.

The metrics carry `approx_ce_sensitivity`'s meanings: `signal_rate`,
`dio_background`, `cosmic_background`, `total_background` in the window,
and `npot`, `stopped_muons_per_pot`, `n_stopped_muons`, `rmue`,
`n_conversions` and `cosmic_rate_per_s_per_mev` behind them. Also
`signal_efficiency` (overall) and `signal_efficiency_window`,
`n_signal_window` (reconstructed CE in the window), and the event counts.
Outputs: `fullsim_sensitivity.log` (with the top windows scanned),
`cut_flow.csv`, and `figures/`: signal vs background, the measured
response, and momentum vs time.

On `nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root`
(20000 generated, 8212 reaching the ntuple: acceptance 0.41; about 6 s),
3010 CE pass the cuts and the time window: efficiency 0.15. For 1e18 POT
(7.67e14 stopped mu-, 46.7 conversions at R_mue = 1e-13) the best window is
[103.5, 104.7] MeV/c with S = 4.69, cosmics 1.90 and DIO 0.0009: with
stopped muons per POT in, the DIO is negligible and the cosmics set the
window. `tests/test_tools.py` locks this in.

Unlike `approx_ce_sensitivity`, whose normalization is its macro's (signal
NPOT x BR and DIO NPOT x (1 - capture fraction), with no stopped muons per
POT), the signal and DIO here are counts for the stopped muons NPOT gives.

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
                      files=outcome.written_root_files, log_path=outcome.log_path)

SPEC = AnalysisSpec(
    name="stops",
    input_kind="art_files",
    fcl=FCL_DIR / "stops.fcl",          # i.e. fcl/stops.fcl in this repo
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

The server sets up mu2e **itself**, once per job, in a fresh bash subprocess,
so it does not care whether the shell that launched it had the Offline
environment. Where that Offline comes from is one setting, fixed when the
server starts — `tools/mu2e_env.py` is the only place that knows how:

| flag | environment variable | what a job runs |
|---|---|---|
| `--work-area <dir>` | `MU2E_WORK_AREA` | `cd <dir> && muse setup` |
| `--musing 'SimJob MDC2025ay'` | `MU2E_MUSING` | `muse setup SimJob MDC2025ay` |
| `--code-tarball <file>` | `MU2E_CODE_TARBALL` | unpack it, then `muse setup` in the tree |

```bash
python3 -m analysis_mcp_server --transport stdio \
    --work-area /exp/mu2e/app/users/mmackenz/mu2eopt
python3 -m analysis_mcp_server --transport stdio --musing 'SimJob MDC2025ay'
python3 -m analysis_mcp_server --transport stdio --code-tarball ~/code.tar
```

The flags are mutually exclusive; with none of them (and no environment
variable) the server falls back to `DEFAULT_WORK_AREA` in `tools/mu2e_env.py`.
A work area that is not a directory, a tarball that is not a file, or a Musing
without a version is a startup error naming itself, not a failure inside the
first job. `list_analyses` reports the environment in use, and every mu2e
result carries it in `metadata.environment`, so a number can be traced to the
code that produced it.

A tarball is unpacked once — into `<output_dir>/code` by default, so runs stay
self-contained, or into `--code-dir <dir>` to share one unpacking between
them. A `<dir>.unpacked` marker beside it means later jobs reuse it rather
than unpacking again. `muse setup` runs in the unpacked root, or in its single
top-level directory if that is all the tarball holds; `--code-subdir` says so
explicitly when it holds something else.

### Which analyses a given environment can run

The analyses here ship their fcl with the server, in `fcl/` (`FCL_DIR` in
`tools/spec.py`), and name it by absolute path — the same file whichever
environment is configured, checked before the job starts and reported by
`list_analyses` as `fcl_exists`. Its `#include`s (`Offline/...`,
`Production/...`) are resolved by art on `FHICL_FILE_PATH`.

An analysis may instead name its fcl **relative** to the configured code. For
a work area or an unpacked tarball that is a path on disk and is checked the
same way; a Musing has no directory of ours to look in, so the relative path
goes to `mu2e`, art resolves it on `FHICL_FILE_PATH`, `fcl_exists` comes back
`null`, and a fcl that is not there fails in the job with art's own
`Can't find file "..."`, which the result carries in `metadata.stdout_tail`.

> **What the analyses here need from the environment**: `EdepAna` is in
> Offline (`Offline/Analyses/src/EdepAna_module.cc`) from v13_39_00, so any
> Musing built on it — `SimJob MDC2025ay` and later — or a work area backed by
> one runs `edep` as is. With an older Offline, art stops with `Library
> specification "EdepAna" does not correspond to any library`, and the result
> carries it in `metadata.stdout_tail`.

## Test

```bash
python3 tests/test_tools.py
```

124 tests, none of which start a mu2e job. (The `ana` env has no pytest, so
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

## A worked example

`examples/simple_client.py` is a small MCP client that does what an agent
would: start the server, list the tools, `list_analyses`, run `edep` over an
art file you name, optionally run `muon_stop_rate` over a target-stop file,
then feed the `nts.*.root` `edep` wrote to `approx_ce_sensitivity`.

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana 2.7.0

# the client starts the server itself over stdio
python3 examples/simple_client.py ../dts.mmackenz.CeEndpoint.<...>.art
```

A quick check that skips the full job (and so the chained sensitivity, whose
per-gen-event normalization `--max-events` invalidates):

```bash
python3 examples/simple_client.py path/to/some.art --max-events 200 --no-chain
```

Against a server you started by hand, same code over HTTP:

```bash
python3 -m analysis_mcp_server --transport streamable-http --port 8000 &
python3 examples/simple_client.py path/to/some.art --url http://127.0.0.1:8000/mcp
```

The input path may be relative or use `~`; the client resolves it, since the
tool itself takes only absolute paths (the job runs in `output_dir`, not in
your shell's directory).

`muon_stop_rate` needs an input of its own, so it runs only when you name
one:

```bash
python3 examples/simple_client.py path/to/a/CeEndpoint/dts.art \
    --stops-file ../sim.mmackenz.TargetStops.<...>.art
```

That run passes `upstream_eff` (`--upstream-eff`, default 0.012) and leaves
`prescale_filter` out, so the server's default target-stop label applies;
`--prescale-filter PolyStopPrescaleFilter` overrides it, for a poly-stop file.

`stop_materials` works the same way, over the ntuple of a job that ran the
stop finders. It needs `--n-gen-events`, the generated events that file is
equivalent to. The art file is optional, so leave it out and nothing but
`stop_materials` runs (no mu2e job):

```bash
python3 examples/simple_client.py \
    --stopmat-file ../nts.mmackenz.mubeam.Run1Bak_local0818120248.001800_00000000.root \
    --n-gen-events 1e5
```

`--stopmat-file` takes several paths too (a shell glob works), combined into
one table; `--n-gen-events` is then their total. `--stop-module
PolyMuonFinder` (or `IPAMuonFinder`) reads another finder's histogram; left
out, the server's default `TargetMuonFinder` applies.

Other flags: `--output-dir` (defaults to `output/example`), `--sig-eff`
(handed to `approx_ce_sensitivity`), `--timeout-s`. There is no default input
file: `edep` runs over any art file with the right products, while
`approx_ce_sensitivity` only means anything for a CE signal sample.

The trigger analyses run only with `--trigger` (`--batch-mode 2BB` for the
two-batch duty factor in the rate; the timing prints the module table). Rate and timing then run
over the server's default pileup sample (or `--trigger-pileup-file`), and the
efficiency over `--trigger-signal-file` when given. The default
`--trigger-paths` are `cpr_TrkDe_80m70p` and `apr_TrkDe_80m70p`. The menu's
TPR (`tpr_`) and MPR (`mpr_`) paths are left out of the defaults and the
tests: they are not meant for the real trigger.

```bash
python3 examples/simple_client.py --trigger --max-events 1000 \
    --trigger-signal-file /pnfs/.../dig.mu2e.CeEndpointOnSpill.<...>.art \
    [--trigger-paths 'apr_TrkDe_80m70p, cpr_TrkDe_80m70p:10, calo_photon:10']
```

The trigger efficiency from EventNtuples needs no mu2e job and runs on its
own with `--trigger-ntuple` (one file or several). It reads the decisions
already in the ntuple's `trig_<path>` branches for `--trigger-paths`, minus
their prescales, over events with a track passing the server's default
selection or `--ntuple-selection`:

```bash
python3 examples/simple_client.py \
    --trigger-ntuple /pnfs/mu2e/tape/phy-nts/nts/mu2e/CeMLeadingLogMix1BB/MDC2025au_best_v1_1-001/root/a8/50/nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root \
    [--ntuple-selection 'status == 1 and pdg == 11 and downstream and 100 < p_front < 110']
```

On that file the defaults give 0.968: 6951 of 7178 selected events pass apr
or cpr `TrkDe_80m70p`.

## Use it from a client

**Claude Code** — the checked-in `.mcp.json` already wires it up, work area
and all; or:

```bash
claude mcp add mu2e-analysis -- python3 -m analysis_mcp_server \
    --transport stdio --work-area /exp/mu2e/app/users/mmackenz/mu2eopt
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

Chaining `edep` into the sensitivity, which is what `produced_by` is for:

```
Run the edep analysis on <a CE signal art file>, then feed the nts.*.root it
produces to approx_ce_sensitivity with sig_eff = 0.1, and tell me the best
momentum window and its S/sqrt(B).
```
