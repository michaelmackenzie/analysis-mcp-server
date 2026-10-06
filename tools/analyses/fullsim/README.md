# Full-simulation analyses (`fullsim`)

Analyses of reconstructed EventNtuple files (`EventNtuple/ntuple`) from
mixed MC samples with MC truth, such as the MDS ensembles. Today there is
one, `fullsim_sensitivity`: RefAna/pyCount's cut-and-count, giving true
signal and background in a momentum-time window and the limits on that
background. The repo README.md, under `fullsim_sensitivity`, says how each
step is computed and where it differs from pyCount.

| module | what it holds |
|---|---|
| `sensitivity.py` | `fullsim_sensitivity` (the registered `SPEC`), its window scan and per-file reduction |
| `eventntuple.py` | the branch set and reader (`read_eventntuple`), surface ids and process codes, and `origin_codes`: what made each event's first track |
| `cuts.py` | RefAna/pyCount's `Analyze.define_cuts` as track-level masks (`cut_masks`), the default set, toggling by name, and the cut flow |
| `limits.py` | the Run-1A analysis' CLs upper limit (exact, or profile likelihood with systematics) and discovery signal; RefAna/pyCount's FC interval, expected upper limit and FC table (`FC.csv`) for comparison |

## Setup

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana
cd /path/to/analysis-mcp-server
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # shared interactive nodes
```

No mu2e job is run. The analysis reads the ROOT files with uproot, so any
EventNtuple `nts.*.root` on disk will do.

## Input: MDS3

The MDS3 ensembles are in `/exp/mu2e/data/users/mu2epro/ensembles/MDS3/`.
For MDS3c, each of `MDS3c/merged_files_1`, `_2` and `_3` is one test set:
88 ensemble files plus the CE split for that set
(`nts.mu2e.CeMLeadingLogMix1BBSplit.*.root`). `merged_files_2` holds two
CE split files, `...Split.3.root` and `...Split.MDC2025an_best_v1_3.2.root`.
Check which one belongs to the set before globbing `nts.*.root` there. The
`filenames_*` lists next to the sets name the `/pnfs` originals.

## Running `fullsim_sensitivity`

From Python, as the MCP server runs it:

```python
import glob
from tools.analysis_tools import run_analysis

D = "/exp/mu2e/data/users/mu2epro/ensembles/MDS3/MDS3c/merged_files_1"
result = run_analysis(
    "fullsim_sensitivity",
    output_dir="output/mds3c_set1",
    data_files=sorted(glob.glob(f"{D}/nts.*.root")),
    parameters={"exposure": 3.4e15},   # captured muons the set is equivalent to
    timeout_s=3600,                    # ~a few seconds per 10k-event file
)
print(result.status, result.message)   # message: one-line summary, or the error
result.metadata["cls_upper_events"]    # every metric is in metadata
```

Through the MCP server, ask the agent to call `run_analysis` with the
same arguments, or `list_analyses` to see every parameter and metric.

For a quick check, pass two files: the CE split file and one ensemble
file. With defaults that gives 162 counted events (92 CE) and the window
[103.5, 104.5] MeV/c x [500, 1650] ns, with 49 CE and 1 cosmic in it.

### Parameters

| parameter | default | what it does |
|---|---|---|
| `sign` | `minus` | `minus` (CE-, signal code 168) or `plus` (CE+, 176). Sets the charge cut, the window and the counting range. |
| `exposure` | 3.398e15 | Captured muons the input is equivalent to: Run-1A's 5.58e15 stopped x the capture fraction 0.609. The background counted in the window *is* the expected background, so this must be the sample's own. It only enters the `*_br` metrics and `ses`. |
| `sig_eff` | 0.12 | Signal efficiency for converting events to a branching ratio. Not measured from the input. |
| `cl` | 0.9 | Confidence level of the CLs, FC and expected limits. |
| `bkg_rel_uncertainty` | 0 | Relative background uncertainty. Nonzero switches the CLs limit to the asymptotic profile likelihood, which comes out *below* the exact one at a few events. |
| `sig_eff_rel_uncertainty` | 0 | Relative signal-efficiency uncertainty, likewise. |
| `signal_window` | empty | `'p_low,p_high,t_low,t_high'` (MeV/c, ns) to fix the window. Empty: for `minus`, scan for the best S/sqrt(B) over windows with both S > 0 and B > 0 (as `approx_ce_sensitivity` scores windows), else fall back to 103.9-105.1 MeV/c x 640-1650 ns; for `plus`, 90-92 MeV/c x 640-1650 ns. |
| `enable_cuts`, `disable_cuts` | empty | Comma-separated cut names to add to or drop from the default set, e.g. `disable_cuts="has_st,no_opa"`. An unknown name is an error that lists the known ones. |
| `trkqual_min`, `trkpid_min` | 0.2, 0.638 | Thresholds of `good_trkqual` and `good_trkpid`. |

The optimized window is tuned on the events it counts, so its background
is biased low. Fix `signal_window` for an unbiased count, for example to
the fixed window `103.9,105.1,640,1650`.

### Cuts

The default set, applied in this order (pyCount's current switches):
the charge cut (`is_reco_electron` / `is_reco_positron`), `has_downstream`,
`has_trk_front_seg`, `good_trkpid`, `good_trkqual`, `within_t0err`,
`has_hits`, `has_st`, `no_opa`, `no_crv_quality`, `no_crv_timewindow`,
`no_crv_veto`, `pz_over_pt`, `good_trigger`. Also available with
`enable_cuts`: `within_t0`, `within_lhr_max`, `within_d0`,
`within_pitch_angle`, `in_mom_range`, `within_t0_early`, `no_reflected`.
`CUT_DESCRIPTIONS` in `cuts.py` gives each one's definition, and the log
prints them next to the cut flow.

### Outputs

In `output_dir`:

- `fullsim_sensitivity.log`: the inputs, the cut flow, the counted events
  by origin, the window and how it was chosen, and every limit.
- `cut_flow.csv`: events with a track passing each cut so far.
- `window_events.csv`: run/subrun/event, origin, momentum and time of
  every event in the window.
- `figures/`: momentum and time with the window (`mom.png`, `time.png`),
  `mom_vs_time.png`, and `window_scan.png` (S/sqrt(B) per window; blank
  where S or B is 0).

The main metrics are `n_signal_window` and `n_background_window`, the
window edges, `cls_upper_events` / `cls_upper_br` (the Run-1A limit),
`discovery_5sigma_events` / `_br`, `ses`, and pyCount's `fc_*`,
`expected_ul_*` and `fc_table_ul90_events`. `metadata` also has
`cut_flow`, `counted_by_origin` and `background_window_by_origin`.

## Using the pieces directly

The modules are plain Python, so they work outside `run_analysis`, for
example in a notebook:

```python
import numpy as np
from tools.analyses.fullsim import cuts, limits, sensitivity
from tools.analyses.fullsim.eventntuple import ORIGIN_NAMES, read_eventntuple

data = read_eventntuple(path)                 # awkward arrays per branch group
active = cuts.active_cuts("minus", enable="", disable="has_st")
masks = cuts.cut_masks(data, "minus")         # cut name -> events x tracks bool
track_mask, flow = cuts.apply_cuts(masks, active, data["trk"]["trk.pdg"])

# one entry per selected event: first track's p, t at the tracker front, origin
ev = sensitivity.reduce_data(data, "minus", active, cuts.TRKQUAL_MIN,
                             cuts.TRKPID_MIN)
best, rows = sensitivity.scan_window(ev["p"], ev["t"], ev["origin"] == 168)
# best is None when no window has both signal and background

limits.cls_upper_limit(n_obs=1, b=1.0, cl=0.9)              # 3.27 events
limits.cls_upper_limit(1, 1.0, 0.9, b_sigma=0.2)            # 2.66, asymptotic
limits.fc_interval(1, 1.0, 0.9)                             # (0.0, 3.357)
limits.required_signal(5, 1.0)                              # 5 sigma: 5.0
```

Read files one at a time (`sensitivity.reduce_file`) and `combine` the
results, so the jagged arrays for many files never sit in memory at once.

## Adding a full-simulation analysis

Read the input with `read_eventntuple`, and add any branches you need to
`BRANCHES` there. Select tracks with `cut_masks` / `apply_cuts`, so every
analysis applies the same cuts. The EventNtuple already pairs each track
with its truth (`trkmcsim`, `trksegsmc`), so no kinematic matching is
needed. Then write a module here that defines a `SPEC`, add
`"fullsim.<module>"` to `_ANALYSIS_MODULES` in `tools/registry.py`, and add
tests. The tests build EventNtuple groups in memory (`_eventntuple` in
`tests/test_tools.py`), because uproot cannot write EventNtuple's nested
branches.

## Tests

`python3 tests/test_tools.py` covers the cuts, origins, reduction, window
scan and limits on synthetic events. It also reruns pyCount's MDS3c
comparison on two ensemble files when they are on disk.
