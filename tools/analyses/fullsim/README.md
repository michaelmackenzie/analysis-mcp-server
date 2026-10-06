# Full-simulation analyses (`fullsim`)

Analyses of reconstructed EventNtuple files (`EventNtuple/ntuple`) from
mixed MC samples with MC truth, such as the MDS ensembles. The repo
README.md has the full description of each, under its own name.

| module | what it holds |
|---|---|
| `eventntuple.py` | the branch set and reader (`read_eventntuple`), surface ids and process codes, and `origin_codes`: what made each event's first track |
| `cuts.py` | RefAna/pyCount's `Analyze.define_cuts` as track-level masks (`cut_masks`), the default set, toggling by name, and the cut flow |
| `limits.py` | the Run-1A analysis' CLs upper limit (exact, or profile likelihood with systematics) and discovery signal; RefAna/pyCount's FC interval, expected upper limit and FC table (`FC.csv`) for comparison |
| `sensitivity.py` | `fullsim_sensitivity`: pyCount's `run_count`, as a registered analysis |

Run it like any other analysis:

```python
run_analysis(
    "fullsim_sensitivity",
    output_dir="/path/to/out",
    data_files=[".../nts.mu2e.ensembleMDS3cMix1BB.MDC2025-001.001430_00000001.root", ...],
    parameters={"exposure": 3.4e15, "disable_cuts": "has_st"},
)
```

## Adding a full-simulation analysis

Read the input with `read_eventntuple`, and add any branches you need to
`BRANCHES` there. Select tracks with `cut_masks` / `apply_cuts`, so every
analysis applies the same cuts. Then write a module here that defines a
`SPEC`, add `"fullsim.<module>"` to `_ANALYSIS_MODULES` in
`tools/registry.py`, and add tests. The tests build EventNtuple groups in
memory (`_eventntuple` in `tests/test_tools.py`), because uproot cannot
write EventNtuple's nested branches.

## Unregistered draft

`eventnuple_validation.py`, `eventnuple_reader.py` and `matching.py` are a
draft truth-matching analysis. They read a `TrkAna/trkTree` with branches
such as `trk_mom` that EventNtuple files do not have, and the SPEC was
missing `summarize`, so importing it stopped the whole server from loading.
They stay out of the registry until they are rewritten against
`read_eventntuple`. The EventNtuple already pairs each track with its truth
(`trkmcsim`, `trksegsmc`), so no kinematic matching is needed.
