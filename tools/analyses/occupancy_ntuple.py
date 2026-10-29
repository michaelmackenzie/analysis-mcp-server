"""Occupancy from EventNtuple files: no job, the counts the ntuple already holds.

The same quantities OccupancyAna counts in art files, read from the branches an
EventNtuple has:

* `hitcount` (filled from the RecoCount, i.e. the whole event, not a
  SelectReco subset): `nsd`, the straw digis, and the straw hits flagged
  energy (`nesel`), radius (`nrsel`), time (`ntsel`) selected and background
  (`nbkg`).
* `calohits` and `caloclusters`: the size of each vector, and the calo hits
  inside a time window and above an energy (as occupancy_reco's params).
* `crvcoincs`: its size; `crvsummary`: `nHitCounters` and `totalPEs`.
* `tcnt.ntrk`: the number of tracks.

Every quantity is optional, since the layout differs between EventNtuple
versions: a leaf is looked up by its full branch path (a bare leaf name such
as `nsd` could be another struct's), under the spellings an older layout used
too, and what the file lacks is reported (extra["missing"]) with metrics of
-1. It is an error only when none of them is there.

No per-generated-event normalization is offered. The ntuple's n_gen_events
histogram is the generated events behind the file, but the ntuple holds only
the events that survived the production's filtering (e.g. 622 of 1e6), so an
occupancy "per generated event" would mix occupancy with that filtering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..spec import AnalysisSpec, RunContext, RunOutcome
from . import occupancy as occ

TREE_PATH = "EventNtuple/ntuple"

# metric prefix -> (kind, leaf spellings, what it is). "scalar" is one number
# per event; "size" is the length of a per-event vector. A leaf is found by its
# full path in the tree, so a path is tried before the next spelling.
QUANTITIES: dict[str, tuple[str, tuple[str, ...], str]] = {
    "strawdigis": ("scalar", ("hitcount/nsd", "hitcount.nsd"),
                   "straw digis (hitcount.nsd)"),
    "strawhits_esel": ("scalar", ("hitcount/nesel", "hitcount.nesel"),
                       "straw hits passing the energy selection (hitcount.nesel)"),
    "strawhits_rsel": ("scalar", ("hitcount/nrsel", "hitcount.nrsel"),
                       "straw hits passing the radius selection (hitcount.nrsel)"),
    "strawhits_tsel": ("scalar", ("hitcount/ntsel", "hitcount.ntsel"),
                       "straw hits passing the time selection (hitcount.ntsel)"),
    "strawhits_bkg": ("scalar", ("hitcount/nbkg", "hitcount.nbkg"),
                      "straw hits flagged background (hitcount.nbkg)"),
    "calohits": ("size", ("calohits/calohits.time_", "calohits.time_"),
                 "calorimeter hits (calohits)"),
    "caloclusters": ("size", ("caloclusters/caloclusters.time_",
                              "caloclusters.time_"),
                     "calorimeter clusters (caloclusters)"),
    "crvcoincs": ("size", ("crvcoincs/crvcoincs.time", "crvcoincs.time"),
                  "CRV coincidence clusters (crvcoincs)"),
    "crv_hit_counters": ("scalar", ("crvsummary/nHitCounters",
                                    "crvsummary.nHitCounters"),
                         "CRV counters with a hit (crvsummary.nHitCounters)"),
    "crv_total_pes": ("scalar", ("crvsummary/totalPEs", "crvsummary.totalPEs"),
                      "CRV photoelectrons in the event (crvsummary.totalPEs)"),
    "ntracks": ("scalar", ("tcnt.ntrk", "tcnt/ntrk"), "tracks (tcnt.ntrk)"),
}
# The calo hit leaves the windowed count needs besides its time.
CALO_ENERGY_LEAVES = ("calohits/calohits.eDep_", "calohits.eDep_")

METRICS = ("n_events",
           *(f"{q}_{s}" for q in QUANTITIES for s in ("mean", "rms", "max")),
           "calohits_sel_mean")

UNITS = {**{f"{q}_{s}": "per event" for q in QUANTITIES
            for s in ("mean", "rms", "max")},
         "calohits_sel_mean": "per event",
         "crv_total_pes_mean": "PE", "crv_total_pes_rms": "PE",
         "crv_total_pes_max": "PE"}


class OccupancyNtupleError(ValueError):
    """An input that cannot be used, worded for the caller."""


def _leaf(tree, spellings: tuple[str, ...]):
    """The branch the first spelling names that the tree has, or None."""
    keys = set(tree.keys())
    for name in spellings:
        if name in keys:
            return tree[name]
    return None


def _per_event(array, kind: str) -> np.ndarray:
    """One number per event: the value, or the length of a per-event vector."""
    import awkward as ak

    if kind == "size" or array.ndim > 1:
        return np.asarray(ak.num(array, axis=1), dtype=np.float64)
    return np.asarray(array, dtype=np.float64)


def read_file(path: Path, window: tuple[float, float, float]
              ) -> tuple[dict[str, np.ndarray], int]:
    """({quantity: per-event values} for what the file has, number of events).

    `calohits_sel` is the calo hits in `window` (time min, time max in ns,
    energy min in MeV), present where the hit time and energy leaves are.
    """
    import awkward as ak
    import uproot

    try:
        tree = uproot.open(path)[TREE_PATH]
    except (KeyError, OSError, ValueError) as exc:
        raise OccupancyNtupleError(
            f"{path}: no {TREE_PATH} ({exc}); is this an EventNtuple "
            "(nts.*.root) file?") from None
    values: dict[str, np.ndarray] = {}
    for name, (kind, spellings, _) in QUANTITIES.items():
        branch = _leaf(tree, spellings)
        if branch is not None:
            values[name] = _per_event(branch.array(library="ak"), kind)
    time_branch = _leaf(tree, QUANTITIES["calohits"][1])
    energy_branch = _leaf(tree, CALO_ENERGY_LEAVES)
    if time_branch is not None and energy_branch is not None:
        time, energy = time_branch.array(library="ak"), energy_branch.array(library="ak")
        tmin, tmax, emin = window
        inside = (time >= tmin) & (time <= tmax) & (energy >= emin)
        values["calohits_sel"] = np.asarray(ak.sum(inside, axis=1), dtype=np.float64)
    return values, tree.num_entries


def occupancy_metrics(per_file: list[dict[str, np.ndarray]]
                      ) -> tuple[dict[str, float], list[str]]:
    """(metrics, quantities missing from some file), over the files together.

    A quantity counts only if every file has it: events from a file without
    it would be missing from its mean.
    """
    metrics: dict[str, float] = {}
    missing: list[str] = []
    for name in (*QUANTITIES, "calohits_sel"):
        columns = [values.get(name) for values in per_file]
        have = all(column is not None for column in columns)
        if not have:
            missing.append(name)
        if name == "calohits_sel":
            metrics[f"{name}_mean"] = (float(np.concatenate(columns).mean())
                                       if have else -1.0)
            continue
        stats = occ.mean_rms_max(np.concatenate(columns)) if have else (-1.0,) * 3
        metrics.update(zip((f"{name}_mean", f"{name}_rms", f"{name}_max"), stats))
    return metrics, missing


def run(context: RunContext) -> RunOutcome:
    """Read the occupancy quantities of the ntuple(s) and summarize them."""
    window = (float(context.params["hit_time_min"]),
              float(context.params["hit_time_max"]),
              float(context.params["calo_hit_energy_min"]))
    if window[0] > window[1]:
        return RunOutcome(error=f"hit_time_min ({window[0]:g}) is above "
                                f"hit_time_max ({window[1]:g}), so no hit is in the window")
    per_file, counts = [], []
    try:
        for path in context.input_paths:
            values, n_events = read_file(path, window)
            per_file.append(values)
            counts.append(n_events)
    except OccupancyNtupleError as exc:
        return RunOutcome(error=str(exc))
    n_events = sum(counts)
    if n_events == 0:
        return RunOutcome(error="the ntuple has no events")
    if not any(name in values for values in per_file for name in QUANTITIES):
        return RunOutcome(
            error="none of the occupancy branches is in the file: looked for "
                  + "; ".join(f"{name} ({' or '.join(s)})"
                              for name, (_, s, _) in QUANTITIES.items())
                  + ". Is this an EventNtuple (nts.*.root) file?")
    metrics, missing = occupancy_metrics(per_file)
    extra: dict = {"window": {"hit_time_min": window[0], "hit_time_max": window[1],
                              "calo_hit_energy_min": window[2]}}
    if missing:
        extra["missing"] = missing
    if len(per_file) > 1:
        extra["per_file_events"] = {str(p): n for p, n in
                                    zip(context.input_paths, counts)}
    return RunOutcome(metrics={"n_events": float(n_events), **metrics}, extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    parts = []
    for name in (*QUANTITIES, "calohits_sel"):
        mean = metrics.get(f"{name}_mean", -1)
        if mean < 0:
            continue                              # not in the file
        text = f"{name} {mean:.4g}"
        if f"{name}_rms" in metrics:
            text += f" +- {metrics[f'{name}_rms']:.3g}"
        parts.append(text)
    return (f"mean per event over {metrics['n_events']:g} events: "
            + ("; ".join(parts) if parts else "no occupancy quantity present")
            + ". Quantities the file lacks are -1 (see 'missing').")


SPEC = AnalysisSpec(
    name="occupancy_ntuple",
    input_kind="root_file",
    combines_files=True,
    run=run,
    description=(
        "Per-event occupancy from EventNtuple file(s), without running a "
        "job: straw digis and the straw hit selections (hitcount, the whole "
        "event), calo hits (also counted inside a time window and above an "
        "energy) and clusters, CRV coincidence clusters, hit counters and "
        "photoelectrons, and tracks, each as mean, rms and max per event. A "
        "branch an EventNtuple version lacks is reported in 'missing' and its "
        "metrics are -1; it errors only if none is there. No per-generated-"
        "event normalization (the ntuple holds only the events that survived "
        "production filtering)."
    ),
    metrics=METRICS,
    units=UNITS,
    parameters=occ.window_params(tracker=False),
    summarize=summarize,
    input_hint=(
        "EventNtuple file(s) (nts.*.root, tree EventNtuple/ntuple), e.g. "
        "nts.mu2e.CeEndpointMix1BB-KL.Run1Baw_best_v1_5.<sequencer>.root; "
        "pass data_files for several, combined into one result."
    ),
)
