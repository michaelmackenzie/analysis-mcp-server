"""Reconstructed momentum at the tracker front, from EventNtuple file(s).

Histograms `p_front` -- |p| at the downstream-going TT_Front crossing -- of
the tracks passing a selection, read with trigger_efficiency_ntuple's track
variables (TRACK_VARIABLES), so the two analyses read a track the same way.
The default keeps every e- fit that has such a crossing. Several files
(data_files) are combined into one histogram, written as nts_momentum.png in
the run's outdir.
"""

from pathlib import Path

import numpy as np

from ..selection import SelectionError, apply_selection
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from .trigger_efficiency_ntuple import (TRACK_VARIABLES, NtupleError,
                                        read_file)

DEFAULT_SELECTION = "pdg == 11 and has_front"
P_RANGE = (95.0, 110.0)
P_BINS = 60  # 0.25 MeV/c


def front_momenta(variables: dict[str, np.ndarray], selection: str) -> np.ndarray:
    """`p_front` (MeV/c) of the tracks passing `selection` that have one."""
    tracks = {k: v for k, v in variables.items() if not k.startswith("_")}
    keep = apply_selection(selection, tracks, variables["_event"].size)
    p = variables["p_front"][keep]
    return p[np.isfinite(p)]


def run(context: RunContext) -> RunOutcome:
    """Histogram p_front of the selected tracks over the input file(s)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selection = context.params["selection"]
    extra: dict = {"selection": selection}
    n_events, n_tracks, momenta = 0, 0, []
    try:
        for path in context.input_paths:
            variables, _, events = read_file(path, [])
            n_events += events
            n_tracks += variables["_event"].size
            momenta.append(front_momenta(variables, selection))
    except (NtupleError, SelectionError) as exc:
        return RunOutcome(error=str(exc), extra=extra)
    p = np.concatenate(momenta) if momenta else np.zeros(0)

    context.outdir.mkdir(parents=True, exist_ok=True)
    png = Path(context.outdir) / "nts_momentum.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p, bins=P_BINS, range=P_RANGE, histtype="step")
    ax.set_xlabel("reconstructed |p| at tracker front [MeV/c]")
    ax.set_ylabel("tracks / 0.25 MeV/c")
    ax.set_title(f"{len(p)} of {n_tracks} tracks in {n_events} events")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)

    metrics = {
        "n_events": float(n_events),
        "n_tracks": float(n_tracks),
        "n_fits": float(len(p)),
        "median_p_front": float(np.median(p)) if len(p) else 0.0,
        "mean_p_front": float(p.mean()) if len(p) else 0.0,
    }
    return RunOutcome(metrics=metrics, files=[str(png)], extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    return (f"{metrics['n_fits']:g} of {metrics['n_tracks']:g} tracks in "
            f"{metrics['n_events']:g} events selected, median |p| at the "
            f"tracker front {metrics['median_p_front']:.3f} MeV/c")


SPEC = AnalysisSpec(
    name="nts_momentum",
    description="Reconstructed |p| at the tracker front of the tracks passing "
                "a selection, from EventNtuple file(s), as a histogram (PNG) "
                "plus counts, median and mean.",
    input_kind="root_file",
    combines_files=True,
    metrics=("n_events", "n_tracks", "n_fits", "median_p_front",
             "mean_p_front"),
    units={"median_p_front": "MeV/c", "mean_p_front": "MeV/c"},
    parameters=(
        ParamSpec(
            name="selection",
            description=(
                "Track selection: comparisons, arithmetic, and/or/not over "
                + ", ".join(TRACK_VARIABLES) + " (trigger_efficiency_ntuple's "
                "track variables). The default keeps every e- fit with a "
                "downstream-going TT_Front crossing. A track without one has "
                "no p_front and is never counted."
            ),
            default=DEFAULT_SELECTION, kind="text", allow_empty=True,
        ),
    ),
    input_hint="EventNtuple ROOT file(s) (nts.*.root) holding "
               "EventNtuple/ntuple.",
    run=run,
    summarize=summarize,
)
