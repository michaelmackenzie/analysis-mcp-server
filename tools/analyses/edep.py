"""Energy deposition: the EdepAna analyzer (fcl/edep.fcl).

Average calorimeter and tracker energy deposition per event and per generated
event, plus the number and rate of events passing a configurable selection.
The averages come from the summary block EdepAna prints at endJob; the
selection is applied to the per-event TTree it writes (EDepAna/tree). The
module lives in Offline (Offline/Analyses/src/EdepAna_module.cc, from
v13_39_00), so it comes with the configured Musing or its backing.

This module also owns the reading of that tree — `read_edep_tree` and
`EDEP_VARIABLES` — so every analysis of EdepAna output selects events the
same way, with the same variable names.
"""

import json
import re
from pathlib import Path

import numpy as np

from ..mu2e_job import run_mu2e_job
from ..selection import SelectionError, apply_selection
from ..spec import FCL_DIR, AnalysisSpec, ParamSpec, RunContext, RunOutcome

FCL = FCL_DIR / "edep.fcl"

# The analyzer's label in edep.fcl, and the tree it books.
TREE_PATH = "EDepAna/tree"

# What the >50 MeV count in the summary block used to be fixed to.
DEFAULT_SELECTION = "event_calo_edep_vis > 50"

# DetectorSystem's origin in the Mu2e frame: x_det = x + 3904 mm. EdepAna's
# primary_start_r histogram is the start radius in the detector frame.
DETECTOR_ORIGIN_X_MM = -3904.0

# --- the EdepAna tree --------------------------------------------------------

_EVENT_BRANCHES = ("event_calo_edep", "event_calo_edep_vis", "event_trk_edep",
                   "weight", "run", "subrun", "event", "nprimaries", "ngen")
_PRIMARY_BRANCHES = (
    "primary_start_x", "primary_start_y", "primary_start_z",
    "primary_start_px", "primary_start_py", "primary_start_pz",
    "primary_start_e", "primary_start_m", "primary_start_pdg",
    "primary_calo_edep", "primary_calo_edep_vis",
    "primary_trk_front_p", "primary_trk_front_energy",
)

# The variables a selection may use, and what each one is. Per-primary
# branches are the *first* primary's value, as EdepAna's histograms use it;
# NaN for an event with no primary, or with no tracker-front step for the
# primary_trk_front_* ones, so any cut on them fails for such an event.
EDEP_VARIABLES: dict[str, str] = {
    "event_calo_edep": "total calo energy deposited in the event (MeV)",
    "event_calo_edep_vis": "total visible (Birks) calo energy in the event (MeV)",
    "event_trk_edep": "total tracker ionizing energy in the event (MeV)",
    "weight": "event weight",
    "run": "run number", "subrun": "subrun number", "event": "event number",
    "nprimaries": "number of primary particles",
    "primary_start_x": "primary start x, Mu2e frame (mm)",
    "primary_start_y": "primary start y, Mu2e frame (mm)",
    "primary_start_z": "primary start z, Mu2e frame (mm)",
    "primary_start_r": "primary start radius, detector frame (mm)",
    "primary_start_px": "primary start px (MeV/c)",
    "primary_start_py": "primary start py (MeV/c)",
    "primary_start_pz": "primary start pz (MeV/c)",
    "primary_start_p": "primary start momentum (MeV/c)",
    "primary_start_e": "primary start energy (MeV)",
    "primary_start_m": "primary mass (MeV)",
    "primary_start_pdg": "primary PDG id",
    "primary_calo_edep": "calo energy from the primary and descendants (MeV)",
    "primary_calo_edep_vis": "visible calo energy from the primary and descendants (MeV)",
    "has_trk_front": "the primary reached the tracker front (true/false)",
    "primary_trk_front_p": "primary momentum at the tracker front (MeV/c)",
    "primary_trk_front_energy": "primary energy at the tracker front (MeV)",
    "primary_trk_front_energy_diff":
        "tracker-front energy minus start energy (MeV, <= 0)",
    "primary_energy_edep_diff": "primary visible calo edep minus start energy (MeV)",
    "primary_trk_front_energy_edep_diff":
        "primary visible calo edep minus tracker-front energy (MeV)",
}


class EdepTreeError(RuntimeError):
    """The input is not an EdepAna file with the tree, worded for the caller."""


def edep_variables(branches: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every EDEP_VARIABLES column from the tree's raw branches.

    `branches` holds the event branches as flat arrays and the primary
    branches as one array per event (uproot's library="np").
    """
    nprim = np.asarray(branches["nprimaries"])
    has_primary = nprim > 0
    out: dict[str, np.ndarray] = {
        name: np.asarray(branches[name], dtype=np.float64)
        for name in _EVENT_BRANCHES if name != "ngen"
    }
    for name in _PRIMARY_BRANCHES:
        first = np.full(nprim.size, np.nan)
        if has_primary.any():
            first[has_primary] = [entry[0] for entry in branches[name][has_primary]]
        out[name] = first

    # EdepAna stores 0 where the primary never reached the tracker front.
    has_front = out["primary_trk_front_p"] > 0.0
    for name in ("primary_trk_front_p", "primary_trk_front_energy"):
        out[name] = np.where(has_front, out[name], np.nan)
    out["has_trk_front"] = has_front

    out["primary_start_r"] = np.hypot(out["primary_start_x"] - DETECTOR_ORIGIN_X_MM,
                                      out["primary_start_y"])
    out["primary_start_p"] = np.sqrt(out["primary_start_px"] ** 2
                                     + out["primary_start_py"] ** 2
                                     + out["primary_start_pz"] ** 2)
    out["primary_trk_front_energy_diff"] = (out["primary_trk_front_energy"]
                                            - out["primary_start_e"])
    out["primary_energy_edep_diff"] = (out["primary_calo_edep_vis"]
                                       - out["primary_start_e"])
    out["primary_trk_front_energy_edep_diff"] = (out["primary_calo_edep_vis"]
                                                 - out["primary_trk_front_energy"])
    return out


def read_edep_tree(paths: list[Path] | Path) -> dict[str, np.ndarray]:
    """EDEP_VARIABLES for every event in the EdepAna file(s), concatenated."""
    import uproot

    paths = [paths] if isinstance(paths, Path) else list(paths)
    parts = []
    for path in paths:
        try:
            with uproot.open(path) as rootfile:
                tree = rootfile[TREE_PATH]
                branches = tree.arrays(_EVENT_BRANCHES + _PRIMARY_BRANCHES,
                                       library="np")
        except KeyError as exc:
            raise EdepTreeError(
                f"{path}: no {TREE_PATH} with the EdepAna branches ({exc}) — is "
                "this an nts.*.root from EdepAna in Offline v13_39_00 or later? "
                "Rerun the 'edep' analysis to make one."
            ) from None
        parts.append(edep_variables(branches))
    return {name: np.concatenate([part[name] for part in parts])
            for name in parts[0]}


def generated_events(paths: list[Path] | Path) -> float:
    """The generated events behind the EdepAna file(s).

    `ngen` is EdepAna's running count, raised by each subrun's GenEventCount
    as the job reaches it, so a file's total is its largest value; files from
    separate jobs add up. A subrun after a file's last event is not in it, nor
    is the rest of a subrun cut off by max_events: read it from a full run
    (read_run_record says whether 'edep' made the file with max_events).
    The count restarts with each job, so a file in which it falls holds
    several jobs' output merged, and is refused. Not every merge shows this
    way: one whose later part starts at or above the earlier part's total
    keeps the count rising and is not caught, so do not merge edep outputs.
    """
    import uproot

    paths = [paths] if isinstance(paths, Path) else list(paths)
    total = 0.0
    for path in paths:
        try:
            with uproot.open(path) as rootfile:
                ngen = rootfile[TREE_PATH]["ngen"].array(library="np")
        except KeyError as exc:
            raise EdepTreeError(
                f"{path}: no {TREE_PATH} with an ngen branch ({exc}) — is "
                "this an nts.*.root from EdepAna in Offline v13_39_00 or later? "
                "Rerun the 'edep' analysis to make one."
            ) from None
        if ngen.size > 1 and (np.diff(ngen) < 0).any():
            raise EdepTreeError(
                f"{path}: EdepAna's running generated-event count (ngen) "
                "falls, so this file holds several edep outputs merged (e.g. "
                "by hadd) and their generated events cannot be added up; run "
                "'edep' once over all the art files instead (and do not merge "
                "its outputs: not every merge can be caught this way)"
            )
        total += float(ngen.max()) if ngen.size else 0.0
    return total


RUN_RECORD_SUFFIX = ".edep.json"


def run_record_path(path: Path) -> Path:
    """Where the 'edep' analysis records how it made the EdepAna file `path`."""
    path = Path(path)
    return path.with_name(path.name + RUN_RECORD_SUFFIX)


def write_run_record(path: Path, max_events: int | None) -> Path:
    """Record, next to an EdepAna file 'edep' wrote, the max_events its job
    ran with (None for a full run). ngen keeps a cut-short subrun's whole
    GenEventCount, so a file from a max_events run undercounts its events
    per generated event, and that cannot be seen from the file itself."""
    record = run_record_path(path)
    record.write_text(json.dumps({"max_events": max_events}) + "\n",
                      encoding="utf-8")
    return record


def read_run_record(path: Path) -> dict | None:
    """The record 'edep' wrote next to `path`, or None for a file it did not
    make (e.g. one from a production job)."""
    record = run_record_path(path)
    if not record.exists():
        return None
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EdepTreeError(f"{record}: unreadable edep run record ({exc})") from None
    if not isinstance(data, dict) or "max_events" not in data:
        raise EdepTreeError(f"{record}: an edep run record holds max_events, "
                            f"got {data!r}")
    return data


def select_events(variables: dict[str, np.ndarray], selection: str) -> np.ndarray:
    """The mask of events passing `selection`, a cut over EDEP_VARIABLES."""
    nevents = variables["event_calo_edep_vis"].size
    return apply_selection(selection, variables, nevents)


def selection_help() -> str:
    """The parameter description shared by every analysis taking a selection."""
    return (
        "Event selection applied to the EdepAna tree, e.g. "
        "'event_calo_edep_vis > 10 && primary_start_z > 5400'. Comparisons, "
        "arithmetic, and/or/not (or &&/||/!) and abs/sqrt/hypot/min/max/log/"
        "exp over: " + ", ".join(EDEP_VARIABLES) + ". Per-primary variables "
        "are the first primary's. An empty string selects every event."
    )

# Matches the block EdepAna_module.cc prints, e.g.:
#   EdepAna summary:
#     Saw 998.5 events (1000 gen events) --> output rate = 0.9985 events / gen event
#     Average calo energy deposition per event: 12.3 MeV
#     Average calo energy deposition per gen event: 12.29 MeV
#     Events with calo Edep > 50 MeV: 42
#     Average tracker energy deposition per event: 3.21 MeV
#     Average tracker energy deposition per gen event: 3.2 MeV
# n_events and n_events_calo_edep_above_50mev are weighted sums (doubles in
# the module), so every field is parsed as a float.
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_SUMMARY_FIELDS = [
    ("n_events", re.compile(rf"Saw\s+({_NUM})\s+events\b")),
    ("n_gen_events", re.compile(rf"\(\s*({_NUM})\s+gen events\)")),
    ("event_rate", re.compile(rf"output rate\s*=\s*({_NUM})\s+events / gen event")),
    ("avg_calo_edep_per_event_mev",
     re.compile(rf"Average calo energy deposition per event:\s*({_NUM})\s*MeV")),
    ("avg_calo_edep_per_gen_event_mev",
     re.compile(rf"Average calo energy deposition per gen event:\s*({_NUM})\s*MeV")),
    ("n_events_calo_edep_above_50mev",
     re.compile(rf"Events with calo Edep > 50 MeV:\s*({_NUM})")),
    ("avg_trk_edep_per_event_mev",
     re.compile(rf"Average tracker energy deposition per event:\s*({_NUM})\s*MeV")),
    ("avg_trk_edep_per_gen_event_mev",
     re.compile(rf"Average tracker energy deposition per gen event:\s*({_NUM})\s*MeV")),
]

_SELECTION_METRICS = ("n_events_selected", "selected_per_gen_event")

METRIC_UNITS = {
    "event_rate": "events / gen event",
    "selected_per_gen_event": "events / gen event",
    "avg_calo_edep_per_event_mev": "MeV",
    "avg_calo_edep_per_gen_event_mev": "MeV",
    "avg_trk_edep_per_event_mev": "MeV",
    "avg_trk_edep_per_gen_event_mev": "MeV",
}


def parse_edep_summary(stdout: str) -> dict[str, float] | None:
    """Extract the 8 EdepAna summary fields from mu2e stdout.

    Returns None if the "EdepAna summary:" block isn't there or is incomplete
    (e.g. the job died before EdepAna::endJob ran).
    """
    if "EdepAna summary:" not in stdout:
        return None
    metrics: dict[str, float] = {}
    for name, pattern in _SUMMARY_FIELDS:
        match = pattern.search(stdout)
        if not match:
            return None
        metrics[name] = float(match.group(1))
    return metrics


def selected_metrics(variables: dict[str, np.ndarray], selection: str,
                     n_gen_events: float) -> dict[str, float]:
    """Weighted count of events passing `selection`, and that per gen event."""
    mask = select_events(variables, selection)
    selected = float(variables["weight"][mask].sum())
    return {
        "n_events_selected": selected,
        "selected_per_gen_event": selected / n_gen_events if n_gen_events > 0 else -1.0,
    }


def summarize_edep(metrics: dict[str, float]) -> str:
    return (
        f"saw {metrics['n_events']:g} events ({metrics['n_gen_events']:g} gen): "
        f"avg calo edep {metrics['avg_calo_edep_per_event_mev']:.4g} MeV/event, "
        f"avg tracker edep {metrics['avg_trk_edep_per_event_mev']:.4g} MeV/event; "
        f"{metrics['n_events_selected']:g} selected "
        f"({metrics['selected_per_gen_event']:.4g} / gen event)."
    )


def run(context: RunContext) -> RunOutcome:
    """Run edep.fcl, parse the summary block, and apply the selection."""
    outcome = run_mu2e_job(
        fcl=FCL,
        input_paths=context.input_paths,
        outdir=context.outdir,
        single=len(context.input_paths) == 1 and not context.wants_file_list,
        timeout_s=context.timeout_s,
        max_events=context.max_events,
    )
    extra = {"returncode": outcome.returncode}
    if outcome.file_list_path is not None:
        extra["file_list_path"] = str(outcome.file_list_path)

    if outcome.timed_out:
        return RunOutcome(
            files=outcome.written_root_files, log_path=outcome.log_path, extra=extra,
            error=f"mu2e timed out after {context.timeout_s}s on "
                  f"{len(context.input_paths)} input file(s)",
        )
    if outcome.failed:
        extra["stdout_tail"] = outcome.stdout_tail()
        return RunOutcome(
            files=outcome.written_root_files, log_path=outcome.log_path, extra=extra,
            error=f"mu2e exited {outcome.returncode}",
        )

    metrics = parse_edep_summary(outcome.stdout)
    if metrics is None:
        extra["stdout_tail"] = outcome.stdout_tail()
        return RunOutcome(
            files=outcome.written_root_files, log_path=outcome.log_path, extra=extra,
            error="EdepAna summary block not found in mu2e output",
        )

    selection = context.params["selection"]
    extra["selection"] = selection
    ntuples = [Path(f) for f in outcome.written_root_files]
    if not ntuples:
        return RunOutcome(log_path=outcome.log_path, extra=extra,
                          error="mu2e wrote no ROOT file, so there is no "
                                f"{TREE_PATH} to apply the selection to")
    # Beside each file, not in `files`: a chained analysis takes the ROOT
    # file alone, and finds the record next to it.
    for ntuple in ntuples:
        write_run_record(ntuple, context.max_events)
    try:
        metrics.update(selected_metrics(read_edep_tree(ntuples), selection,
                                        metrics["n_gen_events"]))
    except (EdepTreeError, SelectionError) as exc:
        return RunOutcome(files=outcome.written_root_files,
                          log_path=outcome.log_path, extra=extra, error=str(exc))
    return RunOutcome(metrics=metrics, files=outcome.written_root_files,
                      log_path=outcome.log_path, extra=extra)


SPEC = AnalysisSpec(
    name="edep",
    fcl=FCL,
    input_kind="art_files",
    run=run,
    description=(
        "Average calorimeter and tracker energy deposition per event and per "
        "generated event (EdepAna), and the number and rate per generated "
        "event of events passing a configurable selection."
    ),
    metrics=tuple(name for name, _ in _SUMMARY_FIELDS) + _SELECTION_METRICS,
    units=METRIC_UNITS,
    parameters=(
        ParamSpec(
            name="selection",
            description=selection_help() + " The default, "
                        f"'{DEFAULT_SELECTION}', is the cut behind "
                        "n_events_calo_edep_above_50mev.",
            default=DEFAULT_SELECTION, kind="text", allow_empty=True,
        ),
    ),
    summarize=summarize_edep,
    input_hint=(
        "art file(s) holding compressDetStepMCs, CaloClusterMaker and "
        "FindMCPrimary — e.g. TargetStops/mcs/dts files."
    ),
)
