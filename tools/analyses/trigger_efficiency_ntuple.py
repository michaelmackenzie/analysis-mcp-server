"""Trigger efficiency from EventNtuple files, for a configurable track selection.

The EventNtuple records each trigger path's decision per event as a boolean
branch, `trig_<path>`. This analysis selects events with at least one track
passing a cut expression (tools/selection.py syntax) and reports the fraction
of them that any of the given paths accepted, and each path's own fraction.
Nothing is rerun: the decisions are the ones made when the ntuple's input was
triggered.

Track variables (TRACK_VARIABLES) come from the `trk`, `trkqual`, `trkpid`
and `trkcalohit` branches and from the track's segments (`trksegs`,
`trksegpars_lh`) at the tracker front and middle. Two details:

* A track can cross TT_Front twice, and the first crossing is usually the
  upstream-going leg. Every `*_front` quantity is the first crossing with
  pz > 0; `downstream` means pz > 0 at TT_Mid.
* The fit parameters (`trksegpars_lh`) share the segments' indexing, so a
  track's `d0`, `maxr`, `t0err`, ... are the entries at its TT_Mid segment.

Simulated files also carry each track's MC truth, offered as the `mc_*`
variables (MC_VARIABLES): the track's own sim particle (`trkmcsim`, whose
first entry is the particle the track was matched to) and its true surface
crossings (`trksegsmc`). The MC TT_Front crossing is the earliest one with
pz > 0, since the true steps are not stored in time order. `mc_npot` is
the event's true beam intensity (protons on target in its microbunch,
`evtinfomc.nprotons`), the same for every track of the event. A file without
these branches (data) is read without them, and naming an `mc_*` variable
then is an error.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..selection import SelectionError, apply_selection
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from .trigger import binomial

TREE_PATH = "EventNtuple/ntuple"
TRIGGER_PREFIX = "trig_"

# SurfaceId values (Offline/DataProducts/inc/SurfaceId.hh).
SID_TT_FRONT, SID_TT_MID = 0, 1

# A downstream electron fit that converged (TrkInfo status 1: Kalman fit
# converged; 2 is only "OK", -1 failed), above 80 MeV/c at the tracker front,
# with at least 15 active hits and chi^2/dof < 5.
DEFAULT_SELECTION = (
    "status == 1 and pdg == 11 and downstream and p_front > 80"
    " and nactive >= 15 and chisq / ndof < 5"
)

# The `trk` (TrkInfo) leaves offered by their own names.
_TRK_FIELDS = ("status", "goodfit", "pdg", "nhits", "ndof", "nactive",
               "ndouble", "nplanes", "nnullambig", "nmat", "nmatactive",
               "nipaup", "nipadown", "nstup", "nstdown", "tsdainter",
               "opainter", "chisq", "fitcon", "maxgap", "avggap", "avgedep",
               "seedalg", "fitalg")
_SEGPAR_FIELDS = ("d0", "maxr", "tanDip", "rad", "t0", "t0err")

TRACK_VARIABLES: dict[str, str] = {
    **{name: f"trk.{name}" for name in _TRK_FIELDS},
    "status": "trk.status: 1 Kalman fit converged, 2 fit OK, -1 failed",
    "charge": "-1 for e-/mu- fits (pdg > 0), +1 for e+/mu+",
    "trkqual": "track quality MVA output (trkqual.result)",
    "pid": "PID MVA output (trkpid.result)",
    "has_front": "has a downstream-going TT_Front crossing",
    "p_front": "momentum at TT_Front, downstream crossing (MeV/c)",
    "pt_front": "transverse momentum at TT_Front (MeV/c)",
    "pz_front": "pz at TT_Front (MeV/c)",
    "cos_front": "cos(theta) = pz/p at TT_Front",
    "tandip_front": "tan(dip) = pz/pT at TT_Front",
    "t_front": "time at TT_Front (ns)",
    "momerr_front": "momentum uncertainty at TT_Front (MeV/c)",
    "downstream": "pz > 0 at TT_Mid",
    "p_mid": "momentum at TT_Mid (MeV/c)",
    "pt_mid": "transverse momentum at TT_Mid (MeV/c)",
    "cos_mid": "cos(theta) at TT_Mid",
    "t_mid": "time at TT_Mid (ns)",
    "d0": "helix d0 at TT_Mid (mm)",
    "maxr": "helix maximum radius at TT_Mid (mm)",
    "tandip": "helix tan(dip) at TT_Mid",
    "rad": "helix radius at TT_Mid (mm)",
    "t0": "helix t0 at TT_Mid (ns)",
    "t0err": "helix t0 uncertainty at TT_Mid (ns)",
    "has_calo": "has a matched calorimeter cluster",
    "calo_edep": "matched cluster energy (MeV)",
    "calo_dt": "track - cluster time difference (ns)",
    "event": "event number", "run": "run number", "subrun": "subrun number",
    "ntrk": "tracks in the event",
}

# MC truth per track, only in simulated files.
MC_VARIABLES: dict[str, str] = {
    "mc_pdg": "PDG id of the track's sim particle",
    "mc_start_code": "creation process code of the track's sim particle "
                     "(ProcessCode.hh, e.g. 167/168 CE-, 166/170 DIO)",
    "mc_gen": "generator id of the track's sim particle",
    "mc_p_start": "true momentum at creation (MeV/c)",
    "mc_t_start": "true creation time (ns)",
    "mc_nhits": "straw hits from the sim particle on the track",
    "mc_nactive": "active straw hits from the sim particle on the track",
    "mc_has_front": "the sim particle has a downstream-going TT_Front crossing",
    "mc_p_front": "true momentum at TT_Front, downstream crossing (MeV/c)",
    "mc_pt_front": "true transverse momentum at TT_Front (MeV/c)",
    "mc_pz_front": "true pz at TT_Front (MeV/c)",
    "mc_cos_front": "true cos(theta) at TT_Front",
    "mc_t_front": "true time at TT_Front (ns)",
    "mc_p_mid": "true momentum at TT_Mid (MeV/c)",
    "mc_t_mid": "true time at TT_Mid (ns)",
    "mc_npot": "true protons on target in the event's microbunch "
               "(evtinfomc.nprotons; the beam intensity, ~1e7-1e8 for Run 1)",
}

# The leaves read, by name (uproot's filter_name): only what TRACK_VARIABLES
# needs. Split struct members are named "<branch>.<member>"; evtinfo's are not.
LEAVES = (
    *(f"trk.{name}" for name in _TRK_FIELDS),
    "trkqual.result", "trkpid.result",
    "trkcalohit.did", "trkcalohit.edep", "trkcalohit.dt",
    "trksegs", "trksegpars_lh", "event", "run", "subrun",
)
# Read too where the file has them (simulation), for MC_VARIABLES.
MC_LEAVES = ("trksegsmc", "trkmcsim")
# The true beam intensity. Read by its full path: evtinfo has a leaf of the
# same name (the reconstructed estimate, 0 in simulation).
NPOT_BRANCH = "evtinfomc/nprotons"


class NtupleError(ValueError):
    """An input or request that cannot be used, worded for the caller."""


# --- the request -----------------------------------------------------------------

def parse_trigger_names(text: str) -> list[str]:
    """'apr_TrkDe_80m70p, trig_cpr_TrkDe_80m70p' -> path names, prefix dropped."""
    names = [n for n in re.split(r"[,\s]+", text.strip()) if n]
    if not names:
        raise NtupleError("no trigger paths given")
    out: list[str] = []
    for name in names:
        name = name[len(TRIGGER_PREFIX):] if name.startswith(TRIGGER_PREFIX) else name
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name):
            raise NtupleError(f"'{name}' does not name a trigger path")
        if name in out:
            raise NtupleError(f"'{name}' is listed twice")
        out.append(name)
    return out


# --- per-track variables ---------------------------------------------------------

def _numpy(array, fill=np.nan) -> np.ndarray:
    import awkward as ak
    return np.asarray(ak.to_numpy(ak.fill_none(array, fill)), dtype=np.float64)


def _flat(array, fill=np.nan) -> np.ndarray:
    """events * tracks -> one value per track, in event order."""
    import awkward as ak
    return _numpy(ak.flatten(array, axis=1), fill)


def _magnitude(vector):
    xyz = vector.fCoordinates
    return np.sqrt(xyz.fX ** 2 + xyz.fY ** 2 + xyz.fZ ** 2)


def mc_track_variables(arrays) -> dict[str, np.ndarray]:
    """MC_VARIABLES but mc_npot, one entry per track, from the trkmcsim and
    trksegsmc leaves (events * tracks * sims, events * tracks * true steps)."""
    import awkward as ak

    out: dict[str, np.ndarray] = {}
    sim = ak.firsts(arrays["trkmcsim"], axis=-1)     # the track's own particle
    out["mc_pdg"] = _flat(sim.pdg)
    out["mc_start_code"] = _flat(sim.startCode)
    out["mc_gen"] = _flat(sim.gen)
    out["mc_p_start"] = _flat(_magnitude(sim.mom))
    out["mc_t_start"] = _flat(sim.time)
    out["mc_nhits"] = _flat(sim.nhits)
    out["mc_nactive"] = _flat(sim.nactive)

    steps = arrays["trksegsmc"]
    steps = steps[ak.argsort(steps.time, axis=-1)]  # stored out of time order
    mom = steps.mom.fCoordinates
    pz = mom.fZ
    pt = np.sqrt(mom.fX ** 2 + mom.fY ** 2)
    p = np.sqrt(mom.fX ** 2 + mom.fY ** 2 + mom.fZ ** 2)

    def first(values, mask):
        return _flat(ak.firsts(values[mask], axis=-1))

    front = (steps.sid == SID_TT_FRONT) & (pz > 0)
    out["mc_has_front"] = _flat(ak.any(front, axis=-1), False).astype(bool)
    out["mc_p_front"] = first(p, front)
    out["mc_pt_front"] = first(pt, front)
    out["mc_pz_front"] = first(pz, front)
    out["mc_t_front"] = first(steps.time, front)
    with np.errstate(all="ignore"):
        out["mc_cos_front"] = out["mc_pz_front"] / out["mc_p_front"]
    mid = steps.sid == SID_TT_MID
    out["mc_p_mid"] = first(p, mid)
    out["mc_t_mid"] = first(steps.time, mid)
    return out


def track_variables(arrays) -> dict[str, np.ndarray]:
    """TRACK_VARIABLES, one entry per track, from one file's leaves.

    `arrays` maps each name in LEAVES to its awkward array (events * tracks
    [* segments]). Also returns `_event`: each track's event index in the file.
    MC_VARIABLES are added when `arrays` has the MC_LEAVES too, and mc_npot
    when it has NPOT_BRANCH (one value per event).
    """
    import awkward as ak

    ntrk = ak.num(arrays["trk.status"], axis=1)
    event_of_track = np.repeat(np.arange(len(ntrk)), ak.to_numpy(ntrk))
    out: dict[str, np.ndarray] = {"_event": event_of_track}
    for name in _TRK_FIELDS:
        out[name] = _flat(arrays[f"trk.{name}"])
    out["charge"] = np.where(out["pdg"] > 0, -1.0, 1.0)
    out["trkqual"] = _flat(arrays["trkqual.result"])
    out["pid"] = _flat(arrays["trkpid.result"])

    segs = arrays["trksegs"]
    sid = segs.sid
    mom = segs.mom.fCoordinates
    pz = mom.fZ
    pt = np.sqrt(mom.fX ** 2 + mom.fY ** 2)
    p = np.sqrt(mom.fX ** 2 + mom.fY ** 2 + mom.fZ ** 2)

    def first(values, mask):
        return _flat(ak.firsts(values[mask], axis=-1))

    front = (sid == SID_TT_FRONT) & (pz > 0)
    out["has_front"] = _flat(ak.any(front, axis=-1), False).astype(bool)
    out["p_front"] = first(p, front)
    out["pt_front"] = first(pt, front)
    out["pz_front"] = first(pz, front)
    out["t_front"] = first(segs.time, front)
    out["momerr_front"] = first(segs.momerr, front)
    with np.errstate(all="ignore"):
        out["cos_front"] = out["pz_front"] / out["p_front"]
        out["tandip_front"] = out["pz_front"] / out["pt_front"]

    mid = sid == SID_TT_MID
    out["p_mid"] = first(p, mid)
    out["pt_mid"] = first(pt, mid)
    out["t_mid"] = first(segs.time, mid)
    pz_mid = first(pz, mid)
    with np.errstate(all="ignore"):
        out["cos_mid"] = pz_mid / out["p_mid"]
    out["downstream"] = pz_mid > 0

    # Fit parameters share the segments' indexing: take the entry at the
    # position of the track's first TT_Mid segment (missing if it has none,
    # or its parameter list is too short to have that entry).
    pars = arrays["trksegpars_lh"]
    mid_index = ak.fill_none(
        ak.firsts(ak.local_index(sid, axis=-1)[mid], axis=-1), -1)
    at_mid = ak.local_index(pars.t0, axis=-1) == mid_index
    for name in _SEGPAR_FIELDS:
        out["tandip" if name == "tanDip" else name] = _flat(
            ak.firsts(pars[name][at_mid], axis=-1))

    has_calo = _flat(arrays["trkcalohit.did"], -1) >= 0
    out["has_calo"] = has_calo
    out["calo_edep"] = np.where(has_calo, _flat(arrays["trkcalohit.edep"]), np.nan)
    out["calo_dt"] = np.where(has_calo, _flat(arrays["trkcalohit.dt"]), np.nan)

    for name in ("event", "run", "subrun"):
        out[name] = _numpy(arrays[name])[event_of_track]
    out["ntrk"] = ak.to_numpy(ntrk).astype(np.float64)[event_of_track]
    if all(leaf in arrays for leaf in MC_LEAVES):
        out.update(mc_track_variables(arrays))
    if NPOT_BRANCH in arrays:
        out["mc_npot"] = _numpy(arrays[NPOT_BRANCH])[event_of_track]
    return out


def check_mc_names(variables: dict, *expressions: str) -> None:
    """Refuse an mc_* variable on a file with no MC truth, saying why,
    rather than as an unknown name."""
    named = {name for text in expressions
             for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)}
    asked = sorted((named & set(MC_VARIABLES)) - set(variables))
    if asked:
        raise NtupleError(
            f"{', '.join(asked)} need MC truth, and this file has no "
            f"{', '.join((*MC_LEAVES, NPOT_BRANCH))} branches (real data, or "
            "an ntuple written without MC)")


# --- reading ---------------------------------------------------------------------

def open_tree(path: Path):
    """The file's EventNtuple tree, or an NtupleError saying it has none."""
    import uproot

    try:
        return uproot.open(path)[TREE_PATH]
    except (KeyError, OSError, ValueError) as exc:
        raise NtupleError(f"{path}: no {TREE_PATH} ({exc}) — is this an "
                          "EventNtuple file?") from None


def read_track_arrays(tree, path: Path) -> dict:
    """The LEAVES, plus MC_LEAVES and NPOT_BRANCH where the file has them,
    as awkward arrays by name (events * tracks ...)."""
    names = {key.rsplit("/", 1)[-1] for key in tree.keys()}
    absent = [leaf for leaf in LEAVES if leaf not in names]
    if absent:
        raise NtupleError(f"{path.name}: missing branch(es) {', '.join(absent)} "
                          "— an EventNtuple version this analysis does not know?")
    mc = [leaf for leaf in MC_LEAVES if leaf in names]
    if len(mc) < len(MC_LEAVES):
        mc = []
    batch = tree.arrays(filter_name=list(LEAVES) + mc, library="ak")
    arrays = {name: batch[name] for name in batch.fields}
    if NPOT_BRANCH in tree.keys():
        arrays[NPOT_BRANCH] = tree[NPOT_BRANCH].array(library="ak")
    return arrays


def read_file(path: Path, paths: list[str]) -> tuple[dict, dict[str, np.ndarray], int]:
    """(track variables, {path: per-event decision}, number of events).

    With no `paths`, only the track variables are read (no trig_ branches
    needed), as selection_efficiency_ntuple does.
    """
    tree = open_tree(path)
    available = [k[len(TRIGGER_PREFIX):] for k in tree.keys(recursive=False)
                 if k.startswith(TRIGGER_PREFIX)]
    missing = [p for p in paths if p not in available]
    if missing:
        raise NtupleError(
            f"{path.name}: no trigger branch for {', '.join(missing)}. "
            f"It has: {', '.join(available) or '(none)'}"
        )
    arrays = read_track_arrays(tree, path)
    triggers: dict[str, np.ndarray] = {}
    if paths:
        decisions = tree.arrays(filter_name=[TRIGGER_PREFIX + p for p in paths],
                                library="np")
        triggers = {p: np.asarray(decisions[TRIGGER_PREFIX + p], dtype=bool)
                    for p in paths}
    return track_variables(arrays), triggers, tree.num_entries


# --- the analysis ------------------------------------------------------------------

def efficiency_counts(variables: dict[str, np.ndarray], triggers: dict[str, np.ndarray],
                      n_events: int, selection: str) -> dict:
    """Selected events, and how many of them each path (and any path) kept."""
    passing = apply_selection(selection, {k: v for k, v in variables.items()
                                          if not k.startswith("_")},
                              variables["_event"].size)
    selected = np.zeros(n_events, dtype=bool)
    selected[variables["_event"][passing]] = True
    any_path = np.zeros(n_events, dtype=bool)
    for decision in triggers.values():
        any_path |= decision
    return {
        "n_events": n_events,
        "n_tracks_selected": int(passing.sum()),
        "n_selected": int(selected.sum()),
        "n_triggered": int((selected & any_path).sum()),
        "per_path": {p: int((selected & d).sum()) for p, d in triggers.items()},
    }


METRICS = ("n_events", "n_selected", "n_triggered", "efficiency",
           "efficiency_err", "n_tracks_selected")


def efficiency_metrics(counts: dict) -> dict[str, float]:
    """The overall efficiency: triggered over selected events."""
    efficiency, err = binomial(counts["n_triggered"], counts["n_selected"])
    return {"n_events": float(counts["n_events"]),
            "n_selected": float(counts["n_selected"]),
            "n_triggered": float(counts["n_triggered"]),
            "efficiency": efficiency, "efficiency_err": err,
            "n_tracks_selected": float(counts["n_tracks_selected"])}


def run(context: RunContext) -> RunOutcome:
    """Select events in the ntuple(s) and count the trigger decisions on them."""
    selection = context.params["selection"]
    extra: dict = {"selection": selection}
    try:
        paths = parse_trigger_names(context.params["trigger_paths"])
        totals = {"n_events": 0, "n_tracks_selected": 0, "n_selected": 0,
                  "n_triggered": 0, "per_path": {p: 0 for p in paths}}
        per_file = []
        for path in context.input_paths:
            variables, triggers, n_events = read_file(path, paths)
            check_mc_names(variables, selection)
            counts = efficiency_counts(variables, triggers, n_events, selection)
            per_file.append({"file": str(path), "n_events": n_events,
                             "n_selected": counts["n_selected"],
                             "n_triggered": counts["n_triggered"]})
            for key in ("n_events", "n_tracks_selected", "n_selected", "n_triggered"):
                totals[key] += counts[key]
            for p in paths:
                totals["per_path"][p] += counts["per_path"][p]
    except (NtupleError, SelectionError) as exc:
        return RunOutcome(error=str(exc), extra=extra)
    if totals["n_selected"] == 0:
        return RunOutcome(
            error=f"no event of {totals['n_events']} has a track passing the "
                  "selection, so there is no efficiency to measure",
            extra=extra)
    paths_out = {}
    for p in paths:
        eff, err = binomial(totals["per_path"][p], totals["n_selected"])
        paths_out[p] = {"passed": totals["per_path"][p],
                        "efficiency": eff, "efficiency_err": err}
    extra.update(paths=paths_out,
                 **({"per_file": per_file} if len(per_file) > 1 else {}))
    return RunOutcome(metrics=efficiency_metrics(totals), extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    return (f"trigger efficiency {metrics['efficiency']:.4g} "
            f"+- {metrics['efficiency_err']:.2g}: {metrics['n_triggered']:g} of "
            f"{metrics['n_selected']:g} selected events (of {metrics['n_events']:g}) "
            "passed any of the paths; per-path efficiencies in metadata.paths.")


SPEC = AnalysisSpec(
    name="trigger_efficiency_ntuple",
    input_kind="root_file",
    combines_files=True,
    run=run,
    description=(
        "Trigger efficiency from EventNtuple file(s): of the events with a "
        "track passing a configurable selection, the fraction any of the "
        "given trigger paths accepted (trig_<path> branches), and each path's "
        "own efficiency (metadata.paths). No job is run."
    ),
    metrics=METRICS,
    parameters=(
        ParamSpec(
            name="trigger_paths",
            description="Trigger paths whose decisions to read, separated by "
                        "commas, e.g. 'apr_TrkDe_80m70p, cpr_TrkDe_80m70p' "
                        "(the trig_ prefix is optional). An event counts as "
                        "triggered if any of them accepted it. A name the "
                        "ntuple has no branch for is reported with the list "
                        "it does have.",
            kind="text",
        ),
        ParamSpec(
            name="selection",
            description=(
                "Track selection: an event is selected if at least one track "
                "passes. Comparisons, arithmetic, and/or/not (or &&/||/!) and "
                "abs/sqrt/hypot/min/max/log/exp over: "
                + ", ".join(TRACK_VARIABLES) + "; and, in simulated files, "
                + ", ".join(MC_VARIABLES) + ". *_front quantities are at "
                "the downstream-going TT_Front crossing; fit parameters (d0, "
                "maxr, tandip, t0err, ...) at TT_Mid; a missing value is NaN "
                "and fails any cut on it. The default is a converged "
                "downstream e- fit with p > 80 MeV/c at the tracker front, "
                ">= 15 active hits and chisq/ndof < 5. An empty string "
                "selects every event."
            ),
            default=DEFAULT_SELECTION, kind="text", allow_empty=True,
        ),
    ),
    summarize=summarize,
    input_hint=(
        "EventNtuple file(s) (nts.*.root, tree EventNtuple/ntuple) with "
        "trig_<path> branches, e.g. nts.mu2e.CeMLeadingLogMix1BB.<version>.root; "
        "pass data_files for several, combined into one result."
    ),
)
