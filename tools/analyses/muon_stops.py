"""Where, when and how fast muons stop: the stop profile of a MuBeam stop file.

A stop file of the MuBeam stage (sim.*.TargetStops.*.art, or PolyStops /
IPAStops) keeps, for every stopped muon, its SimParticle (whose end point is
the stop) and the virtual-detector steps of it and its ancestors. One mu2e
job, fcl/muon_stops.fcl, dumps both with Offline modules: StoppedParticlesDumper
(as Offline's TargetStopsDumper.fcl and Production's Tests/TGTstops.fcl do)
writes each stop's x, y, z, time, pdg and creation/stopping codes, and
StepPointMCDumper writes the VD steps. From those this reports

  the stop profile   z along the DS, radius from the DS axis, time
  momentum at ST_In  the VD just upstream of the stopping target: what the
                     stopped muons arrived with

as metrics, plots (figures/*.png) and histograms (muon_stops_hists.root).

The two dumpers write no event number, so a stop and its ST_In crossing are
not paired: the momentum spectrum is of every muon crossing ST_In
downstream-going in the file, which in a stop file is the stopped muons (one
crossing each, unless one scattered back across it). `n_st_in` says how many
there were, to compare with `n_stops`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..mu2e_job import run_mu2e_job
from ..spec import FCL_DIR, AnalysisSpec, ParamSpec, RunContext, RunOutcome
from ..spectrum import Spectrum

FCL = FCL_DIR / "muon_stops.fcl"
STOPS_TREE = "stoppedMuonDumper/stops"
VD_TREE = "vdDumper/nt"

# The DS axis is at x = -mu2e.solenoidOffset in Mu2e coordinates
# (Offline/Mu2eG4/geom/geom_common*.txt: 3904 mm).
SOLENOID_OFFSET_MM = 3904.0
# VirtualDetectorId ST_In (Offline/DataProducts/inc/VirtualDetectorId.hh).
VD_ST_IN = 9
MUON_PDG = 13

# GenEventCountReader's end-of-job line: the events generated to make the
# file(s), i.e. the resampled MuBeam events behind the stops.
_GEN_COUNT = re.compile(r"GenEventCount total:\s*(\d+)\s+events")

# Bin widths of the histograms and plots; their ranges follow the data,
# with edges on multiples of the width.
BIN_WIDTHS = {"z": 5.0, "r": 2.0, "time": 10.0, "p_st_in": 1.0}

METRICS = ("n_gen_events", "n_stops",
           "stop_z_mean", "stop_z_rms", "stop_z_median",
           "stop_r_mean", "stop_r_rms", "stop_r_median",
           "stop_time_mean", "stop_time_median",
           "n_st_in", "p_st_in_mean", "p_st_in_median", "p_st_in_rms")

UNITS = {**{m: "mm" for m in METRICS if m.startswith(("stop_z", "stop_r"))},
         **{m: "ns" for m in METRICS if m.startswith("stop_time")},
         **{m: "MeV/c" for m in METRICS if m.startswith("p_st_in")},
         "n_gen_events": "events", "n_stops": "stops", "n_st_in": "crossings"}


class MuonStopsError(ValueError):
    """An input or request that cannot be used, worded for the caller."""


# --- the job -----------------------------------------------------------------------

def job_fcl_text(stop_filter: str, base: Path = FCL) -> str:
    """The shipped fcl, pointed at `stop_filter`'s collections.

    FHiCL cannot #include an absolute path, so the run gets a copy with the
    overrides appended rather than a two-line fcl including the shipped one.
    """
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", stop_filter):
        raise MuonStopsError(f"'{stop_filter}' is not a module label")
    return (base.read_text(encoding="utf-8")
            + f'\nphysics.analyzers.stoppedMuonDumper.inputCollection : "{stop_filter}"\n'
            + f'physics.analyzers.vdDumper.hitsInputTag : "{stop_filter}:virtualdetector"\n')


def parse_gen_count(stdout: str) -> int | None:
    """The generated-event total GenEventCountReader printed, if it did."""
    match = _GEN_COUNT.search(stdout)
    return int(match.group(1)) if match else None


# --- reading -----------------------------------------------------------------------

def read_dumps(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(stops, VD steps) from the job's TFileService file, as the numpy
    structured arrays of the dumpers' single leaf-list branches."""
    import uproot

    with uproot.open(path) as f:
        try:
            return (f[STOPS_TREE]["stops"].array(library="np"),
                    f[VD_TREE]["hits"].array(library="np"))
        except KeyError as exc:
            raise MuonStopsError(f"{path.name} has no {exc} — did the job run "
                                 "muon_stops.fcl?") from None


def stop_variables(stops: np.ndarray) -> dict[str, np.ndarray]:
    """StoppedParticlesDumper's fields by name, plus the radius `r` from the
    DS axis."""
    variables = {name: np.asarray(stops[name], dtype=np.float64)
                 for name in ("x", "y", "z", "time", "pdg", "origin", "term")}
    variables["r"] = np.hypot(variables["x"] + SOLENOID_OFFSET_MM, variables["y"])
    return variables


def vd_columns(steps: np.ndarray) -> dict[str, np.ndarray]:
    """The StepPointMCDumper (VDHit) fields st_in_momenta needs."""
    return {name: np.asarray(steps[name], dtype=np.float64)
            for name in ("pz", "pmag", "pdgId", "volumeCopy")}


def st_in_momenta(vd: dict[str, np.ndarray]) -> np.ndarray:
    """|p| of every downstream-going muon crossing of ST_In."""
    at = ((vd["volumeCopy"] == VD_ST_IN) & (np.abs(vd["pdgId"]) == MUON_PDG)
          & (vd["pz"] > 0))
    return vd["pmag"][at]


# --- the analysis ------------------------------------------------------------------

def stop_metrics(stops: dict[str, np.ndarray], p_st_in: np.ndarray,
                 n_gen_events: int) -> dict[str, float]:
    """METRICS, in order, from the stops and the ST_In momenta."""
    def stats(values, *kinds):
        out = {"mean": values.mean(), "rms": values.std(),
               "median": np.median(values)}
        return [float(out[k]) if values.size else float("nan") for k in kinds]

    z_mean, z_rms, z_median = stats(stops["z"], "mean", "rms", "median")
    r_mean, r_rms, r_median = stats(stops["r"], "mean", "rms", "median")
    t_mean, t_median = stats(stops["time"], "mean", "median")
    p_mean, p_median, p_rms = stats(p_st_in, "mean", "median", "rms")
    return dict(zip(METRICS, (
        float(n_gen_events), float(stops["z"].size),
        z_mean, z_rms, z_median, r_mean, r_rms, r_median, t_mean, t_median,
        float(p_st_in.size), p_mean, p_median, p_rms)))


def histogram(values: np.ndarray, width: float, name: str) -> Spectrum:
    """A histogram of `values` with bins of `width`, edges on its multiples."""
    low = np.floor(values.min() / width) * width if values.size else 0.0
    high = np.floor(values.max() / width) * width + width if values.size else width
    nbins = max(1, int(round((high - low) / width)))
    return Spectrum.from_values(values, nbins, low, low + nbins * width, name=name)


def _write_outputs(outdir: Path, stops: dict, p_st_in: np.ndarray) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot

    hists = {"stop_z": histogram(stops["z"], BIN_WIDTHS["z"], "stop_z"),
             "stop_r": histogram(stops["r"], BIN_WIDTHS["r"], "stop_r"),
             "stop_time": histogram(stops["time"], BIN_WIDTHS["time"], "stop_time"),
             "p_st_in": histogram(p_st_in, BIN_WIDTHS["p_st_in"], "p_st_in")}
    z, r = hists["stop_z"], hists["stop_r"]
    rz, _, _ = np.histogram2d(stops["z"], stops["r"], bins=(z.edges(), r.edges()))

    written = []
    root_path = outdir / "muon_stops_hists.root"
    with uproot.recreate(root_path) as f:
        for name, h in hists.items():
            f[name] = (h.values, h.edges())
        f["stop_r_vs_z"] = (rz, z.edges(), r.edges())
    written.append(str(root_path))

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    labels = {"stop_z": ("Stop z (mm)", "Stops along the DS", "mm"),
              "stop_r": ("Stop radius from the DS axis (mm)", "Stop radius", "mm"),
              "stop_time": ("Stop time (ns)", "Stop time", "ns"),
              "p_st_in": ("Momentum at ST_In (MeV/c)",
                          "Momentum of the stopped muons at ST_In", "MeV/c")}
    for name, h in hists.items():
        xlabel, title, unit = labels[name]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.stairs(h.values, h.edges(), color="#2a78d6", lw=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{'Crossings' if name == 'p_st_in' else 'Stops'} / {h.width:g} {unit}")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        path = figdir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    mesh = ax.pcolormesh(z.edges(), r.edges(), rz.T, cmap="Blues",
                         norm=matplotlib.colors.LogNorm(vmin=1) if rz.max() > 0 else None)
    fig.colorbar(mesh, ax=ax, label=f"Stops / ({z.width:g} mm x {r.width:g} mm)")
    ax.set_xlabel("Stop z (mm)")
    ax.set_ylabel("Stop radius from the DS axis (mm)")
    ax.set_title("Stop position", loc="left", fontweight="bold")
    fig.tight_layout()
    path = figdir / "stop_r_vs_z.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))
    return written


def run(context: RunContext) -> RunOutcome:
    """Dump the stops and their VD steps with a mu2e job, then profile them."""
    stop_filter = context.params["stop_filter"]
    extra: dict = {"stop_filter": stop_filter}
    try:
        text = job_fcl_text(stop_filter)
    except MuonStopsError as exc:
        return RunOutcome(error=str(exc), extra=extra)
    context.outdir.mkdir(parents=True, exist_ok=True)
    fcl = context.outdir / "muon_stops.fcl"
    fcl.write_text(text, encoding="utf-8")

    outcome = run_mu2e_job(
        fcl=fcl,
        input_paths=context.input_paths,
        outdir=context.outdir,
        single=len(context.input_paths) == 1 and not context.wants_file_list,
        timeout_s=context.timeout_s,
        max_events=context.max_events,
    )
    extra["returncode"] = outcome.returncode
    files = list(outcome.written_root_files)
    if outcome.timed_out:
        return RunOutcome(files=files, log_path=outcome.log_path, extra=extra,
                          error=f"mu2e timed out after {context.timeout_s}s on "
                                f"{len(context.input_paths)} input file(s)")
    if outcome.failed:
        extra["stdout_tail"] = outcome.stdout_tail()
        return RunOutcome(
            files=files, log_path=outcome.log_path, extra=extra,
            error=f"mu2e exited {outcome.returncode}. If it could not find a "
                  f"product, check stop_filter ('{stop_filter}'): TargetStopFilter "
                  "for a TargetStops file, PolyStopFilter or IPAStopFilter for "
                  "PolyStops or IPAStops")
    n_gen = parse_gen_count(outcome.stdout)
    if n_gen is None:
        extra["stdout_tail"] = outcome.stdout_tail()
        return RunOutcome(files=files, log_path=outcome.log_path, extra=extra,
                          error="GenEventCountReader printed no generated-event "
                                "total: the input has no GenEventCount bookkeeping")
    if not files:
        return RunOutcome(log_path=outcome.log_path, extra=extra,
                          error="mu2e wrote no ROOT file to read the stops from")
    try:
        raw_stops, raw_steps = read_dumps(Path(files[0]))
        stops, vd = stop_variables(raw_stops), vd_columns(raw_steps)
    except MuonStopsError as exc:
        return RunOutcome(files=files, log_path=outcome.log_path, extra=extra,
                          error=str(exc))
    if stops["z"].size == 0:
        return RunOutcome(
            files=files, log_path=outcome.log_path, extra=extra,
            error=f"no stopped particle in {stop_filter}'s SimParticles — is the "
                  "input a stop file of the MuBeam stage?")
    p_st_in = st_in_momenta(vd)
    pdg, count = np.unique(stops["pdg"].astype(int), return_counts=True)
    extra["stops_by_pdg"] = {int(k): int(n) for k, n in zip(pdg, count)}
    if max_events := context.max_events:
        extra["max_events"] = max_events
    files += _write_outputs(context.outdir, stops, p_st_in)
    return RunOutcome(metrics=stop_metrics(stops, p_st_in, n_gen), files=files,
                      log_path=outcome.log_path, extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    return (f"{metrics['n_stops']:g} stops (from {metrics['n_gen_events']:g} "
            f"generated events): z {metrics['stop_z_mean']:.1f} +- "
            f"{metrics['stop_z_rms']:.1f} mm, radius {metrics['stop_r_mean']:.1f} mm "
            f"mean, time median {metrics['stop_time_median']:.0f} ns; momentum at "
            f"ST_In median {metrics['p_st_in_median']:.1f} MeV/c "
            f"({metrics['n_st_in']:g} crossings).")


SPEC = AnalysisSpec(
    name="muon_stops",
    fcl=FCL,
    input_kind="art_files",
    run=run,
    description=(
        "Stopped-muon profile of a MuBeam stop file: where (z along the DS, "
        "radius from its axis), when, and with what momentum at ST_In the "
        "muons stopped, as metrics, plots and ROOT histograms (StoppedParticles"
        "Dumper and StepPointMCDumper in one mu2e job)."
    ),
    metrics=METRICS,
    units=UNITS,
    parameters=(
        ParamSpec(
            name="stop_filter",
            description="Label of the stop filter whose SimParticles and "
                        "virtual-detector steps to read: TargetStopFilter for "
                        "a TargetStops file, PolyStopFilter for PolyStops, "
                        "IPAStopFilter for IPAStops.",
            default="TargetStopFilter", kind="text",
        ),
    ),
    summarize=summarize,
    input_hint=(
        "A MuBeam-stage stop file, e.g. sim.<owner>.TargetStops.<config>."
        "<sequencer>.art (or PolyStops / IPAStops with the matching "
        "stop_filter); several are combined in one job."
    ),
)
