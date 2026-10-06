"""CE sensitivity from one configuration's own files: autoresearch's sob.

The whole chain behind S/sqrt(B) for a stopping-target configuration, as
one analysis over the files its simulation wrote:

1. `count` (print_counts.fcl) over the target-stop files
   (sim.*.TargetStops.*.art): the mu- stops they hold, the MuBeam events
   generated to make them, and the target-stop prescale.
2. EdepAna (edep.fcl) over the CE files (dts.*.CeEndpoint.*.art): the CE
   events seen, the CE events generated, and the nts ROOT file.
3. The absolute CE efficiency,

       ce_abs_eff = input_correction
                    * muminus_stops / (mubeam_sim_total * prescale)
                    * ce_seen / ce_simulated_events

   where input_correction is the generated MuBeam events per POT upstream
   of the stops' stage.
4. approx_ce_sensitivity's window scan on that nts file's EdepAna tree,
   with sig_eff = ce_abs_eff and the caller's cosmic rate, DIO fraction,
   DIO table and selection.

One analysis, not four chained ones: the efficiency needs numbers from both
jobs, and the files of both upstream stages arrive as one list. The
denominators are the files' own generated-event counts (each file's SubRuns
carry its job's GenEventCount), so no events-per-job number has to travel
with them.
"""

import time
from pathlib import Path

from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from . import approx_ce_sensitivity as sens
from . import count
from . import edep

STOPS_TAG = ".TargetStops."
CE_TAG = ".CeEndpoint."
PRESCALE_FILTER = "TargetStopPrescaleFilter"

# The two mu2e jobs, as module attributes so the tests can stand in for them.
count_job = count.run_counts_job
edep_job = edep.run

# Reported in this order; the first is the figure of merit.
METRICS = (
    "s_over_sqrt_b", "ce_abs_eff", "ce_seen", "ce_simulated_events",
    "muminus_stops", "mubeam_sim_total", "prescale",
    "signal_box_low_mev", "signal_box_high_mev", "signal_rate",
    "dio_background", "cosmic_background",
)


class InputError(ValueError):
    """Inputs or counts this analysis cannot turn into an efficiency."""


def split_inputs(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """(target-stop files, CE files), by the Mu2e file name's description.
    Any other file, or an empty group, is an InputError naming it."""
    stops = [p for p in paths if STOPS_TAG in p.name]
    ce = [p for p in paths if CE_TAG in p.name]
    other = [p.name for p in paths
             if STOPS_TAG not in p.name and CE_TAG not in p.name]
    problems = []
    if other:
        problems.append("file(s) neither TargetStops nor CeEndpoint: "
                         + ", ".join(other))
    if not stops:
        problems.append("no TargetStops file (a name containing "
                         f"'{STOPS_TAG}')")
    if not ce:
        problems.append(f"no CeEndpoint file (a name containing '{CE_TAG}')")
    if problems:
        raise InputError("; ".join(problems))
    return stops, ce


def ce_efficiency(*, input_correction: float, muminus_stops: float,
                   mubeam_sim_total: float, prescale: float, ce_seen: float,
                   ce_simulated_events: float) -> float:
    """The absolute CE efficiency (module docstring, step 3)."""
    for name, value in (("muminus_stops", muminus_stops),
                        ("mubeam_sim_total", mubeam_sim_total),
                        ("prescale", prescale), ("ce_seen", ce_seen),
                        ("ce_simulated_events", ce_simulated_events)):
        if not value > 0.0:
            raise InputError(f"{name} is {value:g}: the efficiency needs "
                              f"every count > 0")
    efficiency = (input_correction * muminus_stops
                  / (mubeam_sim_total * prescale)
                  * ce_seen / ce_simulated_events)
    if not 0.0 < efficiency <= 1.0:
        raise InputError(f"CE efficiency {efficiency:g} is outside (0, 1]")
    return efficiency


def assemble_metrics(counts: dict, ce: dict, scan: dict,
                      efficiency: float) -> dict[str, float]:
    """The reported metrics, in METRICS order: the count job's counts,
    EdepAna's on the CE files, the scan's, and the efficiency."""
    return {
        "s_over_sqrt_b": scan["sensitivity"],
        "ce_abs_eff": efficiency,
        "ce_seen": ce["n_events"],
        "ce_simulated_events": ce["n_gen_events"],
        "muminus_stops": counts["n_events"],
        "mubeam_sim_total": counts["n_gen_events"],
        "prescale": counts["prescale"],
        "signal_box_low_mev": scan["signal_box_low_mev"],
        "signal_box_high_mev": scan["signal_box_high_mev"],
        "signal_rate": scan["signal_rate"],
        "dio_background": scan["dio_background"],
        "cosmic_background": scan["cosmic_background"],
    }


def run(context: RunContext) -> RunOutcome:
    try:
        stops, ce_files = split_inputs(context.input_paths)
    except InputError as exc:
        return RunOutcome(error=str(exc))
    if context.max_events is not None:
        return RunOutcome(error=(
            "max_events would leave the generated-event counts covering "
            "events the jobs never read; the efficiency needs every event"))
    params = context.params
    # One budget for both jobs: the caller's timeout covers the analysis.
    deadline = time.monotonic() + context.timeout_s

    def job(paths: list[Path], name: str, job_params: dict) -> RunContext:
        return RunContext(
            input_paths=paths, outdir=context.outdir / name, params=job_params,
            timeout_s=max(1, int(deadline - time.monotonic())),
            wants_file_list=True)

    counts = count_job(job(stops, "count", {}), PRESCALE_FILTER)
    extra = {"count": counts.extra,
             "count_log": str(counts.log_path) if counts.log_path else None}
    if counts.error is not None:
        return RunOutcome(error=f"counting the TargetStops files: {counts.error}",
                           log_path=counts.log_path, extra=extra)

    # edep's own selection only adds its selected-event counts; the scan
    # below applies this analysis' selection to the same tree.
    ce = edep_job(job(ce_files, "edep",
                      {"selection": edep.DEFAULT_SELECTION}))
    extra.update(edep=ce.extra,
                 edep_log=str(ce.log_path) if ce.log_path else None)
    if ce.error is not None:
        return RunOutcome(error=f"EdepAna on the CeEndpoint files: {ce.error}",
                           files=ce.files, log_path=ce.log_path, extra=extra)
    if len(ce.files) != 1:
        return RunOutcome(
            error=(f"EdepAna wrote {len(ce.files)} ROOT files, expected one "
                   f"nts file: {ce.files}"),
            files=ce.files, log_path=ce.log_path, extra=extra)

    try:
        efficiency = ce_efficiency(
            input_correction=params["input_correction"],
            muminus_stops=counts.metrics["n_events"],
            mubeam_sim_total=counts.metrics["n_gen_events"],
            prescale=counts.metrics["prescale"],
            ce_seen=ce.metrics["n_events"],
            ce_simulated_events=ce.metrics["n_gen_events"])
    except InputError as exc:
        return RunOutcome(error=str(exc), files=ce.files,
                           log_path=ce.log_path, extra=extra)

    scan = sens.compute(
        Path(ce.files[0]), context.outdir / "sensitivity",
        sig_eff=efficiency, npot=sens.NPOT,
        mean_pot_per_event=sens.MEAN_POT_PER_EVENT,
        cosmic_rate_per_s_per_mev=params["cosmic_rate_per_s_per_mev"],
        selection=params["selection"],
        dio_table=Path(params["dio_table"]),
        dio_fraction=params["dio_fraction"])
    extra["sensitivity"] = scan.extra
    files = ce.files + scan.files
    if scan.error is not None:
        return RunOutcome(error=f"sensitivity scan: {scan.error}", files=files,
                           log_path=scan.log_path or ce.log_path, extra=extra)
    return RunOutcome(
        metrics=assemble_metrics(counts.metrics, ce.metrics, scan.metrics,
                                  efficiency),
        files=files, log_path=scan.log_path, extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    return (
        f"S/sqrt(B) = {metrics['s_over_sqrt_b']:.4g} at CE efficiency "
        f"{metrics['ce_abs_eff']:.4g} ({metrics['muminus_stops']:g} stops "
        f"from {metrics['mubeam_sim_total']:g} generated, "
        f"{metrics['ce_seen']:g} of {metrics['ce_simulated_events']:g} CE "
        f"events seen)."
    )


SPEC = AnalysisSpec(
    name="ce_sensitivity",
    fcl=edep.FCL,
    input_kind="art_files",
    run=run,
    description=(
        "CE S/sqrt(B) for one configuration from its own files: its mu- "
        "stops (TargetStops) and CE events (CeEndpoint), in one list, turned "
        "into an absolute CE efficiency and then approx_ce_sensitivity's "
        "window scan."
    ),
    metrics=METRICS,
    units={"signal_box_low_mev": "MeV", "signal_box_high_mev": "MeV"},
    parameters=(
        ParamSpec(
            name="input_correction",
            description="Generated MuBeam events per proton on target "
                        "upstream of the stops' stage (autoresearch: "
                        "0.01278168, the fraction of POT reaching the "
                        "MuBeamCat resampler input).",
            minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="cosmic_rate_per_s_per_mev",
            description="Cosmic background rate, flat in momentum, per "
                        "second per MeV/c (approx_ce_sensitivity's). "
                        "Required: the choice moves the answer by orders of "
                        "magnitude.",
            minimum=0.0,
        ),
        ParamSpec(
            name="dio_fraction",
            description="Fraction of stopped muons that decay in orbit "
                        "(approx_ce_sensitivity's dio_fraction).",
            minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="dio_table",
            description="The DIO spectrum table (approx_ce_sensitivity's "
                        "dio_table).",
            kind="text",
        ),
        ParamSpec(
            name="selection",
            description="The EdepAna tree selection the signal shape and "
                        "energy-loss response are taken from "
                        "(approx_ce_sensitivity's selection).",
            default=sens.DEFAULT_SELECTION, kind="text", allow_empty=True,
        ),
    ),
    summarize=summarize,
    input_hint=(
        "One configuration's target-stop files (sim.*.TargetStops.*.art) and "
        "CE files (dts.*.CeEndpoint.*.art), in one list. Each file's SubRuns "
        "must carry its job's GenEventCount, and the target-stop files the "
        "TargetStopPrescaleFilter product. EdepAna must write its tree "
        "(Offline v13_39_00 or later)."
    ),
)
