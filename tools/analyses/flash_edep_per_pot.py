"""Tracker energy from the early beam flash, per proton on target.

EdepAna (edep.py) over the early-flash files gives the tracker StrawGasStep
ionizing energy per generated event. Each generated event is one resampled
beam electron, so dividing by `pot_per_electron` -- protons on target per
resampled electron, a property of the electron-beam dataset the files were
resampled from -- gives energy per POT.

The per-generated-event average needs every event of every file, so
max_events is refused.
"""

from dataclasses import replace

from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from . import edep

# The mu2e job, as a module attribute so the tests can stand in for it.
edep_job = edep.run


def flash_metrics(edep_metrics: dict[str, float],
                   pot_per_electron: float) -> dict[str, float]:
    """flash_edep_per_pot first, then EdepAna's own metrics."""
    return {
        "flash_edep_per_pot":
            edep_metrics["avg_trk_edep_per_gen_event_mev"] / pot_per_electron,
        **edep_metrics,
    }


def run(context: RunContext) -> RunOutcome:
    if context.max_events is not None:
        return RunOutcome(error=(
            "max_events would leave the generated-event counts covering "
            "events EdepAna never read; flash per POT needs every event"))
    pot_per_electron = context.params["pot_per_electron"]
    if not pot_per_electron > 0.0:
        return RunOutcome(
            error=f"pot_per_electron must be > 0, got {pot_per_electron:g}")

    # edep's selection only adds its selected-event counts, reported as is.
    outcome = edep_job(replace(context,
                               params={"selection": edep.DEFAULT_SELECTION}))
    if outcome.error is not None:
        return outcome
    metrics = outcome.metrics
    if not metrics["n_gen_events"] > 0.0:
        return RunOutcome(error="the files report 0 generated events",
                          files=outcome.files, log_path=outcome.log_path,
                          extra=outcome.extra)
    if not metrics["avg_trk_edep_per_gen_event_mev"] > 0.0:
        return RunOutcome(
            error=(f"zero tracker energy in {metrics['n_events']:g} events "
                   f"from {metrics['n_gen_events']:g} generated: too few "
                   f"events reached the early-flash output"),
            files=outcome.files, log_path=outcome.log_path,
            extra=outcome.extra)
    return RunOutcome(metrics=flash_metrics(metrics, pot_per_electron),
                      files=outcome.files, log_path=outcome.log_path,
                      extra=outcome.extra)


def summarize(metrics: dict[str, float]) -> str:
    return (
        f"{metrics['flash_edep_per_pot']:.6g} MeV per POT in the tracker "
        f"({metrics['avg_trk_edep_per_gen_event_mev']:.6g} MeV per generated "
        f"event over {metrics['n_gen_events']:g} generated)."
    )


SPEC = AnalysisSpec(
    name="flash_edep_per_pot",
    fcl=edep.FCL,
    input_kind="art_files",
    run=run,
    description=(
        "Tracker (StrawGasStep) ionizing energy per proton on target from "
        "early-flash files: EdepAna's per-generated-event average divided by "
        "the protons on target per resampled electron."
    ),
    metrics=("flash_edep_per_pot",) + edep.SPEC.metrics,
    units={"flash_edep_per_pot": "MeV / POT", **edep.METRIC_UNITS},
    parameters=(
        ParamSpec(
            name="pot_per_electron",
            description="Protons on target per resampled beam electron, a "
                        "property of the electron-beam dataset (EleBeamCat: "
                        "25000000 / 2166994 = 11.5367).",
            minimum=0.0,
        ),
    ),
    summarize=summarize,
    input_hint=(
        "Early-flash art files (dts.*.EarlyEleBeamFlash.*.art) whose SubRuns "
        "carry the resampling job's GenEventCount."
    ),
)
