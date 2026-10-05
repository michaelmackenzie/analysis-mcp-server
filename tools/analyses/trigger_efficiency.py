"""Trigger efficiency: the fraction of signal events the trigger keeps.

Runs the requested trigger paths, with their prescales, over a signal art file
(trigger.py has the job). The efficiency is relative to the events in the
input: for a sample that was already filtered (e.g. "Triggerable"), fold that
selection's efficiency in separately.
"""

from ..spec import AnalysisSpec, RunContext, RunOutcome
from .trigger import (FCL, TrigReport, TriggerRun, binomial,
                      per_path_fractions, run_trigger_job, trigger_paths_param)

METRICS = ("n_events", "n_triggered", "efficiency", "efficiency_err")


def efficiency_metrics(report: TrigReport) -> dict[str, float]:
    """The overall efficiency from the job's TrigReport."""
    efficiency, err = binomial(report.n_passed, report.n_events)
    return {"n_events": float(report.n_events),
            "n_triggered": float(report.n_passed),
            "efficiency": efficiency, "efficiency_err": err}


def run(context: RunContext) -> RunOutcome:
    """Run the trigger over the signal file(s) and count what it kept."""
    result = run_trigger_job(context)
    if not isinstance(result, TriggerRun):
        return result
    paths = per_path_fractions(result)
    for counts in paths.values():
        counts["efficiency"] = counts.pop("fraction")
        counts["efficiency_err"] = counts.pop("fraction_err")
    return RunOutcome(
        metrics=efficiency_metrics(result.report),
        log_path=result.job.log_path,
        extra={**result.extra, "paths": paths},
    )


def summarize(metrics: dict[str, float]) -> str:
    return (f"trigger efficiency {metrics['efficiency']:.4g} "
            f"+- {metrics['efficiency_err']:.2g} "
            f"({metrics['n_triggered']:g} of {metrics['n_events']:g} events); "
            "per-path efficiencies in metadata.paths.")


SPEC = AnalysisSpec(
    name="trigger_efficiency",
    fcl=FCL,
    input_kind="art_files",
    run=run,
    description=(
        "Trigger efficiency on a signal sample: the fraction of its events "
        "passing any of the given trigger paths, with their prescales, and "
        "each path's own efficiency (metadata.paths)."
    ),
    metrics=METRICS,
    parameters=(trigger_paths_param(),),
    summarize=summarize,
    input_hint=(
        "Signal digi art file(s) the online trigger can run on, e.g. "
        "dig.mu2e.CeEndpointOnSpill.<version>.art. The efficiency is relative "
        "to the events in the input."
    ),
)
