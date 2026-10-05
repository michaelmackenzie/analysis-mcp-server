"""Trigger rate: how often the trigger fires on pileup, overall and per path.

Runs the requested trigger paths, with their prescales, over a pileup art
sample (trigger.py has the job), by default the data-like one-batch sample
mu2e-trig-config's CI uses. Each event is one beam microbunch, so a fraction
of events becomes a rate through the online event rate: the on-spill
microbunch rate (one per 1695 ns) times the spill duty factor, the fraction
of each accelerator cycle that delivers beam. That depends on the number of
proton batches per cycle — 0.322 in one-batch mode (1BB), 0.246 in
two-batch mode (2BB) — or can be given directly. The overall rate counts an
event once however many paths pass it.
"""

from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from .trigger import (DEFAULT_PILEUP_LIST, FCL, TrigReport, TriggerRun,
                      TriggerError, binomial, default_pileup_inputs,
                      per_path_fractions, run_trigger_job, trigger_paths_param)

# On-spill microbunch rate: one event per 1695 ns.
MICROBUNCH_RATE_HZ = 1.0 / 1.695e-6

# Spill duty factor for each batch mode (proton batches per accelerator cycle).
DUTY_FACTORS = {"1BB": 0.322, "2BB": 0.246}
BATCH_MODE = "1BB"

METRICS = ("n_events", "n_triggered", "accept_fraction", "accept_fraction_err",
           "rate_hz", "rate_hz_err", "onspill_rate_hz", "event_rate_hz",
           "duty_factor", "microbunch_rate_hz")


def duty_factor(batch_mode: str, override: float) -> float:
    """The duty factor: `override` if set (> 0), else the batch mode's."""
    if override > 0.0:
        return override
    try:
        return DUTY_FACTORS[batch_mode.strip().upper()]
    except KeyError:
        raise TriggerError(
            f"batch_mode '{batch_mode}' is not one of "
            f"{', '.join(DUTY_FACTORS)}; or give duty_factor directly"
        ) from None


def rate_metrics(report: TrigReport, microbunch_rate: float,
                 duty: float) -> dict[str, float]:
    """The overall accept fraction and rates from the job's TrigReport."""
    fraction, err = binomial(report.n_passed, report.n_events)
    event_rate = microbunch_rate * duty
    return {"n_events": float(report.n_events),
            "n_triggered": float(report.n_passed),
            "accept_fraction": fraction, "accept_fraction_err": err,
            # averaged over the accelerator cycle: what the online system sees
            "rate_hz": fraction * event_rate, "rate_hz_err": err * event_rate,
            # during the spill
            "onspill_rate_hz": fraction * microbunch_rate,
            # the normalization travels with the rates built on it
            "event_rate_hz": event_rate, "duty_factor": duty,
            "microbunch_rate_hz": microbunch_rate}


def run(context: RunContext) -> RunOutcome:
    """Run the trigger over the pileup sample and turn its counts into rates."""
    try:
        duty = duty_factor(context.params["batch_mode"],
                           context.params["duty_factor"])
    except TriggerError as exc:
        return RunOutcome(error=str(exc))
    result = run_trigger_job(context)
    if not isinstance(result, TriggerRun):
        return result
    microbunch_rate = context.params["microbunch_rate_hz"]
    event_rate = microbunch_rate * duty
    paths = per_path_fractions(result)
    for counts in paths.values():
        counts["rate_hz"] = counts["fraction"] * event_rate
        counts["rate_hz_err"] = counts["fraction_err"] * event_rate
        counts["onspill_rate_hz"] = counts["fraction"] * microbunch_rate
    return RunOutcome(
        metrics=rate_metrics(result.report, microbunch_rate, duty),
        log_path=result.job.log_path,
        extra={**result.extra, "paths": paths,
               "batch_mode": context.params["batch_mode"]},
    )


def summarize(metrics: dict[str, float]) -> str:
    return (f"trigger rate {metrics['rate_hz']:.4g} +- {metrics['rate_hz_err']:.2g} Hz "
            f"averaged over the cycle ({metrics['onspill_rate_hz']:.4g} Hz on "
            f"spill); {metrics['n_triggered']:g} of {metrics['n_events']:g} "
            f"events at {metrics['event_rate_hz']:.4g} events/s (duty factor "
            f"{metrics['duty_factor']:.3g}); per-path rates in metadata.paths.")


SPEC = AnalysisSpec(
    name="trigger_rate",
    fcl=FCL,
    input_kind="art_files",
    run=run,
    description=(
        "Trigger rate on pileup: the fraction of events passing any of the "
        "given trigger paths, with their prescales, as a rate in Hz averaged "
        "over the accelerator cycle (spill duty factor from the batch mode, "
        "or given), and each path's own rate (metadata.paths)."
    ),
    metrics=METRICS,
    units={"rate_hz": "Hz", "rate_hz_err": "Hz", "onspill_rate_hz": "Hz",
           "event_rate_hz": "Hz", "microbunch_rate_hz": "Hz"},
    parameters=(
        trigger_paths_param(),
        ParamSpec(
            name="batch_mode",
            description="Proton batches per accelerator cycle, which sets "
                        "the spill duty factor: "
                        + ", ".join(f"'{mode}' -> {duty}"
                                    for mode, duty in DUTY_FACTORS.items())
                        + ". Ignored when duty_factor is given.",
            default=BATCH_MODE, kind="text",
        ),
        ParamSpec(
            name="duty_factor",
            description="Spill duty factor, the fraction of the accelerator "
                        "cycle with beam, overriding batch_mode's. 0 (the "
                        "default) means take it from batch_mode.",
            default=0.0, minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="microbunch_rate_hz",
            description="On-spill events (microbunches) per second. The "
                        "default is one per 1695 ns.",
            default=MICROBUNCH_RATE_HZ, minimum=0.0,
        ),
    ),
    summarize=summarize,
    input_hint=(
        "Pileup (no-primary) digi art file(s), one microbunch per event. "
        f"Defaults to the files in {DEFAULT_PILEUP_LIST} when no data_file(s) "
        "are passed."
    ),
    default_inputs=default_pileup_inputs,
    default_inputs_hint=f"the art files listed in {DEFAULT_PILEUP_LIST} of the "
                        "configured code (mu2e-trig-config's CI pileup sample)",
)
