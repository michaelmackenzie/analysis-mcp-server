"""Trigger timing: the processing time per event of the given trigger paths.

Runs the requested trigger paths, with their prescales, over a pileup art
sample (trigger.py has the job), by default the data-like one-batch sample
mu2e-trig-config's CI uses, and reads the per-event times from the
TimeTracker database the job writes. An event's time is the sum of its
modules' times without fetching the data (the Prefetch module and the input
source), as mu2eTimingPlotsMaker totals it; the fetch is reported apart. The
first `skip_events` events are left out: the first carries the database and
geometry initialization.

The times are wall-clock on whatever node runs the job, so they compare
configurations run on the same node, not absolute online budgets.
"""

import numpy as np

from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from .trigger import (DEFAULT_PILEUP_LIST, FCL, TIMING_DB_NAME, TriggerError,
                      TriggerRun, default_pileup_inputs, read_timing_db,
                      run_trigger_job, trigger_paths_param)

SKIP_EVENTS = 1

METRICS = ("n_events_timed", "mean_time_ms", "mean_time_err_ms",
           "median_time_ms", "p90_time_ms", "p99_time_ms", "max_time_ms",
           "rms_time_ms")


def timing_metrics(times_s: np.ndarray) -> dict[str, float]:
    """Summary statistics of per-event times, in ms."""
    ms = times_s * 1e3
    n = ms.size
    rms = float(ms.std(ddof=1)) if n > 1 else 0.0
    return {
        "n_events_timed": float(n),
        "mean_time_ms": float(ms.mean()),
        "mean_time_err_ms": rms / np.sqrt(n) if n > 1 else 0.0,
        "median_time_ms": float(np.median(ms)),
        "p90_time_ms": float(np.percentile(ms, 90)),
        "p99_time_ms": float(np.percentile(ms, 99)),
        "max_time_ms": float(ms.max()),
        "rms_time_ms": rms,
    }


def run(context: RunContext) -> RunOutcome:
    """Run the trigger over the pileup sample and summarize its event times."""
    result = run_trigger_job(context)
    if not isinstance(result, TriggerRun):
        return result
    skip = int(context.params["skip_events"])
    try:
        timed = read_timing_db(context.outdir / TIMING_DB_NAME, skip)
    except TriggerError as exc:
        return RunOutcome(log_path=result.job.log_path, extra=result.extra,
                          error=str(exc))
    if timed.times.size < 2:
        return RunOutcome(
            log_path=result.job.log_path, extra=result.extra,
            error=f"only {timed.times.size} event(s) left to time after "
                  f"skipping {timed.n_skipped}; process more events",
        )
    return RunOutcome(
        metrics=timing_metrics(timed.times),
        files=[str(context.outdir / TIMING_DB_NAME)],
        log_path=result.job.log_path,
        extra={
            **result.extra,
            "skipped_events": timed.n_skipped,
            # left out of the metrics above, reported so the split is visible
            "fetch_time_mean_ms": float(timed.fetch_times.mean() * 1e3),
            "full_event_time_mean_ms": float(timed.full_event_times.mean() * 1e3),
            # TimeTracker charges a shared module to the first path that ran
            # it, so these are what each path added, in the order they ran.
            "path_time_ms": {p: t * 1e3 for p, t in sorted(timed.per_path.items())},
            # Every module, in the order they ran: N(seen) after upstream
            # filters and prescales, ms per run and per event. The data
            # fetch is listed with counted = false. A shared module's runs
            # are split over the paths that reached it first; the
            # *_all_paths fields add them up.
            "modules": timed.module_table(),
        },
    )


def summarize(metrics: dict[str, float]) -> str:
    return (f"{metrics['mean_time_ms']:.3g} +- {metrics['mean_time_err_ms']:.2g} ms "
            f"per event (median {metrics['median_time_ms']:.3g}, "
            f"p99 {metrics['p99_time_ms']:.3g}, max {metrics['max_time_ms']:.3g} ms) "
            f"over {metrics['n_events_timed']:g} events; per-path and per-module "
            "times in metadata.")


SPEC = AnalysisSpec(
    name="trigger_timing",
    fcl=FCL,
    input_kind="art_files",
    run=run,
    description=(
        "Trigger processing time per event on pileup for the given trigger "
        "paths and prescales, without the time fetching the data: mean, "
        "median, tail and max, plus the time each "
        "path adds (metadata.path_time_ms) and every module's time "
        "(metadata.modules): the events it ran on after upstream filters "
        "(n_seen), its time per run and its time per event."
    ),
    metrics=METRICS,
    units={name: "ms" for name in METRICS if name.endswith("_ms")},
    parameters=(
        trigger_paths_param(),
        ParamSpec(
            name="skip_events",
            description="Events at the start of the job left out of the "
                        "timing; the first carries the database and "
                        "geometry initialization.",
            default=SKIP_EVENTS, minimum=0,
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
