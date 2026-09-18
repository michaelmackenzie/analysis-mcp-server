"""Muon stopping rate: counts from Mu2eOptAna/fcl/print_counts.fcl.

The stopping-target sim file (sim.*.TargetStops.*.art) holds one event per
stopped muon that survived the production job's prescale, and it carries the
bookkeeping needed to turn that count into a rate: the generated-event count
of the stage that made it (MuBeam events resampled), and the
PrescaleFilterFraction products recording what each filter in that job kept.
`print_counts.fcl` prints all three, and this parses them:

    N Event records found                        -> events in the file
    GenEventCount total: N events in M SubRuns   -> generated events
    ProductPrint mu2e::PrescaleFilterFraction_<filter>__<process>
      Fraction passing filter F N Seen S with prescale fraction P

The file keeps only a fraction P of the events that passed, so the rate per
generated event is

    stops_per_gen_event = n_events / (n_gen_events * prescale)

and one more factor converts it to the quantity worth comparing between
beamline configurations — the caller supplies the efficiency of everything
upstream (generated events of this stage per POT, i.e. POT -> MuBeam here):

    stops_per_pot = stops_per_gen_event * upstream_eff

A production job runs several prescale filters (target stops, poly stops, the
early window, ...) and prints one block each; the one that applies to a
TargetStops file is `TargetStopPrescaleFilter`, which is what the
`prescale_filter` parameter defaults to. Pass another label to read a
different stream's rate out of its own file, or to cope with a job that named
its filters differently. Every block the job printed is reported in the
result's metadata either way, so a mismatch is diagnosable without re-running.

That every output stream of the job carries *all* the filters' products is
also why the input is checked by name first: a poly-stop file parses perfectly
well against the target filter and would be divided by the wrong prescale,
giving a rate wrong by whatever the two differ by, with nothing in the output
to show for it. The check is on the filter's own name, so it follows
`prescale_filter`: `PolyStopPrescaleFilter` wants a PolyStops file.
"""

import re
from pathlib import Path

from ..mu2e_job import run_mu2e_job
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome

# Relative: resolved against the configured code (a work area or an
# unpacked tarball), or left to art's FHICL_FILE_PATH for a Musing.
FCL = Path("Mu2eOptAna/fcl/print_counts.fcl")

# The filter whose prescale applies to the target-stop output stream.
PRESCALE_FILTER = "TargetStopPrescaleFilter"

# Module labels end in this; what comes before it names the stream.
FILTER_SUFFIX = "PrescaleFilter"

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

_N_EVENTS = re.compile(r"^\s*(\d+)\s+Event records found\s*$", re.MULTILINE)
_N_GEN_EVENTS = re.compile(rf"GenEventCount total:\s*({_NUM})\s+events\s+in\s+(\d+)\s+SubRuns")
# Module label and process name: the product is <class>_<label>_<instance>_<process>
# and the instance is empty, hence the doubled underscore.
_PRESCALE_BLOCK = re.compile(
    r"ProductPrint\s+mu2e::PrescaleFilterFraction_(?P<label>\w+?)__(?P<process>\w+)\s*\n"
    rf"\s*Fraction passing filter\s+(?P<fraction>{_NUM})\s+"
    rf"N Seen\s+(?P<seen>{_NUM})\s+with prescale fraction\s+(?P<prescale>{_NUM})"
)

METRIC_UNITS = {
    "stops_per_gen_event": "stops / generated event",
    "stops_per_pot": "stops / POT",
}


class CountsError(RuntimeError):
    """A problem with the job's output the caller should see verbatim."""


def dataset_description(path: Path) -> str | None:
    """The description field of a Mu2e file name, or None if it isn't one.

    Mu2e names are <tier>.<owner>.<description>.<configuration>.<sequencer>.<format>
    (sim.mmackenz.TargetStops.Run1Bak_local0813111400.001800_00000000.art), so
    the description is the third field. Anything not shaped like that is left
    to the caller rather than guessed at.
    """
    fields = path.name.split(".")
    return fields[2] if len(fields) == 6 else None


def dataset_hint(filter_label: str) -> str:
    """What a file of this filter's stream should have in its description.

    `TargetStopPrescaleFilter` -> "targetstop", which a TargetStops file's
    description contains. A label not shaped like that gives "", and the
    caller then has nothing to check against.
    """
    stem = filter_label[:-len(FILTER_SUFFIX)] if filter_label.endswith(FILTER_SUFFIX) else ""
    return stem.lower()


def wrong_dataset(paths: list[Path], filter_label: str = PRESCALE_FILTER) -> list[str]:
    """Input files whose name says they are not this filter's stream."""
    hint = dataset_hint(filter_label)
    if not hint:
        return []
    return [
        f"{path.name} ({description})"
        for path in paths
        if (description := dataset_description(path)) is not None
        and hint not in description.lower()
    ]


def parse_prescale_filters(stdout: str) -> dict[str, dict[str, float]]:
    """Every PrescaleFilterFraction block, keyed by the filter's module label."""
    return {
        match["label"]: {
            "process": match["process"],
            "fraction_passing": float(match["fraction"]),
            "n_seen": float(match["seen"]),
            "prescale": float(match["prescale"]),
        }
        for match in _PRESCALE_BLOCK.finditer(stdout)
    }


def parse_counts(stdout: str, filter_label: str = PRESCALE_FILTER) -> dict[str, float]:
    """Events, generated events, and the prescale of `filter_label`.

    Raises CountsError naming what was missing, so a job that ran but printed
    something else is reported rather than silently producing no metrics.
    """
    events = _N_EVENTS.search(stdout)
    if not events:
        raise CountsError("no 'N Event records found' line in the job output")
    gen = _N_GEN_EVENTS.search(stdout)
    if not gen:
        raise CountsError("no 'GenEventCount total:' line in the job output — "
                          "the input has no generated-event bookkeeping")

    filters = parse_prescale_filters(stdout)
    if filter_label not in filters:
        found = ", ".join(sorted(filters)) or "none"
        raise CountsError(
            f"no prescale filter '{filter_label}' in the job output "
            f"(found: {found}) — is this a target-stop file?"
        )

    n_gen_events = float(gen.group(1))
    prescale = filters[filter_label]["prescale"]
    if n_gen_events <= 0.0:
        raise CountsError("the input reports 0 generated events")
    if prescale <= 0.0:
        raise CountsError(f"prescale fraction of '{filter_label}' is {prescale:g}")

    return {
        "n_events": float(events.group(1)),
        "n_gen_events": n_gen_events,
        "prescale": prescale,
    }


def stop_rates(counts: dict[str, float], upstream_eff: float) -> dict[str, float]:
    """The counts plus the two rates derived from them."""
    per_gen_event = counts["n_events"] / (counts["n_gen_events"] * counts["prescale"])
    return {
        **counts,
        "stops_per_gen_event": per_gen_event,
        "stops_per_pot": per_gen_event * upstream_eff,
    }


def summarize(metrics: dict[str, float]) -> str:
    return (
        f"{metrics['n_events']:g} stopped-muon events from "
        f"{metrics['n_gen_events']:g} generated (prescale "
        f"{metrics['prescale']:g}): {metrics['stops_per_gen_event']:.4g} stops "
        f"per generated event, {metrics['stops_per_pot']:.4g} stops / POT."
    )


def run(context: RunContext) -> RunOutcome:
    """Run print_counts.fcl over the input file(s) and turn its counts into rates."""
    filter_label = str(context.params["prescale_filter"])

    # Checked before the job runs: every output stream of a production job
    # carries every filter's product, so the wrong stop file would be scaled
    # by this stream's prescale and look perfectly healthy.
    mismatched = wrong_dataset(context.input_paths, filter_label)
    if mismatched:
        return RunOutcome(error=(
            f"file(s) from another stream: {', '.join(mismatched)}. This "
            f"reports the rate for {filter_label}, whose prescale the "
            "production job writes into every output, so a file from another "
            f"stream would be divided by the wrong one. Pass a "
            f"*.{dataset_hint(filter_label)}*.art file, or set prescale_filter "
            "to this file's own filter"
        ))

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
            log_path=outcome.log_path, extra=extra,
            error=f"mu2e timed out after {context.timeout_s}s on "
                  f"{len(context.input_paths)} input file(s)",
        )
    if outcome.failed:
        extra["stdout_tail"] = outcome.stdout_tail()
        return RunOutcome(log_path=outcome.log_path, extra=extra,
                          error=f"mu2e exited {outcome.returncode}")

    # Every filter the job printed, so a file written with other labels — or
    # the poly-stop and early-window rates — can be read off the result.
    extra["prescale_filter"] = filter_label
    extra["dataset_descriptions"] = sorted(
        {dataset_description(p) or p.name for p in context.input_paths}
    )
    extra["prescale_filters"] = parse_prescale_filters(outcome.stdout)
    try:
        counts = parse_counts(outcome.stdout, filter_label)
    except CountsError as exc:
        extra["stdout_tail"] = outcome.stdout_tail()
        return RunOutcome(log_path=outcome.log_path, extra=extra, error=str(exc))

    return RunOutcome(
        metrics=stop_rates(counts, context.params["upstream_eff"]),
        log_path=outcome.log_path,
        extra=extra,
    )


SPEC = AnalysisSpec(
    name="muon_stop_rate",
    fcl=FCL,
    input_kind="art_files",
    run=run,
    description=(
        "Muon stopping rate from a stopping-target sim file: events, generated "
        "events and the output prescale (print_counts), turned into stops per "
        "generated event and per POT."
    ),
    metrics=("n_events", "n_gen_events", "prescale", "stops_per_gen_event",
             "stops_per_pot"),
    units=METRIC_UNITS,
    parameters=(
        ParamSpec(
            name="prescale_filter",
            description=(
                "Module label of the prescale filter whose stream this file "
                "is, as print_counts names it. The default is the "
                "target-stop stream; pass e.g. 'PolyStopPrescaleFilter' to "
                "read a poly-stop file's own rate."
            ),
            default=PRESCALE_FILTER,
            kind="text",
        ),
        ParamSpec(
            name="upstream_eff",
            description=(
                "Efficiency of everything upstream of this stage: generated "
                "events of this file's stage per POT (POT -> MuBeam here). "
                "Multiplies the per-generated-event rate to give stops / POT."
            ),
            minimum=0.0,
        ),
    ),
    summarize=summarize,
    input_hint=(
        "A stopping-target sim file (sim.*.TargetStops.*.art) with the "
        "generated-event count and the production job's "
        f"{PRESCALE_FILTER} product — or the file of whatever stream "
        "prescale_filter names."
    ),
)
