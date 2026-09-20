"""Event counts from Mu2eOptAna/fcl/print_counts.fcl.

`print_counts.fcl` prints three things, and this parses all three:

    N Event records found                        -> events kept in the file
    GenEventCount total: N events in M SubRuns   -> generated events upstream
    ProductPrint mu2e::PrescaleFilterFraction_<filter>__<process>
      Fraction passing filter F N Seen S with prescale fraction P

The first two are there for any art file with subrun bookkeeping. The third is
not: only a job that ran a prescale filter prints one, and the module label it
prints depends on how that job named its filters. So nothing here assumes a
prescale filter exists, or what it would be called — `prescale_filter` is
optional, and the answer means two different things depending on whether it is
given:

    unset   prescale = 1: the events this file holds per generated event,
            whatever produced it
    given   that filter's prescale is divided out, giving the rate *before*
            the prescale threw events away

        saved_per_gen_event = n_events / (n_gen_events * prescale)

Every prescale block the job printed is reported in the result's metadata
either way, so a file whose stream was prescaled under a label the caller did
not expect is diagnosable without re-running: the block is right there, and
naming it in `prescale_filter` is a second call, not a second job.

A production job runs several prescale filters (target stops, poly stops, the
early window, ...) and writes *all* of their products into *every* output
stream, so a file from one stream parses perfectly well against another's
filter and would be silently divided by the wrong prescale. `wrong_dataset`
therefore checks the input's Mu2e file name against the stream the named
filter belongs to before the job starts. With no filter named there is no
division and so nothing to get wrong, and the check does not apply.

`muon_stop_rate` is this analysis with the target-stop filter required by
default and one more factor applied; it imports everything below rather than
restating it, so the parsing and the input check have one home.
"""

import re
from pathlib import Path

from ..mu2e_job import run_mu2e_job
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome

# Relative: resolved against the configured code (a work area or an
# unpacked tarball), or left to art's FHICL_FILE_PATH for a Musing.
FCL = Path("Mu2eOptAna/fcl/print_counts.fcl")

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
    "saved_per_gen_event": "events / generated event",
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


def dataset_hint(filter_label: str | None) -> str:
    """What a file of this filter's stream should have in its description.

    `TargetStopPrescaleFilter` -> "targetstop", which a TargetStops file's
    description contains. No label, or one not shaped like that, gives "", and
    the caller then has nothing to check against.
    """
    if not filter_label or not filter_label.endswith(FILTER_SUFFIX):
        return ""
    return filter_label[:-len(FILTER_SUFFIX)].lower()


def wrong_dataset(paths: list[Path], filter_label: str | None = None) -> list[str]:
    """Input files whose name says they are not this filter's stream.

    Empty when no filter is named: nothing is being divided out, so no file is
    the wrong one.
    """
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
    """Every PrescaleFilterFraction block, keyed by the filter's module label.

    Empty for a job that ran no prescale filter, which is not an error — see
    the module docstring.
    """
    return {
        match["label"]: {
            "process": match["process"],
            "fraction_passing": float(match["fraction"]),
            "n_seen": float(match["seen"]),
            "prescale": float(match["prescale"]),
        }
        for match in _PRESCALE_BLOCK.finditer(stdout)
    }


def parse_counts(stdout: str, filter_label: str | None = None) -> dict[str, float]:
    """Events, generated events, and the prescale to divide out.

    `filter_label` None (or "") means no prescale filter applies and the
    prescale is 1; a label that was asked for and is not there is an error,
    because silently falling back to 1 would report a prescaled file's rate
    short by whatever the prescale was.

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
    if filter_label:
        if filter_label not in filters:
            found = ", ".join(sorted(filters)) or "none"
            raise CountsError(
                f"no prescale filter '{filter_label}' in the job output "
                f"(found: {found}) — name one of those, or leave "
                f"prescale_filter unset to count the file as it is"
            )
        prescale = filters[filter_label]["prescale"]
    else:
        # No filter asked for: the file's own events, undivided.
        prescale = 1.0

    n_gen_events = float(gen.group(1))
    if n_gen_events <= 0.0:
        raise CountsError("the input reports 0 generated events")
    if prescale <= 0.0:
        raise CountsError(f"prescale fraction of '{filter_label}' is {prescale:g}")

    return {
        "n_events": float(events.group(1)),
        "n_gen_events": n_gen_events,
        "prescale": prescale,
    }


def run_counts_job(context: RunContext, filter_label: str | None) -> RunOutcome:
    """Run print_counts.fcl over the input file(s) and parse what it printed.

    Shared by every analysis built on these counts. On success the outcome's
    `metrics` are the raw counts — `n_events`, `n_gen_events`, `prescale` —
    for the caller to turn into whatever rate it reports; on failure `error`
    says what happened and `extra`/`log_path` say where to look. Either way
    `extra` carries every prescale block the job printed.
    """
    # Checked before the job runs: every output stream of a production job
    # carries every filter's product, so a file from another stream would be
    # scaled by this one's prescale and look perfectly healthy.
    mismatched = wrong_dataset(context.input_paths, filter_label)
    if mismatched:
        return RunOutcome(error=(
            f"file(s) from another stream: {', '.join(mismatched)}. This "
            f"divides out the prescale of {filter_label}, which the production "
            "job writes into every output, so a file from another stream would "
            f"be divided by the wrong one. Pass a "
            f"*.{dataset_hint(filter_label)}*.art file, or set prescale_filter "
            "to this file's own filter — which 'count' with no prescale_filter "
            "will list, since it runs the job without needing to know one"
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

    # Every filter the job printed, so a stream prescaled under a label the
    # caller did not expect can be read off the result rather than re-run.
    if filter_label:
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

    return RunOutcome(metrics=counts, log_path=outcome.log_path, extra=extra)


def saved_rates(counts: dict[str, float]) -> dict[str, float]:
    """The counts plus the rate derived from them."""
    return {
        **counts,
        "saved_per_gen_event":
            counts["n_events"] / (counts["n_gen_events"] * counts["prescale"]),
    }


def summarize(metrics: dict[str, float]) -> str:
    return (
        f"{metrics['n_events']:g} events kept from "
        f"{metrics['n_gen_events']:g} generated (prescale "
        f"{metrics['prescale']:g}): {metrics['saved_per_gen_event']:.4g} "
        f"events per generated event."
    )


def run(context: RunContext) -> RunOutcome:
    """Run print_counts.fcl over the input file(s) and turn its counts into a rate."""
    # "" is a real answer here, not a missing one: no prescale filter applies.
    filter_label = str(context.params["prescale_filter"]) or None

    outcome = run_counts_job(context, filter_label)
    if outcome.error is not None:
        return outcome
    return RunOutcome(metrics=saved_rates(outcome.metrics),
                      log_path=outcome.log_path, extra=outcome.extra)


SPEC = AnalysisSpec(
    name="count",
    fcl=FCL,
    input_kind="art_files",
    run=run,
    description=(
        "Event count and rate for any art file: events kept, generated events "
        "and — if the file's stream was prescaled and you name the filter — "
        "the prescale (print_counts), turned into events per generated event."
    ),
    metrics=("n_events", "n_gen_events", "prescale", "saved_per_gen_event"),
    units=METRIC_UNITS,
    parameters=(
        ParamSpec(
            name="prescale_filter",
            description=(
                "Module label of the prescale filter that thinned this file's "
                "stream, as print_counts names it (e.g. "
                "'TargetStopPrescaleFilter'). Leave it unset for a file that "
                "was not prescaled: the prescale is then 1 and the rate is the "
                "file's own. The result lists every filter the job printed, so "
                "an unset run shows what there was to name."
            ),
            default="",
            kind="text",
            allow_empty=True,
        ),
    ),
    summarize=summarize,
    input_hint=(
        "Any art file whose subrun bookkeeping carries a generated-event "
        "count (GenEventCount) — plus the production job's prescale-filter "
        "product, if prescale_filter names one."
    ),
)
