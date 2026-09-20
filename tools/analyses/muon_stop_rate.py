"""Muon stopping rate: the `count` analysis with a target-stop reading.

The stopping-target sim file (sim.*.TargetStops.*.art) holds one event per
stopped muon that survived the production job's prescale, and it carries the
bookkeeping needed to turn that count into a rate: the generated-event count
of the stage that made it (MuBeam events resampled), and the
PrescaleFilterFraction products recording what each filter in that job kept.
`print_counts.fcl` prints all three, and `count.py` parses them — this module
imports that parsing rather than restating it, and adds the two things that
are about muon stops rather than about counting:

1. **The filter is required.** `count` treats a prescale filter as optional,
   because most files were never prescaled; a target-stop file always was, by
   `TargetStopPrescaleFilter`, so that is this analysis' default and "" is not
   an answer it takes. The file keeps only a fraction P of the events that
   passed, so the rate per generated event is

       stops_per_gen_event = n_events / (n_gen_events * prescale)

2. **One more factor**, converting it to the quantity worth comparing between
   beamline configurations — the caller supplies the efficiency of everything
   upstream (generated events of this stage per POT, i.e. POT -> MuBeam here):

       stops_per_pot = stops_per_gen_event * upstream_eff

Pass another label to read a different stream's rate out of its own file, or
to cope with a job that named its filters differently; the input's name is
checked against whichever filter is named, so `PolyStopPrescaleFilter` wants a
PolyStops file. Why that check exists at all — a production job writes every
filter's product into every output stream — is in `count.py`, along with the
parsing, the input check and the job run this shares with it.

Reach for `count` instead when the file was not prescaled, or when what is
wanted is the events it holds rather than a stopping rate per POT.
"""

from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from .count import FCL, run_counts_job

# The filter whose prescale applies to the target-stop output stream.
PRESCALE_FILTER = "TargetStopPrescaleFilter"

METRIC_UNITS = {
    "stops_per_gen_event": "stops / generated event",
    "stops_per_pot": "stops / POT",
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
    outcome = run_counts_job(context, str(context.params["prescale_filter"]))
    if outcome.error is not None:
        return outcome
    return RunOutcome(
        metrics=stop_rates(outcome.metrics, context.params["upstream_eff"]),
        log_path=outcome.log_path,
        extra=outcome.extra,
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
                "read a poly-stop file's own rate. Required: a target-stop "
                "file is always prescaled, so use the 'count' analysis for a "
                "file that is not."
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
