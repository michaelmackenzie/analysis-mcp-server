"""Muon stopping rate: the `count` analysis with a target-stop reading.

The stopping-target sim file (sim.*.TargetStops.*.art) holds one event per
stopped muon that survived the production job's prescale, and it carries the
bookkeeping needed to turn that count into a rate: the generated-event count
of the stage that made it (MuBeam events resampled), and the
PrescaleFilterFraction products recording what each filter in that job kept.
`stop_counts.fcl` prints all three, as `print_counts.fcl` does, and `count.py`
parses them — this module imports that parsing rather than restating it, and
adds the things that are about muon stops rather than about counting:

1. **The filter is required.** `count` treats a prescale filter as optional,
   because most files were never prescaled; a target-stop file always was, by
   `TargetStopPrescaleFilter`, so that is this analysis' default and "" is not
   an answer it takes. The file keeps only a fraction P of the events that
   passed, so the rate per generated event is

       stops_per_gen_event = n_events / (n_gen_events * prescale)

2. **The charges are counted apart.** A stop file holds mu+ as well as mu-
   stops (about 1% in MDC2025), and only the mu- stops, captured at rest, feed
   a conversion or DIO normalization: Production's MuonStopSelector splits
   them into MuminusStopsCat and MuplusStopsCat before any primary is made.
   `stop_counts.fcl` runs the same two ParticleCodeFilters as trigger paths,
   on the stream's own SimParticle collection (`TargetStopFilter` for
   `TargetStopPrescaleFilter`), and art's TrigReport gives the events each
   passed:

       muminus_stops_per_gen_event = n_muminus / (n_gen_events * prescale)

   and the same for mu+. `stops_per_gen_event` keeps counting every stop.

3. **One more factor**, converting it to the quantity worth comparing between
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

import re
from pathlib import Path

from ..spec import FCL_DIR, AnalysisSpec, ParamSpec, RunContext, RunOutcome
from .count import FILTER_SUFFIX, CountsError, run_counts_job

FCL = FCL_DIR / "stop_counts.fcl"
JOB_FCL_NAME = "stop_counts_job.fcl"

# The filter whose prescale applies to the target-stop output stream.
PRESCALE_FILTER = "TargetStopPrescaleFilter"

# stop_counts.fcl's trigger paths, and their ParticleCodeFilter modules.
SPECIES_PATHS = {"muminus": "selectMuminus", "muplus": "selectMuplus"}
SELECTORS = ("muminusSelector", "muplusSelector")

# A row of art's TrigReport "Trigger-path summary": path ID, run, passed,
# failed, error, name. The per-module rows have the same shape but the
# modules' names, which are not looked up.
_PATH_ROW = re.compile(
    r"^TrigReport\s+\d+\s+(?P<run>\d+)\s+(?P<passed>\d+)\s+(?P<failed>\d+)"
    r"\s+(?P<error>\d+)\s+(?P<name>\w+)\s*$", re.MULTILINE)

METRIC_UNITS = {
    "stops_per_gen_event": "stops / generated event",
    "stops_per_pot": "stops / POT",
    "muminus_stops_per_gen_event": "mu- stops / generated event",
    "muminus_stops_per_pot": "mu- stops / POT",
    "muplus_stops_per_gen_event": "mu+ stops / generated event",
    "muplus_stops_per_pot": "mu+ stops / POT",
}


def stop_collection(filter_label: str) -> str:
    """The SimParticle collection of a stop stream, from its prescale filter:
    TargetStopPrescaleFilter -> TargetStopFilter, as Production names them."""
    prefix = filter_label[:-len(FILTER_SUFFIX)] if filter_label.endswith(FILTER_SUFFIX) \
        else filter_label
    return f"{prefix}Filter"


def job_fcl_text(collection: str, base: Path = FCL) -> str:
    """The run's fcl: stop_counts.fcl with the selectors reading `collection`.

    FHiCL cannot `#include` an absolute path, so this is a copy with the lines
    appended, as the trigger analyses do.
    """
    lines = [base.read_text(encoding="utf-8").rstrip(), "",
             "# --- set by tools/analyses/muon_stop_rate.py for this run ---"]
    lines += [f'physics.filters.{label}.SimParticles : "{collection}"' for label in SELECTORS]
    return "\n".join(lines) + "\n"


def parse_species_counts(stdout: str) -> dict[str, float]:
    """`n_muminus` and `n_muplus`: the events stop_counts.fcl's two selector
    paths passed, from art's TrigReport. Raises CountsError when a path is
    missing (the job ran another fcl, or art's summary was off)."""
    rows = {m["name"]: m for m in _PATH_ROW.finditer(stdout)}
    counts = {}
    for species, path in SPECIES_PATHS.items():
        if path not in rows:
            raise CountsError(f"no TrigReport row for path '{path}' in the job "
                              "output, so the mu-/mu+ stops cannot be counted")
        if int(rows[path]["error"]):
            raise CountsError(f"path '{path}' had {rows[path]['error']} event(s) in "
                              "error, so its count is incomplete")
        counts[f"n_{species}"] = float(rows[path]["passed"])
    return counts


def stop_rates(counts: dict[str, float], upstream_eff: float) -> dict[str, float]:
    """The counts plus the rates derived from them: every stop, and the mu-
    and mu+ stops apart when the counts have them."""
    per_gen = 1.0 / (counts["n_gen_events"] * counts["prescale"])
    split = [s for s in SPECIES_PATHS if f"n_{s}" in counts]
    rates = {
        **{k: v for k, v in counts.items() if not k.startswith(("n_muminus", "n_muplus"))},
        "stops_per_gen_event": counts["n_events"] * per_gen,
        "stops_per_pot": counts["n_events"] * per_gen * upstream_eff,
        **{f"n_{s}": counts[f"n_{s}"] for s in split},
    }
    for species in split:
        rates[f"{species}_stops_per_gen_event"] = counts[f"n_{species}"] * per_gen
        rates[f"{species}_stops_per_pot"] = counts[f"n_{species}"] * per_gen * upstream_eff
    return rates


def summarize(metrics: dict[str, float]) -> str:
    return (
        f"{metrics['n_events']:g} stopped-muon events ({metrics['n_muminus']:g} "
        f"mu-, {metrics['n_muplus']:g} mu+) from {metrics['n_gen_events']:g} "
        f"generated (prescale {metrics['prescale']:g}): "
        f"{metrics['muminus_stops_per_gen_event']:.4g} mu- stops per generated "
        f"event, {metrics['muminus_stops_per_pot']:.4g} mu- stops / POT "
        f"({metrics['stops_per_pot']:.4g} / POT counting mu+ too)."
    )


def run(context: RunContext) -> RunOutcome:
    """Run stop_counts.fcl over the input file(s) and turn its counts into rates."""
    filter_label = str(context.params["prescale_filter"])
    collection = stop_collection(filter_label)
    context.outdir.mkdir(parents=True, exist_ok=True)
    fcl = context.outdir / JOB_FCL_NAME
    fcl.write_text(job_fcl_text(collection), encoding="utf-8")

    outcome = run_counts_job(context, filter_label, fcl=fcl,
                             parse_more=parse_species_counts)
    outcome.extra["job_fcl"] = str(fcl)
    outcome.extra["stop_collection"] = collection
    if outcome.error is not None:
        if "ProductNotFound" in outcome.extra.get("stdout_tail", ""):
            outcome.error = (
                f"{outcome.error}: the input has no '{collection}' SimParticle "
                f"collection, which the mu-/mu+ selectors read (derived from "
                f"prescale_filter {filter_label}). Pass the prescale filter of this "
                "file's own stream, or use 'count' for the plain event count")
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
        "generated event and per POT, for all stops and for the mu- and mu+ "
        "stops apart. Use the muminus_* rates to normalize mu- capture "
        "processes (conversion, DIO, RMC): the file also holds mu+ stops."
    ),
    metrics=("n_events", "n_gen_events", "prescale", "stops_per_gen_event",
             "stops_per_pot", "n_muminus", "n_muplus",
             "muminus_stops_per_gen_event", "muminus_stops_per_pot",
             "muplus_stops_per_gen_event", "muplus_stops_per_pot"),
    units=METRIC_UNITS,
    parameters=(
        ParamSpec(
            name="prescale_filter",
            description=(
                "Module label of the prescale filter whose stream this file "
                "is, as print_counts names it. The default is the "
                "target-stop stream; pass e.g. 'PolyStopPrescaleFilter' to "
                "read a poly-stop file's own rate. It also names the "
                "SimParticle collection the mu-/mu+ split reads "
                "(TargetStopPrescaleFilter -> TargetStopFilter). Required: a "
                "target-stop file is always prescaled, so use the 'count' "
                "analysis for a file that is not."
            ),
            default=PRESCALE_FILTER,
            kind="text",
        ),
        ParamSpec(
            name="upstream_eff",
            description=(
                "Efficiency of everything upstream of this stage: generated "
                "events of this file's stage per POT (POT -> MuBeam here). "
                "Multiplies the per-generated-event rates to give stops / POT."
            ),
            minimum=0.0,
        ),
    ),
    summarize=summarize,
    input_hint=(
        "A stopping-target sim file (sim.*.TargetStops.*.art) with the "
        "generated-event count, the production job's "
        f"{PRESCALE_FILTER} product and its {stop_collection(PRESCALE_FILTER)} "
        "SimParticles — or the file of whatever stream prescale_filter names."
    ),
)
