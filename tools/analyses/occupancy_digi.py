"""Occupancy of the digis: StrawDigis, CaloDigis and CrvDigis per event
(OccupancyAna, fcl/occupancy.fcl). The job and parsing are in occupancy.py."""

from ..spec import AnalysisSpec, RunContext, RunOutcome
from . import occupancy as occ

COLLECTIONS = ("strawdigis", "calodigis", "crvdigis")

# The production labels in dig files (MDC2025 and Run 1B).
DEFAULT_TAGS = {"strawdigis": "makeSD", "calodigis": "CaloDigiMaker",
                "crvdigis": "CrvDigi"}


def run(context: RunContext) -> RunOutcome:
    return occ.run_occupancy(context, COLLECTIONS, window=False)


def summarize(metrics: dict[str, float]) -> str:
    return occ.summarize(COLLECTIONS, metrics)


SPEC = AnalysisSpec(
    name="occupancy_digi",
    fcl=occ.FCL,
    input_kind="art_files",
    run=run,
    description=(
        "Per-event occupancy of the digitized detector (OccupancyAna): the "
        "number of StrawDigis, CaloDigis and CrvDigis per event (mean, rms, "
        "max, and the events that lack the collection), the mean count in the "
        "busiest tracker plane and in each calorimeter disk. For dig files, "
        "raw-data digis included: a dig file holds the full digi collections, "
        "so these are real occupancies. Needs an Offline with OccupancyAna "
        f"(PR #2070). Reco (mcs) files hold only subsets of these collections; "
        "use occupancy_reco for them. No per-generated-event normalization."
    ),
    metrics=occ.metric_names(COLLECTIONS),
    units={**{f"{n}_{s}": "per event" for n in COLLECTIONS
              for s in ("mean", "rms", "max")},
           **{m: "per event" for m in occ.tree_metric_names(COLLECTIONS)}},
    parameters=tuple(occ.tag_param(n, DEFAULT_TAGS[n]) for n in COLLECTIONS),
    summarize=summarize,
    input_hint=(
        "art file(s) holding digis — dig files (dig.*.art) or raw-data digis; "
        "the labels default to makeSD, CaloDigiMaker and CrvDigi. A job "
        "without a collection counts it as missing; check a file's labels "
        "with mu2e -c Offline/Print/fcl/dumpDataProducts.fcl."
    ),
)
