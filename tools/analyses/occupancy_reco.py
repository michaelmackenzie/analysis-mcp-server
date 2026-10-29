"""Occupancy of the reconstructed hits: straw hits, panel ComboHits, calo hits
and clusters per event (OccupancyAna, fcl/occupancy.fcl). The job and parsing
are in occupancy.py."""

from ..spec import AnalysisSpec, RunContext, RunOutcome
from . import occupancy as occ

COLLECTIONS = ("strawhits", "combohits", "calohits", "caloclusters",
               "calorecodigis")

# The production labels in mcs / rec files. CaloRecoDigis are off by default:
# see the description.
DEFAULT_TAGS = {"strawhits": "makeSH", "combohits": "makePH",
                "calohits": "CaloHitMaker", "caloclusters": "CaloClusterMaker",
                "calorecodigis": ""}


def run(context: RunContext) -> RunOutcome:
    return occ.run_occupancy(context, COLLECTIONS, window=True)


def summarize(metrics: dict[str, float]) -> str:
    return occ.summarize(COLLECTIONS, metrics)


SPEC = AnalysisSpec(
    name="occupancy_reco",
    fcl=occ.FCL,
    input_kind="art_files",
    run=run,
    description=(
        "Per-event occupancy of the reconstructed hits (OccupancyAna): straw "
        "hits (single-straw ComboHits), panel ComboHits (and the straw hits "
        "they hold), calorimeter hits and clusters per event (mean, rms, max, "
        "and the events that lack the collection), the mean count in the "
        "busiest tracker plane and in each calorimeter disk, and the counts "
        "inside a hit time window and above a calo hit energy (hit_time_min/"
        "max, calo_hit_energy_min, reported as *_sel_mean). For mcs and rec "
        "files. The digi and CRV pulse/cluster collections in an mcs file are "
        "SelectReco subsets kept for the reconstructed objects, not the full "
        "occupancy, so this analysis does not offer them (CaloRecoDigis only "
        "as an option, off by default, with the same caveat); use "
        "occupancy_digi on dig files for digi occupancy. Needs an Offline "
        "with OccupancyAna (PR #2070). No per-generated-event normalization."
    ),
    metrics=occ.metric_names(COLLECTIONS),
    units={**{f"{n}_{s}": "per event" for n in COLLECTIONS
              for s in ("mean", "rms", "max")},
           **{m: "per event" for m in occ.tree_metric_names(COLLECTIONS)}},
    parameters=(
        *(occ.tag_param(n, DEFAULT_TAGS[n],
                        "A SelectReco subset in an mcs file, not the full "
                        "occupancy. " if n == "calorecodigis" else "")
          for n in COLLECTIONS),
        *occ.window_params(),
    ),
    summarize=summarize,
    input_hint=(
        "reconstructed art file(s), e.g. mcs.*.art or rec.*.art, holding "
        "makeSH, makePH, CaloHitMaker and CaloClusterMaker (the production "
        "labels, set by the *_tag parameters)."
    ),
)
