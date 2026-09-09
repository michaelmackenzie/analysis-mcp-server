"""Energy deposition: the EdepAna analyzer (Mu2eOptAna/fcl/edep.fcl).

Average calorimeter and tracker energy deposition per event and per generated
event. The parser reads the summary block EdepAna_module.cc prints at endJob
(see Mu2eOptAna/src/EdepAna_module.cc:520-528).
"""

import re

from ..mu2e_job import MUSE_WORKAREA
from ..spec import AnalysisSpec

FCL = MUSE_WORKAREA / "Mu2eOptAna" / "fcl" / "edep.fcl"

# Matches the block EdepAna_module.cc prints, e.g.:
#   EdepAna summary:
#     Saw 998.5 events (1000 gen events) --> output rate = 0.9985 events / gen event
#     Average calo energy deposition per event: 12.3 MeV
#     Average calo energy deposition per gen event: 12.29 MeV
#     Events with calo Edep > 50 MeV: 42
#     Average tracker energy deposition per event: 3.21 MeV
#     Average tracker energy deposition per gen event: 3.2 MeV
# n_events and n_events_calo_edep_above_50mev are weighted sums (doubles in
# the module), so every field is parsed as a float.
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_SUMMARY_FIELDS = [
    ("n_events", re.compile(rf"Saw\s+({_NUM})\s+events\b")),
    ("n_gen_events", re.compile(rf"\(\s*({_NUM})\s+gen events\)")),
    ("event_rate", re.compile(rf"output rate\s*=\s*({_NUM})\s+events / gen event")),
    ("avg_calo_edep_per_event_mev",
     re.compile(rf"Average calo energy deposition per event:\s*({_NUM})\s*MeV")),
    ("avg_calo_edep_per_gen_event_mev",
     re.compile(rf"Average calo energy deposition per gen event:\s*({_NUM})\s*MeV")),
    ("n_events_calo_edep_above_50mev",
     re.compile(rf"Events with calo Edep > 50 MeV:\s*({_NUM})")),
    ("avg_trk_edep_per_event_mev",
     re.compile(rf"Average tracker energy deposition per event:\s*({_NUM})\s*MeV")),
    ("avg_trk_edep_per_gen_event_mev",
     re.compile(rf"Average tracker energy deposition per gen event:\s*({_NUM})\s*MeV")),
]

METRIC_UNITS = {
    "event_rate": "events / gen event",
    "avg_calo_edep_per_event_mev": "MeV",
    "avg_calo_edep_per_gen_event_mev": "MeV",
    "avg_trk_edep_per_event_mev": "MeV",
    "avg_trk_edep_per_gen_event_mev": "MeV",
}


def parse_edep_summary(stdout: str) -> dict[str, float] | None:
    """Extract the 8 EdepAna summary fields from mu2e stdout.

    Returns None if the "EdepAna summary:" block isn't there or is incomplete
    (e.g. the job died before EdepAna::endJob ran).
    """
    if "EdepAna summary:" not in stdout:
        return None
    metrics: dict[str, float] = {}
    for name, pattern in _SUMMARY_FIELDS:
        match = pattern.search(stdout)
        if not match:
            return None
        metrics[name] = float(match.group(1))
    return metrics


def summarize_edep(metrics: dict[str, float]) -> str:
    return (
        f"saw {metrics['n_events']:g} events ({metrics['n_gen_events']:g} gen): "
        f"avg calo edep {metrics['avg_calo_edep_per_event_mev']:.4g} MeV/event, "
        f"avg tracker edep {metrics['avg_trk_edep_per_event_mev']:.4g} MeV/event."
    )


SPEC = AnalysisSpec(
    name="edep",
    fcl=FCL,
    description=(
        "Average calorimeter and tracker energy deposition per event and per "
        "generated event (EdepAna)."
    ),
    metrics=tuple(name for name, _ in _SUMMARY_FIELDS),
    units=METRIC_UNITS,
    parse=parse_edep_summary,
    summarize=summarize_edep,
    input_hint=(
        "art file(s) holding compressDetStepMCs, CaloClusterMaker and "
        "FindMCPrimary — e.g. TargetStops/mcs/dts files."
    ),
)
