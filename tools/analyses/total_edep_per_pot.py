"""Total energy deposition per POT from the three beam-related sources.

The dose a detector takes is the sum of what each source deposits, but each
source is simulated from its own sample and normalized to its own generated
event: a resampled beam electron (EleBeamCat), a resampled muon-beam event
(MuBeamCat), or a muon stop. This analysis puts them on one footing, per
proton on target, and adds them:

    per POT = ebeam / pot_per_ebeam_event
            + mubeam * mubeam_events_per_pot
            + mustop * stops_per_pot

Each per-generated-event deposition is an `edep` result
(`avg_trk_edep_per_gen_event_mev` or `avg_calo_edep_per_gen_event_mev`) on
that source's sample: the early electron-beam flash, the early muon-beam
flash, and the muon-stop pileup (MuStopPileup on the target stops). The stop
rate is `muon_stop_rate`'s `stops_per_pot`. Use the same detector's
deposition for all three.

It reads no files: its input_kind is "parameters", so a workflow chains it
by passing earlier analyses' metrics as its parameters.
"""

from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome

# POT per resampled EleBeamCat electron: the Run1Baa electron catalogue holds
# 2.17M electron events from 25.0M POT (sim.mu2e.EleBeamCat.Run1Baa metadata).
POT_PER_EBEAM_EVENT = 11.536718606512062
# MuBeamCat events per POT: the muon_stop_rate default upstream efficiency.
MUBEAM_EVENTS_PER_POT = 0.01278168

SOURCES = ("ebeam", "mubeam", "mustop")
METRICS = (("total_edep_per_pot_mev",)
           + tuple(f"{s}_edep_per_pot_mev" for s in SOURCES)
           + tuple(f"{s}_fraction" for s in SOURCES))
METRIC_UNITS = {m: "MeV / POT" for m in METRICS if m.endswith("_mev")}


def per_pot(params: dict[str, float]) -> dict[str, float]:
    """Each source's deposition per POT, their total and each one's share."""
    parts = {
        "ebeam": params["ebeam_edep_per_gen_event_mev"] / params["pot_per_ebeam_event"],
        "mubeam": params["mubeam_edep_per_gen_event_mev"] * params["mubeam_events_per_pot"],
        "mustop": params["mustop_edep_per_gen_event_mev"] * params["stops_per_pot"],
    }
    total = sum(parts.values())
    metrics = {"total_edep_per_pot_mev": total}
    metrics.update({f"{s}_edep_per_pot_mev": v for s, v in parts.items()})
    metrics.update({f"{s}_fraction": (v / total if total > 0 else 0.0)
                    for s, v in parts.items()})
    return metrics


def run(context: RunContext) -> RunOutcome:
    params = {k: float(v) for k, v in context.params.items()}
    return RunOutcome(metrics=per_pot(params))


def summarize(metrics: dict[str, float]) -> str:
    shares = ", ".join(f"{s} {metrics[f'{s}_fraction'] * 100:.1f}%"
                       for s in SOURCES)
    return (f"total {metrics['total_edep_per_pot_mev']:.4e} MeV/POT "
            f"({shares})")


def _edep_param(source: str, sample: str) -> ParamSpec:
    return ParamSpec(
        name=f"{source}_edep_per_gen_event_mev",
        description=(
            f"Energy deposited per generated event (MeV) in the {sample}: "
            f"'edep''s avg_trk_edep_per_gen_event_mev (or the calo one) on "
            f"that sample. The same detector for all three sources."
        ),
        minimum=0.0,
    )


SPEC = AnalysisSpec(
    name="total_edep_per_pot",
    input_kind="parameters",
    run=run,
    description=(
        "Total energy deposition per POT: the early electron-beam flash, the "
        "early muon-beam flash and the muon-stop pileup, each from its own "
        "'edep' result per generated event, normalized per POT and summed. "
        "Reads no files; its inputs are earlier analyses' metrics."
    ),
    metrics=METRICS,
    units=METRIC_UNITS,
    parameters=(
        _edep_param("ebeam", "early electron-beam flash (EleBeamCat resampled, "
                    "one generated event per beam electron)"),
        ParamSpec(
            name="pot_per_ebeam_event",
            description=(
                "POT per generated electron-beam event. The default is the "
                "Run1Baa EleBeamCat's: 25.0M POT for 2.17M electron events."
            ),
            default=POT_PER_EBEAM_EVENT,
            minimum=0.0,
        ),
        _edep_param("mubeam", "early muon-beam flash (MuBeamCat resampled; an "
                    "unprescaled EarlyMuBeamFlash stream, or scale by its "
                    "prescale)"),
        ParamSpec(
            name="mubeam_events_per_pot",
            description=(
                "Generated muon-beam (MuBeamCat) events per POT. The default "
                "is muon_stop_rate's upstream_eff."
            ),
            default=MUBEAM_EVENTS_PER_POT,
            minimum=0.0,
        ),
        _edep_param("mustop", "muon-stop pileup (MuStopPileup on the target "
                    "stops, one generated event per stop)"),
        ParamSpec(
            name="stops_per_pot",
            description=(
                "Muon stops in the stopping target per POT: muon_stop_rate's "
                "stops_per_pot on the same geometry."
            ),
            minimum=0.0,
        ),
    ),
    input_hint="No input files: pass the three depositions and stops_per_pot "
               "as parameters.",
    produced_by=("edep", "muon_stop_rate"),
    summarize=summarize,
)
