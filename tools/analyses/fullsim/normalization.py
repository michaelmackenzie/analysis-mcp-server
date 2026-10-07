"""Stopped muons per POT and the rates per stopped muon, as Production has them.

Follows Production/JobConfig/ensemble/python/normalizations.py (and its
constants.py): the expected CE count is

    POT x (stopped mu- / POT) x (captures / stopped mu-) x R_mue

and the DIO count is POT x (stopped mu- / POT) x (DIO / stopped mu-),
times the fraction of the DIO spectrum in question.

Stopped mu- per POT comes from the simulation chain that made the stops:
the efficiencies of its MuBeamCat and MuminusStopsCat stages, multiplied,
times 1000 as normalizations.py does. A stage's efficiency is computed from
its dataset, as Production's Scripts/CreateSimEfficiency.sh does with
mu2eGenFilterEff: events in the dataset (SAM event_count) over events
generated to make it (dh.gencount), each summed over the dataset's files.
That needs only SAM, not the SimEfficiencies2 conditions table the script's
output is later loaded into, so it works for a chain that was never put in
the database. Which datasets to use is traced from the input files' SAM
ancestry (stop_chain_of_inputs): a CE file descends from the
MuminusStopsCat file its stops were resampled from, and that from
MuBeamCat. A table that was put in the database (dbTool print-run ...
--table SimEfficiencies2 --content) or mu2eGenFilterEff's output file can
still be read with parse_sim_efficiencies.
"""

import re
from dataclasses import dataclass

from .provenance import (ProvenanceError, ancestor_in_stage, dataset_files,
                         dataset_of, files_metadata, sam_metadata)

# Production constants.py.
CAPTURES_PER_STOPPED_MUON = 0.609
DIO_PER_STOPPED_MUON = 0.391        # 1 - CAPTURES_PER_STOPPED_MUON

# The stages whose efficiencies multiply into stopped mu- per POT, and the
# factor normalizations.py applies to their product.
STOP_CHAIN = ("MuBeamCat", "MuminusStopsCat")
STOP_CHAIN_SCALE = 1000.0

# The MDC2025 stop chain, as CreateSimEfficiency.sh names it: the one the
# MDC2025au_best_v1_1 CeMLeadingLogMix1BB files trace back to in SAM.
STOP_CHAIN_DATASETS = (
    "sim.mu2e.MuBeamCat.MDC2025ab.art",
    "sim.mu2e.MuminusStopsCat.MDC2025ac.art",
)

# A Mu2e dataset name, tier.owner.description.configuration.format; the
# description is the stage. Strict, since the name goes into a SAM query.
DATASET_NAME = re.compile(r"^\w+\.\w+\.([\w-]+)\.[\w-]+\.\w+$")


class NormalizationError(ValueError):
    """Stage efficiencies that cannot give stops per POT, worded for the
    caller."""


@dataclass(frozen=True)
class StageEfficiency:
    """One simulation stage's filter efficiency, as mu2eGenFilterEff sums it."""
    stage: str
    dataset: str
    n_files: int
    passed: int         # events in the dataset
    generated: int      # events generated to make it

    @property
    def efficiency(self) -> float:
        return self.passed / self.generated


# What SAM gives for STOP_CHAIN_DATASETS (4 and 1 files, read 2026-10-07):
# the same rows as SimEfficiencies2 for Sim_best v1_1, run 1430. The
# default for code that takes a number; the analysis recomputes it.
STOP_CHAIN_MDC2025 = {
    "MuBeamCat": StageEfficiency("MuBeamCat", STOP_CHAIN_DATASETS[0],
                                 4, 213816, 100000000),
    "MuminusStopsCat": StageEfficiency("MuminusStopsCat", STOP_CHAIN_DATASETS[1],
                                       1, 1435092, 4000000000),
}


def stage_of(dataset: str) -> str:
    """The stage a dataset holds: its description field."""
    match = DATASET_NAME.match(dataset)
    if not match:
        raise NormalizationError(
            f"'{dataset}' is not a dataset name: expected "
            "tier.owner.description.configuration.format, e.g. "
            f"{STOP_CHAIN_DATASETS[0]}"
        )
    return match.group(1)


def dataset_efficiency(dataset: str, list_files=dataset_files,
                       fetch=files_metadata) -> StageEfficiency:
    """A stage's efficiency from its dataset's SAM metadata, as
    mu2eGenFilterEff computes it. `list_files` and `fetch` read SAM; tests
    pass fakes."""
    stage = stage_of(dataset)
    try:
        names = list_files(dataset)
        metadata = fetch(names) if names else []
    except Exception as exc:
        raise NormalizationError(
            f"cannot read dataset {dataset} from SAM ({exc}), so the "
            f"{stage} efficiency is unknown"
        ) from None
    if not names:
        raise NormalizationError(
            f"SAM has no files in dataset {dataset}: check the name (as "
            f"CreateSimEfficiency.sh writes it, e.g. {STOP_CHAIN_DATASETS[0]})"
        )
    passed = generated = 0
    for record in metadata:
        if not record.get("dh.gencount"):
            raise NormalizationError(
                f"{record.get('file_name', 'a file')} in {dataset} has no "
                "dh.gencount in SAM, so the events generated for the "
                f"{stage} stage are unknown"
            )
        generated += int(record["dh.gencount"])
        # SAM does not store an event_count of zero (INC000001108858), so a
        # missing one is zero, as mu2eGenFilterEff has it.
        passed += int(record.get("event_count") or 0)
    if len(metadata) != len(names):
        raise NormalizationError(
            f"SAM listed {len(names)} files in {dataset} but returned "
            f"metadata for {len(metadata)}, so the {stage} efficiency would "
            "be incomplete"
        )
    return StageEfficiency(stage, dataset, len(names), passed, generated)


def sim_efficiencies(datasets, list_files=dataset_files,
                     fetch=files_metadata) -> dict[str, StageEfficiency]:
    """stage -> its efficiency, one dataset per stage."""
    stages: dict[str, StageEfficiency] = {}
    for dataset in datasets:
        stage = stage_of(dataset)
        if stage in stages:
            raise NormalizationError(
                f"two datasets for the {stage} stage ({stages[stage].dataset} "
                f"and {dataset}): give one per stage"
            )
        stages[stage] = dataset_efficiency(dataset, list_files, fetch)
    return stages


def stop_chain_of(file_name: str, fetch=sam_metadata) -> tuple[str, ...]:
    """The stop chain's datasets, in STOP_CHAIN order, that a file was
    simulated from: its nearest MuminusStopsCat ancestor, then that file's
    nearest MuBeamCat ancestor. `fetch` reads one file's SAM metadata."""
    datasets = []
    current = file_name
    try:
        for stage in reversed(STOP_CHAIN):
            current = ancestor_in_stage(current, stage, fetch)
            datasets.append(dataset_of(current))
    except ProvenanceError as exc:
        raise NormalizationError(str(exc)) from None
    return tuple(reversed(datasets))


def stop_chain_of_inputs(file_names, fetch=sam_metadata) -> tuple[str, ...]:
    """The one stop chain all the input files descend from. A dataset is
    made by one production configuration, so one file per input dataset is
    traced; inputs from chains that differ are an error, since one
    normalization cannot cover them."""
    chains: dict[tuple[str, ...], list[str]] = {}
    traced: set[str] = set()
    for name in file_names:
        dataset = dataset_of(name)
        if dataset is None:
            raise NormalizationError(
                f"{name} is not a SAM file name "
                "(tier.owner.description.configuration.sequencer.format), "
                "so the stop chain it was simulated from cannot be traced"
            )
        if dataset in traced:
            continue
        traced.add(dataset)
        chains.setdefault(stop_chain_of(name, fetch), []).append(dataset)
    if len(chains) > 1:
        raise NormalizationError(
            "the inputs descend from different stop chains ("
            + "; ".join(f"{', '.join(inputs)} from {' + '.join(chain)}"
                        for chain, inputs in chains.items())
            + "): run them separately"
        )
    return next(iter(chains))


def parse_sim_efficiencies(text: str) -> dict[str, float]:
    """stage -> efficiency, from SimEfficiencies2 rows as dbTool prints them
    (or as CreateSimEfficiency.sh writes them): 'stage, passed, generated,
    efficiency'. The efficiency column is taken as written."""
    table = {}
    for line in text.splitlines():
        words = [w.strip() for w in line.split(",")]
        if len(words) < 4 or not words[0]:
            continue
        try:
            table[words[0]] = float(words[3])
        except ValueError:
            continue                    # a header or a comment
    return table


def stopped_muons_per_pot(table: dict) -> float:
    """Target-stopped mu- per POT from stage efficiencies: floats (a parsed
    table) or StageEfficiency (sim_efficiencies)."""
    missing = [stage for stage in STOP_CHAIN if stage not in table]
    if missing:
        raise NormalizationError(
            f"no {', '.join(missing)} efficiency: stopped mu- per POT is the "
            f"product of {' x '.join(STOP_CHAIN)} (times "
            f"{STOP_CHAIN_SCALE:g}), so give a dataset (or table row) for "
            "each of them"
        )
    rate = STOP_CHAIN_SCALE
    for stage in STOP_CHAIN:
        value = table[stage]
        rate *= value.efficiency if isinstance(value, StageEfficiency) else value
    return rate


# The default for code that takes a number: the MDC2025 chain (7.67e-4).
STOPPED_MUONS_PER_POT = stopped_muons_per_pot(STOP_CHAIN_MDC2025)
