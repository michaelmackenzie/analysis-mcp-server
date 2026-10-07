"""Stopped muons per POT and the rates per stopped muon, as Production has them.

Follows Production/JobConfig/ensemble/python/normalizations.py (and its
constants.py): the expected CE count is

    POT x (stopped mu- / POT) x (captures / stopped mu-) x R_mue

and the DIO count is POT x (stopped mu- / POT) x (DIO / stopped mu-),
times the fraction of the DIO spectrum in question.

Stopped mu- per POT comes from the simulation chain that made the stops:
the efficiencies of the MuBeamCat and MuminusStopsCat stages in the
SimEfficiencies2 conditions table (written by Production's
Scripts/CreateSimEfficiency.sh with mu2eGenFilterEff), multiplied, times
1000 as normalizations.py does. Print the table for a campaign with

    muse setup SimJob MDC2025ay
    dbTool print-run --purpose Sim_best --version v1_1 --run 1430 \\
        --table SimEfficiencies2 --content
"""

# Production constants.py.
CAPTURES_PER_STOPPED_MUON = 0.609
DIO_PER_STOPPED_MUON = 0.391        # 1 - CAPTURES_PER_STOPPED_MUON

# The stages whose efficiencies multiply into stopped mu- per POT, and the
# factor normalizations.py applies to their product.
STOP_CHAIN = ("MuBeamCat", "MuminusStopsCat")
STOP_CHAIN_SCALE = 1000.0

# SimEfficiencies2 for Sim_best v1_1, run 1430: the campaign of the
# MDC2025au_best_v1_1 CeMLeadingLogMix1BB files. Only the stop chain's rows,
# as dbTool prints them: stage, events passing, events generated, efficiency.
SIM_EFFICIENCIES_MDC2025_BEST_V1_1 = """\
MuBeamCat,213816,100000000,0.00213816
MuminusStopsCat,1435092,4000000000,0.000358773
"""


class NormalizationError(ValueError):
    """A SimEfficiencies table that cannot give stops per POT, worded for
    the caller."""


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


def stopped_muons_per_pot(table: dict[str, float]) -> float:
    """Target-stopped mu- per POT from a parsed SimEfficiencies table."""
    missing = [stage for stage in STOP_CHAIN if stage not in table]
    if missing:
        raise NormalizationError(
            f"the SimEfficiencies table has no {', '.join(missing)} row: stopped "
            f"mu- per POT is the product of {' x '.join(STOP_CHAIN)} (times "
            f"{STOP_CHAIN_SCALE:g}), so print the campaign's SimEfficiencies2 "
            "with dbTool and pass all of them"
        )
    rate = STOP_CHAIN_SCALE
    for stage in STOP_CHAIN:
        rate *= table[stage]
    return rate


# The default: Sim_best v1_1, run 1430 (7.67e-4).
STOPPED_MUONS_PER_POT = stopped_muons_per_pot(
    parse_sim_efficiencies(SIM_EFFICIENCIES_MDC2025_BEST_V1_1))
