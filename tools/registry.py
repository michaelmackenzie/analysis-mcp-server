"""The catalogue of analyses this server can run.

**To add an analysis**: write `analyses/<name>.py` defining a `SPEC`
(AnalysisSpec: fcl, description, metrics, a stdout parser, a summarizer), then
add its module to `_ANALYSIS_MODULES` below. The `list_analyses` and
`run_analysis` tools, their schemas, and the tests all follow from the
registry — no other file needs touching.
"""

from importlib import import_module

from .spec import AnalysisSpec

# Module names under tools.analyses, each exposing SPEC.
_ANALYSIS_MODULES = ("edep", "muon_stop_rate", "approx_ce_sensitivity")


def _load() -> dict[str, AnalysisSpec]:
    specs: dict[str, AnalysisSpec] = {}
    for module_name in _ANALYSIS_MODULES:
        module = import_module(f"{__package__}.analyses.{module_name}")
        try:
            spec = module.SPEC
        except AttributeError as exc:
            raise RuntimeError(
                f"Analysis module '{module_name}' must define a SPEC "
                "(an AnalysisSpec)."
            ) from exc
        if spec.name in specs:
            raise RuntimeError(f"Duplicate analysis name '{spec.name}'.")
        specs[spec.name] = spec
    return specs


ANALYSES: dict[str, AnalysisSpec] = _load()

# Sorted so the tool schema's enum order is stable across restarts.
ANALYSIS_NAMES: tuple[str, ...] = tuple(sorted(ANALYSES))
