"""What an analysis has to provide, and the result contract every tool returns.

Adding an analysis means writing one module in `analyses/` that defines an
`AnalysisSpec` and listing it in `registry.py`. Nothing else changes: the
`list_analyses` and `run_analysis` tools pick it up from the registry.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel

# A parser turns the job's stdout into named numbers, or None if the expected
# summary output isn't there (job died early, wrong input collections, ...).
Parser = Callable[[str], dict[str, float] | None]

# Formats a one-line human summary from parsed metrics.
Summarizer = Callable[[dict[str, float]], str]


class ArtifactResult(BaseModel):
    """Uniform result contract returned by every tool."""

    status: Literal["success", "error"]
    files: list[str]
    message: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AnalysisSpec:
    """One runnable analysis: an fcl plus how to read its output.

    Attributes:
        name: Short key agents pass as run_analysis(analysis=...).
        fcl: Absolute path to the fcl the job runs.
        description: One line, shown by list_analyses.
        metrics: Metric names the parser returns, in reporting order.
        units: Metric name -> unit, for metrics that have one.
        parse: Turns job stdout into {metric: value}, or None if absent.
        summarize: Builds the human-readable one-line message.
        input_hint: What the input art file(s) must contain.
    """

    name: str
    fcl: Path
    description: str
    metrics: tuple[str, ...]
    parse: Parser
    summarize: Summarizer
    units: dict[str, str] | None = None
    input_hint: str = ""

    def describe(self) -> dict[str, Any]:
        """The registry entry as list_analyses reports it."""
        return {
            "description": self.description,
            "fcl": str(self.fcl),
            "metrics": list(self.metrics),
            "units": dict(self.units or {}),
            "input_hint": self.input_hint,
            "fcl_exists": self.fcl.exists(),
        }
