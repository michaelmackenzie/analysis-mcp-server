"""What an analysis has to provide, and the result contract every tool returns.

An analysis is one entry in the registry. Two kinds exist so far, and the
difference is only what it consumes and what its runner does:

  input_kind="art_files"  a mu2e job over art file(s) whose stdout is parsed
                          (e.g. "edep" runs edep.fcl)
  input_kind="root_file"  Python computation over a ROOT file produced by an
                          earlier analysis (e.g. "approx_ce_sensitivity" reads
                          the nts.*.root that "edep" writes)

Everything generic — argument validation, parameter checking, assembling the
result — lives in analysis_tools.py and works for both. Adding an analysis
means writing a module in `analyses/` with a SPEC and listing it in
registry.py.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel

InputKind = Literal["art_files", "root_file"]


class ArtifactResult(BaseModel):
    """Uniform result contract returned by every tool."""

    status: Literal["success", "error"]
    files: list[str]
    message: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ParamSpec:
    """A physics knob an analysis takes, reported by list_analyses.

    `default=None` means the caller must supply it.
    """

    name: str
    description: str
    default: float | int | str | None = None
    minimum: float | None = None
    maximum: float | None = None

    @property
    def required(self) -> bool:
        return self.default is None

    def describe(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    def check(self, value: Any) -> float:
        """Validate one supplied value, returning it as a float."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"parameter '{self.name}' must be a number, got {value!r}")
        if self.minimum is not None and number < self.minimum:
            raise ValueError(f"parameter '{self.name}'={number:g} is below the "
                             f"minimum {self.minimum:g}")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"parameter '{self.name}'={number:g} is above the "
                             f"maximum {self.maximum:g}")
        return number


@dataclass
class RunContext:
    """Everything a runner needs to do one job."""

    input_paths: list[Path]
    outdir: Path
    params: dict[str, float]
    timeout_s: int
    max_events: int | None = None
    # True when the caller passed data_files (a list) rather than data_file, so
    # a one-element list still goes through mu2e's -S file-list path.
    wants_file_list: bool = False

    @property
    def input_path(self) -> Path:
        """The single input, for root_file analyses."""
        return self.input_paths[0]


@dataclass
class RunOutcome:
    """What a runner produced. `error` set means it did not work."""

    metrics: dict[str, float] | None = None
    files: list[str] = field(default_factory=list)
    log_path: Path | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# A runner does the work described by a RunContext.
Runner = Callable[[RunContext], RunOutcome]

# Formats a one-line human summary from parsed metrics.
Summarizer = Callable[[dict[str, float]], str]


@dataclass(frozen=True)
class AnalysisSpec:
    """One runnable analysis.

    Attributes:
        name: Short key agents pass as run_analysis(analysis=...).
        description: One line, shown by list_analyses.
        input_kind: What it consumes — art file(s) or a ROOT file.
        metrics: Metric names it reports, in reporting order.
        run: Does the work (see Runner).
        summarize: Builds the human-readable one-line message.
        units: Metric name -> unit, for metrics that have one.
        parameters: Physics knobs the caller may/must supply.
        input_hint: What the input file(s) must contain.
        produced_by: Analyses whose output feeds this one, for chaining.
    """

    name: str
    description: str
    input_kind: InputKind
    metrics: tuple[str, ...]
    run: Runner
    summarize: Summarizer
    units: dict[str, str] | None = None
    parameters: tuple[ParamSpec, ...] = ()
    input_hint: str = ""
    produced_by: tuple[str, ...] = ()
    fcl: Path | None = None  # art_files analyses only; shown by list_analyses

    def describe(self) -> dict[str, Any]:
        """The registry entry as list_analyses reports it."""
        entry: dict[str, Any] = {
            "description": self.description,
            "input_kind": self.input_kind,
            "metrics": list(self.metrics),
            "units": dict(self.units or {}),
            "parameters": {p.name: p.describe() for p in self.parameters},
            "input_hint": self.input_hint,
        }
        if self.produced_by:
            entry["produced_by"] = list(self.produced_by)
        if self.fcl is not None:
            entry["fcl"] = str(self.fcl)
            entry["fcl_exists"] = self.fcl.exists()
        return entry

    def resolve_params(self, supplied: dict[str, Any] | None) -> dict[str, float]:
        """Merge supplied parameters over the defaults, validating them.

        Raises ValueError naming the offender for unknown, missing, or
        out-of-range values, so run_analysis can report it verbatim.
        """
        supplied = dict(supplied or {})
        known = {p.name: p for p in self.parameters}
        unknown = sorted(set(supplied) - set(known))
        if unknown:
            raise ValueError(
                f"unknown parameter(s) for '{self.name}': {', '.join(unknown)}. "
                f"Known: {', '.join(known) or '(none)'}"
            )
        resolved: dict[str, float] = {}
        missing = []
        for name, param in known.items():
            if name in supplied:
                resolved[name] = param.check(supplied[name])
            elif param.required:
                missing.append(name)
            else:
                resolved[name] = float(param.default)
        if missing:
            raise ValueError(
                f"missing required parameter(s) for '{self.name}': "
                f"{', '.join(missing)}"
            )
        return resolved
