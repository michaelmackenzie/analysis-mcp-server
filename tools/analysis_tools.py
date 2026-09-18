"""The MCP tool functions: discover analyses, then run one.

Plain Python — no MCP imports. The type hints, Field constraints, and
docstrings below become the MCP tool schema that agents see. Both tools are
driven by `registry.ANALYSES`, so adding an analysis there extends the
`analysis` enum and the list_analyses output automatically.
"""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, validate_call

from .mu2e_env import current as current_env
from .mu2e_job import validate_input_paths
from .registry import ANALYSES, ANALYSIS_NAMES
from .spec import ArtifactResult, RunContext

# Built from the registry, so the schema's enum grows with it. Literal accepts
# a tuple of values at runtime, which is what lets this stay dynamic.
AnalysisName = Literal[ANALYSIS_NAMES]  # type: ignore[valid-type]


@validate_call
def list_analyses() -> ArtifactResult:
    """List the analyses run_analysis can run, with their inputs and metrics.

    Use this tool first to discover valid `analysis` names, what each measures,
    which metric names it reports (and their units), which parameters it takes,
    and what its input file must be. It also names the Offline environment the
    server runs mu2e jobs in — a muse work area, a Musing, or a code tarball —
    which is fixed when the server starts. `input_kind` says what to feed it:

      "art_files"  mu2e art file(s) — pass data_file or data_files
      "root_file"  a ROOT file written by an earlier analysis (see
                   `produced_by`) — pass data_file

    Chaining: an analysis whose `produced_by` names another should be given a
    ROOT file from that one's `files` output.
    """
    env = current_env()
    catalogue = {name: spec.describe() for name, spec in sorted(ANALYSES.items())}
    missing = [
        name for name, entry in catalogue.items()
        if entry.get("fcl_exists") is False
    ]
    message = (f"{len(catalogue)} analyses available: {', '.join(catalogue)}. "
               f"mu2e jobs run against {env.describe()}.")
    if missing:
        message += f" WARNING: fcl file missing for {', '.join(missing)}."
    return ArtifactResult(
        status="success",
        files=[],
        message=message,
        metadata={"analyses": catalogue, "environment": env.describe()},
    )


@validate_call
def run_analysis(
    analysis: AnalysisName,
    output_dir: Annotated[str, Field(min_length=1)],
    data_file: Annotated[str | None, Field(min_length=1)] = None,
    data_files: Annotated[list[str] | None, Field(min_length=1)] = None,
    parameters: dict[str, Any] | None = None,
    max_events: Annotated[int | None, Field(ge=1)] = None,
    timeout_s: Annotated[int, Field(ge=1, le=7200)] = 900,
) -> ArtifactResult:
    """Run one analysis over its input file(s) and return its metrics.

    Call `list_analyses` first if you are unsure of the `analysis` name, the
    parameters it needs, or which metrics it reports; the metrics land in
    `metadata` under the names listed there.

    What happens depends on the analysis' `input_kind`:

    * "art_files" runs a real mu2e job (`mu2e -c <fcl> -s <file>`, or `-S` with
      a generated file list for several inputs). Expect seconds to many
      minutes. Metrics cover the whole input set as ONE job, not one result per
      file — run the tool once per file if you need per-file numbers.
    * "root_file" runs a Python computation over one ROOT file produced by an
      earlier analysis, which is fast. Feed it a path from that analysis'
      `files` output.

    Args:
        analysis: Which analysis to run (see list_analyses), e.g. "edep".
        output_dir: Directory the analysis runs in and writes to — job output,
            logs, figures. Created if missing, and safe to reuse: a rerun
            overwrites the previous run's output rather than failing. Use a
            fresh directory per run only when you want the outputs kept apart.
        data_file: Absolute path to one input file. Pass exactly one of
            data_file or data_files.
        data_files: Absolute paths to several input art files, analyzed
            together in one job. Only for "art_files" analyses.
        parameters: Analysis-specific physics knobs, e.g.
            {"sig_eff": 2.5e-4} for approx_ce_sensitivity. list_analyses reports
            each analysis' parameters, defaults, and which are required.
        max_events: Process at most this many events (mu2e --nevts). Only for
            "art_files" analyses; useful for a quick check before a full run.
            CAUTION: generated-event counts come from the input's subrun
            bookkeeping and cover the whole file either way, so any
            "per gen event" metric is meaningless when this is set.
        timeout_s: Kill the job if it runs longer than this many seconds.
            Raise it when passing many files.
    """
    spec = ANALYSES[analysis]

    def fail(message: str, **extra: Any) -> ArtifactResult:
        return ArtifactResult(
            status="error", files=[], message=f"{analysis}: {message}",
            metadata={"analysis": analysis, **extra},
        )

    if (data_file is None) == (data_files is None):
        return fail("pass exactly one of data_file (one input file) or "
                    "data_files (a list of art files).",
                    data_file=data_file, data_files=data_files)

    if spec.input_kind == "root_file":
        if data_files is not None:
            return fail(
                f"takes a single ROOT file: pass data_file, not data_files. "
                f"{spec.input_hint}"
            )
        if max_events is not None:
            return fail("does not run a mu2e job, so max_events does not apply.")

    # Physics knobs are validated against the spec, which names the offender.
    try:
        params = spec.resolve_params(parameters)
    except ValueError as exc:
        return fail(str(exc), parameters=parameters)

    inputs = [data_file] if data_file is not None else list(data_files)
    paths = [Path(p) for p in inputs]
    problems = validate_input_paths(paths)
    if problems:
        return fail("invalid input file(s): " + "; ".join(problems),
                    data_files=[str(p) for p in paths])

    env = current_env()
    outdir = Path(output_dir).expanduser().resolve()
    # Only code already on disk can be checked up front; for a Musing the
    # answer comes from art when the job runs.
    if spec.fcl is not None and (problem := env.missing_fcl(spec.fcl, outdir)):
        return fail(problem, fcl=str(spec.fcl), environment=env.describe())

    outcome = spec.run(RunContext(
        input_paths=paths,
        outdir=outdir,
        params=params,
        timeout_s=timeout_s,
        max_events=max_events,
        wants_file_list=data_files is not None,
    ))

    # Describes the run in every result, so a caller chaining analyses can tell
    # which inputs and assumptions a number came from without re-reading logs.
    metadata: dict[str, Any] = {
        "analysis": analysis,
        "input_kind": spec.input_kind,
        "data_files": [str(p) for p in paths],
        "n_input_files": len(paths),
        **({"parameters": params} if params else {}),
        **({"max_events": max_events} if max_events is not None else {}),
        **({"fcl": str(spec.fcl), "environment": env.describe()}
           if spec.fcl is not None else {}),
        **outcome.extra,
    }
    if outcome.log_path is not None:
        metadata["log_path"] = str(outcome.log_path)

    if outcome.error is not None or outcome.metrics is None:
        error = outcome.error or "analysis produced no metrics"
        where = f" See {outcome.log_path} for details." if outcome.log_path else ""
        return ArtifactResult(
            status="error", files=outcome.files,
            message=f"{analysis}: {error}.{where}", metadata=metadata,
        )

    return ArtifactResult(
        status="success",
        files=outcome.files,
        message=f"{analysis} over {len(paths)} input file(s): "
                f"{spec.summarize(outcome.metrics)}",
        metadata={**metadata, **outcome.metrics},
    )
