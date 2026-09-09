"""The MCP tool functions: discover analyses, then run one.

Plain Python — no MCP imports. The type hints, Field constraints, and
docstrings below become the MCP tool schema that agents see. Both tools are
driven by `registry.ANALYSES`, so adding an analysis there extends the
`analysis` enum and the list_analyses output automatically.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, validate_call

from .mu2e_job import run_mu2e_job, validate_input_paths
from .registry import ANALYSES, ANALYSIS_NAMES
from .spec import ArtifactResult

# Built from the registry, so the schema's enum grows with it. Literal accepts
# a tuple of values at runtime, which is what lets this stay dynamic.
AnalysisName = Literal[ANALYSIS_NAMES]  # type: ignore[valid-type]


@validate_call
def list_analyses() -> ArtifactResult:
    """List the analyses run_analysis can run, with the metrics each returns.

    Use this tool first to discover valid `analysis` names, what each one
    measures, which metric names it reports (and their units), and what the
    input art file(s) must contain.
    """
    catalogue = {name: spec.describe() for name, spec in sorted(ANALYSES.items())}
    missing = [name for name, entry in catalogue.items() if not entry["fcl_exists"]]
    message = f"{len(catalogue)} analyses available: {', '.join(catalogue)}."
    if missing:
        message += f" WARNING: fcl file missing for {', '.join(missing)}."
    return ArtifactResult(
        status="success",
        files=[],
        message=message,
        metadata={"analyses": catalogue},
    )


@validate_call
def run_analysis(
    analysis: AnalysisName,
    output_dir: Annotated[str, Field(min_length=1)],
    data_file: Annotated[str | None, Field(min_length=1)] = None,
    data_files: Annotated[list[str] | None, Field(min_length=1)] = None,
    max_events: Annotated[int | None, Field(ge=1)] = None,
    timeout_s: Annotated[int, Field(ge=1, le=7200)] = 900,
) -> ArtifactResult:
    """Run one analysis over mu2e art file(s) and return its metrics.

    Runs a real mu2e job — `mu2e -c <the analysis' fcl> -s <file>` for a single
    input, or `-S <file list>` for several — then parses the job's summary
    output into numbers. Expect seconds to many minutes, growing with the
    number of input files and events.

    Call `list_analyses` first if you are unsure of the `analysis` name or
    which metrics it reports; the parsed metrics land in `metadata` under the
    names listed there. Results are per-job: with several input files the
    metrics cover the whole set, NOT one file each — run the tool once per
    file if you need per-file numbers.

    Args:
        analysis: Which analysis to run (see list_analyses), e.g. "edep" for
            average calorimeter/tracker energy deposition.
        output_dir: Directory the job runs in; its ROOT output, the captured
            mu2e.log and (for several inputs) filelist.txt are written here.
            Created if missing. Use a fresh directory per job to keep outputs
            from different runs apart.
        data_file: Absolute path to one input art file. Pass exactly one of
            data_file or data_files.
        data_files: Absolute paths to several input art files, analyzed
            together in one job. Pass exactly one of data_file or data_files.
        max_events: Process at most this many events (mu2e --nevts). Useful
            for a quick check before a full run; omit to process everything.
            CAUTION: generated-event counts come from the input's subrun
            bookkeeping and cover the whole file either way, so any
            "per gen event" metric is meaningless when this is set — use it
            to check that a job runs, not for physics numbers.
        timeout_s: Kill the job if it runs longer than this many seconds.
            Raise it when passing many files.
    """
    spec = ANALYSES[analysis]

    if (data_file is None) == (data_files is None):
        return ArtifactResult(
            status="error", files=[],
            message="Pass exactly one of data_file (one art file) or "
                    "data_files (a list of art files).",
            metadata={"analysis": analysis, "data_file": data_file,
                      "data_files": data_files},
        )

    inputs = [data_file] if data_file is not None else list(data_files)
    paths = [Path(p) for p in inputs]
    problems = validate_input_paths(paths)
    if problems:
        return ArtifactResult(
            status="error", files=[],
            message="Invalid input data file(s): " + "; ".join(problems),
            metadata={"analysis": analysis, "data_files": [str(p) for p in paths]},
        )

    if not spec.fcl.exists():
        return ArtifactResult(
            status="error", files=[],
            message=f"fcl for analysis '{analysis}' not found: {spec.fcl}",
            metadata={"analysis": analysis, "fcl": str(spec.fcl)},
        )

    outcome = run_mu2e_job(
        fcl=spec.fcl,
        input_paths=paths,
        outdir=Path(output_dir).expanduser().resolve(),
        single=data_file is not None,
        timeout_s=timeout_s,
        max_events=max_events,
    )

    # Describes the job in every result, so a caller chaining several analyses
    # can tell which inputs a number came from without re-reading logs.
    job_metadata = {
        "analysis": analysis,
        "fcl": str(spec.fcl),
        "data_files": [str(p) for p in outcome.input_paths],
        "n_input_files": len(outcome.input_paths),
        "log_path": str(outcome.log_path),
    }
    if outcome.file_list_path is not None:
        job_metadata["file_list_path"] = str(outcome.file_list_path)
    if max_events is not None:
        job_metadata["max_events"] = max_events

    if outcome.timed_out:
        return ArtifactResult(
            status="error", files=outcome.new_root_files,
            message=f"{analysis}: mu2e timed out after {timeout_s}s on "
                    f"{len(paths)} input file(s). See {outcome.log_path}.",
            metadata=job_metadata,
        )

    metrics = spec.parse(outcome.stdout)
    if outcome.failed or metrics is None:
        reason = (
            f"mu2e exited {outcome.returncode}" if outcome.failed
            else f"{analysis} summary block not found in mu2e output"
        )
        return ArtifactResult(
            status="error", files=outcome.new_root_files,
            message=f"{analysis}: {reason}. See {outcome.log_path} for the full log.",
            metadata={**job_metadata, "returncode": outcome.returncode,
                      "stdout_tail": outcome.stdout_tail()},
        )

    return ArtifactResult(
        status="success",
        files=outcome.new_root_files,
        message=(
            f"{analysis} over {len(paths)} input file(s): "
            f"{spec.summarize(metrics)}"
        ),
        metadata={**job_metadata, "returncode": outcome.returncode, **metrics},
    )
