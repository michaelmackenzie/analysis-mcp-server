"""Running a mu2e job — the part every analysis shares.

No analysis-specific knowledge lives here: callers supply an fcl and get back
the job's output. Two things about the environment drive the shape of this
module:

* `mu2e` only exists after the Offline environment is set up, so every job
  re-sources it in a fresh bash subprocess instead of assuming the server
  process inherited it.
* Analysis modules like `EdepAna` are *locally built* (not in the Offline
  release yet), so `muse setup` must run from the muse work area — the one
  holding `backing -> .../Musings/SimJob/Run1Baq` and
  `build/<platform>/Mu2eOptAna/lib`. Running it anywhere else leaves the
  module off CET_PLUGIN_PATH and art dies with 'Library specification
  "EdepAna" does not correspond to any library'.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# The muse work area: `muse setup` runs here so the local Mu2eOptAna build and
# the backing release both land on the path.
MUSE_WORKAREA = Path("/exp/mu2e/app/users/mmackenz/mu2eopt")

MU2E_ENV_SETUP = "source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh"


@dataclass
class JobOutcome:
    """What a finished (or timed-out) mu2e job left behind."""

    command: str
    returncode: int | None  # None when the job was killed on timeout
    timed_out: bool
    stdout: str
    stderr: str
    log_path: Path
    input_paths: list[Path]
    input_flag: str  # "-s" or "-S"
    file_list_path: Path | None
    written_root_files: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.timed_out or self.returncode != 0

    def stdout_tail(self, lines: int = 20) -> str:
        return "\n".join((self.stdout + self.stderr).splitlines()[-lines:])


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode("utf-8", "replace")


def build_input_args(
    paths: list[Path], outdir: Path, *, single: bool
) -> tuple[str, str, Path | None]:
    """Pick mu2e's input flag: -s for one file, -S for a written file list.

    Returns (flag, argument, file_list_path or None). For the -S case the list
    is written to outdir/filelist.txt, one absolute path per line — the format
    mu2e expects.
    """
    if single:
        return "-s", str(paths[0]), None
    file_list_path = outdir / "filelist.txt"
    file_list_path.write_text("".join(f"{p}\n" for p in paths), encoding="utf-8")
    return "-S", str(file_list_path), file_list_path


def root_snapshot(outdir: Path) -> dict[str, tuple[float, int]]:
    """Name -> (mtime, size) for every ROOT file in outdir.

    Taken before and after a job so its output is recognized by having been
    *written*, not by being absent beforehand. Running twice into the same
    directory overwrites the previous run's files, and a job whose output was
    overwritten still produced it — comparing names alone would report nothing
    the second time and leave an analysis chained onto it with no input.
    """
    snapshot: dict[str, tuple[float, int]] = {}
    for path in outdir.glob("*.root"):
        try:
            stat = path.stat()
        except OSError:      # vanished between the glob and the stat
            continue
        snapshot[path.name] = (stat.st_mtime, stat.st_size)
    return snapshot


def written_root_files(outdir: Path, before: dict[str, tuple[float, int]]) -> list[str]:
    """The ROOT files in outdir that a job created or rewrote since `before`."""
    return sorted(
        str(outdir / name)
        for name, stamp in root_snapshot(outdir).items()
        if before.get(name) != stamp
    )


def validate_input_paths(paths: list[Path]) -> list[str]:
    """Return one complaint per unusable input path; empty means all good."""
    return [
        f"not an absolute path: {p}" if not p.is_absolute() else f"does not exist: {p}"
        for p in paths
        if not p.is_absolute() or not p.exists()
    ]


def run_mu2e_job(
    *,
    fcl: Path,
    input_paths: list[Path],
    outdir: Path,
    single: bool,
    timeout_s: int,
    max_events: int | None = None,
    log_name: str = "mu2e.log",
) -> JobOutcome:
    """Run `mu2e -c <fcl> -s|-S <input>` in outdir and capture everything.

    Writes the combined stdout/stderr to outdir/<log_name> and reports the
    ROOT files the job wrote there (TFileService output), whether they were
    new or overwrote a previous run's — outputs, logs and file lists are all
    simply overwritten, so a directory can be reused.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / log_name
    before = root_snapshot(outdir)

    input_flag, input_arg, file_list_path = build_input_args(
        input_paths, outdir, single=single
    )
    nevts = f" --nevts {max_events}" if max_events is not None else ""

    # $1 = input (art file or file list), $2 = output dir — passed as bash
    # positional parameters so paths never need shell quoting.
    script = (
        f'cd "{MUSE_WORKAREA}" && '
        f"{MU2E_ENV_SETUP} && "
        "muse setup && "
        'cd "$2" && '
        f'mu2e -c "{fcl}" {input_flag} "$1"{nevts}'
    )
    try:
        proc = subprocess.run(
            ["bash", "-lc", script, "_", input_arg, str(outdir)],
            cwd=outdir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        stdout, stderr, returncode, timed_out = (
            proc.stdout, proc.stderr, proc.returncode, False,
        )
    except subprocess.TimeoutExpired as exc:
        # In text mode these are usually str, but bytes on some versions.
        stdout, stderr = _as_text(exc.stdout), _as_text(exc.stderr)
        returncode, timed_out = None, True

    inputs_note = "\n".join(f"  {p}" for p in input_paths)
    log_path.write_text(
        f"$ {script}\n({len(input_paths)} input file(s), {input_flag})\n"
        f"{inputs_note}\n\n--- stdout ---\n{stdout}\n\n--- stderr ---\n{stderr}\n",
        encoding="utf-8",
    )

    return JobOutcome(
        command=script,
        returncode=returncode,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        log_path=log_path,
        input_paths=list(input_paths),
        input_flag=input_flag,
        file_list_path=file_list_path,
        written_root_files=written_root_files(outdir, before),
    )
