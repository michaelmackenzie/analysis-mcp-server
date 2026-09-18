"""Where the Offline code comes from, and how a job enters it.

`mu2e` exists only after the Offline environment is set up, and which code a
job runs is exactly the question of how that setup is done. Three ways are
supported, one per `kind`:

| kind        | supplied            | what a job runs                                |
|-------------|---------------------|------------------------------------------------|
| `work_area` | a muse work area    | `cd <area> && muse setup`                      |
| `musing`    | a Musing + version  | `muse setup SimJob MDC2025au`                  |
| `tarball`   | a code tarball      | unpack it once, then `muse setup` in the tree  |

All three source `setupmu2e-art.sh` first, in a fresh bash for every job, so
the server does not care what its own shell had.

The difference that matters to an analysis is where its fcl lives. A work area
or an unpacked tarball is a directory on disk, so `Mu2eOptAna/fcl/edep.fcl`
can be checked before a job is started; a Musing is whatever `muse setup` puts
on `FHICL_FILE_PATH`, so the relative path is handed to `mu2e` and art
resolves it — or says plainly that it cannot. Analyses therefore declare their
fcl *relative*, and this module turns it into whatever the environment can
offer.

> **Why the work area matters for the analyses that ship here**: `EdepAna` is
> a locally built module, not in any release. Its library comes from
> `build/<platform>/Mu2eOptAna/lib`, which only lands on `CET_PLUGIN_PATH`
> when `muse setup` runs in the area holding it. Point the server at a bare
> Musing and art dies with `Library specification "EdepAna" does not
> correspond to any library` — the analysis is simply not available in that
> environment.

Configured once, when the server starts (`--work-area`, `--musing`,
`--code-tarball`, or the MU2E_WORK_AREA / MU2E_MUSING / MU2E_CODE_TARBALL
environment variables), not per call: the environment is a property of the
deployment, and an agent chaining analyses should not be able to change
releases halfway through.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MU2E_SETUP = "source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh"

# Used when nothing is configured: the work area this server grew up in.
DEFAULT_WORK_AREA = Path("/exp/mu2e/app/users/mmackenz/mu2eopt")

EnvKind = Literal["work_area", "musing", "tarball"]


class EnvError(ValueError):
    """A configuration that cannot work, reported before any job runs."""


@dataclass(frozen=True)
class Mu2eEnv:
    """How to set up Offline for a job. Build one with the constructors below."""

    kind: EnvKind
    work_area: Path | None = None
    musing: tuple[str, str] | None = None      # (Musing, version)
    tarball: Path | None = None
    code_dir: Path | None = None               # where a tarball is unpacked
    code_subdir: str | None = None             # muse setup runs here inside it

    # --- constructors --------------------------------------------------------

    @classmethod
    def for_work_area(cls, path: str | Path) -> "Mu2eEnv":
        area = Path(path).expanduser()
        if not area.is_absolute():
            area = area.resolve()
        if not area.is_dir():
            raise EnvError(f"work area is not a directory: {area}")
        return cls(kind="work_area", work_area=area)

    @classmethod
    def for_musing(cls, musing: str, version: str | None = None) -> "Mu2eEnv":
        """From ("SimJob", "MDC2025au") or one string, "SimJob MDC2025au"."""
        if version is None:
            parts = musing.replace("/", " ").split()
            if len(parts) != 2:
                raise EnvError(
                    f"expected a Musing and a version, e.g. 'SimJob MDC2025au', "
                    f"got {musing!r}"
                )
            musing, version = parts
        return cls(kind="musing", musing=(musing, version))

    @classmethod
    def for_tarball(cls, path: str | Path, code_dir: str | Path | None = None,
                    code_subdir: str | None = None) -> "Mu2eEnv":
        tarball = Path(path).expanduser().resolve()
        if not tarball.is_file():
            raise EnvError(f"code tarball is not a file: {tarball}")
        return cls(
            kind="tarball",
            tarball=tarball,
            code_dir=Path(code_dir).expanduser().resolve() if code_dir else None,
            code_subdir=code_subdir,
        )

    # --- what a job needs ----------------------------------------------------

    def describe(self) -> str:
        """One line naming this environment, for logs and result metadata."""
        if self.kind == "work_area":
            return f"work area {self.work_area}"
        if self.kind == "musing":
            return f"musing {self.musing[0]} {self.musing[1]}"
        where = f" unpacked in {self.code_dir}" if self.code_dir else ""
        return f"code tarball {self.tarball}{where}"

    def base_dir(self, job_dir: Path | None = None) -> Path | None:
        """The directory relative fcl paths resolve against, if there is one.

        A Musing has none — its fcl lives wherever `muse setup` puts it, so
        `FHICL_FILE_PATH` is the only thing that can find it. Nor has a
        tarball that is unpacked per job, until a job names its directory.
        """
        if self.kind == "work_area":
            return self.work_area
        if self.kind == "tarball":
            if self.code_dir is None and job_dir is None:
                return None
            root = self.unpack_dir(job_dir)
            return root / self.code_subdir if self.code_subdir else root
        return None

    def missing_fcl(self, fcl: Path, job_dir: Path | None = None) -> str | None:
        """A complaint if the fcl can be shown to be missing before running.

        Only code we can already see can say so: for a Musing, or a tarball
        not unpacked yet, the answer comes from art when the job runs.
        """
        if fcl.is_absolute():
            return None if fcl.exists() else f"fcl not found: {fcl}"
        base = self.base_dir(job_dir)
        if base is None or not base.is_dir():
            return None
        return None if (base / fcl).exists() else f"fcl not found: {base / fcl}"

    def resolve_fcl(self, fcl: Path, job_dir: Path | None = None) -> Path:
        """The fcl path to hand `mu2e`: absolute if we can see it, else as given.

        Passing the relative path through is deliberate: art resolves it on
        FHICL_FILE_PATH, which is how a Musing's own fcl is found.
        """
        if fcl.is_absolute():
            return fcl
        base = self.base_dir(job_dir)
        if base is not None and (base / fcl).exists():
            return base / fcl
        return fcl

    def unpack_dir(self, job_dir: Path | None = None) -> Path:
        """Where the tarball is unpacked: the configured dir, else beside the job."""
        return self.code_dir if self.code_dir is not None else job_dir / "code"

    def setup_commands(self, job_dir: Path) -> list[str]:
        """Shell commands that leave the shell inside a set-up environment."""
        if self.kind == "work_area":
            return [f"cd {shlex.quote(str(self.work_area))}", MU2E_SETUP, "muse setup"]
        if self.kind == "musing":
            musing, version = self.musing
            return [
                f"cd {shlex.quote(str(job_dir))}",
                MU2E_SETUP,
                f"muse setup {shlex.quote(musing)} {shlex.quote(version)}",
            ]

        # A tarball is unpacked once and reused: the marker sits outside the
        # tree so it cannot be mistaken for part of the code, and a tree that
        # unpacked into a single directory is entered, which is how most
        # tarballs are rolled.
        root = shlex.quote(str(self.unpack_dir(job_dir)))
        tarball = shlex.quote(str(self.tarball))
        enter = f"cd {root}"
        if self.code_subdir:
            enter += f" && cd {shlex.quote(self.code_subdir)}"
        else:
            enter += (
                ' && if [ "$(ls -1A | wc -l)" -eq 1 ] && [ -d "$(ls -1A)" ];'
                ' then cd "$(ls -1A)"; fi'
            )
        return [
            f'[ -f {root}.unpacked ] || {{ rm -rf {root} && mkdir -p {root} && '
            f'tar -xf {tarball} -C {root} && touch {root}.unpacked; }}',
            enter,
            MU2E_SETUP,
            "muse setup",
        ]


# --- the environment this server is running with -----------------------------

_CONFIGURED: Mu2eEnv | None = None


def configure(env: Mu2eEnv | None) -> None:
    """Set the environment every job runs in (called once, at startup).

    None clears it, which puts the fallback back in charge.
    """
    global _CONFIGURED
    _CONFIGURED = env


def from_environment() -> Mu2eEnv | None:
    """An environment from MU2E_MUSING / MU2E_CODE_TARBALL / MU2E_WORK_AREA."""
    musing = os.environ.get("MU2E_MUSING")
    if musing:
        return Mu2eEnv.for_musing(musing)
    tarball = os.environ.get("MU2E_CODE_TARBALL")
    if tarball:
        return Mu2eEnv.for_tarball(tarball, os.environ.get("MU2E_CODE_DIR"),
                                   os.environ.get("MU2E_CODE_SUBDIR"))
    work_area = os.environ.get("MU2E_WORK_AREA")
    if work_area:
        return Mu2eEnv.for_work_area(work_area)
    return None


def current() -> Mu2eEnv:
    """What jobs run in: whatever was configured, the environment, or the default."""
    if _CONFIGURED is not None:
        return _CONFIGURED
    return from_environment() or Mu2eEnv(kind="work_area", work_area=DEFAULT_WORK_AREA)
