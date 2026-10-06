"""The online trigger over art files: the job and its parsing, shared by the
trigger_efficiency, trigger_rate and trigger_timing analyses.

One job serves all three. `fcl/trigger.fcl` builds on mu2e-trig-config's
timing test (the physics menu, with a Prefetch module so reading the data is
not charged to the first reconstruction module). Each run writes its own fcl
next to the job: that text with `physics.trigger_paths` set to the requested
paths and each path's prescale set. A per-run file rather than an
`#include` because FHiCL only includes through FHICL_FILE_PATH, never by
absolute path.

Prescales are applied by the job itself, by each path's PrescaleEvent filter
(`event % prescale == 0`), exactly as online. So the overall pass count
carries the real correlation between prescaled paths, and a prescaled-away
path costs no reconstruction time. The filter's label follows the menu
generator's convention (generateMenuFromJSON.py): the path name's
underscore-separated words capitalized and joined, then "PS", e.g.
apr_TrkDe_80m70p -> AprTrkDe80m70pPS.

Counts come from art's TrigReport (events, events passing any path, and each
path's run/passed/failed/error). Times come from the TimeTracker database the
job writes (triggerTiming.db), with one row per module per event. An event's
processing time is the sum of its modules' times, leaving out fetching the
data: the Prefetch module (PrefetchDAQData) the timing fcl runs first, and
the input source, which TimeTracker keeps in a table of its own. This is how
mu2eTimingPlotsMaker totals an event (merge_timing_files.py), and it differs
from TimeTracker's own "Full event" time, which includes the fetch. Warm-up
events are left out as well: the first carries the database and geometry
initialization, ~1 s against a few ms for a typical event, and on a few
hundred events it alone would double the mean.
"""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..mu2e_env import current as current_env
from ..mu2e_job import JobOutcome, run_mu2e_job
from ..spec import FCL_DIR, ParamSpec, RunContext, RunOutcome

FCL = FCL_DIR / "trigger.fcl"
JOB_FCL_NAME = "trigger_job.fcl"
TIMING_DB_NAME = "triggerTiming.db"      # set in fcl/trigger.fcl

# The data-like pileup sample mu2e-trig-config's own CI runs the menu over,
# relative to the configured code (its backing release has it).
DEFAULT_PILEUP_LIST = "mu2e-trig-config/ci/data_files.txt"

# Module types whose time is fetching data, not processing it: left out of an
# event's processing time and reported apart.
FETCH_MODULE_TYPES = frozenset({"PrefetchDAQData"})

# Path names are FHiCL labels; anything else could not name a path, and would
# end up written into the job's fcl.
_PATH_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class TriggerError(ValueError):
    """A trigger request or job output that cannot be used, worded for the caller."""


# --- the request -------------------------------------------------------------

def parse_trigger_paths(text: str) -> list[tuple[str, int]]:
    """'apr_TrkDe_80m70p:1, cpr_TrkDe_80m70p:10' -> [(path, prescale), ...].

    Entries are separated by commas or whitespace; a path without ':N' has
    prescale 1. Raises TriggerError naming the bad entry.
    """
    entries = [e for e in re.split(r"[,\s]+", text.strip()) if e]
    if not entries:
        raise TriggerError("no trigger paths given")
    paths: list[tuple[str, int]] = []
    seen: set[str] = set()
    for entry in entries:
        name, _, prescale_text = entry.partition(":")
        if not _PATH_NAME.match(name):
            raise TriggerError(f"'{entry}' does not name a trigger path")
        try:
            prescale = int(prescale_text) if prescale_text else 1
        except ValueError:
            raise TriggerError(f"'{entry}': the prescale must be a whole number")
        if prescale < 1:
            raise TriggerError(f"'{entry}': the prescale must be at least 1 "
                               "(leave a path out to disable it)")
        if name in seen:
            raise TriggerError(f"'{name}' is listed twice")
        seen.add(name)
        paths.append((name, prescale))
    return paths


def prescale_module(path: str) -> str:
    """The PrescaleEvent filter's label in a menu path, e.g. AprTrkDe80m70pPS."""
    return "".join(word[:1].upper() + word[1:] for word in path.split("_")) + "PS"


def job_fcl_text(paths: list[tuple[str, int]], base: Path = FCL) -> str:
    """The run's fcl: the shipped base, then the requested paths and prescales.

    The prescale is set for both spill modes, so it holds whichever the input
    is marked as.
    """
    lines = [base.read_text(encoding="utf-8").rstrip(), "",
             "# --- set by tools/analyses/trigger.py for this run ---",
             "physics.trigger_paths : [ "
             + ", ".join(f'"{name}"' for name, _ in paths) + " ]"]
    for name, prescale in paths:
        lines.append(
            f"physics.filters.{prescale_module(name)}.eventModeConfig : [ "
            f"{{ eventMode: OnSpill prescale: {prescale} }}, "
            f"{{ eventMode: OffSpill prescale: {prescale} }} ]"
        )
    return "\n".join(lines) + "\n"


def default_pileup_inputs() -> list[Path]:
    """The art files in the configured code's mu2e-trig-config CI file list."""
    listing = current_env().find_code_file(DEFAULT_PILEUP_LIST)
    if listing is None:
        raise TriggerError(
            f"no {DEFAULT_PILEUP_LIST} in {current_env().describe()} or its "
            "backing releases; pass data_file(s) for the pileup sample"
        )
    return [Path(line.strip()) for line in listing.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def trigger_paths_param() -> ParamSpec:
    """The parameter every trigger analysis takes."""
    return ParamSpec(
        name="trigger_paths",
        description=(
            "Trigger paths to run, each with its prescale: "
            "'path:prescale' entries separated by commas, e.g. "
            "'apr_TrkDe_80m70p:1, cpr_TrkDe_80m70p:10, calo_photon'. "
            "A path without ':N' has prescale 1. Names are the menu's paths "
            "(mu2e-trig-config physMenu, e.g. apr_/cpr_ track paths, "
            "calo_photon; the menu's tpr_ and mpr_ paths are not meant for "
            "the real trigger); an unknown one fails the job with art's "
            "'Unknown path' message. Prescales are applied in the job, as "
            "online (event number % prescale == 0)."
        ),
        kind="text",
    )


# --- what the job printed ----------------------------------------------------

@dataclass
class PathCounts:
    run: int
    passed: int
    failed: int
    error: int


@dataclass
class TrigReport:
    n_events: int
    n_passed: int
    paths: dict[str, PathCounts] = field(default_factory=dict)


_EVENT_SUMMARY = re.compile(
    r"TrigReport Events total = (\d+) passed = (\d+) failed = (\d+)")
_PATH_LINE = re.compile(
    r"^TrigReport\s+\S+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s*$")


def parse_trig_report(stdout: str) -> TrigReport | None:
    """The event and trigger-path summaries art prints at the end of a job.

    None if the event summary is not there (the job died before it).
    """
    summary = _EVENT_SUMMARY.search(stdout)
    if summary is None:
        return None
    report = TrigReport(n_events=int(summary.group(1)),
                        n_passed=int(summary.group(2)))
    in_paths = False
    for line in stdout.splitlines():
        if "Trigger-path summary" in line:
            in_paths = True
            continue
        if in_paths:
            if "----------" in line:          # the next block's header
                break
            match = _PATH_LINE.match(line)
            if match:
                run, passed, failed, error = map(int, match.groups()[:4])
                report.paths[match.group(5)] = PathCounts(run, passed, failed, error)
    return report


def binomial(k: float, n: float) -> tuple[float, float]:
    """The fraction k/n and its uncertainty (0, 0 for n = 0).

    The uncertainty is the width of the fraction's posterior for a uniform
    prior, sqrt((k+1)(n-k+1) / ((n+2)^2 (n+3))): close to the usual
    sqrt(p(1-p)/n) away from the edges, but not zero for k = 0 or k = n,
    where a rare trigger on a short sample would otherwise claim to be
    exactly zero.
    """
    if n <= 0:
        return 0.0, 0.0
    return k / n, math.sqrt((k + 1) * (n - k + 1) / ((n + 2) ** 2 * (n + 3)))


@dataclass
class ModuleTimes:
    """One module's times (s), on the timed events it ran on."""

    path: str        # the first path that ran it, which TimeTracker charges
    label: str
    type: str
    times: list[float] = field(default_factory=list)

    @property
    def counted(self) -> bool:
        """False for a data fetch, which is left out of the event time."""
        return self.type not in FETCH_MODULE_TYPES

    def summary(self, n_events: int) -> dict:
        """N(seen) and the time per run and per event, in ms.

        N(seen) is the number of timed events the module ran on: fewer than
        all of them when a filter upstream in its path, or the path's
        prescale, stopped the rest.
        """
        ms = np.asarray(self.times) * 1e3
        n_seen = ms.size
        return {
            "path": self.path, "label": self.label, "type": self.type,
            "n_seen": n_seen,
            "seen_fraction": n_seen / n_events if n_events else 0.0,
            "mean_ms": float(ms.mean()),             # per event it ran on
            "median_ms": float(np.median(ms)),
            "rms_ms": float(ms.std(ddof=1)) if n_seen > 1 else 0.0,
            "max_ms": float(ms.max()),
            "ms_per_event": float(ms.sum()) / n_events if n_events else 0.0,
            "counted": self.counted,
        }


@dataclass
class EventTimes:
    """Per-event processing times (s) from the TimeTracker database."""

    times: np.ndarray            # processing time of each timed event, in order
    fetch_times: np.ndarray      # time fetching its data (left out of `times`)
    full_event_times: np.ndarray  # TimeTracker's whole-event time, fetch included
    n_skipped: int
    per_path: dict[str, float]   # mean processing s/event charged to each path
    # every module that ran on a timed event, fetch included, in the order
    # they first ran, keyed "path:label:type"
    modules: dict[str, ModuleTimes] = field(default_factory=dict)

    def module_table(self) -> list[dict]:
        """ModuleTimes.summary for every module, in the order they ran.

        A module shared by several paths runs once per event, charged to
        whichever path reached it first on that event — so its runs can be
        split over several rows (e.g. CaloHitMakerFast under calo_photon on
        the events its prescale let through, under apr_ on the rest).
        `n_seen_all_paths` and `ms_per_event_all_paths` add those rows up:
        the events the module ran on at all, and its whole cost.
        """
        n = self.times.size
        rows = [m.summary(n) for m in self.modules.values()]
        totals: dict[tuple[str, str], list[float]] = {}
        for m in self.modules.values():
            seen, cost = totals.setdefault((m.label, m.type), [0, 0.0])
            totals[(m.label, m.type)] = [seen + len(m.times),
                                         cost + sum(m.times) * 1e3]
        for row in rows:
            seen, cost = totals[(row["label"], row["type"])]
            row["n_seen_all_paths"] = seen
            row["ms_per_event_all_paths"] = cost / n if n else 0.0
        return rows


def read_timing_db(db: Path, skip_events: int) -> EventTimes:
    """Per-event processing times, leaving out the first `skip_events` events.

    An event's processing time is the sum of its modules' times, without the
    data fetch (FETCH_MODULE_TYPES) and the input source, which is timed
    apart. A module shared by several paths runs once per event, and
    TimeTracker charges it to the first path that ran it — so a path's time
    is what it added to the event, given the paths before it, not what it
    would cost alone.
    """
    if not db.exists():
        raise TriggerError(f"the job wrote no timing database ({db})")
    with sqlite3.connect(db) as con:
        events = con.execute(
            "SELECT Run, SubRun, Event, Time FROM TimeEvent "
            "ORDER BY rowid").fetchall()
        modules = con.execute(
            "SELECT Run, SubRun, Event, Path, ModuleLabel, ModuleType, Time "
            "FROM TimeModule ORDER BY rowid").fetchall()
    timed = [(r, s, e) for r, s, e, _ in events[skip_events:]]
    index = {key: i for i, key in enumerate(timed)}
    n = max(len(timed), 1)
    times = np.zeros(len(timed))
    fetch = np.zeros(len(timed))
    per_path: dict[str, float] = {}
    per_module: dict[str, ModuleTimes] = {}
    for run, subrun, event, path, label, mtype, t in modules:
        i = index.get((run, subrun, event))
        if i is None:
            continue
        key = f"{path}:{label}:{mtype}"
        if key not in per_module:
            per_module[key] = ModuleTimes(path, label, mtype)
        per_module[key].times.append(t)
        if mtype in FETCH_MODULE_TYPES:
            fetch[i] += t
            continue
        times[i] += t
        per_path[path] = per_path.get(path, 0.0) + t / n
    return EventTimes(
        times=times, fetch_times=fetch,
        full_event_times=np.array([t for *_, t in events[skip_events:]],
                                  dtype=np.float64),
        n_skipped=min(skip_events, len(events)),
        per_path=per_path, modules=per_module,
    )


# --- running it ----------------------------------------------------------------

@dataclass
class TriggerRun:
    """A finished trigger job: what was asked, and what came back."""

    paths: list[tuple[str, int]]
    job: JobOutcome
    report: TrigReport
    extra: dict


def run_trigger_job(context: RunContext) -> TriggerRun | RunOutcome:
    """Write the run's fcl, run it, and parse the TrigReport.

    Returns a RunOutcome carrying the error when anything fails, so a runner
    can hand it straight back.
    """
    try:
        paths = parse_trigger_paths(context.params["trigger_paths"])
    except TriggerError as exc:
        return RunOutcome(error=str(exc))
    context.outdir.mkdir(parents=True, exist_ok=True)
    fcl = context.outdir / JOB_FCL_NAME
    fcl.write_text(job_fcl_text(paths), encoding="utf-8")
    # A stale database must not pass for this run's.
    (context.outdir / TIMING_DB_NAME).unlink(missing_ok=True)

    job = run_mu2e_job(
        fcl=fcl,
        input_paths=context.input_paths,
        outdir=context.outdir,
        single=len(context.input_paths) == 1 and not context.wants_file_list,
        timeout_s=context.timeout_s,
        max_events=context.max_events,
    )
    extra = {"returncode": job.returncode, "job_fcl": str(fcl),
             "prescales": dict(paths)}
    if job.file_list_path is not None:
        extra["file_list_path"] = str(job.file_list_path)
    if job.timed_out:
        return RunOutcome(log_path=job.log_path, extra=extra,
                          error=f"mu2e timed out after {context.timeout_s}s")
    if job.failed:
        extra["stdout_tail"] = job.stdout_tail()
        return RunOutcome(log_path=job.log_path, extra=extra,
                          error=f"mu2e exited {job.returncode}")
    report = parse_trig_report(job.stdout)
    if report is None:
        extra["stdout_tail"] = job.stdout_tail()
        return RunOutcome(log_path=job.log_path, extra=extra,
                          error="no TrigReport event summary in the mu2e output")
    missing = [name for name, _ in paths if name not in report.paths]
    if missing:
        return RunOutcome(log_path=job.log_path, extra=extra,
                          error=f"TrigReport has no line for {', '.join(missing)}")
    if report.n_events == 0:
        return RunOutcome(log_path=job.log_path, extra=extra,
                          error="the job processed no events")
    return TriggerRun(paths=paths, job=job, report=report, extra=extra)


def per_path_fractions(run: TriggerRun) -> dict[str, dict[str, float]]:
    """Each path's pass count and fraction of all events, with its prescale."""
    out = {}
    for name, prescale in run.paths:
        counts = run.report.paths[name]
        fraction, err = binomial(counts.passed, run.report.n_events)
        out[name] = {"prescale": prescale, "passed": counts.passed,
                     "fraction": fraction, "fraction_err": err,
                     "errors": counts.error}
    return out
