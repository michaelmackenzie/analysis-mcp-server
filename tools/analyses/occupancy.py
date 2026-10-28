"""Occupancy: the OccupancyAna job and its parsing, shared by the
occupancy_digi and occupancy_reco analyses (and the statistics helper of
occupancy_ntuple).

OccupancyAna (Offline/Analyses/src/OccupancyAna_module.cc, from Offline PR
#2070) counts the size of up to ten collections per event: tracker, calorimeter
and CRV digis, straw hits (single-straw ComboHits), panel ComboHits, calorimeter
hits, clusters. A collection with no tag in its configuration is skipped. It
needs no geometry or conditions. It reports two things:

* a summary table at endJob, one row per configured collection: the number of
  events with the collection and without it, and the mean, rms and maximum of
  its size over the events that have it. These are the metrics' main source.
* a per-event tree (OccupancyAna/tree) with the same counts, the per-plane
  (tracker) and per-disk (calorimeter) breakdown, and the counts inside a hit
  time / calorimeter energy window. A count is -1 where its collection is
  skipped or missing from the event.

Each run writes its own fcl next to the job: `fcl/occupancy.fcl` (which
configures no collection) with the requested tags and window appended. A
per-run file rather than an `#include`, since FHiCL only includes through
FHICL_FILE_PATH, never by absolute path.

Metrics of a collection that is skipped, or present in no event, are -1 (and
its events_found / events_missing are -1 when skipped), so every metric a
spec lists is always reported. A configured collection missing from every
event almost always means a wrong tag: it is flagged in extra["warnings"] and
in the message.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..mu2e_job import run_mu2e_job
from ..spec import FCL_DIR, ParamSpec, RunContext, RunOutcome

FCL = FCL_DIR / "occupancy.fcl"
JOB_FCL_NAME = "occupancy_job.fcl"

# The analyzer's label in occupancy.fcl, and the tree it books.
ANALYZER = "OccupancyAna"
TREE_PATH = "OccupancyAna/tree"

# Offline PR that added the module (merged 2026-10-28), for the error message.
PR = "https://github.com/Mu2e/Offline/pull/2070"


@dataclass(frozen=True)
class Collection:
    """One collection OccupancyAna can count."""

    name: str          # the summary row's name; the tree branch is n<name>
    fcl_param: str     # its tag parameter in the analyzer's fcl
    what: str          # for the tag's description


COLLECTIONS: dict[str, Collection] = {c.name: c for c in (
    Collection("strawdigis", "StrawDigiCollection", "StrawDigis"),
    Collection("strawhits", "StrawHitCollection",
               "single-straw ComboHits (the straw hits)"),
    Collection("combohits", "ComboHitCollection", "panel ComboHits"),
    Collection("calodigis", "CaloDigiCollection", "CaloDigis"),
    Collection("calorecodigis", "CaloRecoDigiCollection", "CaloRecoDigis"),
    Collection("calohits", "CaloHitCollection", "CaloHits"),
    Collection("caloclusters", "CaloClusterCollection", "CaloClusters"),
    Collection("crvdigis", "CrvDigiCollection", "CrvDigis"),
    Collection("crvpulses", "CrvRecoPulseCollection", "CrvRecoPulses"),
    Collection("crvclusters", "CrvCoincidenceClusterCollection",
               "CrvCoincidenceClusters"),
)}

# Tree arrays: per tracker plane and per calorimeter disk, by collection.
PLANE_ARRAYS = {"strawdigis": "strawdigis_plane", "strawhits": "strawhits_plane"}
DISK_ARRAYS = {"calodigis": "calodigis_disk", "calohits": "calohits_disk"}
# Windowed counts in the tree, and the collection each belongs to.
SELECTED_BRANCHES = {"strawhits": "nstrawhits_sel", "combohits": "ncombohits_sel",
                     "calohits": "ncalohits_sel"}
# The calorimeter has two disks (CaloConst::_nDisk); the tree arrays are that wide.
NDISKS = 2
# Sum of the straw hits in the panel ComboHits.
COMBOHIT_STRAWS_BRANCH = "ncombohit_straws"

# What OccupancyAna's window parameters default to: everything.
FULL_WINDOW = 1.0e9

# The fcl numbers go through %.9g, and a tag becomes a quoted FHiCL string, so
# it may only be what an art InputTag is: label[:instance[:process]].
_TAG = re.compile(r"^[A-Za-z0-9_]+(?::[A-Za-z0-9_]*){0,2}$")


class OccupancyError(ValueError):
    """A request or job output that cannot be used, worded for the caller."""


# --- the request -------------------------------------------------------------

def tag_param(name: str, default: str, note: str = "") -> ParamSpec:
    """The parameter that sets one collection's tag; '' skips the collection."""
    collection = COLLECTIONS[name]
    return ParamSpec(
        name=f"{name}_tag",
        description=(
            f"Input tag of the {collection.what} collection to count "
            f"('label', 'label:instance' or 'label:instance:process'). "
            + (f"The default '{default}' is the production label. " if default
               else "Off by default. ")
            + note + "Empty string skips this collection: its metrics are "
            f"then -1. If the tag does not match a collection in the file, "
            f"{name}_events_found is 0 and the run warns."
        ),
        default=default, kind="text", allow_empty=True,
    )


def window_params(tracker: bool = True) -> tuple[ParamSpec, ...]:
    """The window the windowed (*_sel) counts apply, as OccupancyAna's own.

    `tracker` False for an analysis that windows only the calo hits.
    """
    hits = ("straw hit and panel ComboHit corrected time, calo hit time"
            if tracker else "calo hit time")
    return (
        ParamSpec(
            name="hit_time_min",
            description=(
                "Earliest hit time (ns) counted in the windowed *_sel_mean "
                "metrics: " + hits + ". Inclusive. The default (-1e9) has no "
                "lower bound, so the windowed count is the whole collection. "
                "The plain *_mean metrics never use the window."),
            default=-FULL_WINDOW,
        ),
        ParamSpec(
            name="hit_time_max",
            description=(
                "Latest hit time (ns) counted in the windowed *_sel_mean "
                "metrics (same hits as hit_time_min). Inclusive. The default "
                "(1e9) has no upper bound."),
            default=FULL_WINDOW,
        ),
        ParamSpec(
            name="calo_hit_energy_min",
            description=(
                "Minimum calo hit energy (MeV) counted in calohits_sel_mean, on "
                "top of the time window. Inclusive. Default 0: every hit."),
            default=0.0, minimum=0.0,
        ),
    )


def parse_tags(params: dict[str, float | str], names: tuple[str, ...]
               ) -> dict[str, str]:
    """{collection name: tag} for the collections of `names` that are not
    skipped. Raises OccupancyError for a tag that is not an InputTag, or when
    every collection is skipped."""
    tags: dict[str, str] = {}
    for name in names:
        tag = str(params.get(f"{name}_tag", "")).strip()
        if not tag:
            continue
        if not _TAG.match(tag):
            raise OccupancyError(
                f"{name}_tag '{tag}' is not an art input tag; use 'label', "
                "'label:instance' or 'label:instance:process' (letters, digits "
                "and underscores)")
        tags[name] = tag
    if not tags:
        raise OccupancyError(
            "every collection is skipped (all tags are empty), so there is "
            "nothing to count; give at least one of "
            + ", ".join(f"{n}_tag" for n in names))
    return tags


def job_fcl_text(tags: dict[str, str], window: tuple[float, float, float] | None
                 = None, base: Path = FCL) -> str:
    """The run's fcl: the base, then the collection tags and the window.

    `window` is (hit time min ns, hit time max ns, calo hit energy min MeV);
    None leaves the base's (no window).
    """
    lines = [base.read_text(encoding="utf-8").rstrip(), "",
             "# --- set by tools/analyses/occupancy.py for this run ---"]
    for name, tag in tags.items():
        lines.append(f'physics.analyzers.{ANALYZER}.'
                     f'{COLLECTIONS[name].fcl_param} : "{tag}"')
    if window is not None:
        tmin, tmax, emin = window
        for key, value in (("HitTimeMin", tmin), ("HitTimeMax", tmax),
                           ("CaloHitEnergyMin", emin)):
            lines.append(f"physics.analyzers.{ANALYZER}.{key} : {value:.9g}")
    return "\n".join(lines) + "\n"


# --- what the job printed ----------------------------------------------------

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_HEADER = re.compile(r"Occupancy summary over\s+(\d+)\s+events")
# "  strawhits      makeSH      622  0  724.141  35.2  901"
_ROW = re.compile(rf"^\s+(\w+)\s+(\S+)\s+(\d+)\s+(\d+)\s+({_NUM})\s+({_NUM})\s+(\d+)\s*$")


@dataclass
class Row:
    """One configured collection's line of the summary table."""

    tag: str
    found: int
    missing: int
    mean: float
    rms: float
    max: int


@dataclass
class Summary:
    n_events: int
    rows: dict[str, Row]


def parse_summary(stdout: str) -> Summary | None:
    """The endJob table from mu2e stdout, or None if it is not there (e.g. the
    job died before endJob)."""
    header = _HEADER.search(stdout)
    if header is None:
        return None
    rows: dict[str, Row] = {}
    for line in stdout[header.end():].splitlines():
        match = _ROW.match(line)
        if match and match.group(1) in COLLECTIONS:
            name, tag, found, missing, mean, rms, top = match.groups()
            rows[name] = Row(tag, int(found), int(missing), float(mean),
                             float(rms), int(top))
    return Summary(int(header.group(1)), rows) if rows else None


def table_metrics(names: tuple[str, ...], summary: Summary) -> dict[str, float]:
    """Mean, rms, max and found/missing event counts for every collection of
    `names`; -1 for a skipped one, and -1 mean/rms/max where no event had it."""
    metrics: dict[str, float] = {}
    for name in names:
        row = summary.rows.get(name)
        if row is None:
            values = (-1.0,) * 5
        elif row.found == 0:
            values = (-1.0, -1.0, -1.0, 0.0, float(row.missing))
        else:
            values = (row.mean, row.rms, float(row.max), float(row.found),
                      float(row.missing))
        metrics.update(zip(
            (f"{name}_mean", f"{name}_rms", f"{name}_max",
             f"{name}_events_found", f"{name}_events_missing"), values))
    return metrics


def table_metric_names(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{name}_{stat}" for name in names
                 for stat in ("mean", "rms", "max", "events_found",
                              "events_missing"))


def warnings(names: tuple[str, ...], summary: Summary) -> list[str]:
    """One line per configured collection that no event had."""
    return [f"{name} ({summary.rows[name].tag}) was missing from all "
            f"{summary.rows[name].missing} events: the tag is probably wrong "
            "for this file (see mu2e's dumpDataProducts.fcl for its labels)"
            for name in names if name in summary.rows
            and summary.rows[name].found == 0 and summary.n_events > 0]


# --- the per-event tree --------------------------------------------------------

def mean_rms_max(values) -> tuple[float, float, float]:
    """Mean, rms (population, as OccupancyAna's) and maximum of `values`;
    (-1, -1, -1) for none."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return -1.0, -1.0, -1.0
    mean = float(values.mean())
    return mean, math.sqrt(max(0.0, float((values ** 2).mean()) - mean ** 2)), \
        float(values.max())


def column_mean(values) -> float:
    """Mean over the events that have a count (>= 0); -1 if none do."""
    values = np.asarray(values, dtype=np.float64)
    values = values[values >= 0]
    return float(values.mean()) if values.size else -1.0


def read_tree(paths: list[Path]) -> dict[str, np.ndarray]:
    """The branches of OccupancyAna/tree the metrics use, concatenated over
    the files. Per-plane/disk branches are (events, n) arrays. A branch the
    tree lacks is left out."""
    import uproot

    wanted = ([f"n{name}" for name in COLLECTIONS]
              + list(PLANE_ARRAYS.values()) + list(DISK_ARRAYS.values())
              + list(SELECTED_BRANCHES.values()) + [COMBOHIT_STRAWS_BRANCH])
    parts: list[dict[str, np.ndarray]] = []
    for path in paths:
        try:
            with uproot.open(path) as rootfile:
                tree = rootfile[TREE_PATH]
                present = [b for b in wanted if b in tree]
                arrays = tree.arrays(present, library="np")
        except (KeyError, OSError, ValueError) as exc:
            raise OccupancyError(f"{path}: no {TREE_PATH} ({exc})") from None
        parts.append({k: np.stack(list(v)) if v.dtype == object else v
                      for k, v in arrays.items()})
    common = set.intersection(*(set(p) for p in parts)) if parts else set()
    return {k: np.concatenate([p[k] for p in parts]) for k in common}


def tree_metrics(names: tuple[str, ...], tree: dict[str, np.ndarray]
                 ) -> tuple[dict[str, float], dict]:
    """(metrics, extra) from the tree: the busiest tracker plane's mean count,
    each calorimeter disk's, and the windowed counts, for the collections of
    `names`. Every mean is over the events that have the collection."""
    metrics: dict[str, float] = {}
    extra: dict = {}
    for name in names:
        if name in PLANE_ARRAYS:
            per_plane = [column_mean(col) for col in
                         tree.get(PLANE_ARRAYS[name], np.zeros((0, 0))).T]
            metrics[f"{name}_busiest_plane_mean"] = max(per_plane, default=-1.0)
            extra.setdefault("per_plane_mean", {})[name] = per_plane
            extra.setdefault("busiest_plane", {})[name] = (
                int(np.argmax(per_plane)) if per_plane else -1)
        if name in DISK_ARRAYS:
            disks = tree.get(DISK_ARRAYS[name])
            for disk in range(NDISKS):
                metrics[f"{name}_disk{disk}_mean"] = (
                    column_mean(disks[:, disk])
                    if disks is not None and disks.shape[1] > disk else -1.0)
        if name in SELECTED_BRANCHES:
            metrics[f"{name}_sel_mean"] = column_mean(
                tree.get(SELECTED_BRANCHES[name], []))
        if name == "combohits":
            metrics["combohit_straws_mean"] = column_mean(
                tree.get(COMBOHIT_STRAWS_BRANCH, []))
    return metrics, extra


def tree_metric_names(names: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for name in names:
        if name in PLANE_ARRAYS:
            out.append(f"{name}_busiest_plane_mean")
        if name in DISK_ARRAYS:
            out += [f"{name}_disk{d}_mean" for d in range(NDISKS)]
        if name in SELECTED_BRANCHES:
            out.append(f"{name}_sel_mean")
        if name == "combohits":
            out.append("combohit_straws_mean")
    return tuple(out)


# --- the job -------------------------------------------------------------------

def module_unknown(text: str) -> bool:
    """Does the job's output say OccupancyAna is not in this Offline? art says
    'Library specification "OccupancyAna" does not correspond to any library'."""
    return ANALYZER in text and "does not correspond to any library" in text


def run_occupancy(context: RunContext, names: tuple[str, ...],
                  window: bool) -> RunOutcome:
    """Write the run's fcl, run it, parse the summary table, read the tree.

    `names` are the collections this analysis may count, `window` whether it
    passes the hit window on.
    """
    try:
        tags = parse_tags(context.params, names)
    except OccupancyError as exc:
        return RunOutcome(error=str(exc))
    win = ((float(context.params["hit_time_min"]),
            float(context.params["hit_time_max"]),
            float(context.params["calo_hit_energy_min"])) if window else None)
    if win is not None and win[0] > win[1]:
        return RunOutcome(error=f"hit_time_min ({win[0]:g}) is above "
                                f"hit_time_max ({win[1]:g}), so no hit is in the window")

    context.outdir.mkdir(parents=True, exist_ok=True)
    fcl = context.outdir / JOB_FCL_NAME
    fcl.write_text(job_fcl_text(tags, win), encoding="utf-8")
    job = run_mu2e_job(
        fcl=fcl,
        input_paths=context.input_paths,
        outdir=context.outdir,
        single=len(context.input_paths) == 1 and not context.wants_file_list,
        timeout_s=context.timeout_s,
        max_events=context.max_events,
    )
    extra: dict = {"returncode": job.returncode, "job_fcl": str(fcl),
                   "tags": tags}
    if job.file_list_path is not None:
        extra["file_list_path"] = str(job.file_list_path)
    if job.timed_out:
        return RunOutcome(log_path=job.log_path, extra=extra,
                          error=f"mu2e timed out after {context.timeout_s}s on "
                                f"{len(context.input_paths)} input file(s)")
    if job.failed:
        extra["stdout_tail"] = job.stdout_tail()
        if module_unknown(job.stdout + job.stderr):
            return RunOutcome(
                log_path=job.log_path, extra=extra,
                error=f"the configured Offline has no {ANALYZER} module, so "
                      f"this analysis cannot run in it: {ANALYZER} needs an "
                      f"Offline that includes PR #2070 ({PR}), merged "
                      "2026-10-28 and not in a Musing release yet. Use a work "
                      "area built on Offline with that PR")
        return RunOutcome(log_path=job.log_path, extra=extra,
                          error=f"mu2e exited {job.returncode}")
    summary = parse_summary(job.stdout)
    if summary is None:
        extra["stdout_tail"] = job.stdout_tail()
        return RunOutcome(log_path=job.log_path, extra=extra,
                          error=f"{ANALYZER} summary table not found in mu2e "
                                "output")
    if summary.n_events == 0:
        return RunOutcome(log_path=job.log_path, extra=extra,
                          error="the job processed no events")

    metrics = {"n_events": float(summary.n_events),
               **table_metrics(names, summary)}
    extra["collections"] = {n: {"tag": r.tag, "events_found": r.found,
                                "events_missing": r.missing}
                            for n, r in summary.rows.items()}
    extra["skipped"] = [n for n in names if n not in tags]
    problems = warnings(names, summary)
    if problems:
        extra["warnings"] = problems
    root_files = [Path(f) for f in job.written_root_files]
    if not root_files:
        return RunOutcome(files=job.written_root_files, log_path=job.log_path,
                          extra=extra,
                          error=f"mu2e wrote no ROOT file, so there is no "
                                f"{TREE_PATH} for the per-plane and windowed "
                                "counts")
    try:
        tree_values, tree_extra = tree_metrics(names, read_tree(root_files))
    except OccupancyError as exc:
        return RunOutcome(files=job.written_root_files, log_path=job.log_path,
                          extra=extra, error=str(exc))
    metrics.update(tree_values)
    extra.update(tree_extra)
    return RunOutcome(metrics=metrics, files=job.written_root_files,
                      log_path=job.log_path, extra=extra)


def metric_names(names: tuple[str, ...]) -> tuple[str, ...]:
    """Every metric run_occupancy reports for the collections of `names`, in
    reporting order."""
    return ("n_events", *table_metric_names(names), *tree_metric_names(names))


def summarize(names: tuple[str, ...], metrics: dict[str, float]) -> str:
    """One line: each counted collection's mean, rms and max per event; a
    collection present in no event is called out."""
    parts = []
    for name in names:
        found = metrics.get(f"{name}_events_found", -1)
        if found < 0:
            continue                          # skipped
        if found == 0:
            parts.append(f"{name} MISSING from every event (wrong tag?)")
            continue
        text = (f"{name} {metrics[f'{name}_mean']:.4g} +- "
                f"{metrics[f'{name}_rms']:.3g} (max {metrics[f'{name}_max']:g})")
        selected = metrics.get(f"{name}_sel_mean", -1)
        if selected >= 0:
            text += f", {selected:.4g} in the window"
        parts.append(text)
    return (f"mean +- rms per event over {metrics['n_events']:g} events: "
            + "; ".join(parts) + ".")
