"""Turning catalogued inputs into paths: SAM definitions, metacat datasets,
and registered file names.

A caller can name its input the way Mu2e names data, instead of by path:

  a dataset   a SAM definition (e.g. nts.mu2e.CeMLeadingLogMix1BB.
              MDC2025au_best_v1_1.root) or a metacat dataset DID
              (mmackenz:dts.mmackenz.CeEndpoint.MDC2025ax.art)
  a file      a registered file name, bare or as a metacat DID

and gets back absolute /pnfs paths the job can read. Where a file is comes
from SAM's location records when it has them, then from Mu2e's standard
dCache layout (mdh's `url`): <prefix>/<family>/<tier>/<owner>/<description>/
<configuration>/<format>/<h0h1>/<h2h3>/<name>, h the sha256 of the name. A
file catalogued only in metacat (Rucio keeps its replicas) is found by the
second. Disk copies are preferred to tape.

A tape file that is not staged would block the job until dCache fetches it,
so it is refused up front with the command that stages it, from dCache's
locality, which NFS exposes per file for free.

Both catalogs' read APIs are plain HTTPS without authentication, so this
needs neither samweb nor metacat-client; only the network. Nothing here
knows about physics.
"""

import hashlib
import json
from collections import Counter
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

SAM_API = "https://sammu2e.fnal.gov:8483/sam/mu2e/api"
METACAT_API = "https://metacat.fnal.gov:9443/mu2e_meta_prod/app"
TIMEOUT_S = 30.0
# Names per bulk SAM locations request: mu2eGenFilterEff's chunk size.
LOCATIONS_CHUNK = 100
# A dataset larger than this needs max_files: each file costs a stat and a
# locality read, and no single job here should take thousands of files.
MAX_DATASET_FILES = 2000

# mdh's dCache locations, in the order a copy is preferred: a disk copy
# reads straight away, a tape one may need staging first.
DCACHE_PREFIXES = {
    "disk": "/pnfs/mu2e/persistent/datasets",
    "scratch": "/pnfs/mu2e/scratch/datasets",
    "tape": "/pnfs/mu2e/tape",
}
# mdh's file families: data tier -> directory, for production (owner mu2e)
# and user files. None: that tier has no production family.
FILE_FAMILIES = {
    "raw": ("phy-raw", "phy-raw"), "rec": ("phy-rec", "usr-dat"),
    "ntd": ("phy-ntd", "usr-dat"), "ext": (None, "usr-dat"),
    "rex": (None, "usr-dat"), "xnt": (None, "usr-dat"),
    "cnf": ("phy-etc", "usr-etc"), "sim": ("phy-sim", "usr-sim"),
    "dts": ("phy-sim", "usr-sim"), "mix": ("phy-sim", "usr-sim"),
    "dig": ("phy-sim", "usr-sim"), "mcs": ("phy-sim", "usr-sim"),
    "nts": ("phy-nts", "usr-nts"), "log": ("phy-etc", "usr-etc"),
    "bck": ("phy-etc", "usr-etc"), "etc": ("phy-etc", "usr-etc"),
}
# dCache localities a file can be read in without waiting for tape.
ONLINE = {"ONLINE", "ONLINE_AND_NEARLINE"}


class CatalogError(ValueError):
    """An input could not be resolved, worded for the caller."""


def is_catalog_name(entry: str) -> bool:
    """True for a file named by the catalog rather than by path: no '/'."""
    return "/" not in entry


def split_did(name: str) -> tuple[str | None, str]:
    """('ns', 'name') for a DID 'ns:name', (None, name) for a bare name."""
    namespace, sep, rest = name.partition(":")
    return (namespace, rest) if sep else (None, name)


def standard_paths(name: str) -> list[Path]:
    """Where Mu2e's layout puts file `name` in each dCache location, in
    preference order; empty for a name outside the convention."""
    words = name.split(".")
    if len(words) != 6 or not all(words) or words[0] not in FILE_FAMILIES:
        return []
    tier, owner, description, configuration, _, fmt = words
    family = FILE_FAMILIES[tier][0 if owner == "mu2e" else 1]
    if family is None:
        return []
    digest = hashlib.sha256(name.encode()).hexdigest()
    subdir = "/".join([family, tier, owner, description, configuration, fmt,
                       digest[0:2], digest[2:4], name])
    return [Path(prefix) / subdir for prefix in DCACHE_PREFIXES.values()]


def location_rank(path: Path) -> int:
    """Preference of a copy by where it is: disk, scratch, tape, other."""
    for rank, prefix in enumerate(DCACHE_PREFIXES.values()):
        if str(path).startswith(prefix + "/"):
            return rank
    return len(DCACHE_PREFIXES)


def on_tape(path: Path) -> bool:
    return str(path).startswith(DCACHE_PREFIXES["tape"] + "/")


def _fetch(url: str, data: bytes | None = None) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, data=data),
                                timeout=TIMEOUT_S) as response:
        return response.read().decode()


def _json_records(text: str) -> list[dict]:
    """metacat's query output: a JSON text sequence (RS-separated records),
    or one per line, or a JSON list."""
    text = text.strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(chunk)
            for chunk in text.replace("\x1e", "\n").splitlines()
            if chunk.strip()]


class Catalogs:
    """Read access to SAM, metacat and dCache. Every method a test needs to
    fake is here, so resolution itself never touches the network or disk
    except through one of these."""

    def sam_definition_files(self, defname: str) -> list[str] | None:
        """The file names of SAM definition `defname`; None if SAM has no
        such definition."""
        query = urllib.parse.urlencode({"dims": f"defname:{defname}",
                                        "format": "json"})
        try:
            return json.loads(_fetch(f"{SAM_API}/files/list?{query}"))
        except urllib.error.HTTPError as exc:
            # SAM answers 400 "No existing dataset definition" for an
            # unknown name, and 404 elsewhere.
            if exc.code in (400, 404):
                return None
            raise

    def metacat_dataset_files(self, did: str) -> list[str] | None:
        """The file names of metacat dataset `did`; None if there is none."""
        try:
            _fetch(f"{METACAT_API}/data/dataset?"
                      + urllib.parse.urlencode({"dataset": did}))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        text = _fetch(f"{METACAT_API}/data/query?with_meta=no",
                         data=f"files from {did}".encode())
        return [record["name"] for record in _json_records(text)]

    def sam_locations(self, names: list[str]) -> dict[str, list[Path]]:
        """Each name's file paths as SAM records them. A name SAM does not
        know is left out."""
        found: dict[str, list[Path]] = {}
        for start in range(0, len(names), LOCATIONS_CHUNK):
            form = urllib.parse.urlencode(
                [("file_name", n) for n in names[start:start + LOCATIONS_CHUNK]])
            reply = json.loads(_fetch(f"{SAM_API}/files/locations?format=json",
                                         data=form.encode()))
            for name, locations in reply.items():
                # full_path is "<system>:<directory>", e.g. enstore:/pnfs/...
                found[name] = [Path(loc["full_path"].split(":", 1)[-1]) / name
                               for loc in locations or []
                               if loc.get("full_path")]
        return found

    def exists(self, path: Path) -> bool:
        return path.exists()

    def locality(self, path: Path) -> str | None:
        """dCache's locality of a file (ONLINE, NEARLINE, ...), read from
        its NFS magic file; None where that cannot be read."""
        try:
            return (path.parent / f".(get)({path.name})(locality)").read_text().strip()
        except OSError:
            return None


CLIENT = Catalogs()


@dataclass
class ResolvedInputs:
    """Paths for the job, and where they came from, for the result."""

    paths: list[Path]
    source: dict = field(default_factory=dict)


def dataset_file_names(dataset: str, client: Catalogs) -> tuple[list[str], str]:
    """A dataset's file names and which catalog listed them. A DID goes to
    metacat; a bare name is tried as a SAM definition, then as a metacat
    dataset in its owner's namespace (mdh's default)."""
    namespace, name = split_did(dataset)
    tried = []
    try:
        if namespace is None:
            tried.append(f"SAM definition {name}")
            names = client.sam_definition_files(name)
            if names is not None:
                return names, "sam"
            owner = name.split(".")[1] if name.count(".") >= 1 else None
            did = f"{owner}:{name}" if owner else None
        else:
            did = dataset
        if did is not None:
            tried.append(f"metacat dataset {did}")
            names = client.metacat_dataset_files(did)
            if names is not None:
                return names, "metacat"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise CatalogError(
            f"cannot list dataset {dataset} ({exc}); SAM and metacat are read "
            "over HTTPS, so check this node can reach them") from None
    raise CatalogError(f"no such dataset: tried {' and '.join(tried)}")


def locate_files(names: list[str], client: Catalogs) -> tuple[list[Path], list[str]]:
    """Each catalogued file's best existing copy, and one complaint per file
    with none. Disk copies win over tape; SAM's records are tried before the
    standard layout."""
    bare = [split_did(n)[1] for n in names]
    try:
        recorded = client.sam_locations(sorted(set(bare)))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise CatalogError(
            f"cannot look up where the input files are in SAM ({exc})") from None
    paths, problems = [], []
    for entry, name in zip(names, bare):
        candidates = sorted(recorded.get(name, []), key=location_rank)
        candidates += [p for p in standard_paths(name) if p not in candidates]
        found = next((p for p in candidates if client.exists(p)), None)
        if found is not None:
            paths.append(found)
        elif candidates:
            problems.append(f"{entry} has no copy on disk (looked in "
                            f"{', '.join(str(p.parent) for p in candidates)})")
        else:
            problems.append(
                f"{entry} is neither registered in SAM nor a Mu2e file name "
                "(tier.owner.description.configuration.sequencer.format); "
                "pass an absolute path for a file outside the catalog")
    return paths, problems


def unstaged(paths: list[Path], client: Catalogs) -> list[Path]:
    """The tape files that are not readable without staging first. A file
    whose locality cannot be read is let through: the job finds out."""
    return [p for p in paths
            if on_tape(p) and (where := client.locality(p)) is not None
            and where not in ONLINE]


def resolve_inputs(
    entries: list[str] | None = None,
    dataset: str | None = None,
    max_files: int | None = None,
    client: Catalogs | None = None,
) -> ResolvedInputs:
    """Paths for a run's inputs: `entries` (absolute paths, or catalogued
    file names to locate) or a `dataset`, cut to its first `max_files` by
    name. Raises CatalogError saying what is wrong."""
    client = client or CLIENT
    source: dict = {}
    if dataset is not None:
        names, catalog = dataset_file_names(dataset, client)
        if not names:
            raise CatalogError(f"dataset {dataset} has no files ({catalog})")
        names = sorted(names)
        if max_files is None and len(names) > MAX_DATASET_FILES:
            raise CatalogError(
                f"dataset {dataset} has {len(names)} files, more than the "
                f"{MAX_DATASET_FILES} one run takes; pass max_files to run over "
                "the first ones by name")
        source = {"dataset": dataset, "catalog": catalog,
                  "n_files_in_dataset": len(names)}
        entries = names[:max_files] if max_files is not None else names
    entries = list(entries or [])

    catalogued = [e for e in entries if is_catalog_name(e)]
    if not catalogued:
        return ResolvedInputs([Path(e) for e in entries], source)

    located, problems = locate_files(catalogued, client)
    if problems:
        shown = "; ".join(problems[:5])
        more = f"; and {len(problems) - 5} more" if len(problems) > 5 else ""
        raise CatalogError(f"cannot find {len(problems)} of {len(catalogued)} "
                           f"catalogued input file(s): {shown}{more}")
    by_name = dict(zip(catalogued, located))
    paths = [by_name[e] if e in by_name else Path(e) for e in entries]

    waiting = unstaged(located, client)
    if waiting:
        what = dataset or " ".join(p.name for p in waiting[:3])
        hint = (f"`samweb prestage-dataset --defname={dataset}`"
                if source.get("catalog") == "sam" else f"`mdh prestage-files {what}`")
        raise CatalogError(
            f"{len(waiting)} of {len(located)} input file(s) are on tape and not "
            f"staged (e.g. {waiting[0]}), so the job would wait on tape. Stage "
            f"them first with {hint}, then rerun")
    where = [*DCACHE_PREFIXES, "other"]
    source["locations"] = dict(Counter(where[location_rank(p)] for p in located))
    source["n_catalogued_files"] = len(located)
    return ResolvedInputs(paths, source)
