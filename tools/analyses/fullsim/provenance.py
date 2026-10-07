"""How many events were generated to make an input file, from SAM.

An EventNtuple file holds only the events that survived the chain before
it: generation, then digitization (whose filter keeps the events with
enough tracker activity), then reconstruction. The efficiency must count
against what was generated, so each file's generated-event count is looked
up in SAM: the first ancestor carrying `dh.gencount` (for a
CeMLeadingLogMix1BB nts file, its parent mcs file). This is the per-file
form of the acceptance pyfitter takes from samDatasetsSummary.sh
(Triggered / Generated for the mcs dataset).

The same metadata gives the simulation chain's stage efficiencies
(normalization.py): a dataset's files are listed, and their metadata read
in bulk, as mu2efiletools' mu2eGenFilterEff does. Which datasets those are
is found the same way, by walking a file's parents up to a given stage.

SAM's read API is plain HTTPS without authentication, so this needs no
samweb client; only the network.
"""

import json
import urllib.parse
import urllib.request

SAM_API = "https://sammu2e.fnal.gov:8483/sam/mu2e/api"
SAM_TIMEOUT_S = 30.0
# nts -> mcs -> dig -> ...: the count is on the mcs file, one step up. Stop
# well before a chain could loop.
MAX_ANCESTRY = 6
# Generations searched for an ancestor in a given stage. nts -> mcs -> dig
# -> dts -> MuminusStopsCat is four, MuminusStopsCat -> TargetStopsCat ->
# TargetStops -> MuBeamCat three; leave room for a longer chain.
MAX_STAGE_SEARCH = 8
# Files per bulk metadata request: mu2eGenFilterEff's default chunk size.
METADATA_CHUNK = 100


class ProvenanceError(RuntimeError):
    """The generated-event count could not be found, worded for the caller."""


def sam_metadata(name: str) -> dict:
    """One file's SAM metadata, as SAM's JSON."""
    url = f"{SAM_API}/files/name/{urllib.parse.quote(name)}/metadata?format=json"
    with urllib.request.urlopen(url, timeout=SAM_TIMEOUT_S) as response:
        return json.load(response)


def dataset_of(file_name: str) -> str | None:
    """The dataset of a SAM file name, tier.owner.description.configuration.
    sequencer.format without its sequencer; None if it is not one."""
    words = file_name.split(".")
    if len(words) != 6 or not all(words):
        return None
    return ".".join(words[:4] + words[5:])


def ancestor_in_stage(name: str, stage: str, fetch=sam_metadata) -> str:
    """The nearest ancestor of file `name` whose dataset's description is
    `stage` (e.g. MuminusStopsCat). A file can have thousands of parents
    (a concatenation), all from one or a few datasets, so each generation
    follows one file per parent dataset not seen before. Two datasets of
    `stage` in the same generation are ambiguous and an error."""
    generation = [name]
    seen: set[str] = set()
    for _ in range(MAX_STAGE_SEARCH):
        parents: dict[str, str] = {}        # dataset -> one of its files
        for current in generation:
            try:
                metadata = fetch(current)
            except Exception as exc:
                raise ProvenanceError(
                    f"cannot read the SAM metadata of {current} ({exc}), so "
                    f"the {stage} dataset {name} descends from is unknown"
                ) from None
            for parent in metadata.get("parents") or []:
                dataset = dataset_of(parent["file_name"])
                if dataset and dataset not in seen:
                    parents.setdefault(dataset, parent["file_name"])
        found = sorted(d for d in parents if d.split(".")[2] == stage)
        if len(found) > 1:
            raise ProvenanceError(
                f"{name} descends from more than one {stage} dataset "
                f"({', '.join(found)}), so which one it was made from is "
                "ambiguous"
            )
        if found:
            return parents[found[0]]
        seen.update(parents)
        generation = list(parents.values())
        if not generation:
            break
    raise ProvenanceError(
        f"no {stage} dataset among the SAM ancestors of {name} (searched "
        f"{MAX_STAGE_SEARCH} generations, through "
        f"{', '.join(sorted(seen)) or 'no parents'})"
    )


def dataset_files(dataset: str) -> list[str]:
    """The SAM file names of a dataset, wherever they are stored (the query
    mu2eGenFilterEff makes)."""
    query = urllib.parse.urlencode({
        "dims": f"dh.dataset={dataset} with availability anylocation",
        "format": "json",
    })
    with urllib.request.urlopen(f"{SAM_API}/files/list?{query}",
                                timeout=SAM_TIMEOUT_S) as response:
        return json.load(response)


def files_metadata(names: list[str]) -> list[dict]:
    """The SAM metadata of many files, METADATA_CHUNK per request."""
    metadata = []
    for start in range(0, len(names), METADATA_CHUNK):
        form = urllib.parse.urlencode(
            [("file_name", name) for name in names[start:start + METADATA_CHUNK]])
        request = urllib.request.Request(f"{SAM_API}/files/metadata",
                                         data=form.encode())
        with urllib.request.urlopen(request, timeout=SAM_TIMEOUT_S) as response:
            metadata.extend(json.load(response))
    return metadata


def generated_events(name: str, fetch=sam_metadata) -> int:
    """Events generated to make file `name` (a SAM file name): the
    `dh.gencount` of the file or its nearest ancestor that has one.
    `fetch` returns a file's metadata; tests pass a fake."""
    current = name
    for _ in range(MAX_ANCESTRY):
        try:
            metadata = fetch(current)
        except Exception as exc:
            raise ProvenanceError(
                f"cannot read the SAM metadata of {current} ({exc}), so the "
                f"events generated for {name} are unknown"
            ) from None
        if metadata.get("dh.gencount"):
            return int(metadata["dh.gencount"])
        parents = metadata.get("parents") or []
        if len(parents) != 1:
            break
        current = parents[0]["file_name"]
    raise ProvenanceError(
        f"no dh.gencount in SAM for {name} or its parents (stopped at "
        f"{current}), so the events generated for it are unknown"
    )
