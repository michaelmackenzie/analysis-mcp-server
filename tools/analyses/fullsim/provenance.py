"""How many events were generated to make an input file, from SAM.

An EventNtuple file holds only the events that survived the chain before
it: generation, then digitization (whose filter keeps the events with
enough tracker activity), then reconstruction. The efficiency must count
against what was generated, so each file's generated-event count is looked
up in SAM: the first ancestor carrying `dh.gencount` (for a
CeMLeadingLogMix1BB nts file, its parent mcs file). This is the per-file
form of the acceptance pyfitter takes from samDatasetsSummary.sh
(Triggered / Generated for the mcs dataset).

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


class ProvenanceError(RuntimeError):
    """The generated-event count could not be found, worded for the caller."""


def sam_metadata(name: str) -> dict:
    """One file's SAM metadata, as SAM's JSON."""
    url = f"{SAM_API}/files/name/{urllib.parse.quote(name)}/metadata?format=json"
    with urllib.request.urlopen(url, timeout=SAM_TIMEOUT_S) as response:
        return json.load(response)


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
