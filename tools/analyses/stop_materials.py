"""Muon stops per material, from a stop finder's `stopmat` histogram.

The stage that finds stopped muons (the MuBeam job's TargetMuonFinder,
PolyMuonFinder, IPAMuonFinder, ...) books a TH1 `<module>/stopmat` in its
TFileService output (nts.*.root): one alphanumeric bin per stopping material,
labelled with the material's name, filled once per stopped muon. This reads
that histogram and turns each material's count into a rate per generated
event:

    stops_per_gen_event[material] = stops[material] / n_gen_events

The ntuple carries no generated-event bookkeeping of its own, so
`n_gen_events` is the caller's to supply — the generated events the file is
equivalent to (for a resampled or prescaled stage, the equivalent count
upstream of it, whatever the rate should be per).

Several files (data_files) are combined into one table, as `hadd` would:
bins are matched by material name, not bin number, since each file's axis
is labelled in the order its own job met the materials, and a material some
file never saw is simply zero there. `n_gen_events` is then the total for the
whole set. The per-file stop counts are reported beside the combined table.

ROOT grows an alphanumeric axis by doubling it, so the histogram usually has
more bins than labels; unlabelled bins are empty and are dropped. An
unlabelled bin that is *not* empty, or content in the under/overflow, is
reported rather than dropped, since it means stops the table does not name.
"""

import math
from pathlib import Path

from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome

STOP_MODULE = "TargetMuonFinder"
HIST_NAME = "stopmat"


class StopMaterialsError(RuntimeError):
    """Raised for a problem the caller should see verbatim."""


def read_stopmat(rootfile, module: str) -> tuple[list[tuple[str, float, float]], dict]:
    """(material, stops, error) per labelled bin, plus what did not fit.

    `rootfile` is an open uproot file. The labels are TObjStrings whose
    fUniqueID is the bin they name.
    """
    path = f"{module}/{HIST_NAME}"
    try:
        hist = rootfile[path]
    except KeyError:
        available = sorted({key.split(";")[0].rsplit("/", 1)[0]
                            for key in rootfile.keys()
                            if key.split(";")[0].endswith(f"/{HIST_NAME}")})
        hint = (f" Modules with a {HIST_NAME} histogram in this file: "
                f"{', '.join(available)}" if available
                else f" No module in this file has a {HIST_NAME} histogram")
        raise StopMaterialsError(f"histogram '{path}' not found.{hint}")
    if not hist.classname.startswith("TH1"):
        raise StopMaterialsError(f"'{path}' is a {hist.classname}, not a TH1")

    values = hist.values(flow=True)
    errors = hist.errors(flow=True)
    labels = hist.member("fXaxis").member("fLabels") or []
    names = {int(label.member("@fUniqueID")): str(label) for label in labels}
    if not names:
        raise StopMaterialsError(
            f"'{path}' has no bin labels, so its bins cannot be named as "
            "materials"
        )

    nbins = len(values) - 2
    table, unlabelled = [], {}
    for ibin in range(1, nbins + 1):
        if ibin in names:
            table.append((names[ibin], float(values[ibin]), float(errors[ibin])))
        elif values[ibin] != 0.0:
            unlabelled[f"bin{ibin}"] = float(values[ibin])
    return table, {
        "unlabelled_bins": unlabelled,
        "underflow": float(values[0]),
        "overflow": float(values[-1]),
        "hist_entries": float(hist.member("fEntries")),
        "hist_path": path,
    }


def combine_tables(tables: list[list[tuple[str, float, float]]]
                   ) -> list[tuple[str, float, float]]:
    """Sum per-material tables by name, errors in quadrature, first seen first."""
    stops: dict[str, float] = {}
    err2: dict[str, float] = {}
    for table in tables:
        for name, count, err in table:
            stops[name] = stops.get(name, 0.0) + count
            err2[name] = err2.get(name, 0.0) + err ** 2
    return [(name, stops[name], math.sqrt(err2[name])) for name in stops]


def material_rates(table: list[tuple[str, float, float]],
                   n_gen_events: float) -> list[dict[str, float | str]]:
    """Per-material stops, rate per generated event and fraction, most first."""
    total = sum(stops for _, stops, _ in table)
    rows = [
        {
            "material": name,
            "stops": stops,
            "stops_err": err,
            "stops_per_gen_event": stops / n_gen_events,
            "stops_per_gen_event_err": err / n_gen_events,
            "fraction": stops / total if total > 0.0 else 0.0,
        }
        for name, stops, err in table
    ]
    return sorted(rows, key=lambda row: -row["stops"])


def run(context: RunContext) -> RunOutcome:
    """Tabulate one stop finder's stops per material over the input file(s)."""
    import uproot

    module = context.params["stop_module"]
    n_gen_events = context.params["n_gen_events"]
    outdir = context.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    tables, per_file = [], []
    for path in context.input_paths:
        try:
            with uproot.open(path) as rootfile:
                table, leftovers = read_stopmat(rootfile, module)
        except StopMaterialsError as exc:
            where = f" in {path}" if len(context.input_paths) > 1 else ""
            return RunOutcome(error=f"{exc}{where}")
        tables.append(table)
        per_file.append({
            "file": str(path),
            "n_stops": sum(stops for _, stops, _ in table),
            **leftovers,
            "unnamed_stops": (sum(leftovers["unlabelled_bins"].values())
                              + leftovers["underflow"] + leftovers["overflow"]),
        })

    rows = material_rates(combine_tables(tables), n_gen_events)
    # Everything named goes into the total; unnamed content is reported
    # beside it rather than silently added to or dropped from it.
    n_stops = sum(row["stops"] for row in rows)
    n_stops_err = math.sqrt(sum(row["stops_err"] ** 2 for row in rows))
    metrics = {
        "n_stops": n_stops,
        "n_gen_events": float(n_gen_events),
        "stops_per_gen_event": n_stops / n_gen_events,
        "stops_per_gen_event_err": n_stops_err / n_gen_events,
        "n_materials": float(sum(1 for row in rows if row["stops"] > 0.0)),
    }
    unnamed = sum(entry["unnamed_stops"] for entry in per_file)
    hist_entries = sum(entry["hist_entries"] for entry in per_file)

    log_path = outdir / "stop_materials.log"
    lines = ["stop_materials",
             f"  histogram      {module}/{HIST_NAME} "
             f"({hist_entries:g} entries in {len(per_file)} file(s))"]
    lines += [f"  input          {entry['file']}  ({entry['n_stops']:g} stops)"
              for entry in per_file]
    lines += [
        f"  n_gen_events   {n_gen_events:g}",
        "",
        f"  {'material':<24} {'stops':>10} {'stops/gen event':>28} {'fraction':>9}",
    ]
    lines += [
        f"  {row['material']:<24} {row['stops']:>10g} "
        f"{row['stops_per_gen_event']:>13.4e} +- {row['stops_per_gen_event_err']:<10.2e} "
        f"{row['fraction']:>9.4f}"
        for row in rows
    ]
    lines += [
        f"  {'total':<24} {n_stops:>10g} "
        f"{metrics['stops_per_gen_event']:>13.4e} +- "
        f"{metrics['stops_per_gen_event_err']:<10.2e}",
    ]
    for entry in per_file:
        if entry["unnamed_stops"]:
            lines += ["", f"  WARNING: {entry['unnamed_stops']:g} stops outside "
                          f"the labelled bins of {entry['file']} (unlabelled "
                          f"{entry['unlabelled_bins']}, underflow "
                          f"{entry['underflow']:g}, overflow "
                          f"{entry['overflow']:g}), not in the total"]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return RunOutcome(
        metrics=metrics,
        files=[str(log_path)],
        log_path=log_path,
        extra={"stop_module": module, "hist_path": f"{module}/{HIST_NAME}",
               "materials": rows, "hist_entries": hist_entries,
               "unnamed_stops": unnamed, "per_file": per_file},
    )


def summarize(metrics: dict[str, float]) -> str:
    return (
        f"{metrics['n_stops']:g} stops in {metrics['n_materials']:g} materials "
        f"for {metrics['n_gen_events']:g} generated events -> "
        f"{metrics['stops_per_gen_event']:.4g} +- "
        f"{metrics['stops_per_gen_event_err']:.2g} stops / generated event "
        "in total; per-material stops and rates are in metadata.materials."
    )


SPEC = AnalysisSpec(
    name="stop_materials",
    description=(
        "Muon stops per stopping material from a stop finder's stopmat "
        "histogram, as counts and as rates per generated event."
    ),
    input_kind="root_file",
    combines_files=True,
    metrics=(
        "n_stops", "n_gen_events", "stops_per_gen_event",
        "stops_per_gen_event_err", "n_materials",
    ),
    units={
        "stops_per_gen_event": "stops / generated event",
        "stops_per_gen_event_err": "stops / generated event",
    },
    parameters=(
        ParamSpec(
            name="n_gen_events",
            description="Generated events the input is equivalent to — "
                        "the total over all files when several are passed; "
                        "every rate is stops divided by this. The ntuple does "
                        "not record it, so it must be supplied.",
            minimum=1.0,
        ),
        ParamSpec(
            name="stop_module",
            description="Module label of the stop finder whose "
                        f"<module>/{HIST_NAME} histogram to read, e.g. "
                        "PolyMuonFinder or IPAMuonFinder.",
            default=STOP_MODULE, kind="text",
        ),
    ),
    input_hint=(
        "The TFileService ROOT file(s) (nts.*.root) of the job that ran the stop "
        "finder — pass data_files for several, combined into one table. Each "
        f"must hold <stop_module>/{HIST_NAME}, a TH1 with the "
        "material names as bin labels, e.g. "
        "nts.mmackenz.mubeam.Run1Bak_local0818120248.001800_00000000.root."
    ),
    run=run,
    summarize=summarize,
)
