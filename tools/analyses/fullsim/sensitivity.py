"""Full-simulation CE sensitivity from EventNtuple: S/sqrt(B) in the best window.

The full-simulation counterpart of approx_ce_sensitivity: the same figure
of merit, the best S/sqrt(B) over signal windows, with the signal and
background in it, but counted from reconstructed tracks in a mixed MC
sample (e.g. an MDS ensemble) instead of folded from theory spectra. The
selection and window follow RefAna/pyCount's process.py `run_count`.

1. Select CE-like tracks with pyCount's cuts (cuts.py), file by file.
2. Each surviving event is represented by its first selected track: its
   momentum and time at the tracker front, and its MC origin
   (eventntuple.origin_codes). Events whose first track has more than one
   tracker-front segment are skipped, as pyCount's Count.CheckMCTruth does.
3. The signal window in (momentum, time). For e- by default it is optimized:
   pyCount's optimize_momentum_time scan of a 1 MeV/c momentum window and
   the time window's start, for the best S/sqrt(B) with S the true CE. The
   figure of merit follows approx_ce_sensitivity's scan_signal_box: a window
   counts only with both S > 0 and B > 0, and the first best one scanned
   wins. A sample with no such window has nothing to optimize, so it falls
   back to the fixed window. For e+ the window is fixed. `signal_window`
   overrides either.
4. Count true signal and background (everything else, split into DIO,
   cosmic and other) in the window, and report S/sqrt(B) there. The counts
   are the sample's own: nothing is rescaled.
"""

from pathlib import Path

import numpy as np

from ...spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from .cuts import (CUT_DESCRIPTIONS, DEFAULT_CUTS, TRKPID_MIN, TRKQUAL_MIN,
                   CutError, active_cuts, apply_cuts, cut_masks, cut_names)
from .eventntuple import (ORIGIN_NAMES, SID_TT_FRONT, TREE_PATH,
                          EventNtupleError, origin_codes, read_eventntuple)

# Which origin label is signal, per sign searched for.
SIGNAL_ORIGIN = {"minus": 168, "plus": 176}

# Fixed signal windows, (p_low, p_high) MeV/c and (t_low, t_high) ns.
FIXED_WINDOW = {"minus": ((103.9, 105.1), (640.0, 1650.0)),
                "plus": ((90.0, 92.0), (640.0, 1650.0))}

# pyCount's optimize_momentum_time scan, for e- only.
SCAN_MOM_STARTS = (103.0, 105.5, 0.1)    # first, last, step (MeV/c)
SCAN_MOM_WIDTH = 1.0                     # MeV/c
SCAN_TIME_STARTS = (500.0, 700.0, 10.0)  # first, last, step (ns)
SCAN_TIME_END = 1650.0                   # ns

# The region tracks are counted (and plotted) in at all: Count's plot range.
PLOT_MOM_RANGE = {"minus": (95.0, 110.0), "plus": (85.0, 95.0)}
PLOT_TIME_RANGE = (0.0, 1695.0)

# Background origins reported on their own, as approx_ce_sensitivity
# reports DIO and cosmics; everything else is "other".
DIO_ORIGINS = ("DIO", "IPA DIO")
COSMIC_ORIGINS = ("cosmic",)


class FullsimError(RuntimeError):
    """Raised for a problem the caller should see verbatim."""


# --- per-file reduction ------------------------------------------------------

def reduce_file(path: Path, sign: str, active: list[str], trkqual_min: float,
                trkpid_min: float) -> dict:
    """One file's selected events, reduced to flat per-event arrays, and its
    cut flow. Done per file so the jagged arrays never all sit in memory."""
    return reduce_data(read_eventntuple(path), sign, active, trkqual_min,
                       trkpid_min)


def reduce_data(data: dict, sign: str, active: list[str], trkqual_min: float,
                trkpid_min: float) -> dict:
    """reduce_file for EventNtuple groups already in memory."""
    import awkward as ak

    masks = cut_masks(data, sign, trkqual_min, trkpid_min)
    track_mask, flow = apply_cuts(masks, active, data["trk"]["trk.pdg"])
    keep = ak.to_numpy(ak.any(track_mask, axis=-1))

    segs = data["trkfit"]["trksegs"][track_mask][keep]
    sims = data["trkmc"]["trkmcsim"][track_mask][keep]
    evt = data["evt"][keep]

    front = segs[segs["sid"] == SID_TT_FRONT]
    mom = front["mom"]["fCoordinates"]
    p = np.sqrt(mom["fX"] ** 2 + mom["fY"] ** 2 + mom["fZ"] ** 2)
    t = front["time"]
    finite = np.isfinite(p) & np.isfinite(t)
    p, t = p[finite], t[finite]

    first_p = ak.firsts(p, axis=1)                  # events x front segments
    first_t = ak.firsts(t, axis=1)
    nfront = ak.to_numpy(ak.fill_none(ak.num(first_p, axis=1), 0))
    single = nfront == 1

    def one(values):
        return ak.to_numpy(ak.fill_none(ak.firsts(values, axis=1), np.nan)
                           ).astype(np.float64)

    return {
        "n_events": len(data["evt"]),
        "flow": flow,
        "n_selected": int(keep.sum()),
        "p": one(first_p)[single] if single.any() else np.zeros(0),
        "t": one(first_t)[single] if single.any() else np.zeros(0),
        "origin": origin_codes(sims)[single],
        "run": ak.to_numpy(evt["run"])[single],
        "subrun": ak.to_numpy(evt["subrun"])[single],
        "event": ak.to_numpy(evt["event"])[single],
        # every front segment of every selected track, as pyCount's
        # Count.ExtractReco counts them
        "all_p": ak.to_numpy(ak.flatten(p, axis=None)),
        "all_t": ak.to_numpy(ak.flatten(t, axis=None)),
    }


def combine(parts: list[dict]) -> dict:
    out = {key: np.concatenate([part[key] for part in parts])
           for key in ("p", "t", "origin", "run", "subrun", "event",
                       "all_p", "all_t")}
    out["n_events"] = sum(part["n_events"] for part in parts)
    out["n_selected"] = sum(part["n_selected"] for part in parts)
    out["flow"] = list(np.sum([part["flow"] for part in parts], axis=0)
                       ) if parts[0]["flow"] else []
    return out


# --- the signal window -------------------------------------------------------

def scan_window(p: np.ndarray, t: np.ndarray, is_signal: np.ndarray
                ) -> tuple[dict | None, list[dict]]:
    """pyCount's optimize_momentum_time windows [p0, p0 + 1 MeV/c] x
    [t0, 1650 ns], edges inclusive, scored as approx_ce_sensitivity's
    scan_signal_box scores its windows: S/sqrt(B), only where S > 0 and
    B > 0 (NaN elsewhere), the first best one scanned winning a tie.

    The best window is None when no window has both signal and background.
    """
    first, last, step = SCAN_MOM_STARTS
    mom_starts = np.arange(first, last + 1e-9, step)
    first, last, step = SCAN_TIME_STARTS
    time_starts = np.arange(first, last + 1e-9, step)

    rows: list[dict] = []
    best = None
    for p0 in mom_starts:
        p1 = p0 + SCAN_MOM_WIDTH
        in_mom = (p >= p0) & (p <= p1)
        for t0 in time_starts:
            inside = in_mom & (t >= t0) & (t <= SCAN_TIME_END)
            s = int(np.sum(inside & is_signal))
            b = int(np.sum(inside & ~is_signal))
            fom = s / np.sqrt(b) if s > 0 and b > 0 else np.nan
            row = {"mom_low": round(float(p0), 3), "mom_high": round(float(p1), 3),
                   "time_low": round(float(t0), 3), "time_high": SCAN_TIME_END,
                   "n_signal": s, "n_background": b, "s_over_sqrt_b": float(fom)}
            rows.append(row)
            if np.isfinite(fom) and (best is None
                                     or fom > best["s_over_sqrt_b"]):
                best = row
    return best, rows


def parse_window(text: str) -> tuple[tuple[float, float], tuple[float, float]]:
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        values = []
    if len(values) != 4 or not (values[0] < values[1] and values[2] < values[3]):
        raise FullsimError(
            f"signal_window {text!r} must be four comma-separated numbers "
            "'p_low,p_high,t_low,t_high' (MeV/c, ns) with each low below its high"
        )
    return (values[0], values[1]), (values[2], values[3])


# --- plots -------------------------------------------------------------------

def _write_plots(outdir: Path, sign: str, ev: dict, counted: np.ndarray,
                 is_signal: np.ndarray, window, rows: list[dict] | None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written = []
    (p_lo, p_hi), (t_lo, t_hi) = window
    mom_range = PLOT_MOM_RANGE[sign]
    sig, bkg = counted & is_signal, counted & ~is_signal

    for key, rng, (lo, hi), xlabel, name in (
        ("p", mom_range, (p_lo, p_hi), "Reconstructed momentum at tracker front (MeV/c)", "mom"),
        ("t", PLOT_TIME_RANGE, (t_lo, t_hi), "Time at tracker front (ns)", "time"),
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist([ev[key][sig], ev[key][bkg]], bins=50, range=rng, stacked=True,
                color=["tab:blue", "tab:red"], alpha=0.6,
                label=["True signal", "True background"])
        ax.axvspan(lo, hi, color="0.85", zorder=0, label="signal window")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Events")
        ax.legend(fontsize="small")
        fig.tight_layout()
        path = figdir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ev["p"][bkg], ev["t"][bkg], s=8, color="tab:red", alpha=0.5,
               label="True background")
    ax.scatter(ev["p"][sig], ev["t"][sig], s=12, color="tab:blue", marker="x",
               label="True signal")
    ax.add_patch(matplotlib.patches.Rectangle((p_lo, t_lo), p_hi - p_lo, t_hi - t_lo,
                                              fill=False, color="k", lw=1.5))
    ax.set_xlim(*mom_range)
    ax.set_ylim(400.0, PLOT_TIME_RANGE[1])
    ax.set_xlabel("Reconstructed momentum at tracker front (MeV/c)")
    ax.set_ylabel("Time at tracker front (ns)")
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = figdir / "mom_vs_time.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    if rows:
        moms = sorted({r["mom_low"] for r in rows})
        times = sorted({r["time_low"] for r in rows})
        grid = np.full((len(times), len(moms)), np.nan)
        for r in rows:
            fom = r["s_over_sqrt_b"]
            grid[times.index(r["time_low"]), moms.index(r["mom_low"])] = (
                fom if np.isfinite(fom) else np.nan)
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(grid, origin="lower", aspect="auto",
                          extent=(moms[0], moms[-1], times[0], times[-1]))
        ax.plot([p_lo], [t_lo], "o", color="white")
        ax.set_xlabel(f"Momentum window start (MeV/c), width {SCAN_MOM_WIDTH:g}")
        ax.set_ylabel(f"Time window start (ns), end {SCAN_TIME_END:g}")
        ax.set_title("S/sqrt(B) (blank: S = 0 or B = 0)")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        path = figdir / "window_scan.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))
    return written


# --- the runner --------------------------------------------------------------

def run(context: RunContext) -> RunOutcome:
    params = context.params
    sign = str(params["sign"]).strip().lower()
    outdir = context.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    extra = {"sign": sign}

    try:
        if sign not in SIGNAL_ORIGIN:
            raise FullsimError(f"sign must be 'minus' (CE-) or 'plus' (CE+), got {sign!r}")
        active = active_cuts(sign, params["enable_cuts"], params["disable_cuts"])
        extra["cuts"] = active
        parts = [reduce_file(path, sign, active, params["trkqual_min"],
                             params["trkpid_min"])
                 for path in context.input_paths]
        ev = combine(parts)
        if ev["n_selected"] == 0:
            raise FullsimError(
                f"no event in {ev['n_events']} has a track passing the cuts "
                f"({', '.join(active) or 'none'}). Check the input is a "
                f"reconstructed sample for sign='{sign}', or loosen the cuts "
                "with disable_cuts."
            )

        is_signal = ev["origin"] == SIGNAL_ORIGIN[sign]
        rows = None
        optimized = False
        if params["signal_window"].strip():
            window = parse_window(params["signal_window"])
        else:
            window = FIXED_WINDOW[sign]
            if sign == "minus":
                best, rows = scan_window(ev["p"], ev["t"], is_signal)
                extra["window_scan_best"] = best
                if best is not None:
                    window = ((best["mom_low"], best["mom_high"]),
                              (best["time_low"], best["time_high"]))
                    optimized = True
    except (FullsimError, EventNtupleError, CutError) as exc:
        return RunOutcome(error=str(exc), extra=extra)

    # Count.CheckMCTruth: open-interval cuts on the plot region and window.
    (p_lo, p_hi), (t_lo, t_hi) = window
    mom_range = PLOT_MOM_RANGE[sign]
    p, t = ev["p"], ev["t"]
    counted = ((p > mom_range[0]) & (p < mom_range[1])
               & (t > PLOT_TIME_RANGE[0]) & (t < PLOT_TIME_RANGE[1]))
    in_window = counted & (p > p_lo) & (p < p_hi) & (t > t_lo) & (t < t_hi)
    n_sig = int(np.sum(in_window & is_signal))
    n_bkg = int(np.sum(in_window & ~is_signal))
    n_reco_window = int(np.sum((ev["all_p"] > p_lo) & (ev["all_p"] < p_hi)
                               & (ev["all_t"] > t_lo) & (ev["all_t"] < t_hi)))

    def by_origin(mask):
        codes, counts = np.unique(ev["origin"][mask], return_counts=True)
        return {ORIGIN_NAMES.get(int(c), str(int(c))): int(n)
                for c, n in zip(codes, counts)}

    background = by_origin(in_window & ~is_signal)
    n_dio = sum(background.get(name, 0) for name in DIO_ORIGINS)
    n_cosmic = sum(background.get(name, 0) for name in COSMIC_ORIGINS)
    # As in approx_ce_sensitivity, S/sqrt(B) only means something with both;
    # a given or fixed window without them reports NaN.
    sensitivity = n_sig / np.sqrt(n_bkg) if n_sig > 0 and n_bkg > 0 else float("nan")

    metrics = {
        "sensitivity": float(sensitivity),
        "n_events": float(ev["n_events"]),
        "n_events_selected": float(ev["n_selected"]),
        "n_events_counted": float(counted.sum()),
        "signal_mom_low_mevc": float(p_lo),
        "signal_mom_high_mevc": float(p_hi),
        "signal_time_low_ns": float(t_lo),
        "signal_time_high_ns": float(t_hi),
        "window_optimized": float(optimized),
        "n_signal_window": float(n_sig),
        "n_background_window": float(n_bkg),
        "dio_background": float(n_dio),
        "cosmic_background": float(n_cosmic),
        "other_background": float(n_bkg - n_dio - n_cosmic),
    }

    descriptions = dict(zip(cut_names(sign), CUT_DESCRIPTIONS.values()))
    flow_rows = [("No cuts", ev["n_events"], "no selection")] + [
        (name, n, descriptions[name]) for name, n in zip(active, ev["flow"])]
    flow_path = outdir / "cut_flow.csv"
    flow_path.write_text(
        "cut,events_passing,description\n"
        + "".join(f"{name},{n},\"{desc}\"\n" for name, n, desc in flow_rows),
        encoding="utf-8")
    events_path = outdir / "window_events.csv"
    events_path.write_text(
        "run,subrun,event,origin,mom_mevc,time_ns\n" + "".join(
            f"{int(r)},{int(s)},{int(e)},{ORIGIN_NAMES.get(int(o), int(o))},{pp:.4f},{tt:.2f}\n"
            for r, s, e, o, pp, tt in zip(ev["run"][in_window], ev["subrun"][in_window],
                                          ev["event"][in_window], ev["origin"][in_window],
                                          p[in_window], t[in_window])),
        encoding="utf-8")

    log_path = outdir / "fullsim_sensitivity.log"
    lines = [
        "fullsim_sensitivity",
        *(f"  input            {path}" for path in context.input_paths),
        f"  sign             {sign} (signal: {ORIGIN_NAMES[SIGNAL_ORIGIN[sign]]})",
        "",
        "Cut flow (events with a track passing every cut so far):",
        *(f"  {name:<20s} {n:>9d}   {desc}" for name, n, desc in flow_rows),
        "",
        f"{int(counted.sum())} of {ev['n_selected']} selected events counted "
        f"(first track has one tracker-front segment, in "
        f"{mom_range[0]:g}-{mom_range[1]:g} MeV/c and "
        f"{PLOT_TIME_RANGE[0]:g}-{PLOT_TIME_RANGE[1]:g} ns).",
        f"  by origin: {by_origin(counted)}",
        "",
        f"Signal window [{p_lo:g}, {p_hi:g}] MeV/c x [{t_lo:g}, {t_hi:g}] ns "
        + ("(optimized)" if optimized else
           "(given)" if params["signal_window"].strip() else
           "(fixed: no scanned window had both signal and background)"
           if rows else "(fixed)"),
        f"  S = {n_sig}, DIO = {n_dio}, cosmic = {n_cosmic}, "
        f"other = {n_bkg - n_dio - n_cosmic} -> B = {n_bkg}, "
        f"S/sqrt(B) = {sensitivity:.4g}",
        f"  background by origin: {background}",
        f"  reconstructed front segments in window, all selected tracks: {n_reco_window}",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    files = [str(flow_path), str(events_path)]
    files += _write_plots(outdir, sign, ev, counted, is_signal, window, rows)

    extra.update({
        "cut_flow": {name: int(n) for name, n, _ in flow_rows},
        "counted_by_origin": by_origin(counted),
        "background_window_by_origin": background,
        "n_reco_segments_window": n_reco_window,
    })
    return RunOutcome(metrics=metrics, files=files, log_path=log_path, extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    return (
        f"S/sqrt(B) = {metrics['sensitivity']:.4g} in "
        f"[{metrics['signal_mom_low_mevc']:g}, {metrics['signal_mom_high_mevc']:g}] MeV/c x "
        f"[{metrics['signal_time_low_ns']:g}, {metrics['signal_time_high_ns']:g}] ns "
        f"(S = {metrics['n_signal_window']:.0f}, "
        f"B = {metrics['n_background_window']:.0f}: "
        f"DIO {metrics['dio_background']:.0f}, "
        f"cosmic {metrics['cosmic_background']:.0f}, "
        f"other {metrics['other_background']:.0f}) "
        f"from {metrics['n_events']:.0f} events."
    )


SPEC = AnalysisSpec(
    name="fullsim_sensitivity",
    description=(
        "Full-simulation CE sensitivity S/sqrt(B) from reconstructed "
        "EventNtuple files, the counterpart of approx_ce_sensitivity: select "
        "CE-like tracks (RefAna/pyCount's cuts), find the best momentum-time "
        "window, and count the true signal and the DIO, cosmic and other "
        "background in it."
    ),
    input_kind="root_file",
    combines_files=True,
    metrics=(
        "sensitivity",
        "signal_mom_low_mevc", "signal_mom_high_mevc",
        "signal_time_low_ns", "signal_time_high_ns", "window_optimized",
        "n_signal_window", "n_background_window",
        "dio_background", "cosmic_background", "other_background",
        "n_events", "n_events_selected", "n_events_counted",
    ),
    units={
        "n_events": "events", "n_events_selected": "events",
        "n_events_counted": "events",
        "signal_mom_low_mevc": "MeV/c", "signal_mom_high_mevc": "MeV/c",
        "signal_time_low_ns": "ns", "signal_time_high_ns": "ns",
        "n_signal_window": "events", "n_background_window": "events",
        "dio_background": "events", "cosmic_background": "events",
        "other_background": "events",
    },
    parameters=(
        ParamSpec(
            name="sign",
            description="Which conversion to search for: 'minus' (mu- -> e-, "
                        "signal = CE- process codes 167/168) or 'plus' "
                        "(mu- -> e+, signal = 169/176). Sets the track charge "
                        "cut, the signal window and the counting range.",
            default="minus", kind="text",
        ),
        ParamSpec(
            name="signal_window",
            description="'p_low,p_high,t_low,t_high' (MeV/c, ns) to fix the "
                        "signal window. Empty (default): for 'minus', scan for "
                        "the best S/sqrt(B) window over pyCount's windows, "
                        "scored as approx_ce_sensitivity scores its windows "
                        "(only windows with both signal and background "
                        "count), falling back to 103.9-105.1 MeV/c x "
                        "640-1650 ns if none has both; for 'plus', 90-92 MeV/c x "
                        "640-1650 ns. An optimized window is tuned on the "
                        "same events it counts, so its background is biased "
                        "low. Fix the window for an unbiased count.",
            default="", kind="text", allow_empty=True,
        ),
        ParamSpec(
            name="enable_cuts",
            description="Comma-separated cuts to apply on top of the default "
                        "set: the charge cut (is_reco_electron for sign "
                        "'minus', is_reco_positron for 'plus'), "
                        f"{', '.join(DEFAULT_CUTS[1:])}. The others: "
                        + ", ".join(n for n in cut_names("minus")
                                    if n not in DEFAULT_CUTS[1:]
                                    and n != "is_reco_electron") + ".",
            default="", kind="text", allow_empty=True,
        ),
        ParamSpec(
            name="disable_cuts",
            description="Comma-separated cuts to drop from the default set, "
                        "e.g. 'has_st,no_opa'.",
            default="", kind="text", allow_empty=True,
        ),
        ParamSpec(
            name="trkqual_min",
            description="good_trkqual threshold: keep tracks with TrkQual "
                        "above this.",
            default=TRKQUAL_MIN, minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="trkpid_min",
            description="good_trkpid threshold: keep tracks with TrkPID above "
                        "this.",
            default=TRKPID_MIN, minimum=0.0, maximum=1.0,
        ),
    ),
    input_hint=(
        f"Reconstructed EventNtuple file(s) (nts.*.root with {TREE_PATH}) "
        "from a mixed MC sample with MC truth, e.g. an MDS ensemble: "
        "/exp/mu2e/data/users/mu2epro/ensembles/MDS3/MDS3c/merged_files_1/. "
        "Several files are combined into one count. Each takes a few seconds "
        "per 10k events."
    ),
    run=run,
    summarize=summarize,
)
