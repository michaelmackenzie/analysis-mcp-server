"""Full-simulation cut-and-count sensitivity from EventNtuple: RefAna/pyCount.

A port of pyCount's process.py `run_count` path: the counting analysis run
over reconstructed tracks in a mixed MC sample (e.g. an MDS ensemble), with
the MC truth saying what each selected track was.

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
4. Count true signal and background (everything else) in the window. The
   background count *is* the expected background: the sample is taken to
   be the experiment, at the exposure the `exposure` parameter names.
5. Limits on that background (limits.py), for n_obs = B as Run-1A quotes
   them: the Run-1A analysis' CLs upper limit (exact Poisson, or profile
   likelihood when the background or efficiency uncertainty is set) and
   5 sigma discovery signal Z sqrt(B + sigma_B^2); and, to compare with
   pyCount, the Feldman-Cousins interval, the expected classical upper
   limit and the 90% CL FC table value. Each also as a branching ratio,
   over captured muons x signal efficiency.
"""

from pathlib import Path

import numpy as np

from ...spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from ..approx_ce_sensitivity import MUON_CAPTURE_RATE
from . import limits
from .cuts import (CUT_DESCRIPTIONS, DEFAULT_CUTS, TRKPID_MIN, TRKQUAL_MIN,
                   CutError, active_cuts, apply_cuts, cut_masks, cut_names)
from .eventntuple import (ORIGIN_NAMES, SID_TT_FRONT, TREE_PATH,
                          EventNtupleError, origin_codes, read_eventntuple)

# pyCount run_count's normalization: captured muons for Run-1A, 28 days at
# 3.84 kW (Run-1A-Analysis/Normalization.md: 5.58e15 stopped, 3.4e15
# captured), and a signal efficiency it took as given. The captures use
# approx_ce_sensitivity's capture fraction, so the two analyses agree on it.
STOPPED_MUONS = 5.58e15
EXPOSURE = STOPPED_MUONS * MUON_CAPTURE_RATE
SIG_EFF = 0.12
CL = 0.9

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

DISCOVERY_SIGMA = 5


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
    exposure = float(params["exposure"])
    sig_eff = float(params["sig_eff"])
    cl = float(params["cl"])
    bkg_rel_unc = float(params["bkg_rel_uncertainty"])
    eff_rel_unc = float(params["sig_eff_rel_uncertainty"])
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

    # The background count is an integer, so n_obs = B exactly, as Run-1A
    # quotes its expected limit.
    b_sigma = bkg_rel_unc * n_bkg
    cls_ul = limits.cls_upper_limit(n_bkg, float(n_bkg), cl, b_sigma, eff_rel_unc)
    cls_method = ("exact Poisson CLs" if b_sigma <= 0 and eff_rel_unc <= 0
                  else "asymptotic profile-likelihood CLs")
    fc_low, fc_high = limits.fc_interval(n_bkg, float(n_bkg), cl)
    expected_ul = limits.expected_upper_limit(float(n_bkg), cl)
    table_ul = limits.fc_table_upper_limit(float(n_bkg))
    s_discovery = limits.required_signal(DISCOVERY_SIGMA, float(n_bkg), b_sigma)
    per_br = exposure * sig_eff
    ses = 1.0 / per_br if per_br > 0 else float("inf")

    def by_origin(mask):
        codes, counts = np.unique(ev["origin"][mask], return_counts=True)
        return {ORIGIN_NAMES.get(int(c), str(int(c))): int(n)
                for c, n in zip(codes, counts)}

    metrics = {
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
        "cls_upper_events": cls_ul,
        "cls_upper_br": cls_ul * ses,
        "fc_lower_events": fc_low,
        "fc_upper_events": fc_high,
        "fc_upper_br": fc_high * ses,
        "expected_ul_events": expected_ul,
        "expected_ul_br": expected_ul * ses,
        "fc_table_ul90_events": table_ul,
        "discovery_5sigma_events": s_discovery,
        "discovery_5sigma_br": s_discovery * ses,
        "ses": ses,
        "exposure": exposure,
        "sig_eff": sig_eff,
        "cl": cl,
        "bkg_rel_uncertainty": bkg_rel_unc,
        "sig_eff_rel_uncertainty": eff_rel_unc,
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
        f"  exposure         {exposure:.4g} captured muons   sig_eff {sig_eff:g}   CL {cl:g}",
        f"  uncertainties    background {bkg_rel_unc:g}, signal efficiency "
        f"{eff_rel_unc:g} (relative)",
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
        f"  true signal {n_sig}, background {n_bkg}: {by_origin(in_window & ~is_signal)}",
        f"  reconstructed front segments in window, all selected tracks: {n_reco_window}",
        "",
        f"Run-1A: {cl:g} CL upper limit, {cls_method}, n_obs = b = {n_bkg}"
        f" (sigma_b = {b_sigma:.4g}): {cls_ul:.4g} events -> BR < {cls_ul * ses:.4g}",
        f"Run-1A: {DISCOVERY_SIGMA} sigma discovery needs {s_discovery:.4g} signal "
        f"events -> BR {s_discovery * ses:.4g}",
        f"SES = 1 / (exposure x sig_eff) = {ses:.4g}",
        "",
        "pyCount, no systematics:",
        f"  FC {cl:g} CL interval for n_obs = {n_bkg}: "
        f"[{fc_low:.4g}, {fc_high:.4g}] events -> BR < {fc_high * ses:.4g}",
        f"  Expected upper limit ({cl:g} CL): {expected_ul:.4g} events -> BR {expected_ul * ses:.4g}",
        f"  FC table 90% CL upper limit: {table_ul:.4g} events",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    files = [str(flow_path), str(events_path)]
    files += _write_plots(outdir, sign, ev, counted, is_signal, window, rows)

    extra.update({
        "cut_flow": {name: int(n) for name, n, _ in flow_rows},
        "counted_by_origin": by_origin(counted),
        "background_window_by_origin": by_origin(in_window & ~is_signal),
        "n_reco_segments_window": n_reco_window,
        "cls_method": cls_method,
    })
    return RunOutcome(metrics=metrics, files=files, log_path=log_path, extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    return (
        f"{metrics['n_signal_window']:.0f} signal and "
        f"{metrics['n_background_window']:.0f} background events in "
        f"[{metrics['signal_mom_low_mevc']:g}, {metrics['signal_mom_high_mevc']:g}] MeV/c x "
        f"[{metrics['signal_time_low_ns']:g}, {metrics['signal_time_high_ns']:g}] ns; "
        f"{metrics['cl']:g} CL CLs upper limit {metrics['cls_upper_events']:.3g} events "
        f"(BR < {metrics['cls_upper_br']:.3g}), SES {metrics['ses']:.3g}, for "
        f"{metrics['exposure']:.3g} captured muons and signal efficiency "
        f"{metrics['sig_eff']:g}."
    )


SPEC = AnalysisSpec(
    name="fullsim_sensitivity",
    description=(
        "Full-simulation CE cut-and-count from reconstructed EventNtuple "
        "files (RefAna/pyCount): select CE-like tracks, count true signal and "
        "background in a momentum-time window, and give the Run-1A CLs upper "
        "limit (with optional background and efficiency uncertainties), the "
        "5 sigma discovery signal and SES, plus pyCount's Feldman-Cousins "
        "limits for comparison."
    ),
    input_kind="root_file",
    combines_files=True,
    metrics=(
        "n_events", "n_events_selected", "n_events_counted",
        "signal_mom_low_mevc", "signal_mom_high_mevc",
        "signal_time_low_ns", "signal_time_high_ns", "window_optimized",
        "n_signal_window", "n_background_window",
        "cls_upper_events", "cls_upper_br",
        "discovery_5sigma_events", "discovery_5sigma_br", "ses",
        "fc_lower_events", "fc_upper_events", "fc_upper_br",
        "expected_ul_events", "expected_ul_br", "fc_table_ul90_events",
        "exposure", "sig_eff", "cl",
        "bkg_rel_uncertainty", "sig_eff_rel_uncertainty",
    ),
    units={
        "n_events": "events", "n_events_selected": "events",
        "n_events_counted": "events",
        "signal_mom_low_mevc": "MeV/c", "signal_mom_high_mevc": "MeV/c",
        "signal_time_low_ns": "ns", "signal_time_high_ns": "ns",
        "n_signal_window": "events", "n_background_window": "events",
        "cls_upper_events": "events",
        "fc_lower_events": "events", "fc_upper_events": "events",
        "expected_ul_events": "events", "fc_table_ul90_events": "events",
        "discovery_5sigma_events": "events",
        "cls_upper_br": "branching ratio",
        "fc_upper_br": "branching ratio", "expected_ul_br": "branching ratio",
        "discovery_5sigma_br": "branching ratio", "ses": "branching ratio",
        "exposure": "captured muons",
        "bkg_rel_uncertainty": "fraction", "sig_eff_rel_uncertainty": "fraction",
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
            name="exposure",
            description="Captured muons the input sample is equivalent to (the "
                        "conversion rate is normalized to captures). The "
                        "background counted in the window is taken as the "
                        "expected background at this exposure, so it must be "
                        "the sample's own. Converts event limits to branching "
                        "ratios with sig_eff. Default: Run-1A, 28 days at "
                        "3.84 kW (5.58e15 stopped muons).",
            default=EXPOSURE, minimum=0.0,
        ),
        ParamSpec(
            name="sig_eff",
            description="Signal efficiency (0-1) for converting event limits "
                        "to branching ratios. Not measured from the input. "
                        "The default is the value pyCount assumes.",
            default=SIG_EFF, minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="cl",
            description="Confidence level for the CLs upper limit, the "
                        "Feldman-Cousins interval and the expected upper "
                        "limit (fc_table_ul90_events is always 90%).",
            default=CL, minimum=0.5, maximum=0.999,
        ),
        ParamSpec(
            name="bkg_rel_uncertainty",
            description="Relative systematic uncertainty on the background "
                        "count, e.g. 0.2 for 20%: a Gaussian constraint on it "
                        "in the CLs limit, and sigma_B in the discovery "
                        "signal. 0 (default) with sig_eff_rel_uncertainty 0 "
                        "gives the exact Poisson CLs limit; any uncertainty "
                        "switches to the asymptotic profile likelihood, which "
                        "at a background of a few events can come out below "
                        "the exact limit.",
            default=0.0, minimum=0.0, maximum=5.0,
        ),
        ParamSpec(
            name="sig_eff_rel_uncertainty",
            description="Relative systematic uncertainty on the signal "
                        "efficiency, e.g. 0.04 for 4%: a Gaussian constraint "
                        "on it in the CLs limit.",
            default=0.0, minimum=0.0, maximum=1.0,
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
