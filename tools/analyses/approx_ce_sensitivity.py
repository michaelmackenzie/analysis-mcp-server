"""Approximate Run-1A CE sensitivity: signal = CE, background = DIO + cosmics.

A Python conversion of Mu2eOptAna/scripts/rough_run1a_sensitivity.C. It reads
the histograms EdepAna writes (nts.*.root) and estimates S/sqrt(B) for the
best momentum window.

The chain, following the original macro:

1. Signal shape: `trk_front_energy` from the "edep 10 MeV" histogram set —
   the energy of the primary at the front of the tracker for events leaving
   >10 MeV in the calorimeter. Rebinned x2 and scaled to a rate for NPOT
   protons at the assumed branching ratio and signal efficiency.
2. Signal smearing: convolved with a Gaussian tracker resolution.
3. DIO background: the Heeck/Szafron theoretical spectrum, scaled to a rate,
   then convolved with the *measured* energy-loss response
   (`trk_front_energy_diff`, energy at the tracker minus energy at birth) and
   the same tracker resolution.
4. Cosmic background: flat in momentum at a rough rate per MeV/c, scaled by
   the live on-spill time implied by NPOT.
5. Scan every window [x1, x2] with x1 >= 50 MeV and keep the one maximizing
   S/sqrt(B).

The numbers are rough by construction — this is a figure of merit for
comparing beamline configurations, not a sensitivity calculation.
"""

from pathlib import Path

import numpy as np

from ..root_hist import Hist1D
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome

# --- assumptions carried over from the macro ---------------------------------

NPOT = 1.0e18                     # protons on target assumed
SIGNAL_BR = 1.0e-13 / 0.609       # CE branching ratio for R_mue = 1e-9
MEAN_POT_PER_EVENT = 1.6e7        # 1BB
ONSPILL_SECONDS_PER_EVENT = 1.695e-6
COSMIC_RATE_PER_SECOND_PER_MEV = 2.0e4 / 1.1e7  # rough, per second per MeV/c
DIO_RATE_FRACTION = 0.39          # DIO fraction feeding the rate normalization
TRK_RESOLUTION_SIGMA_MEV = 0.2
SIGNAL_BOX_MIN_MEV = 50.0         # below this DIO swamps everything anyway
SIGNAL_REBIN = 2

# The Heeck (2016) / Szafron DIO spectrum, finely binned.
DIO_TABLE = Path(
    "/exp/mu2e/app/users/mmackenz/run1b/Run1BAna/data/"
    "heeck_finer_binning_2016_szafron.tbl"
)
DIO_NBINS, DIO_EMIN, DIO_EMAX = 11000, 0.0, 110.0

# Where the histograms live in the EdepAna output. hist_2 is the "edep 10 MeV"
# set (see EdepAna_module.cc bookHistograms).
HIST_DIR = "EDepAna/hist_2"
SIGNAL_HIST = "trk_front_energy"
RESPONSE_HIST = "trk_front_energy_diff"
EXTRA_PLOT_HISTS = ("primary_start_z", "primary_start_r")


class SensitivityError(RuntimeError):
    """Raised for a problem the caller should see verbatim."""


# --- pieces of the calculation -----------------------------------------------

def load_dio_spectrum(table: Path = DIO_TABLE) -> Hist1D:
    """The theoretical DIO spectrum as a probability density in energy.

    The table holds (energy, weight) rows on a 0.01 MeV grid; each row sets the
    bin containing that energy. Normalized so the result integrates to 1 over
    energy.
    """
    if not table.exists():
        raise SensitivityError(f"DIO spectrum table not found: {table}")
    data = np.loadtxt(table)
    if data.ndim != 2 or data.shape[1] < 2:
        raise SensitivityError(f"DIO table {table} is not two columns of numbers")
    energy, weight = data[:, 0], data[:, 1]

    dio = Hist1D(DIO_NBINS, DIO_EMIN, DIO_EMAX, name="h_dio", title="DIO spectrum")
    # Nudge left by half a bin so a tabulated energy lands in the bin it ends.
    bins = dio.find_bin(energy - dio.bin_width / 2.0)
    dio.contents[bins] = weight

    total = dio.integral()
    if total <= 0.0:
        raise SensitivityError(f"DIO table {table} summed to zero weight")
    return dio.scale(1.0 / (dio.bin_width * total))


def tracker_resolution(sigma: float = TRK_RESOLUTION_SIGMA_MEV,
                       nbins: int = 5000, half_range: float = 5.0) -> Hist1D:
    """Gaussian tracker response as a density in energy offset."""
    res = Hist1D(nbins, -half_range, half_range, name="response", title="response")
    centers = res.centers()
    res.values()[:] = np.exp(-0.5 * (centers / sigma) ** 2) / (
        sigma * np.sqrt(2.0 * np.pi)
    )
    return res


def convolve(true_hist: Hist1D, response: Hist1D, name: str = "") -> Hist1D:
    """Smear `true_hist` by `response`, a density of energy offsets.

    Keeps the binning of `true_hist`: each true bin's content is redistributed
    to bins at (true energy + offset), weighted by the response probability
    mass in each offset bin. Content pushed off the axis lands in the flow
    bins rather than piling up at the edges.
    """
    reco = Hist1D(true_hist.nbins, true_hist.xmin, true_hist.xmax,
                  name=name or f"{true_hist.name}_reco", title=true_hist.title)

    true_centers = true_hist.centers()
    true_values = true_hist.values()
    # Bins with nothing in them contribute nothing; skipping them is what makes
    # the 11000 x 5000 DIO convolution affordable.
    filled = np.flatnonzero(true_values != 0.0)
    if filled.size == 0:
        return reco
    centers, weights = true_centers[filled], true_values[filled]

    offsets = response.centers()
    probabilities = response.values() * response.bin_width
    active = np.flatnonzero(probabilities != 0.0)

    for joffset in active:
        target = reco.find_bin(centers + offsets[joffset])
        np.add.at(reco.contents, target, weights * probabilities[joffset])
    return reco


def mpv_fwhm(hist: Hist1D) -> tuple[float, float]:
    """Rough most-probable value and full width at half maximum."""
    max_bin = hist.get_maximum_bin()
    mpv = float(hist.bin_center(max_bin))
    half = hist.content(max_bin) / 2.0
    first = hist.find_first_bin_above(half)
    last = max(first, hist.find_last_bin_above(half))
    if first < 0:
        return mpv, 0.0
    return mpv, hist.bin_up_edge(last) - hist.bin_low_edge(first)


def scan_signal_box(signal: Hist1D, dio: Hist1D, cosmic: Hist1D,
                    box_min_mev: float = SIGNAL_BOX_MIN_MEV,
                    top_n: int = 10) -> tuple[dict[str, float], list[dict]]:
    """Find the window [x1, x2] maximizing S/sqrt(B).

    Scans every pair of signal bins with x1 >= box_min_mev.

    Window sums are accumulated from each window's own lower edge outward,
    never as a difference of whole-spectrum prefix sums: the DIO spectrum spans
    ~18 orders of magnitude, so subtracting two such totals to get a count of
    order 1 loses the answer entirely to float cancellation.
    """
    centers = signal.centers()
    first_bin = int(np.searchsorted(centers, box_min_mev)) + 1

    # Tail sums for the loop guards, summed inward from the top edge.
    signal_tail = np.cumsum(signal.values()[::-1])[::-1]

    def running(hist: Hist1D, lo: int) -> np.ndarray:
        """Sums from bin `lo` outward: running[k] covers bins lo..lo+k."""
        return np.cumsum(hist.contents[max(lo, 0):])

    best: dict[str, float] | None = None
    tried: list[dict] = []

    for ibin in range(first_bin, signal.nbins + 1):
        # Nothing left to the right: no wider window can help.
        if signal_tail[ibin - 1] <= 0.0:
            break
        x1 = float(centers[ibin - 1])
        s_lo, d_lo, c_lo = (h.find_bin(x1) for h in (signal, dio, cosmic))
        s_run, d_run, c_run = (running(h, lo) for h, lo in
                               ((signal, s_lo), (dio, d_lo), (cosmic, c_lo)))

        def window(run: np.ndarray, lo: int, hi: int) -> float:
            """Sum of bins lo..hi, clipped to what the histogram holds."""
            index = min(max(hi - lo, 0), run.size - 1)
            return float(run[index]) if run.size else 0.0

        for jbin in range(ibin, signal.nbins + 1):
            if signal_tail[jbin - 1] <= 0.0:
                break
            x2 = float(centers[jbin - 1])
            s = window(s_run, s_lo, signal.find_bin(x2))
            d = window(d_run, d_lo, dio.find_bin(x2))
            c = window(c_run, c_lo, cosmic.find_bin(x2))
            background = d + c
            if s <= 0.0 or background <= 0.0:
                continue
            entry = {
                "low_mev": x1, "high_mev": x2, "signal": s, "dio": d,
                "cosmic": c, "background": background,
                "sensitivity": s / np.sqrt(background),
            }
            tried.append(entry)
            if best is None or entry["sensitivity"] > best["sensitivity"]:
                best = entry

    if best is None:
        raise SensitivityError(
            "no signal window had both signal and background > 0 — the input "
            "histograms are probably empty above "
            f"{box_min_mev:g} MeV"
        )
    tried.sort(key=lambda e: e["sensitivity"], reverse=True)
    return best, tried[:top_n]


# --- plots -------------------------------------------------------------------

def _write_plots(outdir: Path, signal: Hist1D, dio_reco: Hist1D, cosmic: Hist1D,
                 response: Hist1D, resolution: Hist1D, dio_true: Hist1D,
                 extras: dict[str, Hist1D], best: dict[str, float],
                 mpv: float, fwhm: float) -> list[str]:
    """Reproduce the macro's figures. Returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def step(ax, hist: Hist1D, **kwargs):
        edges = np.linspace(hist.xmin, hist.xmax, hist.nbins + 1)
        ax.stairs(hist.values(), edges, **kwargs)

    # signal vs background — the one that matters
    fig, ax = plt.subplots(figsize=(8, 6))
    step(ax, signal, color="tab:blue", lw=1.8, label="Signal (CE)")
    step(ax, dio_reco, color="tab:red", lw=1.8, label="DIO")
    step(ax, cosmic, color="tab:green", lw=1.8, label="Cosmic")
    ax.axvspan(best["low_mev"], best["high_mev"], color="0.85", zorder=0,
               label=f"box [{best['low_mev']:.1f}, {best['high_mev']:.1f}]")
    ax.set_yscale("log")
    ax.set_xlim(min(95.0, mpv - 1.5 * fwhm), max(105.0, mpv + 1.5 * fwhm))
    ax.set_ylim(1e-3, 1e4)
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel(f"Rate / {signal.bin_width:.1f} MeV")
    ax.set_title(f"Signal vs. background — S/sqrt(B) = {best['sensitivity']:.3g}")
    ax.legend(ncols=2, fontsize="small")
    fig.tight_layout()
    path = figdir / "sig_vs_bkg.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # DIO before/after smearing
    fig, ax = plt.subplots(figsize=(8, 6))
    step(ax, dio_true, color="tab:blue", lw=1.5, label="DIO (true)")
    step(ax, dio_reco, color="tab:red", lw=1.5, label="DIO (smeared)")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Rate")
    ax.set_title("DIO spectrum")
    ax.set_ylim(1.e-10, 1.e20)
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = figdir / "dio.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # inputs: measured energy-loss response and the assumed resolution
    for hist, filename, title, xlabel in (
        (response, "response.png", "Energy loss before the tracker",
         "Energy difference (MeV)"),
        (resolution, "res.png",
         f"Tracker resolution (sigma = {TRK_RESOLUTION_SIGMA_MEV:g} MeV)",
         "Energy offset (MeV)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        step(ax, hist, color="tab:blue", lw=1.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        fig.tight_layout()
        path = figdir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    # optional primary-vertex distributions, as the macro drew them
    for hist_name, hist in extras.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        step(ax, hist, color="tab:blue", lw=1.5)
        ax.set_xlabel(hist_name)
        ax.set_title(hist.title or hist_name)
        fig.tight_layout()
        path = figdir / f"ce_{hist_name.replace('primary_start_', '')}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    return written


# --- the runner --------------------------------------------------------------

def _read_hist(rootfile, path: str, required: bool = True) -> Hist1D | None:
    try:
        obj = rootfile[path]
    except KeyError:
        if required:
            raise SensitivityError(
                f"histogram '{path}' not found — is this an EdepAna nts.*.root file?"
            )
        return None
    return Hist1D.from_uproot(obj, name=path.rsplit("/", 1)[-1])


def run(context: RunContext) -> RunOutcome:
    """Compute the approximate CE sensitivity for one EdepAna ROOT file."""
    import uproot

    sig_eff = context.params["sig_eff"]
    npot = context.params["npot"]
    outdir = context.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        with uproot.open(context.input_path) as rootfile:
            signal = _read_hist(rootfile, f"{HIST_DIR}/{SIGNAL_HIST}")
            response = _read_hist(rootfile, f"{HIST_DIR}/{RESPONSE_HIST}")
            extras = {
                name: hist
                for name in EXTRA_PLOT_HISTS
                if (hist := _read_hist(rootfile, f"{HIST_DIR}/{name}", required=False))
                is not None
            }

        # Both normalizations divide by the entry count, so empty input is a
        # clean error rather than a division by zero.
        for hist, label in ((signal, SIGNAL_HIST), (response, RESPONSE_HIST)):
            if hist.entries <= 0:
                raise SensitivityError(
                    f"'{HIST_DIR}/{label}' has no entries: no event in this file "
                    "left >10 MeV in the calorimeter, so there is no signal "
                    "shape to work with. Run this on a CE (signal) sample."
                )

        # 1-2. signal shape -> rate, then smeared by the tracker resolution
        signal.rebin(SIGNAL_REBIN)
        signal.scale(npot * SIGNAL_BR * sig_eff / signal.entries)
        response.scale(sig_eff / response.entries / response.bin_width)
        resolution = tracker_resolution()
        signal_reco = convolve(signal, resolution, name="signal_reco")
        mpv, fwhm = mpv_fwhm(signal_reco)

        # 3. DIO: theory spectrum -> rate, smeared by energy loss + resolution
        dio_true = load_dio_spectrum()
        dio_true.scale(DIO_RATE_FRACTION * sig_eff * npot)
        dio_reco = convolve(convolve(dio_true, response), resolution, name="dio_reco")
        dio_reco.rebin(int(signal_reco.bin_width / dio_reco.bin_width))

        # 4. cosmics: flat rate per MeV/c over the implied on-spill time
        events = npot / MEAN_POT_PER_EVENT
        onspill_seconds = events * ONSPILL_SECONDS_PER_EVENT
        cosmic_rate = COSMIC_RATE_PER_SECOND_PER_MEV * onspill_seconds
        cosmic = Hist1D(signal_reco.nbins, signal_reco.xmin, signal_reco.xmax,
                        name="cosmic")
        cosmic.values()[:] = cosmic_rate * cosmic.bin_width

        # 5. best window
        best, top = scan_signal_box(signal_reco, dio_reco, cosmic)
    except SensitivityError as exc:
        return RunOutcome(error=str(exc))

    metrics = {
        "sensitivity": float(best["sensitivity"]),
        "signal_box_low_mev": float(best["low_mev"]),
        "signal_box_high_mev": float(best["high_mev"]),
        "signal_rate": float(best["signal"]),
        "dio_background": float(best["dio"]),
        "cosmic_background": float(best["cosmic"]),
        "total_background": float(best["background"]),
        "signal_mpv_mev": float(mpv),
        "signal_fwhm_mev": float(fwhm),
    }

    log_path = outdir / "approx_ce_sensitivity.log"
    lines = [
        "approx_ce_sensitivity",
        f"  input            {context.input_path}",
        f"  sig_eff          {sig_eff:g}",
        f"  NPOT             {npot:g}",
        f"  signal BR        {SIGNAL_BR:.4g}  (R_mue = 1e-9)",
        f"  cosmic rate      {cosmic_rate:.4g} per MeV/c "
        f"({onspill_seconds:.4g} s on-spill)",
        f"  signal entries   {signal.entries:g}",
        f"  MPV / FWHM       {mpv:.3f} / {fwhm:.3f} MeV",
        "",
        f"Best window [{best['low_mev']:.1f}, {best['high_mev']:.1f}] MeV: "
        f"S = {best['signal']:.3g}, DIO = {best['dio']:.3g}, "
        f"cosmic = {best['cosmic']:.3g} -> B = {best['background']:.3g}, "
        f"S/sqrt(B) = {best['sensitivity']:.4g}",
        "",
        f"Top {len(top)} windows scanned:",
    ]
    lines += [
        f"  [{e['low_mev']:6.1f}, {e['high_mev']:6.1f}] MeV  S = {e['signal']:9.3g}  "
        f"DIO = {e['dio']:9.3g}  cosmic = {e['cosmic']:9.3g}  "
        f"S/sqrt(B) = {e['sensitivity']:.4g}"
        for e in top
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    files = _write_plots(outdir, signal_reco, dio_reco, cosmic, response,
                         resolution, dio_true, extras, best, mpv, fwhm)

    return RunOutcome(
        metrics=metrics,
        files=files,
        log_path=log_path,
        extra={
            "npot": npot,
            "sig_eff": sig_eff,
            "signal_br": SIGNAL_BR,
            "cosmic_rate_per_mev": cosmic_rate,
            "onspill_seconds": onspill_seconds,
            "signal_hist_entries": float(signal.entries),
            "dio_table": str(DIO_TABLE),
        },
    )


def summarize(metrics: dict[str, float]) -> str:
    return (
        f"S/sqrt(B) = {metrics['sensitivity']:.4g} in "
        f"[{metrics['signal_box_low_mev']:.1f}, "
        f"{metrics['signal_box_high_mev']:.1f}] MeV "
        f"(S = {metrics['signal_rate']:.3g}, B = {metrics['total_background']:.3g})."
    )


SPEC = AnalysisSpec(
    name="approx_ce_sensitivity",
    description=(
        "Approximate Run-1A conversion-electron sensitivity S/sqrt(B) from "
        "EdepAna histograms, with DIO and cosmic backgrounds."
    ),
    input_kind="root_file",
    produced_by=("edep",),
    metrics=(
        "sensitivity", "signal_box_low_mev", "signal_box_high_mev",
        "signal_rate", "dio_background", "cosmic_background",
        "total_background", "signal_mpv_mev", "signal_fwhm_mev",
    ),
    units={
        "signal_box_low_mev": "MeV",
        "signal_box_high_mev": "MeV",
        "signal_mpv_mev": "MeV",
        "signal_fwhm_mev": "MeV",
    },
    parameters=(
        ParamSpec(
            name="sig_eff",
            description="Signal (CE) reconstruction+selection efficiency, 0-1. "
                        "Scales signal and both backgrounds.",
            minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="npot",
            description="Protons on target to assume for the rate normalization.",
            default=NPOT, minimum=0.0,
        ),
    ),
    input_hint=(
        "An EdepAna nts.*.root file from a CE (signal) sample — it must have "
        f"{HIST_DIR}/{SIGNAL_HIST} and /{RESPONSE_HIST} filled, i.e. events "
        "leaving >10 MeV in the calorimeter."
    ),
    run=run,
    summarize=summarize,
)
