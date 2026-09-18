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
4. Cosmic background: flat in momentum at a rough rate per second per MeV/c
   (the `cosmic_rate_per_s_per_mev` parameter), scaled by the live on-spill
   time implied by NPOT.
5. Scan every window [x1, x2] with x1 >= 50 MeV and keep the one maximizing
   S/sqrt(B).

The numbers are rough by construction — this is a figure of merit for
comparing beamline configurations, not a sensitivity calculation.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..spectrum import Kernel, Spectrum
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome

# --- assumptions carried over from the macro ---------------------------------

MUON_CAPTURE_RATE = 0.609         # N(muon captures) / N(muon stops) on aluminum
NPOT = 1.0e18                     # protons on target assumed
SIGNAL_BR = 1.0e-13 / MUON_CAPTURE_RATE # CE branching ratio for R_mue = 1e-13
MEAN_POT_PER_EVENT = 1.6e7        # 1BB
ONSPILL_SECONDS_PER_EVENT = 1.695e-6
COSMIC_RATE_PER_SECOND_PER_MEV = 10. / 7.8e5  # rough, per second per MeV/c, taken from Run 1A mu- --> e- analysis
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

def load_dio_spectrum(table: Path = DIO_TABLE) -> Spectrum:
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

    width = (DIO_EMAX - DIO_EMIN) / DIO_NBINS
    values = np.zeros(DIO_NBINS)
    # Nudge left by half a bin so a tabulated energy lands in the bin it ends.
    index = np.floor((energy - width / 2.0 - DIO_EMIN) / width).astype(np.int64)
    inside = (index >= 0) & (index < DIO_NBINS)
    values[index[inside]] = weight[inside]

    total = values.sum()
    if total <= 0.0:
        raise SensitivityError(f"DIO table {table} summed to zero weight")
    return Spectrum(values / (total * width), DIO_EMIN, width,
                    name="dio", title="DIO spectrum")


def mpv_fwhm(spectrum: Spectrum) -> tuple[float, float]:
    """Rough most-probable value and full width at half maximum."""
    peak = int(np.argmax(spectrum.values))
    mpv = float(spectrum.centers()[peak])
    above = np.flatnonzero(spectrum.values > spectrum.values[peak] / 2.0)
    if above.size == 0:
        return mpv, 0.0
    edges = spectrum.edges()
    return mpv, float(edges[above[-1] + 1] - edges[above[0]])


def scan_signal_box(signal: Spectrum, dio: Spectrum, cosmic: Spectrum,
                    box_min_mev: float = SIGNAL_BOX_MIN_MEV,
                    top_n: int = 10) -> tuple[dict[str, float], list[dict]]:
    """Find the window [x1, x2] maximizing S/sqrt(B).

    The three spectra share one binning, so a window is a slice and the scan is
    a cumulative sum per lower edge. Those sums run outward from each window's
    own lower edge, never as a difference of whole-spectrum prefix sums: the
    DIO spectrum spans ~18 orders of magnitude, so subtracting two such totals
    to get a count of order 1 loses the answer entirely to float cancellation.
    """
    for other in (dio, cosmic):
        if (other.nbins, other.xmin, other.width) != (signal.nbins, signal.xmin,
                                                      signal.width):
            raise SensitivityError(
                f"'{other.name}' is not on the signal's binning; regrid it first"
            )

    centers = signal.centers()
    first_bin = int(np.searchsorted(centers, box_min_mev))
    best: dict[str, float] | None = None
    per_edge: list[dict] = []

    for ibin in range(first_bin, signal.nbins):
        s = np.cumsum(signal.values[ibin:])
        d = np.cumsum(dio.values[ibin:])
        c = np.cumsum(cosmic.values[ibin:])
        background = d + c
        usable = np.flatnonzero((s > 0.0) & (background > 0.0))
        if usable.size == 0:
            continue
        ratio = s[usable] / np.sqrt(background[usable])
        jbest = usable[int(np.argmax(ratio))]
        entry = {
            "low_mev": float(centers[ibin]),
            "high_mev": float(centers[ibin + jbest]),
            "signal": float(s[jbest]), "dio": float(d[jbest]),
            "cosmic": float(c[jbest]), "background": float(background[jbest]),
            "sensitivity": float(s[jbest] / np.sqrt(background[jbest])),
        }
        per_edge.append(entry)
        if best is None or entry["sensitivity"] > best["sensitivity"]:
            best = entry

    if best is None:
        raise SensitivityError(
            "no signal window had both signal and background > 0 — the input "
            "histograms are probably empty above "
            f"{box_min_mev:g} MeV"
        )
    # The best window for each lower edge, ranked -- more informative in the
    # log than the top 10 overall, which only ever differ by a bin.
    per_edge.sort(key=lambda e: e["sensitivity"], reverse=True)
    return best, per_edge[:top_n]


# --- plots -------------------------------------------------------------------

def _write_plots(outdir: Path, signal: Spectrum, dio_reco: Spectrum,
                 cosmic: Spectrum, response: Spectrum, resolution: Spectrum,
                 dio_true: Spectrum, extras: dict[str, Spectrum],
                 best: dict[str, float], mpv: float, fwhm: float) -> list[str]:
    """Reproduce the macro's figures. Returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def step(ax, spectrum: Spectrum, **kwargs):
        ax.stairs(spectrum.values, spectrum.edges(), **kwargs)

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
    ax.set_ylabel(f"Rate / {signal.width:.1f} MeV")
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

def _read_hist(rootfile, path: str, required: bool = True) -> Spectrum | None:
    try:
        obj = rootfile[path]
    except KeyError:
        if required:
            raise SensitivityError(
                f"histogram '{path}' not found — is this an EdepAna nts.*.root file?"
            )
        return None
    return Spectrum.from_uproot(obj, name=path.rsplit("/", 1)[-1])


def run(context: RunContext) -> RunOutcome:
    """Compute the approximate CE sensitivity for one EdepAna ROOT file."""
    import uproot

    sig_eff = context.params["sig_eff"]
    npot = context.params["npot"]
    cosmic_rate_per_s_per_mev = context.params["cosmic_rate_per_s_per_mev"]
    mean_pot_per_event = context.params["mean_pot_per_event"]
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
        signal = signal.rebin(SIGNAL_REBIN)
        signal = signal.scaled(npot * SIGNAL_BR * sig_eff / signal.entries)
        # A density in energy offset. The efficiency is folded in here, so it
        # rides along with the response into the smeared DIO spectrum.
        response = response.scaled(sig_eff / response.entries / response.width)
        signal_reco = signal.smear(
            Kernel.gaussian(signal.width, TRK_RESOLUTION_SIGMA_MEV)
        )
        mpv, fwhm = mpv_fwhm(signal_reco)

        # 3. DIO: theory spectrum -> rate, smeared by the energy loss and then
        # by the resolution on its own fine binning, then put on signal's bins.
        dio_true = load_dio_spectrum().scaled((1. - MUON_CAPTURE_RATE) * sig_eff * npot)
        resolution = Kernel.gaussian(dio_true.width, TRK_RESOLUTION_SIGMA_MEV)
        dio_reco = (dio_true
                    .smear(Kernel.from_density(response, dio_true.width))
                    .smear(resolution)
                    .regrid(signal_reco))

        # 4. cosmics: flat rate per MeV/c over the implied on-spill time
        events = npot / mean_pot_per_event
        onspill_seconds = events * ONSPILL_SECONDS_PER_EVENT
        cosmic_rate = cosmic_rate_per_s_per_mev * onspill_seconds
        cosmic = replace(
            signal_reco, name="cosmic", title="Cosmics",
            values=np.full(signal_reco.nbins, cosmic_rate * signal_reco.width),
        )

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
        # The assumptions the rates above are built on, reported with them so
        # a number never travels without the normalization behind it.
        "npot": float(npot),
        "cosmic_rate_per_s_per_mev": float(cosmic_rate_per_s_per_mev),
    }

    log_path = outdir / "approx_ce_sensitivity.log"
    lines = [
        "approx_ce_sensitivity",
        f"  input            {context.input_path}",
        f"  sig_eff          {sig_eff:g}",
        f"  NPOT             {npot:g}",
        f"  signal BR        {SIGNAL_BR:.4g}  (R_mue = 1e-9)",
        f"  cosmic rate      {cosmic_rate_per_s_per_mev:.4g} per s per MeV/c "
        f"-> {cosmic_rate:.4g} per MeV/c ({onspill_seconds:.4g} s on-spill)",
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
                         resolution.as_spectrum(dio_true.width, "resolution"),
                         dio_true, extras, best, mpv, fwhm)

    return RunOutcome(
        metrics=metrics,
        files=files,
        log_path=log_path,
        extra={
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
        f"(S = {metrics['signal_rate']:.3g}, B = {metrics['total_background']:.3g}) "
        f"for {metrics['npot']:.3g} POT and a cosmic rate of "
        f"{metrics['cosmic_rate_per_s_per_mev']:.4g} per s per MeV/c."
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
        "npot", "cosmic_rate_per_s_per_mev",
    ),
    units={
        "signal_box_low_mev": "MeV",
        "signal_box_high_mev": "MeV",
        "signal_mpv_mev": "MeV",
        "signal_fwhm_mev": "MeV",
        "npot": "POT",
        "cosmic_rate_per_s_per_mev": "per second per MeV/c",
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
        ParamSpec(
            name="mean_pot_per_event",
            description="Mean number of protons on target per event.",
            default=MEAN_POT_PER_EVENT, minimum=1.0,
        ),
        ParamSpec(
            name="cosmic_rate_per_s_per_mev",
            description="Cosmic-ray background rate, flat in momentum, per "
                        "second per MeV/c. The default is the rough Run-1A "
                        "mu- -> e- number; scaled by the on-spill live time "
                        "that npot and mean npot per event implies.",
            default=COSMIC_RATE_PER_SECOND_PER_MEV, minimum=0.0,
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
