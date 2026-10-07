"""Tests for the science tools as plain Python — no MCP layer, no mu2e run.

The `pyenv ana` environment has no pytest, so these are bare asserts:

    python3 tests/test_tools.py

The registry tests loop over every registered analysis, so a new analysis is
covered by them as soon as it is added to registry.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools import ANALYSES, list_analyses, run_analysis
from tools.analyses import approx_ce_sensitivity as sens
from tools.analyses import edep as edep_mod
from tools.analyses.edep import parse_edep_summary
from tools.analyses.fullsim import cuts as fs_cuts
from tools.analyses.fullsim import sensitivity as fs_sens
from tools.analyses.fullsim.eventntuple import (DEFAULT_TRIGGERS, TRIGGER_PREFIX,
                                                origin_codes)
from tools.analyses.count import (CountsError, dataset_description,
                                  dataset_hint, parse_counts,
                                  parse_prescale_filters, saved_rates,
                                  wrong_dataset)
from tools.analyses.muon_stop_rate import PRESCALE_FILTER, stop_rates
from tools.analyses.stop_materials import combine_tables, material_rates
from tools.analyses import trigger as trig
from tools.analyses.trigger_efficiency import efficiency_metrics
from tools.analyses import trigger_efficiency_ntuple as ntrig
from tools.analyses.trigger_rate import duty_factor, rate_metrics
from tools.analyses.trigger_timing import timing_metrics
from tools.mu2e_env import EnvError, Mu2eEnv, configure, current
from tools.mu2e_job import (build_input_args, root_snapshot,
                            validate_input_paths, written_root_files)
from tools.selection import SelectionError, apply_selection
from tools.spectrum import Kernel, Spectrum
from tools.spec import FCL_DIR, ParamSpec

# Verbatim shape of the block EdepAna_module.cc prints, with art's usual
# surrounding noise.
SAMPLE_EDEP_STDOUT = """\
%MSG-i ArtReport:  PostBeginJob 09-Sep-2026 11:22:33 CDT
Begin processing the 1st record. run: 1200 subRun: 0 event: 1
EdepAna summary:
  Saw 998.5 events (1000 gen events) --> output rate = 0.9985 events / gen event
  Average calo energy deposition per event: 12.3456 MeV
  Average calo energy deposition per gen event: 12.3271 MeV
  Events with calo Edep > 50 MeV: 42
  Average tracker energy deposition per event: 3.21 MeV
  Average tracker energy deposition per gen event: 3.2052 MeV
Art has completed and will exit with status 0.
"""

# Verbatim shape of what print_counts.fcl prints, with art's usual noise. One
# prescale block per filter the production job ran; only the target-stop one
# applies to a TargetStops file.
SAMPLE_COUNTS_STDOUT = """\
18-Sep-2026 12:04:13 CDT  Opened input file "sim.mmackenz.TargetStops.Run1Bak_local0813111400.001800_00000000.art"

ProductPrint mu2e::PrescaleFilterFraction_PolyStopPrescaleFilter__MuBeamResampler
 Fraction passing filter 0.000800 N Seen 12500 with prescale fraction 0.001000

ProductPrint mu2e::PrescaleFilterFraction_TargetStopPrescaleFilter__MuBeamResampler
 Fraction passing filter 1.000000 N Seen 12500 with prescale fraction 1.000000

ProductPrint mu2e::PrescaleFilterFraction_EarlyPrescaleFilter__MuBeamResampler
 Fraction passing filter 0.034800 N Seen 12500 with prescale fraction 0.033333

Begin processing the 1st record. run: 1800 subRun: 0 event: 3 at 18-Sep-2026 12:04:13 CDT
GenEventCount: 12500 events in run: 1800 subRun: 0
     1 BeginRun records found
     1 Subrun records found
   735 Event records found
GenEventCount total: 12500 events in 1 SubRuns

Art has completed and will exit with status 0.
"""

# The end of a trigger job's stdout: art's TrigReport, in the format of a run
# of three menu paths (cpr_ at prescale 3) over 200 CE digi events.
SAMPLE_TRIGGER_STDOUT = """\
Full event                                                           0.00106055     0.0163061      1.10655      0.0102812     0.077483        200
TrigReport ---------- Event summary -------------
TrigReport Events total = 200 passed = 182 failed = 18

TrigReport ---------- Trigger-path summary ------------
TrigReport    Path ID        Run     Passed     Failed      Error Name
TrigReport        400        200        181         19          0 calo_photon
TrigReport        160        200         40        160          0 cpr_TrkDe_80m70p
TrigReport        210        200        171         29          0 apr_TrkDe_80m70p

TrigReport ---------- End-path summary ---------
TrigReport        Run    Success      Error

TrigReport ---------- Modules in path: apr_TrkDe_80m70p ------------
TrigReport    Path ID    Visited     Passed     Failed      Error Name
TrigReport        210        200        200          0          0 Prefetch

Art has completed and will exit with status 0.
"""

# One stdout sample per art_files analysis, so the registry test can exercise
# each parser. Add an entry when adding such an analysis.
SAMPLE_STDOUT = {"edep": SAMPLE_EDEP_STDOUT,
                 "count": SAMPLE_COUNTS_STDOUT,
                 "muon_stop_rate": SAMPLE_COUNTS_STDOUT,
                 "trigger_efficiency": SAMPLE_TRIGGER_STDOUT,
                 "trigger_rate": SAMPLE_TRIGGER_STDOUT,
                 "trigger_timing": SAMPLE_TRIGGER_STDOUT}


# --- the edep parser ---------------------------------------------------------

def test_edep_parses_all_eight_fields():
    metrics = parse_edep_summary(SAMPLE_EDEP_STDOUT)
    assert metrics == {
        "n_events": 998.5,
        "n_gen_events": 1000.0,
        "event_rate": 0.9985,
        "avg_calo_edep_per_event_mev": 12.3456,
        "avg_calo_edep_per_gen_event_mev": 12.3271,
        "n_events_calo_edep_above_50mev": 42.0,
        "avg_trk_edep_per_event_mev": 3.21,
        "avg_trk_edep_per_gen_event_mev": 3.2052,
    }, metrics


def test_edep_parses_scientific_notation():
    stdout = SAMPLE_EDEP_STDOUT.replace("12.3456 MeV", "1.23456e-02 MeV")
    assert parse_edep_summary(stdout)["avg_calo_edep_per_event_mev"] == 0.0123456


def test_edep_parse_returns_none_without_summary():
    assert parse_edep_summary("Art has completed and will exit with status 1.") is None


def test_edep_parse_returns_none_on_truncated_summary():
    truncated = SAMPLE_EDEP_STDOUT.split("Events with calo Edep")[0]
    assert parse_edep_summary(truncated) is None


# --- the shared count parser (count.py) --------------------------------------

def test_counts_parses_events_gen_events_and_the_named_prescale():
    assert parse_counts(SAMPLE_COUNTS_STDOUT, PRESCALE_FILTER) == {
        "n_events": 735.0,
        "n_gen_events": 12500.0,
        "prescale": 1.0,        # the TargetStop filter's, not PolyStop's 0.001
    }


def test_counts_takes_the_prescale_of_whichever_filter_is_named():
    counts = parse_counts(SAMPLE_COUNTS_STDOUT, "PolyStopPrescaleFilter")
    assert counts["prescale"] == 0.001
    assert counts["n_events"] == 735.0      # the file's events, either way


def test_counts_needs_no_prescale_filter_at_all():
    """No filter named: prescale 1, and the blocks that ARE there are ignored.

    The point of `count`: nothing assumes a prescale module exists. The sample
    holds three prescale blocks and none of them may be picked up by accident.
    """
    for unset in (None, ""):
        assert parse_counts(SAMPLE_COUNTS_STDOUT, unset) == {
            "n_events": 735.0,
            "n_gen_events": 12500.0,
            "prescale": 1.0,
        }
    # ... and the counts still parse out of a job that printed no block at all
    bare = SAMPLE_COUNTS_STDOUT[SAMPLE_COUNTS_STDOUT.index("Begin processing"):]
    assert parse_prescale_filters(bare) == {}
    assert parse_counts(bare)["n_events"] == 735.0


def test_counts_never_falls_back_to_no_prescale_for_a_filter_that_is_missing():
    """A named-but-absent filter is an error, not a silent prescale of 1.

    Falling back would report a prescaled file's rate short by exactly the
    prescale, with nothing in the output to show for it.
    """
    without = SAMPLE_COUNTS_STDOUT.replace("PolyStopPrescaleFilter", "OtherFilter")
    try:
        parse_counts(without, "PolyStopPrescaleFilter")
    except CountsError as exc:
        assert "PolyStopPrescaleFilter" in str(exc)
    else:
        raise AssertionError("a named filter that is absent must be reported")


def test_counts_reads_every_prescale_block():
    filters = parse_prescale_filters(SAMPLE_COUNTS_STDOUT)
    assert set(filters) == {"PolyStopPrescaleFilter", "TargetStopPrescaleFilter",
                            "EarlyPrescaleFilter"}
    poly = filters["PolyStopPrescaleFilter"]
    assert poly["prescale"] == 0.001 and poly["fraction_passing"] == 0.0008
    assert poly["n_seen"] == 12500.0 and poly["process"] == "MuBeamResampler"


def test_counts_rejects_output_without_the_target_stop_filter():
    without = SAMPLE_COUNTS_STDOUT.replace("TargetStopPrescaleFilter",
                                           "IPAStopPrescaleFilter")
    try:
        parse_counts(without, PRESCALE_FILTER)
    except CountsError as exc:
        assert "TargetStopPrescaleFilter" in str(exc)
        assert "IPAStopPrescaleFilter" in str(exc)      # names what it did find
    else:
        raise AssertionError("a file without the filter must be reported")


def test_counts_rejects_output_without_the_generated_event_count():
    without = SAMPLE_COUNTS_STDOUT.replace("GenEventCount total: 12500 events", "")
    try:
        parse_counts(without, PRESCALE_FILTER)
    except CountsError as exc:
        assert "generated" in str(exc)
    else:
        raise AssertionError("missing gen-event bookkeeping must be reported")


def test_dataset_description_reads_mu2e_names_and_passes_on_others():
    name = Path("sim.mmackenz.TargetStops.Run1Bak_local0813111400.001800_00000000.art")
    assert dataset_description(name) == "TargetStops"
    assert dataset_description(Path("my_stops.art")) is None   # not a Mu2e name


def test_dataset_hint_comes_from_the_filter_label():
    assert dataset_hint("TargetStopPrescaleFilter") == "targetstop"
    assert dataset_hint("PolyStopPrescaleFilter") == "polystop"
    assert dataset_hint("SomethingElse") == ""      # nothing to check against
    assert dataset_hint(None) == "" and dataset_hint("") == ""   # no filter at all


def test_wrong_dataset_follows_the_filter_it_is_given():
    stem = "Run1Bak_local0813111400.001800_00000000.art"
    target = Path(f"sim.mmackenz.TargetStops.{stem}")
    poly = Path(f"sim.mmackenz.PolyStops.{stem}")
    unnamed = Path("stops.art")
    # A poly-stop file parses fine and carries the target filter's product, so
    # only the name says it is the wrong input.
    assert wrong_dataset([target, unnamed], PRESCALE_FILTER) == []
    assert wrong_dataset([target, poly], PRESCALE_FILTER) == [
        f"sim.mmackenz.PolyStops.{stem} (PolyStops)"
    ]
    # With no filter named nothing is divided out, so no file is the wrong one.
    assert wrong_dataset([target, poly]) == []
    assert wrong_dataset([target, poly], "") == []
    # ... and the check follows the filter: with the poly filter it is the
    # target-stop file that is out of place.
    assert wrong_dataset([target, poly], "PolyStopPrescaleFilter") == [
        f"sim.mmackenz.TargetStops.{stem} (TargetStops)"
    ]
    assert wrong_dataset([target, poly], "OddlyNamedFilter") == []


def test_saved_rates_divide_out_the_prescale():
    counts = {"n_events": 735.0, "n_gen_events": 12500.0, "prescale": 0.5}
    assert abs(saved_rates(counts)["saved_per_gen_event"] - 0.1176) < 1e-6
    # an unprescaled file (prescale 1) is just the ratio
    plain = {"n_events": 735.0, "n_gen_events": 12500.0, "prescale": 1.0}
    assert abs(saved_rates(plain)["saved_per_gen_event"] - 0.0588) < 1e-6


def test_stop_rates_divide_out_the_prescale_and_scale_by_the_upstream_efficiency():
    counts = {"n_events": 735.0, "n_gen_events": 12500.0, "prescale": 0.5}
    rates = stop_rates(counts, upstream_eff=2.0e-3)
    # the prescale kept half the events, so the true rate is twice 735/12500
    assert abs(rates["stops_per_gen_event"] - 0.1176) < 1e-6
    assert abs(rates["stops_per_pot"] - 0.1176 * 2.0e-3) < 1e-9


# --- the spectrum helper -----------------------------------------------------

def test_spectrum_axis_and_total():
    s = Spectrum(np.full(10, 2.0), 0.0, 0.1)
    assert s.nbins == 10 and abs(s.xmax - 1.0) < 1e-12
    assert abs(s.centers()[0] - 0.05) < 1e-12
    assert s.values.sum() == 20.0


def test_spectrum_rebin_merges_groups_and_drops_leftovers():
    s = Spectrum(np.arange(1.0, 11.0), 0.0, 1.0).rebin(3)
    assert s.nbins == 3 and s.width == 3.0 and abs(s.xmax - 9.0) < 1e-12
    assert list(s.values) == [6.0, 15.0, 24.0]   # the 10th bin is dropped


def test_spectrum_regrid_sums_onto_a_coarser_axis():
    fine = Spectrum(np.ones(100), 0.0, 0.1)          # 10 units over [0, 10)
    coarse = fine.regrid(Spectrum(np.zeros(5), 0.0, 1.0))
    assert coarse.nbins == 5 and list(coarse.values) == [10.0] * 5
    assert coarse.values.sum() == 50.0               # half of it fell off the axis


def test_gaussian_kernel_is_normalized_and_centered():
    kernel = Kernel.gaussian(width=0.05, sigma=0.2)
    assert abs(kernel.mass.sum() - 1.0) < 1e-12
    spectrum = kernel.as_spectrum(0.05)
    mpv, fwhm = sens.mpv_fwhm(spectrum)
    assert abs(mpv) < 0.03                                  # centered on zero
    assert abs(fwhm - 2.355 * 0.2) < 0.06                   # FWHM = 2.355 sigma


def test_kernel_from_density_keeps_the_response_total():
    # A response carrying only half the probability (an efficiency folded in),
    # on a coarser grid than the target.
    response = Spectrum(np.full(10, 0.5 / (10 * 0.2)), -1.0, 0.2)
    kernel = Kernel.from_density(response, width=0.05)
    assert abs(kernel.mass.sum() - 0.5) < 1e-9


def test_dio_spectrum_is_a_normalized_density():
    dio = sens.load_dio_spectrum()
    # Density in energy: sum(contents) * bin width == 1
    assert abs(dio.values.sum() * dio.width - 1.0) < 1e-9
    # Falls steeply toward the CE endpoint
    at = lambda e: dio.values[int((e - dio.xmin) / dio.width)]
    assert at(60.0) > at(100.0) > at(104.5)


def test_smear_conserves_total_and_shifts_by_the_response_mean():
    true = Spectrum(np.zeros(200), 0.0, 0.1)
    true.values[100] = 1.0                                  # delta at 10 MeV
    response = Spectrum(np.zeros(200), -5.0, 0.05)          # delta at -2 MeV
    response.values[59] = 1.0 / response.width

    reco = true.smear(Kernel.from_density(response, true.width))
    assert abs(reco.values.sum() - 1.0) < 1e-9              # probability kept
    assert abs(reco.centers()[int(np.argmax(reco.values))] - 8.0) < 0.2


def test_smear_drops_content_pushed_off_the_axis():
    true = Spectrum(np.zeros(100), 0.0, 0.1)
    true.values[10] = 1.0                                   # 1 MeV
    response = Spectrum(np.zeros(100), -5.0, 0.1)
    response.values[9] = 1.0 / response.width               # shift by -4 MeV
    reco = true.smear(Kernel.from_density(response, true.width))
    assert reco.values.sum() < 1e-12                        # left the axis


def test_scan_finds_the_best_window():
    signal, dio, cosmic = (Spectrum(np.zeros(100), 50.0, 1.0, name=n)
                           for n in ("signal", "dio", "cosmic"))
    signal.values[45:55] = 10.0          # a signal bump at ~95-105 MeV
    dio.values[:] = 1.0
    cosmic.values[:] = 1.0
    best, top = sens.scan_signal_box(signal, dio, cosmic)
    assert best["low_mev"] >= 50.0
    # The bump spans bins 46..55 -> centers 95.5..104.5
    assert 94.0 <= best["low_mev"] <= 96.0, best
    assert 104.0 <= best["high_mev"] <= 106.0, best
    assert top[0]["sensitivity"] == best["sensitivity"]


def test_scan_keeps_a_tiny_window_count_against_a_huge_spectrum_total():
    """Regression: window sums must not be prefix-sum differences.

    The DIO spectrum spans ~18 orders of magnitude. Differencing whole-spectrum
    prefix sums to get a count of order 1 loses it completely to float
    cancellation, which silently reported dio_background = 0.
    """
    signal, dio, cosmic = (Spectrum(np.zeros(100), 50.0, 1.0, name=n)
                           for n in ("signal", "dio", "cosmic"))
    signal.values[50:60] = 10.0
    cosmic.values[:] = 1.0
    dio.values[0] = 4.0e17            # enormous, far below the signal window
    dio.values[50:60] = 2.0e-1        # what we must still be able to see

    best, _ = sens.scan_signal_box(signal, dio, cosmic)
    assert best["dio"] > 0.0, "tiny DIO count was lost to cancellation"
    assert abs(best["dio"] - 0.2 * 10) < 1e-6 or best["dio"] > 0.1, best


def test_sensitivity_rejects_a_file_without_the_tree(tmp_dir):
    """A non-EdepAna ROOT file (or a pre-tree one) is reported, not a crash."""
    import uproot
    path = Path(tmp_dir) / "empty.root"
    with uproot.recreate(path) as f:
        f["something_else"] = np.histogram(np.zeros(1), bins=2)
    result = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=tmp_dir, parameters={"sig_eff": 0.1})
    assert result.status == "error"
    assert "EDepAna/tree" in result.message


# --- selections and the EdepAna tree ------------------------------------------

def test_selection_takes_python_and_root_spellings_alike():
    x = {"a": np.array([1.0, 5.0, 20.0]), "b": np.array([0.0, 1.0, 2.0])}
    expect = [False, True, False]
    assert apply_selection("a > 2 and a < 10", x, 3).tolist() == expect
    assert apply_selection("a > 2 && a < 10", x, 3).tolist() == expect
    assert apply_selection("2 < a < 10", x, 3).tolist() == expect
    assert apply_selection("!(a <= 2 || a >= 10)", x, 3).tolist() == expect
    assert apply_selection("abs(a - 2*b) > 10", x, 3).tolist() == [False, False, True]
    assert apply_selection("b != 1", x, 3).tolist() == [True, False, True]
    assert apply_selection("", x, 3).all()                  # no cut at all


def test_selection_fails_a_cut_on_a_missing_value():
    x = {"a": np.array([np.nan, 5.0])}
    assert apply_selection("a > 0", x, 2).tolist() == [False, True]
    assert apply_selection("a <= 0", x, 2).tolist() == [False, False]


def test_selection_refuses_what_it_should_not_run():
    x = {"a": np.array([1.0, 2.0])}
    for bad, words in (("a > ", "does not parse"),
                       ("c > 1", "unknown variable"),
                       ("a.__class__ > 1", "not allowed"),
                       ("__import__('os') > 1", "unknown variable"),
                       ("a + 1", "true/false")):
        try:
            apply_selection(bad, x, 2)
        except SelectionError as exc:
            assert words in str(exc), (bad, str(exc))
        else:
            raise AssertionError(f"{bad!r} should have been refused")


def _edep_branches():
    """Three events: one with no primary, one that never reached the tracker
    front, one that did — raw branches as uproot hands them over."""
    def per_event(*values):
        out = np.empty(len(values), dtype=object)
        out[:] = [np.asarray(v, dtype=np.float32) for v in values]
        return out
    branches = {
        "nprimaries": np.array([0, 1, 2]),
        "event_calo_edep": np.array([0.0, 30.0, 90.0]),
        "event_calo_edep_vis": np.array([0.0, 25.0, 80.0]),
        "event_trk_edep": np.array([0.0, 0.1, 0.2]),
        "weight": np.ones(3), "run": np.ones(3), "subrun": np.ones(3),
        "event": np.arange(3), "ngen": np.array([5, 5, 5]),
    }
    first = {"primary_start_x": -3904.0 + 30.0, "primary_start_y": 40.0,
             "primary_start_z": 5500.0, "primary_start_px": 0.0,
             "primary_start_py": 0.0, "primary_start_pz": 104.0,
             "primary_start_e": 104.97, "primary_start_m": 0.511,
             "primary_start_pdg": 11, "primary_calo_edep": 80.0,
             "primary_calo_edep_vis": 75.0}
    for name, value in first.items():
        branches[name] = per_event([], [value], [value, 0.0])
    branches["primary_trk_front_p"] = per_event([], [0.0], [103.0, 0.0])
    branches["primary_trk_front_energy"] = per_event([], [0.0], [103.5, 0.0])
    return branches


def test_edep_variables_take_the_first_primary_and_mark_what_is_missing():
    v = edep_mod.edep_variables(_edep_branches())
    assert np.isnan(v["primary_start_e"][0])                # no primary
    assert v["primary_start_e"][2] == np.float32(104.97)    # first primary's
    assert v["has_trk_front"].tolist() == [False, False, True]
    assert np.isnan(v["primary_trk_front_energy"][1])       # never got there
    assert abs(v["primary_trk_front_energy_diff"][2] - (103.5 - 104.97)) < 1e-4
    assert abs(v["primary_start_r"][1] - 50.0) < 1e-3       # detector frame
    assert set(v) == set(edep_mod.EDEP_VARIABLES)
    # every advertised variable is usable in a selection
    for name in edep_mod.EDEP_VARIABLES:
        edep_mod.select_events(v, f"{name} == {name}")


def test_edep_selection_counts_weighted_events_per_gen_event():
    v = edep_mod.edep_variables(_edep_branches())
    assert edep_mod.selected_metrics(v, "event_calo_edep_vis > 50", 10.0) == {
        "n_events_selected": 1.0, "selected_per_gen_event": 0.1}
    assert edep_mod.selected_metrics(v, "", 10.0)["n_events_selected"] == 3.0


def _write_edep_tree(path: Path, n: int = 4000, ngen_total: int | None = None) -> None:
    """A CE-like EdepAna tree: ~105 MeV electrons losing a little on the way.

    `ngen` is EdepAna's running generated-event count, raised at each subrun;
    with `ngen_total` it climbs to that over the file, else it is n throughout.
    """
    import awkward as ak
    import uproot
    rng = np.random.default_rng(1)
    e0 = np.full(n, 104.97, dtype=np.float32)
    front = (e0 - rng.exponential(0.6, n)).astype(np.float32)
    calo = rng.uniform(0.0, 100.0, n).astype(np.float32)
    def one(values):
        return ak.unflatten(values, np.ones(n, dtype=np.int64))
    tree = {
        "nprimaries": np.ones(n, dtype=np.int32),
        "event_calo_edep": calo, "event_calo_edep_vis": calo,
        "event_trk_edep": np.zeros(n, dtype=np.float32),
        "weight": np.ones(n, dtype=np.float32),
        "run": np.ones(n, dtype=np.int32), "subrun": np.ones(n, dtype=np.int32),
        "event": np.arange(n, dtype=np.int32),
        "ngen": (np.full(n, n, dtype=np.int64) if ngen_total is None else
                 np.ceil(np.arange(1, n + 1) * ngen_total / n).astype(np.int64)),
        "primary_trk_front_p": one(front), "primary_trk_front_energy": one(front),
        "primary_start_e": one(e0), "primary_start_pdg": one(np.full(n, 11, np.int32)),
    }
    zeros = np.zeros(n, dtype=np.float32)
    for name in ("x", "y", "z", "px", "py", "pz", "m"):
        tree[f"primary_start_{name}"] = one(zeros)
    for name in ("primary_calo_edep", "primary_calo_edep_vis"):
        tree[name] = one(calo)
    # mktree, not assignment: this uproot writes a dict as an RNTuple
    types = {name: ("var * " + str(values.layout.content.dtype)
                    if isinstance(values, ak.Array) else values.dtype)
             for name, values in tree.items()}
    with uproot.recreate(path) as f:
        f.mktree("EDepAna/tree", types).extend(tree)


def test_signal_is_r_mue_times_the_captures_per_stop(tmp_dir):
    """R_mue is per muon capture, so the CEs per stopped muon are R_mue times
    the fraction of stops that capture, not R_mue divided by it."""
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path)
    result = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=tmp_dir, parameters={"sig_eff": 1e-3})
    assert result.status == "success", result.message
    assert result.metadata["signal_br"] == 1.0e-13 * 0.609, result.metadata["signal_br"]
    log = (Path(tmp_dir) / "approx_ce_sensitivity.log").read_text()
    assert "R_mue = 1e-13" in log and "1e-9" not in log, log


def test_sensitivity_runs_on_the_tree_with_the_selection_it_is_given(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path)
    results = {}
    for cut in ("event_calo_edep_vis > 10", "event_calo_edep_vis > 50"):
        result = run_analysis(analysis="approx_ce_sensitivity",
                              data_file=str(path), output_dir=tmp_dir,
                              parameters={"sig_eff": 0.5, "selection": cut})
        assert result.status == "success", result.message
        results[cut] = result.metadata
    loose, tight = results.values()
    # the shape normalization is per selected event, so the selection matters
    assert loose["n_events_selected"] > tight["n_events_selected"] > 0
    assert loose["selection"] == sens.DEFAULT_SELECTION

    nothing = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                           output_dir=tmp_dir,
                           parameters={"sig_eff": 0.5,
                                       "selection": "event_calo_edep_vis > 1000"})
    assert nothing.status == "error" and "0 of 4000" in nothing.message
    typo = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                        output_dir=tmp_dir,
                        parameters={"sig_eff": 0.5, "selection": "calo > 10"})
    assert typo.status == "error" and "unknown variable" in typo.message


# --- approx_ce_sensitivity: the efficiency from the stopping rate --------------

def test_generated_events_is_each_files_last_running_count(tmp_dir):
    a, b = Path(tmp_dir) / "a.root", Path(tmp_dir) / "b.root"
    _write_edep_tree(a, n=400, ngen_total=1000)
    _write_edep_tree(b, n=300, ngen_total=900)
    assert edep_mod.generated_events(a) == 1000
    assert edep_mod.generated_events([a, b]) == 1900


def test_sensitivity_takes_exactly_one_of_sig_eff_and_stops_per_pot(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path)
    for params in ({}, {"sig_eff": 0.1, "stops_per_pot": 1e-3}):
        result = run_analysis(analysis="approx_ce_sensitivity",
                              data_file=str(path), output_dir=tmp_dir,
                              parameters=params)
        assert result.status == "error", params
        assert "one of sig_eff or stops_per_pot" in result.message, result.message
    declared = list_analyses().metadata["analyses"]["approx_ce_sensitivity"]["parameters"]
    assert not declared["sig_eff"]["required"]
    assert not declared["stops_per_pot"]["required"]


def test_stops_per_pot_is_sig_eff_times_the_files_ce_acceptance(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=4000, ngen_total=8000)     # acceptance 0.5
    by_stops = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                            output_dir=str(Path(tmp_dir) / "stops"),
                            parameters={"stops_per_pot": 2e-3, "selection": ""})
    by_eff = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=str(Path(tmp_dir) / "eff"),
                          parameters={"sig_eff": 1e-3, "selection": ""})
    assert by_stops.status == "success", by_stops.message
    assert by_eff.status == "success", by_eff.message
    meta = by_stops.metadata
    assert meta["sensitivity"] == by_eff.metadata["sensitivity"]
    assert (meta["sig_eff"], meta["ce_acceptance"], meta["n_gen_events"],
            meta["stops_per_pot"]) == (1e-3, 0.5, 8000.0, 2e-3), meta


def test_signal_efficiency_is_stops_per_pot_times_selected_per_generated():
    variables = {"weight": np.ones(400)}
    mask = np.arange(400) < 300
    assert sens.signal_efficiency(variables, mask, 1000.0, 2e-3) == (2e-3 * 0.3, 0.3)


def test_signal_efficiency_refuses_a_selection_no_event_passes():
    try:
        sens.signal_efficiency({"weight": np.ones(10)}, np.zeros(10, bool),
                               100.0, 2e-3)
    except sens.SensitivityError as exc:
        assert "no event" in str(exc) and "selection" in str(exc), exc
    else:
        raise AssertionError("an acceptance of 0 was accepted")


def test_signal_efficiency_refuses_an_acceptance_above_one():
    try:
        sens.signal_efficiency({"weight": np.ones(400)}, np.ones(400, bool),
                               100.0, 2e-3)
    except sens.SensitivityError as exc:
        assert "acceptance" in str(exc) and "4" in str(exc), exc
    else:
        raise AssertionError("an acceptance of 4 was accepted")


def test_signal_efficiency_refuses_an_empty_file_for_what_it_is():
    try:
        sens.signal_efficiency({"weight": np.ones(0)}, np.ones(0, bool),
                               0.0, 2e-3)
    except sens.SensitivityError as exc:
        assert "no events" in str(exc), exc
    else:
        raise AssertionError("an empty file was accepted")


def test_generated_events_refuses_a_merged_file(tmp_dir):
    """ngen restarts with each job, so a falling count means several edep
    outputs were merged into one file: their totals cannot be told apart."""
    import uproot
    path = Path(tmp_dir) / "merged.root"
    with uproot.recreate(path) as f:
        f.mktree("EDepAna/tree", {"ngen": np.int64}).extend(
            {"ngen": np.array([1000, 2000, 2000, 500, 1500], dtype=np.int64)})
    try:
        edep_mod.generated_events(path)
    except edep_mod.EdepTreeError as exc:
        assert "merged" in str(exc), exc
    else:
        raise AssertionError("a merged file was accepted")


def test_sensitivity_refuses_a_nan_efficiency_input(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=100, ngen_total=200)
    for params in ({"sig_eff": 0.1, "stops_per_pot": "nan"},
                   {"sig_eff": "nan", "stops_per_pot": 2e-3}):
        result = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                              output_dir=tmp_dir, parameters=params)
        assert result.status == "error", params
        assert "nan" in result.message.lower(), result.message


def test_stops_per_pot_needs_a_generated_count(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=100, ngen_total=0)
    result = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=tmp_dir, parameters={"stops_per_pot": 2e-3})
    assert result.status == "error"
    assert "generated" in result.message, result.message


def test_ce_acceptance_counts_only_events_passing_the_selection(tmp_dir):
    """The selection sets the signal's size, not only its shape: a tighter
    cut must lower the efficiency stops_per_pot works out."""
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=4000, ngen_total=8000)
    cut = "event_calo_edep_vis > 50"
    passing = float((edep_mod.read_edep_tree(path)["event_calo_edep_vis"] > 50).sum())
    acceptance = passing / 8000.0
    assert 0.0 < acceptance < 0.45, acceptance       # the cut removes events
    by_stops = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                            output_dir=str(Path(tmp_dir) / "stops"),
                            parameters={"stops_per_pot": 2e-3, "selection": cut})
    by_eff = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=str(Path(tmp_dir) / "eff"),
                          parameters={"sig_eff": 2e-3 * acceptance, "selection": cut})
    assert by_stops.status == "success", by_stops.message
    assert by_eff.status == "success", by_eff.message
    meta = by_stops.metadata
    assert (meta["ce_acceptance"], meta["sig_eff"]) == (acceptance, 2e-3 * acceptance), meta
    assert meta["sensitivity"] == by_eff.metadata["sensitivity"]


def _fake_edep_job(tmp_dir):
    """A stand-in for run_mu2e_job: writes an EdepAna tree where the job
    would and reports the summary block, recording the max_events it got."""
    from tools.mu2e_job import JobOutcome
    calls = []

    def fake(*, fcl, input_paths, outdir, single, timeout_s, max_events=None, **_):
        calls.append(max_events)
        outdir.mkdir(parents=True, exist_ok=True)
        root = outdir / "nts.owner.edep.test.root"
        _write_edep_tree(root, n=100)
        return JobOutcome(command="mu2e", returncode=0, timed_out=False,
                          stdout=SAMPLE_EDEP_STDOUT, stderr="",
                          log_path=outdir / "mu2e.log", input_paths=input_paths,
                          input_flag="-s", environment="test", file_list_path=None,
                          written_root_files=[str(root)])
    return fake, calls


def test_edep_records_max_events_next_to_its_file(tmp_dir):
    from tools.spec import RunContext
    fake, calls = _fake_edep_job(tmp_dir)
    real = edep_mod.run_mu2e_job
    edep_mod.run_mu2e_job = fake
    try:
        for max_events in (100, None):
            outdir = Path(tmp_dir) / f"run{max_events}"
            outcome = edep_mod.run(RunContext(
                input_paths=[Path(tmp_dir) / "in.art"], outdir=outdir,
                params={"selection": ""}, timeout_s=60, max_events=max_events))
            assert outcome.error is None, outcome.error
            root = Path(outcome.files[0])
            assert edep_mod.read_run_record(root) == {"max_events": max_events}
            # the record rides next to the file; it is not an output to chain
            assert outcome.files == [str(root)], outcome.files
    finally:
        edep_mod.run_mu2e_job = real
    assert calls == [100, None], calls


def test_stops_per_pot_refuses_a_file_from_a_max_events_run(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=4000, ngen_total=8000)
    edep_mod.write_run_record(path, max_events=100)
    result = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=tmp_dir, parameters={"stops_per_pot": 2e-3})
    assert result.status == "error"
    assert "max_events" in result.message, result.message


def test_stops_per_pot_says_when_it_cannot_confirm_a_full_run(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=4000, ngen_total=8000)
    unrecorded = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                              output_dir=str(Path(tmp_dir) / "a"),
                              parameters={"stops_per_pot": 2e-3})
    assert unrecorded.status == "success", unrecorded.message
    assert unrecorded.metadata["edep_run_recorded"] is False
    log = (Path(tmp_dir) / "a" / "approx_ce_sensitivity.log").read_text()
    assert "cannot confirm" in log, log
    edep_mod.write_run_record(path, max_events=None)
    recorded = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                            output_dir=str(Path(tmp_dir) / "b"),
                            parameters={"stops_per_pot": 2e-3})
    assert recorded.status == "success", recorded.message
    assert recorded.metadata["edep_run_recorded"] is True
    log = (Path(tmp_dir) / "b" / "approx_ce_sensitivity.log").read_text()
    assert "cannot confirm" not in log, log


def test_stops_per_pot_is_at_most_one(tmp_dir):
    declared = list_analyses().metadata["analyses"]["approx_ce_sensitivity"]["parameters"]
    assert declared["stops_per_pot"]["maximum"] == 1.0
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=100, ngen_total=200)
    result = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=tmp_dir, parameters={"stops_per_pot": 1.5})
    assert result.status == "error"
    assert "above the maximum" in result.message, result.message


# --- the trigger ---------------------------------------------------------------

def test_trigger_paths_take_prescales_and_default_to_one():
    assert trig.parse_trigger_paths(
        "calo_photon:1, cpr_TrkDe_80m70p:10 apr_TrkDe_80m70p") == [
        ("calo_photon", 1), ("cpr_TrkDe_80m70p", 10), ("apr_TrkDe_80m70p", 1)]


def test_trigger_paths_refuse_what_cannot_be_run():
    for bad, words in (("", "no trigger paths"),
                       ("apr_TrkDe:0", "at least 1"),
                       ("apr_TrkDe:x", "whole number"),
                       ("apr_TrkDe:2.5", "whole number"),
                       ("apr_TrkDe, apr_TrkDe:2", "listed twice"),
                       ('bad"name', "does not name"),
                       ("1abc", "does not name")):
        try:
            trig.parse_trigger_paths(bad)
        except trig.TriggerError as exc:
            assert words in str(exc), (bad, str(exc))
        else:
            raise AssertionError(f"{bad!r} should have been refused")


def test_prescale_module_follows_the_menu_generator():
    assert trig.prescale_module("cpr_TrkDe_80m70p") == "CprTrkDe80m70pPS"
    assert trig.prescale_module("apr_TrkDe_80m70p_D0200") == "AprTrkDe80m70pD0200PS"
    assert trig.prescale_module("calo_photon") == "CaloPhotonPS"


def test_prescale_module_names_every_path_in_the_published_menu():
    """Against the generated menu itself, where it is on disk."""
    gen = Path("/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ay/build/"
               "al9-prof-e29-p107/mu2e-trig-config/gen")
    menu, ps = gen / "trig_physMenu_OnSpill.fcl", gen / "trig_physMenuPSConfig_OnSpill.fcl"
    if not (menu.exists() and ps.exists()):
        return
    import re
    # tpr_/mpr_ paths are in the menu but not meant for the real trigger
    paths = [p for p in re.findall(r'"\d+:(\w+)"', menu.read_text())
             if not p.startswith(("tpr_", "mpr_"))]
    labels = set(re.findall(r"^\s*(\w+PS):", ps.read_text(), re.M))
    assert paths and {trig.prescale_module(p) for p in paths} <= labels


def test_job_fcl_sets_the_paths_and_their_prescales(tmp_dir):
    base = Path(tmp_dir) / "base.fcl"
    base.write_text('#include "mu2e-trig-config/test/timingTest.fcl"\n')
    text = trig.job_fcl_text([("apr_TrkDe_80m70p", 1), ("cpr_TrkDe_80m70p", 10)], base)
    assert text.startswith('#include "mu2e-trig-config/test/timingTest.fcl"')
    assert 'physics.trigger_paths : [ "apr_TrkDe_80m70p", "cpr_TrkDe_80m70p" ]' in text
    assert ("physics.filters.CprTrkDe80m70pPS.eventModeConfig : [ "
            "{ eventMode: OnSpill prescale: 10 }, "
            "{ eventMode: OffSpill prescale: 10 } ]") in text
    # the shipped base is the one runs use
    assert trig.FCL.exists() and "timingTest.fcl" in trig.FCL.read_text()


def test_trig_report_parses_events_and_every_path():
    report = trig.parse_trig_report(SAMPLE_TRIGGER_STDOUT)
    assert (report.n_events, report.n_passed) == (200, 182)
    assert list(report.paths) == ["calo_photon", "cpr_TrkDe_80m70p",
                                  "apr_TrkDe_80m70p"]
    assert report.paths["cpr_TrkDe_80m70p"] == trig.PathCounts(200, 40, 160, 0)
    # the per-module blocks after it are not mistaken for paths
    assert "Prefetch" not in report.paths
    assert trig.parse_trig_report("Art has completed and will exit with status 1.") is None


def test_binomial_is_never_exactly_certain():
    p, err = trig.binomial(0, 150)
    assert p == 0.0 and 0.0 < err < 0.02
    p, err = trig.binomial(150, 150)
    assert p == 1.0 and 0.0 < err < 0.02
    p, err = trig.binomial(500, 1000)
    assert abs(err - (0.25 / 1000) ** 0.5) < 1e-3       # the usual, mid-range
    assert trig.binomial(0, 0) == (0.0, 0.0)


def test_rate_scales_the_accept_fraction_by_the_duty_cycled_event_rate():
    report = trig.TrigReport(n_events=1000, n_passed=10)
    metrics = rate_metrics(report, 5.0e5, 0.3)
    assert metrics["accept_fraction"] == 0.01
    assert abs(metrics["onspill_rate_hz"] - 5000.0) < 1e-9
    assert abs(metrics["event_rate_hz"] - 1.5e5) < 1e-6
    assert abs(metrics["rate_hz"] - 1500.0) < 1e-9          # averaged over the cycle
    assert abs(metrics["rate_hz_err"] - metrics["accept_fraction_err"] * 1.5e5) < 1e-9
    assert metrics["duty_factor"] == 0.3


def test_duty_factor_comes_from_the_batch_mode_unless_given():
    assert duty_factor("1BB", 0.0) == 0.322
    assert duty_factor("2bb", 0.0) == 0.246
    assert duty_factor("2BB", 0.5) == 0.5                   # given: overrides
    try:
        duty_factor("3BB", 0.0)
    except trig.TriggerError as exc:
        assert "1BB" in str(exc) and "duty_factor" in str(exc)
    else:
        raise AssertionError("an unknown batch mode should be refused")
    params = ANALYSES["trigger_rate"].resolve_params({"trigger_paths": "apr_TrkDe_80m70p"})
    assert params["batch_mode"] == "1BB" and params["duty_factor"] == 0.0


def _write_timing_db(path: Path) -> None:
    """Event 7 is the slow first one, then three ordinary events. Each runs a
    2 ms Prefetch, then module A (cpr) and B (apr) for its processing time;
    module C (apr, behind a filter) runs only on events 9 and 10, for 4 ms.
    TimeTracker's whole-event time adds 1 ms of framework on top."""
    import sqlite3
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE TimeEvent(Run, SubRun, Event, Time)")
        con.execute("CREATE TABLE TimeModule(Run, SubRun, Event, Path, "
                    "ModuleLabel, ModuleType, Time)")
        for event, t in ((7, 1.0), (8, 0.010), (9, 0.020), (10, 0.030)):
            c = 0.004 if event in (9, 10) else 0.0
            con.execute("INSERT INTO TimeEvent VALUES (1, 0, ?, ?)",
                        (event, t + c + 0.002 + 0.001))
            con.execute("INSERT INTO TimeModule VALUES (1, 0, ?, 'apr', "
                        "'Prefetch', 'PrefetchDAQData', 0.002)", (event,))
            con.execute("INSERT INTO TimeModule VALUES (1, 0, ?, 'cpr', 'A', "
                        "'TypeA', ?)", (event, t * 0.75))
            con.execute("INSERT INTO TimeModule VALUES (1, 0, ?, 'apr', 'B', "
                        "'TypeB', ?)", (event, t * 0.25))
            if c:
                con.execute("INSERT INTO TimeModule VALUES (1, 0, ?, 'apr', "
                            "'C', 'TypeC', ?)", (event, c))


def test_timing_leaves_out_the_data_fetch_and_the_warm_up_events(tmp_dir):
    db = Path(tmp_dir) / "triggerTiming.db"
    _write_timing_db(db)
    timed = trig.read_timing_db(db, skip_events=1)
    assert timed.n_skipped == 1
    # processing only: no Prefetch, no framework time
    assert np.allclose(timed.times, [0.010, 0.024, 0.034])
    assert np.allclose(timed.fetch_times, 0.002)
    assert np.allclose(timed.full_event_times, [0.013, 0.027, 0.037])
    assert abs(timed.per_path["cpr"] - 0.015) < 1e-12     # 0.75 * mean 0.020
    assert abs(timed.per_path["apr"] - (0.005 + 0.008 / 3)) < 1e-12  # no Prefetch
    everything = trig.read_timing_db(db, skip_events=0)
    assert everything.times.size == 4 and everything.times[0] == 1.0
    metrics = timing_metrics(timed.times)
    assert abs(metrics["mean_time_ms"] - 68.0 / 3) < 1e-9
    assert abs(metrics["median_time_ms"] - 24.0) < 1e-9
    assert abs(metrics["max_time_ms"] - 34.0) < 1e-9
    try:
        trig.read_timing_db(Path(tmp_dir) / "missing.db", 1)
    except trig.TriggerError as exc:
        assert "no timing database" in str(exc)
    else:
        raise AssertionError("a missing database should be reported")


def test_module_timing_counts_the_events_each_module_ran_on(tmp_dir):
    db = Path(tmp_dir) / "triggerTiming.db"
    _write_timing_db(db)
    table = trig.read_timing_db(db, skip_events=1).module_table()
    # every module, in the order they first ran, the fetch included but flagged
    assert [(r["path"], r["label"]) for r in table] == [
        ("apr", "Prefetch"), ("cpr", "A"), ("apr", "B"), ("apr", "C")]
    rows = {r["label"]: r for r in table}
    assert rows["Prefetch"]["counted"] is False and rows["A"]["counted"] is True
    # C sat behind a filter: it ran on 2 of the 3 timed events
    assert rows["A"]["n_seen"] == 3 and rows["C"]["n_seen"] == 2
    assert abs(rows["C"]["seen_fraction"] - 2 / 3) < 1e-12
    assert abs(rows["C"]["mean_ms"] - 4.0) < 1e-9          # per event it ran on
    assert abs(rows["C"]["ms_per_event"] - 8.0 / 3) < 1e-9  # per timed event
    assert abs(rows["A"]["max_ms"] - 22.5) < 1e-9 and rows["A"]["median_ms"] == 15.0
    # the warm-up event is not counted as a run of anything
    assert rows["Prefetch"]["n_seen"] == 3


def test_a_shared_module_is_added_up_over_the_paths_that_ran_it(tmp_dir):
    """CaloHit runs once per event, charged to whichever path got there first."""
    import sqlite3
    db = Path(tmp_dir) / "triggerTiming.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE TimeEvent(Run, SubRun, Event, Time)")
        con.execute("CREATE TABLE TimeModule(Run, SubRun, Event, Path, "
                    "ModuleLabel, ModuleType, Time)")
        for event in range(1, 6):
            con.execute("INSERT INTO TimeEvent VALUES (1, 0, ?, 0.01)", (event,))
            first = "calo_photon" if event % 2 == 0 else "apr"
            con.execute("INSERT INTO TimeModule VALUES (1, 0, ?, ?, 'CaloHit', "
                        "'CaloHitMakerFast', 0.001)", (event, first))
    rows = trig.read_timing_db(db, skip_events=1).module_table()
    split = {r["path"]: r for r in rows}
    assert split["calo_photon"]["n_seen"] == 2 and split["apr"]["n_seen"] == 2
    for row in rows:
        assert row["n_seen_all_paths"] == 4
        assert abs(row["ms_per_event_all_paths"] - 1.0) < 1e-12


def test_code_files_are_found_down_the_backing_chain(tmp_dir):
    root = Path(tmp_dir)
    area, musing, offline = root / "area", root / "musing", root / "offline"
    for d in (area, musing, offline):
        d.mkdir()
    (area / "backing").symlink_to(musing)
    (musing / "backing").symlink_to(offline)
    target = musing / "mu2e-trig-config" / "ci"
    target.mkdir(parents=True)
    (target / "data_files.txt").write_text("/data/a.art\n\n# note\n/data/b.art\n")

    env = Mu2eEnv.for_work_area(area)
    found = env.find_code_file("mu2e-trig-config/ci/data_files.txt")
    assert found == area / "backing" / "mu2e-trig-config/ci/data_files.txt"
    assert env.find_code_file("nowhere/at/all.txt") is None

    before = current()
    try:
        configure(env)
        assert trig.default_pileup_inputs() == [Path("/data/a.art"), Path("/data/b.art")]
        configure(Mu2eEnv.for_work_area(offline))
        try:
            trig.default_pileup_inputs()
        except trig.TriggerError as exc:
            assert "pass data_file" in str(exc)
        else:
            raise AssertionError("no CI file list should be reported")
    finally:
        configure(None)
        assert current().describe() == before.describe()


def test_a_musing_is_searched_from_its_published_directory():
    env = Mu2eEnv.for_musing("SimJob MDC2025ay")
    found = env.find_code_file(trig.DEFAULT_PILEUP_LIST)
    musing_dir = Path("/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ay")
    if musing_dir.is_dir():
        assert found == musing_dir / trig.DEFAULT_PILEUP_LIST


def test_only_analyses_with_default_inputs_run_without_data_files(tmp_dir):
    result = run_analysis(analysis="trigger_efficiency", output_dir=tmp_dir,
                          parameters={"trigger_paths": "apr_TrkDe_80m70p"})
    assert result.status == "error" and "exactly one of" in result.message
    listed = list_analyses().metadata["analyses"]
    assert "default_inputs" in listed["trigger_rate"]
    assert "default_inputs" in listed["trigger_timing"]
    assert "default_inputs" not in listed["trigger_efficiency"]


# --- trigger efficiency from EventNtuples -------------------------------------

# A CE EventNtuple with trig_<path> branches; tests using it are skipped where
# it is not on disk.
NTUPLE_FILE = Path("/pnfs/mu2e/tape/phy-nts/nts/mu2e/CeMLeadingLogMix1BB/"
                   "MDC2025au_best_v1_1-001/root/a8/50/nts.mu2e.CeMLeadingLogMix1BB."
                   "MDC2025au_best_v1_1-001.001430_00000000.root")


def test_ntuple_trigger_names_drop_the_prefix_and_refuse_typos():
    assert ntrig.parse_trigger_names("apr_TrkDe_80m70p, trig_cpr_TrkDe_80m70p") == [
        "apr_TrkDe_80m70p", "cpr_TrkDe_80m70p"]
    for bad in ("", "apr_TrkDe, trig_apr_TrkDe", "bad-name"):
        try:
            ntrig.parse_trigger_names(bad)
        except ntrig.NtupleError:
            pass
        else:
            raise AssertionError(f"{bad!r} should have been refused")


def _ntuple_arrays():
    """Two events in the EventNtuple's layout. Event 0: a downstream e- that
    crosses TT_Front twice (upstream leg first), at 104 MeV/c, plus a track
    with no segments at all. Event 1: an upstream-going e+."""
    import awkward as ak

    def seg(sid, px, py, pz, time, momerr=0.2):
        return {"mom": {"fCoordinates": {"fX": px, "fY": py, "fZ": pz}},
                "time": time, "momerr": momerr, "sid": sid}

    def pars(d0, t0err):
        return {"d0": d0, "maxr": 600.0, "tanDip": 0.7, "rad": 250.0,
                "t0": 900.0, "t0err": t0err}

    good = [seg(0, 0.0, 30.0, -80.0, 850.0),        # upstream-going TT_Front leg
            seg(0, 0.0, 80.0, 66.4529, 900.0),       # downstream TT_Front: p = 104
            seg(1, 0.0, 79.0, 66.0, 902.0)]          # TT_Mid, pz > 0
    upstream = [seg(1, 0.0, 70.0, -60.0, 700.0)]
    tracks = {
        "trk.status": [[1, 1], [1]], "trk.goodfit": [[1, 0], [1]],
        "trk.pdg": [[11, 11], [-11]], "trk.nactive": [[30, 5], [25]],
        "trk.chisq": [[40.0, 9.0], [30.0]], "trk.ndof": [[30, 3], [20]],
        "trkqual.result": [[0.9, 0.1], [0.8]], "trkpid.result": [[0.7, 0.0], [0.6]],
        "trkcalohit.did": [[12, -1], [-1]], "trkcalohit.edep": [[95.0, 0.0], [0.0]],
        "trkcalohit.dt": [[1.5, 0.0], [0.0]],
        "trksegs": [[good, []], [upstream]],
        "trksegpars_lh": [[[pars(10.0, 9.0), pars(20.0, 9.0), pars(30.0, 0.5)], []],
                          [[pars(40.0, 0.4)]]],
        "event": [5, 6], "run": [1, 1], "subrun": [0, 0],
    }
    for name in ntrig._TRK_FIELDS:
        tracks.setdefault(f"trk.{name}", [[0, 0], [0]])
    return {name: ak.Array(values) for name, values in tracks.items()}


def test_ntuple_track_variables_take_the_downstream_front_and_the_mid_fit():
    v = ntrig.track_variables(_ntuple_arrays())
    assert v["_event"].tolist() == [0, 0, 1]
    assert abs(v["p_front"][0] - 104.0) < 0.01               # not the upstream leg
    assert v["t_front"][0] == 900.0 and v["downstream"].tolist() == [True, False, False]
    assert abs(v["tandip_front"][0] - 66.4529 / 80.0) < 1e-6
    assert v["d0"][0] == 30.0 and v["t0err"][0] == 0.5       # the TT_Mid entry
    assert np.isnan(v["p_front"][1]) and np.isnan(v["d0"][1])  # no segments
    assert np.isnan(v["p_front"][2]) and v["d0"][2] == 40.0  # upstream: no front
    assert v["has_calo"].tolist() == [True, False, False]
    assert v["calo_edep"][0] == 95.0 and np.isnan(v["calo_edep"][1])
    assert v["charge"].tolist() == [-1.0, -1.0, 1.0]
    assert v["ntrk"].tolist() == [2.0, 2.0, 1.0] and v["event"].tolist() == [5, 5, 6]
    assert set(v) - {"_event"} == set(ntrig.TRACK_VARIABLES)
    for name in ntrig.TRACK_VARIABLES:            # every advertised one is usable
        apply_selection(f"{name} == {name}", v, v["_event"].size)


def test_ntuple_efficiency_is_over_events_with_a_selected_track():
    v = ntrig.track_variables(_ntuple_arrays())
    triggers = {"a": np.array([True, True]), "b": np.array([False, True])}
    counts = ntrig.efficiency_counts(v, triggers, 2, ntrig.DEFAULT_SELECTION)
    assert counts["n_selected"] == 1 and counts["n_tracks_selected"] == 1
    assert counts["n_triggered"] == 1 and counts["per_path"] == {"a": 1, "b": 0}
    everything = ntrig.efficiency_counts(v, triggers, 2, "")
    assert everything["n_selected"] == 2 and everything["per_path"]["b"] == 1
    metrics = ntrig.efficiency_metrics(counts)
    assert metrics["efficiency"] == 1.0 and metrics["efficiency_err"] > 0.0


def test_ntuple_trigger_efficiency_on_a_real_file(tmp_dir):
    if not NTUPLE_FILE.exists():
        return
    result = run_analysis(analysis="trigger_efficiency_ntuple",
                          data_file=str(NTUPLE_FILE), output_dir=tmp_dir,
                          parameters={"trigger_paths": "apr_TrkDe_80m70p, "
                                                       "trig_cpr_TrkDe_80m70p"})
    assert result.status == "success", result.message
    meta = result.metadata
    assert meta["n_events"] == 8212 and 0 < meta["n_selected"] < meta["n_events"]
    assert meta["n_triggered"] <= meta["n_selected"]
    for row in meta["paths"].values():                       # OR >= each path
        assert row["passed"] <= meta["n_triggered"]
    missing = run_analysis(analysis="trigger_efficiency_ntuple",
                           data_file=str(NTUPLE_FILE), output_dir=tmp_dir,
                           parameters={"trigger_paths": "not_a_path"})
    assert missing.status == "error" and "It has: " in missing.message


# --- stop_materials ----------------------------------------------------------

# A MuBeam-stage ntuple with TargetMuonFinder/, PolyMuonFinder/ and
# IPAMuonFinder/stopmat; the test using it is skipped where it is not on disk.
STOPMAT_FILE = Path("/exp/mu2e/app/users/mmackenz/mu2eopt/"
                    "nts.mmackenz.mubeam.Run1Bak_local0818120248.001800_00000000.root")


def test_material_rates_divide_by_n_gen_and_sort_most_first():
    rows = material_rates([("Al", 30.0, 30.0 ** 0.5), ("Steel", 70.0, 70.0 ** 0.5),
                           ("Ti", 0.0, 0.0)], n_gen_events=1000.0)
    assert [r["material"] for r in rows] == ["Steel", "Al", "Ti"]
    assert abs(rows[0]["stops_per_gen_event"] - 0.07) < 1e-12
    assert abs(rows[1]["stops_per_gen_event_err"] - 30.0 ** 0.5 / 1000.0) < 1e-12
    assert abs(rows[0]["fraction"] - 0.7) < 1e-12 and rows[2]["fraction"] == 0.0


def test_combine_tables_matches_materials_by_name_not_bin():
    """Each file labels its axis in its own order; a missing material is 0."""
    combined = dict((name, (stops, err)) for name, stops, err in combine_tables([
        [("Al", 3.0, 3.0), ("Steel", 4.0, 4.0)],
        [("Steel", 12.0, 3.0), ("Ti", 1.0, 1.0)],
    ]))
    assert combined["Steel"] == (16.0, 5.0)       # errors add in quadrature
    assert combined["Al"] == (3.0, 3.0) and combined["Ti"] == (1.0, 1.0)


def test_stop_materials_names_the_modules_it_could_have_read(tmp_dir):
    """A module without a stopmat is an error listing the ones that have one."""
    import uproot
    path = Path(tmp_dir) / "stops.root"
    with uproot.recreate(path) as f:
        f["PolyMuonFinder/stopmat"] = np.histogram(np.zeros(1), bins=2)
    result = run_analysis(analysis="stop_materials", data_file=str(path),
                          output_dir=tmp_dir, parameters={"n_gen_events": 10})
    assert result.status == "error"
    assert "TargetMuonFinder/stopmat" in result.message
    assert "PolyMuonFinder" in result.message


def test_stop_materials_needs_n_gen_events(tmp_dir):
    result = run_analysis(analysis="stop_materials", data_file="/a.root",
                          output_dir=tmp_dir)
    assert result.status == "error" and "n_gen_events" in result.message


def test_stop_materials_reads_the_labelled_bins_of_a_real_file(tmp_dir):
    if not STOPMAT_FILE.exists():
        print(f"     (skipped: {STOPMAT_FILE} not on disk)")
        return
    result = run_analysis(analysis="stop_materials", data_file=str(STOPMAT_FILE),
                          output_dir=tmp_dir, parameters={"n_gen_events": 1e6})
    assert result.status == "success", result.message
    meta = result.metadata
    assert meta["stop_module"] == "TargetMuonFinder"
    materials = {row["material"]: row for row in meta["materials"]}
    assert "StoppingTarget_Al" in materials
    # every stop is in a named bin, and the total is what the rates add up to
    assert meta["unnamed_stops"] == 0.0
    assert meta["n_stops"] == meta["hist_entries"]
    total = sum(row["stops_per_gen_event"] for row in meta["materials"])
    assert abs(total - meta["stops_per_gen_event"]) < 1e-12
    assert abs(meta["stops_per_gen_event"] - meta["n_stops"] / 1e6) < 1e-15

    poly = run_analysis(analysis="stop_materials", data_file=str(STOPMAT_FILE),
                        output_dir=tmp_dir,
                        parameters={"n_gen_events": 1e6,
                                    "stop_module": "PolyMuonFinder"})
    assert poly.status == "success", poly.message
    assert poly.metadata["hist_path"] == "PolyMuonFinder/stopmat"

    # the same file twice: twice the stops over twice the generated events
    both = run_analysis(analysis="stop_materials",
                        data_files=[str(STOPMAT_FILE)] * 2, output_dir=tmp_dir,
                        parameters={"n_gen_events": 2e6})
    assert both.status == "success", both.message
    assert both.metadata["n_stops"] == 2 * meta["n_stops"]
    assert abs(both.metadata["stops_per_gen_event"]
               - meta["stops_per_gen_event"]) < 1e-15
    assert [f["n_stops"] for f in both.metadata["per_file"]] == [meta["n_stops"]] * 2
    doubled = {row["material"]: row for row in both.metadata["materials"]}
    al = materials["StoppingTarget_Al"]
    assert doubled["StoppingTarget_Al"]["stops"] == 2 * al["stops"]
    assert abs(doubled["StoppingTarget_Al"]["stops_err"]
               - 2 ** 0.5 * al["stops_err"]) < 1e-9


def test_stop_materials_names_the_file_in_a_list_that_lacks_the_histogram(tmp_dir):
    import uproot
    bad = Path(tmp_dir) / "bad.root"
    with uproot.recreate(bad) as f:
        f["other"] = np.histogram(np.zeros(1), bins=2)
    if not STOPMAT_FILE.exists():
        print(f"     (skipped: {STOPMAT_FILE} not on disk)")
        return
    result = run_analysis(analysis="stop_materials",
                          data_files=[str(STOPMAT_FILE), str(bad)],
                          output_dir=tmp_dir, parameters={"n_gen_events": 10})
    assert result.status == "error"
    assert str(bad) in result.message and str(STOPMAT_FILE) not in result.message


# --- fullsim_sensitivity -----------------------------------------------------

def _seg(sid, time=900.0, p=(0.0, 85.0, 60.0), t0err=0.5, maxr=500.0, d0=50.0,
         tandip=0.7):
    """One track segment and its loop-helix parameters. The default momentum
    is CE-like: |p| = 104 MeV/c, p_z/p_T = 0.71."""
    return ({"sid": sid, "time": time,
             "mom": {"fCoordinates": {"fX": p[0], "fY": p[1], "fZ": p[2]}}},
            {"t0err": t0err, "maxr": maxr, "d0": d0, "tanDip": tandip})


def _track(pdg=11, nactive=30, qual=0.9, pid=0.9, status=1, goodfit=1, segs=None,
           sims=((168, 0, 10.0),)):
    """A track passing every default cut unless told otherwise: segments at
    the tracker front and middle, an ST boundary and the ST foils. `sims` are
    (startCode, gen, rho) for the track's particle, then its ancestors."""
    if segs is None:
        segs = [_seg(0), _seg(1), _seg(100), _seg(104)]
    return {"pdg": pdg, "nactive": nactive, "qual": qual, "pid": pid,
            "status": status, "goodfit": goodfit, "segs": segs,
            "sims": [{"startCode": code, "gen": gen,
                      "pos": {"fCoordinates": {"fX": rho, "fY": 0.0}},
                      "mom": {"fCoordinates": {"fX": 0.0, "fY": 84.0, "fZ": 63.0}}}
                     for code, gen, rho in sims]}


def _eventntuple(events):
    """EventNtuple groups, as read_eventntuple returns them, from events of
    {"tracks": [...], "crv": [(time, PEs, nHits, start, end)], "trig": 0/1,
    "calo": [cluster energies]}. "trig" sets every default trigger path; a
    dict {path: 0/1} sets each. "calo" defaults to one 50 MeV cluster.
    Every event carries a far-off coincidence so the CRV arrays have a type."""
    import awkward as ak
    far = (-5000.0, 1.0, 1, -5000.0, -4990.0)
    crv = [ev.get("crv", []) + [far] for ev in events]
    tracks = [ev["tracks"] for ev in events]
    return {
        "evt": ak.Array([{"run": 1, "subrun": 0, "event": i,
                          **{TRIGGER_PREFIX + name: trig.get(name, 0)
                             if isinstance(trig := ev.get("trig", 1), dict) else trig
                             for name in DEFAULT_TRIGGERS}}
                         for i, ev in enumerate(events)]),
        "crv": ak.Array([{f"crvcoincs.{field}": [c[k] for c in coincs]
                          for k, field in enumerate(("time", "PEs", "nHits",
                                                     "timeStart", "timeEnd"))}
                         for coincs in crv]),
        "calo": ak.Array([{"caloclusters.energyDep_": [float(e) for e in
                                                       ev.get("calo", [50.0])]}
                          for ev in events]),
        "trk": ak.Array([{"trk.pdg": [t["pdg"] for t in trks],
                          "trk.nactive": [t["nactive"] for t in trks],
                          "trk.status": [t["status"] for t in trks],
                          "trk.goodfit": [t["goodfit"] for t in trks],
                          "trkqual.result": [t["qual"] for t in trks],
                          "trkpid.result": [t["pid"] for t in trks]}
                         for trks in tracks]),
        "trkfit": ak.Array([{"trksegs": [[s for s, _ in t["segs"]] for t in trks],
                             "trksegpars_lh": [[lh for _, lh in t["segs"]] for t in trks]}
                            for trks in tracks]),
        "trkmc": ak.Array([{"trkmcsim": [t["sims"] for t in trks]} for trks in tracks]),
    }


def _failed_cuts(event, active=fs_cuts.DEFAULT_CUTS, sign="minus", track=0):
    masks = fs_cuts.cut_masks(_eventntuple([event]), sign)
    return sorted(name for name in active if not masks[name][0][track])


def test_fullsim_a_ce_like_track_passes_every_default_cut():
    assert _failed_cuts({"tracks": [_track()]}) == []


def test_fullsim_each_default_cut_rejects_what_it_should():
    front, mid, boundary, foils = _seg(0), _seg(1), _seg(100), _seg(104)
    cases = [
        (["is_good_track"], {"tracks": [_track(status=-1)]}),
        (["is_good_track"], {"tracks": [_track(goodfit=0)]}),
        # without a front segment there is no direction or tan(dip) either
        (["has_downstream", "has_trk_front_seg", "pz_over_pt"],
         {"tracks": [_track(segs=[mid, boundary, foils])]}),
        (["charge_selection", "is_reco_electron_or_positron"], {"tracks": [_track(pdg=13)]}),
        (["has_downstream"],
         {"tracks": [_track(segs=[_seg(0, p=(0, 85, -60)), mid, boundary, foils])]}),
        (["charge_selection"], {"tracks": [_track(pdg=-11)]}),
        (["or_trigger"], {"tracks": [_track()], "trig": 0}),
        (["good_trkpid"], {"tracks": [_track(pid=0.5)]}),
        (["good_trkpid"], {"tracks": [_track()], "calo": []}),
        (["pz_over_pt"],
         {"tracks": [_track(segs=[_seg(0, tandip=0.9), mid, boundary, foils])]}),
        (["st_boundary"], {"tracks": [_track(segs=[front, mid, foils])]}),
        (["has_st"], {"tracks": [_track(segs=[front, mid, boundary])]}),
        (["no_opa"], {"tracks": [_track(segs=[front, mid, boundary, foils, _seg(95)])]}),
        (["good_trkqual"], {"tracks": [_track(qual=0.1)]}),
        (["has_hits"], {"tracks": [_track(nactive=19)]}),
        # t0err counts at the tracker middle, not the front
        (["within_t0err"],
         {"tracks": [_track(segs=[front, _seg(1, t0err=1.0), boundary, foils])]}),
        ([], {"tracks": [_track(segs=[_seg(0, t0err=1.0), mid, boundary, foils])]}),
        # the CRV veto is asymmetric: a coincidence 50 ns before the track
        # vetoes it, one 50 ns after does not
        (["no_crv_veto"], {"tracks": [_track()], "crv": [(850.0, 5.0, 3, 800.0, 900.0)]}),
        ([], {"tracks": [_track()], "crv": [(950.0, 5.0, 3, 900.0, 1000.0)]}),
    ]
    for expected, event in cases:
        assert _failed_cuts(event) == sorted(expected), (expected, _failed_cuts(event))


def test_fullsim_vetoes_pair_tracks_in_time():
    def at(time, pz=60.0, pdg=11):
        return _track(pdg=pdg, segs=[_seg(0, time=time, p=(0, 85, pz)), _seg(1),
                                     _seg(100), _seg(104)])
    # an upstream track 40-110 ns before a downstream one is its reflection
    assert _failed_cuts({"tracks": [at(900.0), at(830.0, pz=-60)]}) == ["upstream_veto"]
    assert _failed_cuts({"tracks": [at(900.0), at(870.0, pz=-60)]}) == []
    # two downstream e+- within 150 ns veto each other, for sign minus only
    pair = {"tracks": [at(900.0), at(1000.0, pdg=-11)]}
    assert _failed_cuts(pair) == ["no_multi_trk_veto"]
    assert _failed_cuts(pair, track=1) == ["charge_selection", "no_multi_trk_veto"]
    assert _failed_cuts({"tracks": [at(900.0), at(1100.0)]}) == []
    assert _failed_cuts(pair, sign="plus") == ["charge_selection"]


def test_fullsim_trigger_passes_on_any_of_the_paths():
    apr, cpr = DEFAULT_TRIGGERS
    assert _failed_cuts({"tracks": [_track()], "trig": {apr: 1}}) == []
    assert _failed_cuts({"tracks": [_track()], "trig": {cpr: 1}}) == []
    assert _failed_cuts({"tracks": [_track()], "trig": {}}) == ["or_trigger"]
    # only the paths asked for count
    data = _eventntuple([{"tracks": [_track()], "trig": {apr: 1}}])
    masks = fs_cuts.cut_masks(data, "minus", triggers=(cpr,))
    assert not masks["or_trigger"][0][0]
    from tools.analyses.fullsim.eventntuple import EventNtupleError, parse_trigger_paths
    assert parse_trigger_paths(f"trig_{apr}, {cpr} {apr}") == [apr, cpr]
    for bad in ("", " , ", "bad-name"):
        try:
            parse_trigger_paths(bad)
        except EventNtupleError:
            pass
        else:
            raise AssertionError(f"{bad!r} should have been refused")


def test_fullsim_a_segment_cut_applies_only_at_its_surface():
    # a late time at the ST does not matter; at the front it does
    st_late = {"tracks": [_track(segs=[_seg(0), _seg(1), _seg(100), _seg(104, time=50.0)])]}
    front_early = {"tracks": [_track(segs=[_seg(0, time=50.0), _seg(1), _seg(100),
                                           _seg(104)])]}
    active = fs_cuts.DEFAULT_CUTS + ("within_t0",)
    assert _failed_cuts(st_late, active) == []
    assert _failed_cuts(front_early, active) == ["within_t0"]


def test_fullsim_active_cuts_toggle_by_name_and_refuse_unknown_ones():
    active = fs_cuts.active_cuts("within_t0", "has_st, no_opa")
    assert "within_t0" in active and "has_st" not in active and "no_opa" not in active
    assert "in_mom_range" not in active and active[0] == "has_a_track"
    # the cut flow follows pyfitter's order whatever order they are named in
    assert active == [n for n in fs_cuts.CUT_DESCRIPTIONS if n in active]
    try:
        fs_cuts.active_cuts("not_a_cut", "")
    except fs_cuts.CutError as exc:
        assert "Known:" in str(exc), exc
    else:
        raise AssertionError("not_a_cut was accepted")


def test_fullsim_origin_is_the_first_classified_particle_in_the_chain():
    events = [
        {"tracks": [_track(sims=[(168, 0, 10.0)])]},              # CE
        {"tracks": [_track(sims=[(167, 0, 10.0)])]},              # CE endpoint
        {"tracks": [_track(sims=[(166, 0, 30.0)])]},              # DIO on the ST
        {"tracks": [_track(sims=[(166, 0, 200.0)])]},             # DIO on the IPA
        {"tracks": [_track(sims=[(170, 0, 30.0)])]},              # leading-log DIO
        {"tracks": [_track(sims=[(12, 0, 0.0), (171, 0, 0.0)])]},  # conversion of an iRMC photon
        {"tracks": [_track(sims=[(12, 44, 0.0)])]},                # cosmic
        {"tracks": [_track(sims=[(12, 0, 0.0), (13, 0, 0.0)])]},   # nothing known
    ]
    origin = origin_codes(_eventntuple(events)["trkmc"]["trkmcsim"])
    assert origin.tolist() == [168, 168, 166, 0, 166, 171, -1, -2]


def test_fullsim_reduce_counts_the_first_selected_track_of_each_event():
    data = _eventntuple([
        {"tracks": [_track()]},
        # the first track fails, so the second (a DIO, 200 ns later so the
        # multi-track veto leaves it) represents the event
        {"tracks": [_track(qual=0.0),
                    _track(segs=[_seg(0, time=1100.0, p=(0, 80, 60)), _seg(1), _seg(100),
                                 _seg(104)], sims=[(166, 0, 20.0)])]},
        {"tracks": [_track(nactive=5)]},                            # cut away
        # selected, but with two front segments, so not counted (a reflected
        # track would be, but pz_over_pt removes those)
        {"tracks": [_track(segs=[_seg(0), _seg(0, time=950.0), _seg(1), _seg(100),
                                 _seg(104)])]},
        # likewise when the second front segment's time is NaN: it still counts
        {"tracks": [_track(segs=[_seg(0), _seg(0, time=np.nan), _seg(1), _seg(100),
                                 _seg(104)])]},
    ])
    data["n_processed"] = 7          # the job processed more than it kept
    active = fs_cuts.active_cuts("", "")
    reduced = fs_sens.reduce_data(data, active, fs_cuts.TRKQUAL_MIN, fs_cuts.TRKPID_MIN)
    assert reduced["n_events"] == 5 and reduced["n_selected"] == 4
    assert reduced["n_processed"] == 7
    assert reduced["flow"][-1] == 4 and len(reduced["flow"]) == len(active)
    assert reduced["origin"].tolist() == [168, 166]
    assert np.allclose(reduced["p"], [np.hypot(85, 60), 100.0])
    assert np.allclose(reduced["p_true"], [105.0, 105.0])
    assert reduced["event"].tolist() == [0, 1]


def test_fullsim_rates_follow_production_normalization():
    """Signal = NPOT x stops/POT x captures/stop x R_mue x efficiency; the DIO
    is the theory spectrum for NPOT x stops/POT x DIO/stop, folded with the
    measured response, which carries the same efficiency; cosmics are flat
    per MeV/c over the on-spill time."""
    from tools.analyses.fullsim import normalization as norm
    from tools.analyses import approx_ce_sensitivity as fast
    if not fast.DIO_TABLE.exists():
        print("     (skipped: DIO table not on disk)")
        return
    rng = np.random.default_rng(1)
    p_true = np.full(400, 104.97)
    p_reco = p_true - rng.exponential(0.3, p_true.size) + rng.normal(0, 0.15, p_true.size)
    spectra = fs_sens.build_spectra(p_reco, p_true, n_generated=1000,
                                    upstream_eff=0.5, npot=1e18,
                                    mean_pot_per_event=1.6e7,
                                    cosmic_rate_per_s_per_mev=1e-5,
                                    stopped_muons_per_pot=1e-3, rmue=1e-13)
    eff = 400 / 1000 * 0.5
    stops = 1e18 * 1e-3
    assert np.isclose(spectra["efficiency"], eff)
    assert np.isclose(spectra["stopped_muons"], stops)
    assert np.isclose(spectra["signal"].values.sum(), stops * 0.609 * 1e-13 * eff)
    assert np.isclose(spectra["response"].values.sum() * spectra["response"].width, eff)
    # the response only moves DIO a little, so the folded total is the
    # theory total times the efficiency
    dio_total = norm.DIO_PER_STOPPED_MUON * stops * eff
    assert np.isclose(spectra["dio"].values.sum(), dio_total, rtol=1e-3)
    width = spectra["signal"].width
    onspill = 1e18 / 1.6e7 * fast.ONSPILL_SECONDS_PER_EVENT
    assert np.allclose(spectra["cosmic"].values, 1e-5 * onspill * width)
    # the scan is approx_ce_sensitivity's, and finds the peak
    best, _ = fast.scan_signal_box(spectra["signal"], spectra["dio"], spectra["cosmic"])
    assert 103.5 < best["low_mev"] < best["high_mev"] < 105.5


# SimEfficiencies2 for Sim_best v1_1, run 1430, as dbTool prints it.
SIM_EFFICIENCIES_TABLE = """\
MuBeamCat,213816,100000000,0.00213816
EleBeamCat,5532579,100000000,0.05532579
MuminusStopsCat,1435092,4000000000,0.000358773
MuplusStopsCat,7578,4000000000,0.0000018945
IPAStopsCat,22519,3584800000,0.00000628180093729078
PiTotalLifeimeWeight_filter,0,0,38536.77997060446
"""


def test_fullsim_stopped_muons_per_pot_follow_the_sim_chain():
    from tools.analyses.fullsim import normalization as norm
    table = norm.parse_sim_efficiencies(SIM_EFFICIENCIES_TABLE)
    # Production's normalizations.py: MuBeamCat x MuminusStopsCat x 1000
    assert np.isclose(norm.stopped_muons_per_pot(table),
                      0.00213816 * 0.000358773 * 1000)
    assert np.isclose(norm.STOPPED_MUONS_PER_POT, norm.stopped_muons_per_pot(table))
    assert np.isclose(norm.CAPTURES_PER_STOPPED_MUON + norm.DIO_PER_STOPPED_MUON, 1.0)
    del table["MuminusStopsCat"]
    try:
        norm.stopped_muons_per_pot(table)
    except norm.NormalizationError as exc:
        assert "MuminusStopsCat" in str(exc), exc
    else:
        raise AssertionError("a table without MuminusStopsCat was accepted")


# The MDC2025 stop chain's files as SAM has them (2026-10-07): MuBeamCat's
# four, one with event_count left out as SAM does for a zero, and
# MuminusStopsCat's one.
MUBEAM_FILES = [("sim.mu2e.MuBeamCat.MDC2025ab.001430_00020851.art", 25000000, 53513),
                ("sim.mu2e.MuBeamCat.MDC2025ab.001430_00032620.art", 25000000, 53834),
                ("sim.mu2e.MuBeamCat.MDC2025ab.001430_00020001.art", 25000000, 53262),
                ("sim.mu2e.MuBeamCat.MDC2025ab.001430_00020000.art", 25000000, 53207)]
MUSTOPS_FILES = [("sim.mu2e.MuminusStopsCat.MDC2025ac.001430_00000000.art",
                  4000000000, 1435092)]


def _fake_sam(datasets):
    """list_files and fetch over {dataset: [(file, gencount, event_count)]};
    an event_count of None is left out of the metadata."""
    records = {}
    for files in datasets.values():
        for name, gencount, count in files:
            records[name] = {"file_name": name, "dh.gencount": gencount}
            if count is not None:
                records[name]["event_count"] = count
    return ((lambda ds: [f[0] for f in datasets.get(ds, [])]),
            (lambda names: [records[n] for n in names]))


def test_fullsim_stage_efficiencies_come_from_the_datasets_without_the_db():
    """As CreateSimEfficiency.sh (mu2eGenFilterEff): events in the dataset
    over events generated, summed over its files, so the MDC2025 chain gives
    the SimEfficiencies2 rows without reading them."""
    from tools.analyses.fullsim import normalization as norm
    beam, stops = norm.STOP_CHAIN_DATASETS
    list_files, fetch = _fake_sam({beam: MUBEAM_FILES, stops: MUSTOPS_FILES})
    stages = norm.sim_efficiencies(norm.STOP_CHAIN_DATASETS, list_files, fetch)
    assert stages == norm.STOP_CHAIN_MDC2025
    table = norm.parse_sim_efficiencies(SIM_EFFICIENCIES_TABLE)
    for stage in norm.STOP_CHAIN:
        assert np.isclose(stages[stage].efficiency, table[stage]), stage
    assert np.isclose(norm.stopped_muons_per_pot(stages), norm.STOPPED_MUONS_PER_POT)
    # a missing event_count is zero events, not an error
    zero = [MUBEAM_FILES[0][:2] + (None,)] + MUBEAM_FILES[1:]
    list_files, fetch = _fake_sam({beam: zero})
    assert norm.dataset_efficiency(beam, list_files, fetch).passed == 213816 - 53513


def test_fullsim_stage_efficiency_errors_say_what_is_wrong():
    from tools.analyses.fullsim import normalization as norm
    beam, stops = norm.STOP_CHAIN_DATASETS
    no_gencount = [(MUBEAM_FILES[0][0], 0, 5)]
    def unreachable(_):
        raise OSError("network down")
    cases = [
        (["MuBeamCat"], _fake_sam({}), "not a dataset name"),
        (["sim.mu2e.MuBeamCat.MDC2025ab.art with availability x"], _fake_sam({}),
         "not a dataset name"),
        ([beam], _fake_sam({}), "no files"),
        ([beam], _fake_sam({beam: no_gencount}), "no dh.gencount"),
        ([beam], (unreachable, None), "network down"),
        ([beam, "sim.mu2e.MuBeamCat.MDC2026a.art"], _fake_sam({beam: MUBEAM_FILES}),
         "two datasets"),
    ]
    for datasets, (list_files, fetch), needle in cases:
        try:
            norm.sim_efficiencies(datasets, list_files, fetch)
        except norm.NormalizationError as exc:
            assert needle in str(exc), (needle, str(exc))
        else:
            raise AssertionError(f"{needle}: no error")
    # one stage only cannot give stops per POT
    stages = norm.sim_efficiencies([beam], *_fake_sam({beam: MUBEAM_FILES}))
    try:
        norm.stopped_muons_per_pot(stages)
    except norm.NormalizationError as exc:
        assert "MuminusStopsCat" in str(exc), exc
    else:
        raise AssertionError("a chain without MuminusStopsCat was accepted")


def _ce_ancestry(config="MDC2025", stops="MDC2025ac", beam="MDC2025ab", seq="001430_00000000"):
    """A SAM catalog shaped like the CE mix file's real ancestry: nts -> mcs
    -> dig (many dts parents) -> dts -> MuminusStopsCat -> TargetStopsCat ->
    TargetStops -> MuBeamCat. Returns (nts file name, catalog)."""
    nts = f"nts.mu2e.CeMLeadingLogMix1BB.{config}au_best_v1_1-001.{seq}.root"
    mcs = f"mcs.mu2e.CeMLeadingLogMix1BB.{config}au_best_v1_1.{seq}.art"
    dig = f"dig.mu2e.CeMLeadingLogMix1BB.{config}au_best_v1_3.{seq}.art"
    dts = [f"dts.mu2e.CeMLeadingLog.{config}ap.001430_{i:08d}.art" for i in range(3)]
    mustops = f"sim.mu2e.MuminusStopsCat.{stops}.001430_00000000.art"
    tscat = f"sim.mu2e.TargetStopsCat.{stops}.001430_00000000.art"
    ts = [f"sim.mu2e.TargetStops.{stops}.001430_{i:08d}.art" for i in range(3)]
    mubeam = f"sim.mu2e.MuBeamCat.{beam}.001430_00020001.art"
    beam_parents = [f"sim.mu2e.Beam.{beam}.001430_{i:08d}.art" for i in range(3)]
    def parents(*names):
        return {"parents": [{"file_name": n} for n in names]}
    catalog = {nts: parents(mcs), mcs: parents(dig), dig: parents(*dts),
               **{d: parents(mustops) for d in dts},
               mustops: parents(tscat), tscat: parents(*ts),
               **{t: parents(mubeam) for t in ts},
               mubeam: parents(*beam_parents)}
    return nts, catalog


def test_fullsim_stop_chain_is_traced_from_the_inputs_ancestry():
    from tools.analyses.fullsim import normalization as norm
    nts, catalog = _ce_ancestry()
    fetched = []
    def fetch(name):
        fetched.append(name)
        return catalog[name]
    # the CE mix's chain is the one CreateSimEfficiency.sh names for MDC2025
    assert norm.stop_chain_of(nts, fetch) == norm.STOP_CHAIN_DATASETS
    # one file per parent dataset is followed, not every parent
    assert len(fetched) == 7, fetched
    # files of one dataset are traced once
    other = nts.replace("00000000.root", "00000001.root")
    catalog[other] = catalog[nts]
    fetched.clear()
    assert norm.stop_chain_of_inputs([nts, other], fetch) == norm.STOP_CHAIN_DATASETS
    assert len(fetched) == 7, fetched
    # a later iteration is traced to its own chain
    nts26, catalog26 = _ce_ancestry("MDC2026", "MDC2026c", "MDC2026b")
    assert norm.stop_chain_of(nts26, catalog26.__getitem__) == (
        "sim.mu2e.MuBeamCat.MDC2026b.art", "sim.mu2e.MuminusStopsCat.MDC2026c.art")


def test_fullsim_stop_chain_tracing_errors_say_what_is_wrong():
    from tools.analyses.fullsim import normalization as norm
    nts, catalog = _ce_ancestry()
    nts26, catalog26 = _ce_ancestry("MDC2026", "MDC2026c", "MDC2026b")
    both = {**catalog, **catalog26}
    # a dts parent from a second stops dataset makes the stops ambiguous
    ambiguous = dict(catalog)
    dts0 = catalog[catalog[catalog[nts]["parents"][0]["file_name"]]["parents"][0]["file_name"]]
    first_dts = dts0["parents"][0]["file_name"]
    ambiguous[first_dts] = {"parents": [
        {"file_name": "sim.mu2e.MuminusStopsCat.MDC2025ac.001430_00000000.art"},
        {"file_name": "sim.mu2e.MuminusStopsCat.MDC2025zz.001430_00000000.art"}]}
    orphan = {nts: {"parents": []}}
    cases = [
        ([nts, nts26], both.__getitem__, "different stop chains"),
        ([nts], ambiguous.__getitem__, "more than one MuminusStopsCat"),
        ([nts], orphan.__getitem__, "no MuminusStopsCat dataset"),
        ([nts], {}.__getitem__, "cannot read the SAM metadata"),
        (["my_ce_ntuple.root"], catalog.__getitem__, "not a SAM file name"),
    ]
    for names, fetch, needle in cases:
        try:
            norm.stop_chain_of_inputs(names, fetch)
        except norm.NormalizationError as exc:
            assert needle in str(exc), (needle, str(exc))
        else:
            raise AssertionError(f"{needle}: no error")


def test_fullsim_generated_events_come_from_the_nearest_ancestor_with_gencount():
    from tools.analyses.fullsim import provenance
    nts = "nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root"
    mcs = "mcs.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1.001430_00000000.art"
    catalog = {nts: {"parents": [{"file_name": mcs}]},
               mcs: {"dh.gencount": 20000, "event_count": 8212,
                     "parents": [{"file_name": "dig.x"}]}}
    assert provenance.generated_events(nts, fetch=catalog.__getitem__) == 20000
    # no ancestor with a count, or no SAM: an error that says which file
    orphan = {nts: {"parents": []}}
    for fetch, needle in ((orphan.__getitem__, "no dh.gencount"),
                          ({}.__getitem__, "cannot read the SAM metadata")):
        try:
            provenance.generated_events(nts, fetch=fetch)
        except provenance.ProvenanceError as exc:
            assert needle in str(exc) and nts in str(exc), exc
        else:
            raise AssertionError(f"{needle}: no error")


def test_fullsim_run_reports_a_file_that_is_not_an_eventntuple(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=10)
    result = run_analysis(analysis="fullsim_sensitivity", data_file=str(path),
                          output_dir=tmp_dir)
    assert result.status == "error" and "EventNtuple/ntuple" in result.message
    window = run_analysis(analysis="fullsim_sensitivity", data_file=str(path),
                          output_dir=tmp_dir, parameters={"time_window": "1650,640"})
    assert window.status == "error" and "time_window" in window.message


# A CeMLeadingLogMix1BB EventNtuple file: CE- mixed with pileup; skipped
# where not on disk.
CE_MIX_FILE = Path("/pnfs/mu2e/tape/phy-nts/nts/mu2e/CeMLeadingLogMix1BB/"
                   "MDC2025au_best_v1_1-001/root/a8/50/nts.mu2e.CeMLeadingLogMix1BB."
                   "MDC2025au_best_v1_1-001.001430_00000000.root")


def test_fullsim_measures_the_efficiency_on_ce_mix(tmp_dir):
    if not CE_MIX_FILE.exists():
        print("     (skipped: CE mix file not on disk)")
        return
    # the file's parent mcs file has dh.gencount 20000 in SAM, and the stop
    # chain gives STOPPED_MUONS_PER_POT; both passed here so the test needs
    # no network
    from tools.analyses.fullsim import normalization as norm
    generated = {"n_generated": 20000,
                 "stopped_muons_per_pot": norm.STOPPED_MUONS_PER_POT}
    result = run_analysis(analysis="fullsim_sensitivity", data_file=str(CE_MIX_FILE),
                          output_dir=tmp_dir, parameters=generated)
    assert result.status == "success", result.message
    md = result.metadata
    assert md["n_events_processed"] == 8212 and md["n_events_generated"] == 20000
    assert np.isclose(md["acceptance"], 8212 / 20000)
    assert (md["n_events_selected"], md["n_events_counted"]) == (4181, 3010)
    # the efficiency counts against what was generated, so it carries the
    # digitization filter's acceptance
    assert np.isclose(md["signal_efficiency"], 3010 / 20000)
    assert np.isclose(md["signal_rate"], 4.686, rtol=1e-3)
    # the cut flow is pyfitter's for cut-set 80_1d on this file
    assert list(md["cut_flow"].values())[-1] == 4181
    assert md["cut_flow"]["no_multi_trk_veto"] == 6909 and md["cut_flow"]["pz_over_pt"] == 4550
    assert np.allclose((md["signal_mom_low_mevc"], md["signal_mom_high_mevc"]), (103.5, 104.7))
    # 1e18 POT x 7.67e-4 stopped mu-/POT x 0.609 captures x R_mue 1e-13
    assert np.isclose(md["n_conversions"], 46.717, rtol=1e-4)
    assert md["total_background"] == md["dio_background"] + md["cosmic_background"]
    # a pileup DIO track that passes the cuts is not signal
    assert md["selected_by_origin"] == {"DIO": 1, "CE-": 4180}
    # upstream_eff scales the signal and the DIO alike, not the cosmics
    half = run_analysis(analysis="fullsim_sensitivity", data_file=str(CE_MIX_FILE),
                        output_dir=tmp_dir, parameters={**generated, "upstream_eff": 0.5})
    hm = half.metadata
    assert np.isclose(hm["signal_efficiency"], 0.5 * 3010 / 20000)
    assert hm["cosmic_rate_per_s_per_mev"] == md["cosmic_rate_per_s_per_mev"]
    bad = run_analysis(analysis="fullsim_sensitivity", data_file=str(CE_MIX_FILE),
                       output_dir=tmp_dir, parameters={**generated, "trigger_paths": "apr_Nope"})
    assert bad.status == "error" and "trig_apr_Nope" in bad.message
    few = run_analysis(analysis="fullsim_sensitivity", data_file=str(CE_MIX_FILE),
                       output_dir=tmp_dir, parameters={**generated, "n_generated": 100})
    assert few.status == "error" and "n_generated 100" in few.message


# --- the registry (loops over every analysis) --------------------------------

def test_registry_includes_every_analysis():
    assert {"edep", "count", "muon_stop_rate", "approx_ce_sensitivity",
            "stop_materials", "fullsim_sensitivity"} <= set(ANALYSES)


def test_every_spec_is_self_consistent():
    for name, spec in ANALYSES.items():
        assert spec.name == name, f"{name}: SPEC.name is {spec.name}"
        assert spec.description.strip(), f"{name}: needs a description"
        assert spec.metrics, f"{name}: needs at least one metric"
        assert callable(spec.run) and callable(spec.summarize)
        assert spec.input_kind in ("art_files", "root_file"), spec.input_kind
        for metric in (spec.units or {}):
            assert metric in spec.metrics, f"{name}: unit for unknown '{metric}'"
        for param in spec.parameters:
            assert param.description.strip(), f"{name}: {param.name} needs a description"
        # art_files analyses run an fcl shipped in this repo's fcl/;
        # root_file ones must not claim one at all
        if spec.input_kind == "art_files":
            assert spec.fcl is not None and spec.fcl.parent == FCL_DIR, name
            assert current().missing_fcl(spec.fcl) is None, \
                f"{name}: {current().missing_fcl(spec.fcl)}"
        else:
            assert spec.fcl is None, f"{name}: root_file analysis should have no fcl"
            # its input comes from another analysis here, or from elsewhere
            # (a production job's ntuple) — either way, say which
            assert spec.produced_by or spec.input_hint.strip(), \
                f"{name}: say what produces its input"


def test_every_summarizer_handles_its_own_metrics():
    """summarize() must cope with exactly the metric names the spec declares."""
    for name, spec in ANALYSES.items():
        metrics = {metric: 1.0 for metric in spec.metrics}
        assert spec.summarize(metrics).strip(), f"{name}: empty summary"


def test_every_art_analysis_parser_matches_its_declared_metrics():
    for name, spec in ANALYSES.items():
        if spec.input_kind != "art_files":
            continue
        sample = SAMPLE_STDOUT.get(name)
        assert sample is not None, f"{name}: add a sample stdout to SAMPLE_STDOUT"
        if name == "edep":
            # the summary block, then what the selection adds from the tree
            assert (tuple(parse_edep_summary(sample)) + edep_mod._SELECTION_METRICS
                    == spec.metrics)
        elif name == "count":
            assert tuple(saved_rates(parse_counts(sample))) == spec.metrics
        elif name == "muon_stop_rate":
            parsed = stop_rates(parse_counts(sample, PRESCALE_FILTER),
                                upstream_eff=1.0)
            assert tuple(parsed) == spec.metrics
        elif name == "trigger_efficiency":
            report = trig.parse_trig_report(sample)
            assert tuple(efficiency_metrics(report)) == spec.metrics
        elif name == "trigger_rate":
            report = trig.parse_trig_report(sample)
            assert tuple(rate_metrics(report, 1.0, 0.5)) == spec.metrics
        elif name == "trigger_timing":
            # its numbers come from the timing database, not stdout
            assert tuple(timing_metrics(np.array([0.01, 0.02]))) == spec.metrics
        else:
            raise AssertionError(f"{name}: add its parser to this test")


def test_list_analyses_reports_every_registered_analysis():
    result = list_analyses()
    assert result.status == "success"
    catalogue = result.metadata["analyses"]
    assert set(catalogue) == set(ANALYSES)

    edep = catalogue["edep"]
    assert edep["input_kind"] == "art_files"
    assert edep["fcl"] == str(FCL_DIR / "edep.fcl")
    assert edep["fcl_exists"] is True
    assert edep["units"]["avg_calo_edep_per_event_mev"] == "MeV"

    stops = catalogue["muon_stop_rate"]
    assert stops["input_kind"] == "art_files"
    assert stops["fcl"] == str(FCL_DIR / "print_counts.fcl")
    assert stops["fcl_exists"] is True
    assert stops["parameters"]["upstream_eff"]["required"] is True
    assert stops["units"]["stops_per_pot"] == "stops / POT"

    counts = catalogue["count"]
    assert counts["input_kind"] == "art_files"
    assert counts["fcl"] == str(FCL_DIR / "print_counts.fcl")
    assert counts["fcl_exists"] is True
    # the whole point of `count`: naming a prescale filter is optional
    assert counts["parameters"]["prescale_filter"]["required"] is False
    assert counts["parameters"]["prescale_filter"]["default"] == ""
    assert counts["units"]["saved_per_gen_event"] == "events / generated event"

    ce = catalogue["approx_ce_sensitivity"]
    assert ce["input_kind"] == "root_file"
    assert ce["produced_by"] == ["edep"]          # chaining is discoverable
    # one of the two, checked when it runs: 0 means not given
    assert ce["parameters"]["sig_eff"]["required"] is False
    assert ce["parameters"]["stops_per_pot"]["required"] is False
    assert ce["parameters"]["npot"]["required"] is False
    cosmic = ce["parameters"]["cosmic_rate_per_s_per_mev"]
    assert cosmic["required"] is False
    assert cosmic["default"] == sens.COSMIC_RATE_PER_SECOND_PER_MEV
    # the normalization a result was built on is reported with it
    assert {"npot", "cosmic_rate_per_s_per_mev"} <= set(ce["metrics"])
    assert ce["units"]["npot"] == "POT"
    assert "fcl" not in ce

    mats = catalogue["stop_materials"]
    assert mats["input_kind"] == "root_file"
    assert mats["parameters"]["n_gen_events"]["required"] is True
    assert mats["parameters"]["stop_module"]["default"] == "TargetMuonFinder"
    # a root_file analysis says whether it takes data_files; art ones always do
    assert mats["takes_data_files"] is True
    assert ce["takes_data_files"] is False
    assert edep["takes_data_files"] is True


# --- parameter handling ------------------------------------------------------

def test_params_resolve_defaults_and_validate():
    spec = ANALYSES["approx_ce_sensitivity"]
    resolved = spec.resolve_params({"sig_eff": 0.25})
    assert resolved["sig_eff"] == 0.25
    assert resolved["npot"] == sens.NPOT           # default filled in


def test_missing_required_parameter_is_reported(tmp_dir):
    result = run_analysis(analysis="trigger_efficiency_ntuple",
                          data_file="/nonexistent/x.root", output_dir=tmp_dir)
    assert result.status == "error"
    assert "trigger_paths" in result.message and "missing" in result.message


def test_out_of_range_parameter_is_reported(tmp_dir):
    result = run_analysis(analysis="approx_ce_sensitivity",
                          data_file="/nonexistent/x.root", output_dir=tmp_dir,
                          parameters={"sig_eff": 5.0})
    assert result.status == "error"
    assert "sig_eff" in result.message and "maximum" in result.message


def test_unknown_parameter_is_reported(tmp_dir):
    result = run_analysis(analysis="approx_ce_sensitivity",
                          data_file="/nonexistent/x.root", output_dir=tmp_dir,
                          parameters={"sig_eff": 0.1, "wat": 1.0})
    assert result.status == "error"
    assert "wat" in result.message and "unknown" in result.message


def test_cosmic_rate_defaults_to_the_modules_assumption():
    spec = ANALYSES["approx_ce_sensitivity"]
    params = spec.resolve_params({"sig_eff": 0.1})
    assert params["cosmic_rate_per_s_per_mev"] == sens.COSMIC_RATE_PER_SECOND_PER_MEV
    # and a supplied value is what reaches the run
    params = spec.resolve_params({"sig_eff": 0.1, "cosmic_rate_per_s_per_mev": 5.0e-3})
    assert params["cosmic_rate_per_s_per_mev"] == 5.0e-3
    try:
        spec.resolve_params({"sig_eff": 0.1, "cosmic_rate_per_s_per_mev": -1.0})
    except ValueError as exc:
        assert "cosmic_rate_per_s_per_mev" in str(exc)
    else:
        raise AssertionError("a negative rate should be rejected")


def test_text_param_defaults_and_validates():
    spec = ANALYSES["muon_stop_rate"]
    # left out, it falls back to the target-stop stream
    assert (spec.resolve_params({"upstream_eff": 0.012})["prescale_filter"]
            == "TargetStopPrescaleFilter")
    assert (spec.resolve_params({"upstream_eff": 0.012,
                                 "prescale_filter": "PolyStopPrescaleFilter"})
            ["prescale_filter"] == "PolyStopPrescaleFilter")
    for bad in ("", "   ", 7):
        try:
            spec.resolve_params({"upstream_eff": 0.012, "prescale_filter": bad})
        except ValueError as exc:
            assert "prescale_filter" in str(exc)
        else:
            raise AssertionError(f"{bad!r} should not be a valid label")


def test_optional_text_param_takes_the_empty_string_as_an_answer():
    """count's prescale_filter: "" means "no such filter", not a typo.

    muon_stop_rate's same-named knob (above) still refuses it — the difference
    is ParamSpec.allow_empty, not the name.
    """
    spec = ANALYSES["count"]
    assert spec.resolve_params(None)["prescale_filter"] == ""
    assert spec.resolve_params({"prescale_filter": ""})["prescale_filter"] == ""
    assert spec.resolve_params({"prescale_filter": "  "})["prescale_filter"] == ""
    assert (spec.resolve_params({"prescale_filter": "PolyStopPrescaleFilter"})
            ["prescale_filter"] == "PolyStopPrescaleFilter")
    # allow_empty is about blank text, not about text: a number is still wrong
    try:
        spec.resolve_params({"prescale_filter": 7})
    except ValueError as exc:
        assert "prescale_filter" in str(exc)
    else:
        raise AssertionError("a non-string label should be rejected")


def test_param_spec_rejects_non_numbers():
    param = ParamSpec(name="p", description="d", default=1.0)
    try:
        param.check("not a number")
    except ValueError as exc:
        assert "must be a number" in str(exc)
        return
    raise AssertionError("a non-numeric parameter was accepted")


# --- input handling (-s / -S) ------------------------------------------------

def test_single_input_uses_dash_s_and_no_filelist(tmp_dir):
    flag, arg, file_list = build_input_args(
        [Path("/data/one.art")], Path(tmp_dir), single=True
    )
    assert (flag, arg) == ("-s", "/data/one.art")
    assert file_list is None
    assert not (Path(tmp_dir) / "filelist.txt").exists()


def test_multiple_inputs_write_one_path_per_line_for_dash_S(tmp_dir):
    outdir = Path(tmp_dir)
    paths = [Path("/data/a.art"), Path("/data/b.art"), Path("/data/c.art")]
    flag, arg, file_list = build_input_args(paths, outdir, single=False)

    assert flag == "-S"
    assert file_list == outdir / "filelist.txt"
    assert arg == str(file_list)
    # mu2e wants one absolute path per line, trailing newline included.
    assert file_list.read_text() == "/data/a.art\n/data/b.art\n/data/c.art\n"


# --- where Offline comes from ------------------------------------------------

def test_musing_is_taken_as_a_name_and_a_version():
    for spelling in ("SimJob MDC2025au", "SimJob/MDC2025au"):
        env = Mu2eEnv.for_musing(spelling)
        assert env.musing == ("SimJob", "MDC2025au")
        assert env.describe() == "musing SimJob MDC2025au"
    # and the version is not optional
    for bad in ("SimJob", "SimJob MDC2025au extra"):
        try:
            Mu2eEnv.for_musing(bad)
        except EnvError as exc:
            assert "SimJob MDC2025au" in str(exc)      # shows the spelling wanted
        else:
            raise AssertionError(f"{bad!r} should not be a Musing")


def test_a_musing_sets_itself_up_and_leaves_the_fcl_to_art(tmp_dir):
    env = Mu2eEnv.for_musing("SimJob MDC2025au")
    commands = env.setup_commands(Path(tmp_dir))
    assert commands[-1] == "muse setup SimJob MDC2025au"
    assert any("setupmu2e-art.sh" in c for c in commands)
    # No directory of ours to resolve against: art finds it on FHICL_FILE_PATH,
    # so the relative path is passed through and cannot be pre-checked.
    fcl = Path("Mu2eOptAna/fcl/edep.fcl")
    assert env.base_dir(Path(tmp_dir)) is None
    assert env.resolve_fcl(fcl, Path(tmp_dir)) == fcl
    assert env.missing_fcl(fcl, Path(tmp_dir)) is None


def test_a_shipped_fcl_is_used_as_is_and_checked_under_a_musing(tmp_dir):
    env = Mu2eEnv.for_musing("SimJob MDC2025au")
    fcl = FCL_DIR / "edep.fcl"
    assert env.resolve_fcl(fcl, Path(tmp_dir)) == fcl
    assert env.missing_fcl(fcl, Path(tmp_dir)) is None
    assert "not found" in env.missing_fcl(FCL_DIR / "nope.fcl", Path(tmp_dir))


def test_a_work_area_is_set_up_in_place_and_resolves_its_own_fcl(tmp_dir):
    area = Path(tmp_dir)
    (area / "Mu2eOptAna" / "fcl").mkdir(parents=True)
    (area / "Mu2eOptAna" / "fcl" / "edep.fcl").write_text("# fcl")

    env = Mu2eEnv.for_work_area(area)
    assert env.setup_commands(area) == [
        f"cd {area}",
        "source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh",
        "muse setup",
    ]
    assert env.resolve_fcl(Path("Mu2eOptAna/fcl/edep.fcl")) == area / "Mu2eOptAna/fcl/edep.fcl"
    assert env.missing_fcl(Path("Mu2eOptAna/fcl/edep.fcl")) is None
    # a missing one is caught before any job starts
    assert "not found" in env.missing_fcl(Path("Mu2eOptAna/fcl/nope.fcl"))


def test_a_work_area_that_is_not_a_directory_is_rejected(tmp_dir):
    try:
        Mu2eEnv.for_work_area(Path(tmp_dir) / "nowhere")
    except EnvError as exc:
        assert "not a directory" in str(exc)
    else:
        raise AssertionError("a missing work area should be rejected")


def test_a_tarball_is_unpacked_once_beside_the_job_then_set_up(tmp_dir):
    tarball = Path(tmp_dir) / "code.tar"
    tarball.write_bytes(b"not really a tarball, only its path is used here")
    job = Path(tmp_dir) / "job"

    env = Mu2eEnv.for_tarball(tarball)
    assert env.unpack_dir(job) == job / "code"          # self-contained by default
    commands = env.setup_commands(job)
    unpack, enter = commands[0], commands[1]
    assert str(tarball) in unpack and "tar -xf" in unpack
    assert f"{job / 'code'}.unpacked" in unpack         # the marker that makes it once
    assert str(job / "code") in enter
    assert commands[-1] == "muse setup"

    # a shared unpack directory is used as given, and can be pre-checked
    shared = Mu2eEnv.for_tarball(tarball, code_dir=Path(tmp_dir) / "shared")
    assert shared.unpack_dir(job) == Path(tmp_dir) / "shared"
    assert shared.base_dir() == Path(tmp_dir) / "shared"
    # nothing unpacked yet, so nothing can be said about the fcl
    assert shared.missing_fcl(Path("Mu2eOptAna/fcl/edep.fcl")) is None

    sub = Mu2eEnv.for_tarball(tarball, code_dir=Path(tmp_dir) / "shared",
                              code_subdir="Code")
    assert sub.base_dir() == Path(tmp_dir) / "shared" / "Code"


def test_configured_environment_is_what_analyses_see(tmp_dir):
    before = current()
    try:
        configure(Mu2eEnv.for_musing("SimJob MDC2025au"))
        catalogue = list_analyses()
        assert "musing SimJob MDC2025au" in catalogue.message
        assert catalogue.metadata["environment"] == "musing SimJob MDC2025au"
        edep = catalogue.metadata["analyses"]["edep"]
        # shipped with the server, so the same file under any environment
        assert edep["fcl"] == str(FCL_DIR / "edep.fcl")
        assert edep["fcl_exists"] is True
    finally:
        configure(None)
        assert current().describe() == before.describe()


def test_written_root_files_reports_a_rerun_that_overwrote_its_output(tmp_dir):
    """Rerunning into the same directory must still report the job's output."""
    outdir = Path(tmp_dir)
    stale = outdir / "nts.owner.edep.Run1B.001800_00000000.root"
    stale.write_bytes(b"from an earlier run")
    before = root_snapshot(outdir)

    # nothing touched yet
    assert written_root_files(outdir, before) == []

    # the job overwrites the file it wrote last time, and adds another
    stale.write_bytes(b"from this run, a different size")
    fresh = outdir / "nts.owner.edep.Run1B.001801_00000000.root"
    fresh.write_bytes(b"new this time")
    assert written_root_files(outdir, before) == sorted([str(stale), str(fresh)])


def test_written_root_files_ignores_files_the_job_left_alone(tmp_dir):
    outdir = Path(tmp_dir)
    untouched = outdir / "someone_elses.root"
    untouched.write_bytes(b"not ours")
    before = root_snapshot(outdir)
    (outdir / "notes.txt").write_text("not a ROOT file")   # nor is this
    assert written_root_files(outdir, before) == []


def test_validate_input_paths_flags_only_bad_ones(tmp_dir):
    good = Path(tmp_dir) / "good.art"
    good.touch()
    problems = validate_input_paths([good, Path("rel.art"), Path("/nope/x.art")])
    assert len(problems) == 2
    assert any("not an absolute path" in p and "rel.art" in p for p in problems)
    assert any("does not exist" in p and "x.art" in p for p in problems)


# --- run_analysis argument validation (no job, no computation) ---------------

def test_relative_data_file_is_rejected(tmp_dir):
    result = run_analysis(analysis="edep", data_file="some/relative.art",
                          output_dir=tmp_dir)
    assert result.status == "error"
    assert "absolute" in result.message


def test_missing_data_file_is_rejected(tmp_dir):
    result = run_analysis(analysis="edep", data_file="/nonexistent/nope.art",
                          output_dir=tmp_dir)
    assert result.status == "error"
    assert "does not exist" in result.message


def test_one_bad_path_in_data_files_is_reported(tmp_dir):
    good = Path(tmp_dir) / "good.art"
    good.touch()
    result = run_analysis(analysis="edep",
                          data_files=[str(good), "/nonexistent/nope.art"],
                          output_dir=tmp_dir)
    assert result.status == "error"
    assert "does not exist" in result.message and "nope.art" in result.message
    assert "good.art" not in result.message  # the valid one isn't flagged


def test_neither_input_is_rejected_without_default_inputs(tmp_dir):
    result = run_analysis(analysis="edep", output_dir=tmp_dir)
    assert result.status == "error"
    assert "exactly one" in result.message


def test_both_inputs_at_once_are_rejected(tmp_dir):
    result = run_analysis(analysis="edep", data_file="/a.art",
                          data_files=["/b.art"], output_dir=tmp_dir)
    assert result.status == "error"
    assert "not both" in result.message


def test_root_file_analysis_rejects_a_file_list(tmp_dir):
    result = run_analysis(analysis="approx_ce_sensitivity",
                          data_files=["/a.root", "/b.root"], output_dir=tmp_dir,
                          parameters={"sig_eff": 0.1})
    assert result.status == "error"
    assert "single ROOT file" in result.message


def test_root_file_analysis_rejects_max_events(tmp_dir):
    result = run_analysis(analysis="approx_ce_sensitivity",
                          data_file="/a.root", output_dir=tmp_dir,
                          parameters={"sig_eff": 0.1}, max_events=10)
    assert result.status == "error"
    assert "max_events" in result.message


def test_unknown_analysis_is_rejected(tmp_dir):
    """The Literal enum from the registry must reject unknown names."""
    try:
        run_analysis(analysis="not_an_analysis", data_file="/a.art",
                     output_dir=tmp_dir)
    except Exception:
        return  # pydantic rejected it, as intended
    raise AssertionError("unknown analysis name was not rejected")


def main() -> int:
    import tempfile
    import traceback

    failures = 0
    for name, func in sorted(globals().items()):
        if not name.startswith("test_") or not callable(func):
            continue
        # A fresh directory per test — tests that write filelist.txt must not
        # be visible to the ones asserting it is absent.
        with tempfile.TemporaryDirectory() as tmp_dir:
            kwargs = {"tmp_dir": tmp_dir} if "tmp_dir" in func.__code__.co_varnames else {}
            try:
                func(**kwargs)
                print(f"ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc or '(no message)'}")
            except Exception:
                failures += 1
                print(f"ERROR {name}:\n{traceback.format_exc()}")
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
