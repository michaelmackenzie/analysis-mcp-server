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
from tools.analyses.edep import parse_edep_summary
from tools.analyses.muon_stop_rate import (CountsError, dataset_description,
                                           dataset_hint, parse_counts,
                                           parse_prescale_filters, stop_rates,
                                           wrong_dataset)
from tools.mu2e_env import EnvError, Mu2eEnv, configure, current
from tools.mu2e_job import (build_input_args, root_snapshot,
                            validate_input_paths, written_root_files)
from tools.spectrum import Kernel, Spectrum
from tools.spec import ParamSpec

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

# One stdout sample per art_files analysis, so the registry test can exercise
# each parser. Add an entry when adding such an analysis.
SAMPLE_STDOUT = {"edep": SAMPLE_EDEP_STDOUT,
                 "muon_stop_rate": SAMPLE_COUNTS_STDOUT}


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


# --- the muon_stop_rate parser -----------------------------------------------

def test_counts_parses_events_gen_events_and_the_target_stop_prescale():
    assert parse_counts(SAMPLE_COUNTS_STDOUT) == {
        "n_events": 735.0,
        "n_gen_events": 12500.0,
        "prescale": 1.0,        # the TargetStop filter's, not PolyStop's 0.001
    }


def test_counts_takes_the_prescale_of_whichever_filter_is_named():
    counts = parse_counts(SAMPLE_COUNTS_STDOUT, "PolyStopPrescaleFilter")
    assert counts["prescale"] == 0.001
    assert counts["n_events"] == 735.0      # the file's events, either way


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
        parse_counts(without)
    except CountsError as exc:
        assert "TargetStopPrescaleFilter" in str(exc)
        assert "IPAStopPrescaleFilter" in str(exc)      # names what it did find
    else:
        raise AssertionError("a file without the filter must be reported")


def test_counts_rejects_output_without_the_generated_event_count():
    without = SAMPLE_COUNTS_STDOUT.replace("GenEventCount total: 12500 events", "")
    try:
        parse_counts(without)
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


def test_wrong_dataset_follows_the_filter_it_is_given():
    stem = "Run1Bak_local0813111400.001800_00000000.art"
    target = Path(f"sim.mmackenz.TargetStops.{stem}")
    poly = Path(f"sim.mmackenz.PolyStops.{stem}")
    unnamed = Path("stops.art")
    # A poly-stop file parses fine and carries the target filter's product, so
    # only the name says it is the wrong input.
    assert wrong_dataset([target, unnamed]) == []
    assert wrong_dataset([target, poly]) == [f"sim.mmackenz.PolyStops.{stem} (PolyStops)"]
    # ... and the check follows the filter: with the poly filter it is the
    # target-stop file that is out of place.
    assert wrong_dataset([target, poly], "PolyStopPrescaleFilter") == [
        f"sim.mmackenz.TargetStops.{stem} (TargetStops)"
    ]
    assert wrong_dataset([target, poly], "OddlyNamedFilter") == []


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


def test_sensitivity_rejects_a_file_without_the_histograms(tmp_dir):
    """A non-EdepAna ROOT file must be reported clearly, not crash."""
    import uproot
    path = Path(tmp_dir) / "empty.root"
    with uproot.recreate(path) as f:
        f["something_else"] = np.histogram(np.zeros(1), bins=2)
    result = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=tmp_dir, parameters={"sig_eff": 0.1})
    assert result.status == "error"
    assert "trk_front_energy" in result.message


# --- the registry (loops over every analysis) --------------------------------

def test_registry_includes_every_analysis():
    assert {"edep", "muon_stop_rate", "approx_ce_sensitivity"} <= set(ANALYSES)


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
        # art_files analyses name an fcl relative to the configured code;
        # root_file ones must not claim one at all
        if spec.input_kind == "art_files":
            assert spec.fcl is not None and not spec.fcl.is_absolute(), name
            assert current().missing_fcl(spec.fcl) is None, \
                f"{name}: {current().missing_fcl(spec.fcl)}"
        else:
            assert spec.fcl is None, f"{name}: root_file analysis should have no fcl"
            assert spec.produced_by, f"{name}: say which analysis produces its input"


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
            assert tuple(parse_edep_summary(sample)) == spec.metrics
        elif name == "muon_stop_rate":
            parsed = stop_rates(parse_counts(sample), upstream_eff=1.0)
            assert tuple(parsed) == spec.metrics


def test_list_analyses_reports_every_registered_analysis():
    result = list_analyses()
    assert result.status == "success"
    catalogue = result.metadata["analyses"]
    assert set(catalogue) == set(ANALYSES)

    edep = catalogue["edep"]
    assert edep["input_kind"] == "art_files"
    assert edep["fcl"].endswith("Mu2eOptAna/fcl/edep.fcl")
    assert edep["fcl_exists"] is True
    assert edep["units"]["avg_calo_edep_per_event_mev"] == "MeV"

    stops = catalogue["muon_stop_rate"]
    assert stops["input_kind"] == "art_files"
    assert stops["fcl"].endswith("Mu2eOptAna/fcl/print_counts.fcl")
    assert stops["fcl_exists"] is True
    assert stops["parameters"]["upstream_eff"]["required"] is True
    assert stops["units"]["stops_per_pot"] == "stops / POT"

    ce = catalogue["approx_ce_sensitivity"]
    assert ce["input_kind"] == "root_file"
    assert ce["produced_by"] == ["edep"]          # chaining is discoverable
    assert ce["parameters"]["sig_eff"]["required"] is True
    assert ce["parameters"]["npot"]["required"] is False
    cosmic = ce["parameters"]["cosmic_rate_per_s_per_mev"]
    assert cosmic["required"] is False
    assert cosmic["default"] == sens.COSMIC_RATE_PER_SECOND_PER_MEV
    # the normalization a result was built on is reported with it
    assert {"npot", "cosmic_rate_per_s_per_mev"} <= set(ce["metrics"])
    assert ce["units"]["npot"] == "POT"
    assert "fcl" not in ce


# --- parameter handling ------------------------------------------------------

def test_params_resolve_defaults_and_validate():
    spec = ANALYSES["approx_ce_sensitivity"]
    resolved = spec.resolve_params({"sig_eff": 0.25})
    assert resolved["sig_eff"] == 0.25
    assert resolved["npot"] == sens.NPOT           # default filled in


def test_missing_required_parameter_is_reported(tmp_dir):
    result = run_analysis(analysis="approx_ce_sensitivity",
                          data_file="/nonexistent/x.root", output_dir=tmp_dir)
    assert result.status == "error"
    assert "sig_eff" in result.message and "missing" in result.message


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
        assert edep["fcl"] == "Mu2eOptAna/fcl/edep.fcl"   # relative, art resolves it
        assert edep["fcl_exists"] is None                 # unknowable from here
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


def test_neither_input_is_rejected(tmp_dir):
    result = run_analysis(analysis="edep", output_dir=tmp_dir)
    assert result.status == "error"
    assert "exactly one" in result.message


def test_both_inputs_at_once_are_rejected(tmp_dir):
    result = run_analysis(analysis="edep", data_file="/a.art",
                          data_files=["/b.art"], output_dir=tmp_dir)
    assert result.status == "error"
    assert "exactly one" in result.message


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
