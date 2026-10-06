"""The CE-like track selection: RefAna/pyCount's Analyze.define_cuts.

Every cut is a track-level mask (events x tracks). A cut on a segment
quantity applies at the tracker-front (TT_Front) segment, as in pyCount:
`ak.all(~at_front | condition, axis=-1)`, so a track with no front segment
passes it vacuously and `has_trk_front_seg` is what requires one. An
event-level cut (the trigger) is broadcast to the event's tracks.

The cuts and their order are pyCount's, so the cut flow lines up with its
cut_stats.csv. DEFAULT_CUTS is its current ("new") switch set.
"""

import numpy as np

from .eventntuple import (SID_OPA, SID_ST_FOILS, SID_TT_FRONT, SID_TT_MID,
                          TRIGGERS)

# Thresholds, from pyCount analyze.py.
TRKQUAL_MIN = 0.2
TRKPID_MIN = 0.638
T0_WINDOW_NS = (640.0, 1650.0)
T0ERR_MAX_NS = 0.9
MIN_ACTIVE_HITS = 20
MAXR_WINDOW_MM = (450.0, 680.0)
D0_MAX_MM = 100.0
TANDIP_WINDOW = (0.5577350, 1.0)
CRV_DT_NS = 150.0
# A CRV coincidence counts as high quality with more PEs and hits than this
# and a shorter span; and as in-time when it starts and ends inside the window.
CRV_QUALITY_MIN_PES = 25.0
CRV_QUALITY_MIN_HITS = 15
CRV_QUALITY_MAX_SPAN_NS = 175.0
CRV_TIME_WINDOW_NS = (429.0, 1700.0)
PZ_OVER_PT_WINDOW = (0.5, 1.0)
MOM_RANGE_MEVC = (95.0, 115.0)
EARLY_T0_WINDOW_NS = (0.0, 700.0)

# name -> description, in pyCount's order. The charge cut is named for the
# sign searched for: is_reco_electron for "minus", is_reco_positron for "plus".
CUT_DESCRIPTIONS: dict[str, str] = {
    "is_reco_lepton": "track fit hypothesis is the signal lepton (e- or e+)",
    "has_downstream": "downstream: p_z > 0 at the tracker middle",
    "has_trk_front_seg": "track has a segment at the tracker front (TT_Front)",
    "good_trkpid": f"TrkPID > {TRKPID_MIN:g} (trkpid_min)",
    "good_trkqual": f"TrkQual > {TRKQUAL_MIN:g} (trkqual_min)",
    "within_t0": "{:g} < t0 < {:g} ns at the tracker front".format(*T0_WINDOW_NS),
    "within_t0err": f"t0 error < {T0ERR_MAX_NS:g} ns",
    "has_hits": f"at least {MIN_ACTIVE_HITS} active tracker hits",
    "within_lhr_max": "{:g} < loop-helix max radius < {:g} mm".format(*MAXR_WINDOW_MM),
    "within_d0": f"loop-helix d0 < {D0_MAX_MM:g} mm",
    "within_pitch_angle": "{:.4g} < tan(dip) < {:g}".format(*TANDIP_WINDOW),
    "has_st": "extrapolates to the stopping-target foils",
    "no_opa": "does not extrapolate to the outer proton absorber",
    "no_crv_quality": (f"no high-quality CRV coincidence (PEs > {CRV_QUALITY_MIN_PES:g}, "
                       f"nHits >= {CRV_QUALITY_MIN_HITS}, span < "
                       f"{CRV_QUALITY_MAX_SPAN_NS:g} ns) within {CRV_DT_NS:g} ns"),
    "no_crv_timewindow": ("no CRV coincidence starting after {:g} and ending before "
                          "{:g} ns within {:g} ns").format(*CRV_TIME_WINDOW_NS, CRV_DT_NS),
    "no_crv_veto": f"no CRV coincidence within {CRV_DT_NS:g} ns of the track",
    "pz_over_pt": "{:g} < p_z/p_T < {:g} at the tracker front".format(*PZ_OVER_PT_WINDOW),
    "good_trigger": "event passed all of " + ", ".join(TRIGGERS),
    "in_mom_range": "{:g} < p < {:g} MeV/c at the tracker front".format(*MOM_RANGE_MEVC),
    "within_t0_early": "{:g} < t0 < {:g} ns at the tracker front".format(*EARLY_T0_WINDOW_NS),
    "no_reflected": "not reflected (no up- and downstream front segments both)",
}

# pyCount's "new" switch set — the cuts it applies today.
DEFAULT_CUTS = (
    "is_reco_lepton", "has_downstream", "has_trk_front_seg", "good_trkpid",
    "good_trkqual", "within_t0err", "has_hits", "has_st", "no_opa",
    "no_crv_quality", "no_crv_timewindow", "no_crv_veto", "pz_over_pt",
    "good_trigger",
)


class CutError(ValueError):
    """A cut list that names unknown cuts, worded for the caller."""


def charge_cut_name(sign: str) -> str:
    return "is_reco_electron" if sign == "minus" else "is_reco_positron"


def cut_names(sign: str) -> list[str]:
    """Every cut, as the caller names it for this sign, in order."""
    return [charge_cut_name(sign) if name == "is_reco_lepton" else name
            for name in CUT_DESCRIPTIONS]


def active_cuts(sign: str, enable: str, disable: str) -> list[str]:
    """DEFAULT_CUTS plus `enable` minus `disable` (comma-separated names),
    in cut order."""
    known = cut_names(sign)
    active = {charge_cut_name(sign) if n == "is_reco_lepton" else n
              for n in DEFAULT_CUTS}
    for text, add in ((enable, True), (disable, False)):
        names = [n.strip() for n in text.split(",") if n.strip()]
        unknown = [n for n in names if n not in known]
        if unknown:
            other = {"is_reco_electron", "is_reco_positron"} & set(unknown)
            hint = (f" ({', '.join(other)} is the charge cut for the other "
                    f"sign; with sign='{sign}' it is {charge_cut_name(sign)})"
                    if other else "")
            raise CutError(f"unknown cut(s) {', '.join(unknown)}{hint}. "
                           f"Known: {', '.join(known)}")
        active = active | set(names) if add else active - set(names)
    return [n for n in known if n in active]


def _at_front_all(at_front, condition):
    """Track-level: `condition` holds at every tracker-front segment."""
    import awkward as ak
    return ak.all(~at_front | condition, axis=-1)


def cut_masks(data: dict, sign: str, trkqual_min: float = TRKQUAL_MIN,
              trkpid_min: float = TRKPID_MIN) -> dict:
    """name -> track-level boolean mask for every cut, in cut order."""
    import awkward as ak

    trk = data["trk"]
    segs = data["trkfit"]["trksegs"]
    lh = data["trkfit"]["trksegpars_lh"]
    crv = data["crv"]
    evt = data["evt"]

    sid = segs["sid"]
    at_front = sid == SID_TT_FRONT
    at_mid = sid == SID_TT_MID
    time = segs["time"]
    mom = segs["mom"]["fCoordinates"]
    px, py, pz = mom["fX"], mom["fY"], mom["fZ"]
    p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    pt = np.hypot(px, py)
    up, down = pz < 0, pz > 0

    masks = {}
    masks[charge_cut_name(sign)] = trk["trk.pdg"] == (11 if sign == "minus" else -11)
    masks["has_downstream"] = ak.all(~at_mid | down, axis=-1)
    masks["has_trk_front_seg"] = ak.any(at_front, axis=-1)
    masks["good_trkpid"] = trk["trkpid.result"] > trkpid_min
    masks["good_trkqual"] = trk["trkqual.result"] > trkqual_min
    masks["within_t0"] = _at_front_all(
        at_front, (T0_WINDOW_NS[0] < time) & (time < T0_WINDOW_NS[1]))
    masks["within_t0err"] = _at_front_all(at_front, lh["t0err"] < T0ERR_MAX_NS)
    masks["has_hits"] = trk["trk.nactive"] >= MIN_ACTIVE_HITS
    masks["within_lhr_max"] = _at_front_all(
        at_front, (MAXR_WINDOW_MM[0] < lh["maxr"]) & (lh["maxr"] < MAXR_WINDOW_MM[1]))
    masks["within_d0"] = _at_front_all(at_front, lh["d0"] < D0_MAX_MM)
    masks["within_pitch_angle"] = _at_front_all(
        at_front, (TANDIP_WINDOW[0] < lh["tanDip"]) & (lh["tanDip"] < TANDIP_WINDOW[1]))
    masks["has_st"] = ak.sum(sid == SID_ST_FOILS, axis=-1) > 0
    masks["no_opa"] = ak.sum(sid == SID_OPA, axis=-1) == 0

    # CRV: |t_front - t_coinc| for every front segment against every
    # coincidence in the event: events x tracks x front segments x coincs.
    front_times = time[at_front]
    near = abs(front_times[:, :, :, None] - crv["crvcoincs.time"][:, None, None, :]) < CRV_DT_NS
    start, end = crv["crvcoincs.timeStart"], crv["crvcoincs.timeEnd"]
    quality = ((crv["crvcoincs.PEs"] > CRV_QUALITY_MIN_PES)
               & (crv["crvcoincs.nHits"] >= CRV_QUALITY_MIN_HITS)
               & ((end - start) < CRV_QUALITY_MAX_SPAN_NS))
    in_window = (start > CRV_TIME_WINDOW_NS[0]) & (end < CRV_TIME_WINDOW_NS[1])

    def vetoed(coinc_ok=None):
        hit = near if coinc_ok is None else near & coinc_ok[:, None, None, :]
        return ak.any(ak.any(hit, axis=3), axis=2)

    masks["no_crv_quality"] = ~vetoed(quality)
    masks["no_crv_timewindow"] = ~vetoed(in_window)
    masks["no_crv_veto"] = ~vetoed()

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ak.where(pt > 0, pz / pt, 0.0)
    masks["pz_over_pt"] = _at_front_all(
        at_front, (PZ_OVER_PT_WINDOW[0] < ratio) & (ratio < PZ_OVER_PT_WINDOW[1]))

    triggered = evt[TRIGGERS[0]] == 1
    for name in TRIGGERS[1:]:
        triggered = triggered & (evt[name] == 1)
    masks["good_trigger"] = ak.broadcast_arrays(triggered, trk["trk.pdg"])[0]

    masks["in_mom_range"] = _at_front_all(
        at_front, (MOM_RANGE_MEVC[0] < p) & (p < MOM_RANGE_MEVC[1]))
    masks["within_t0_early"] = _at_front_all(
        at_front, (EARLY_T0_WINDOW_NS[0] < time) & (time < EARLY_T0_WINDOW_NS[1]))
    reflected = ak.any(up & at_front, axis=-1) & ak.any(down & at_front, axis=-1)
    masks["no_reflected"] = ~reflected

    return {name: masks[name] for name in cut_names(sign)}


def apply_cuts(masks: dict, active: list[str], ntracks) -> tuple:
    """(track mask, events passing after each active cut in turn).

    `ntracks`: any events x tracks array, giving the shape when no cut is
    active. The cut flow counts events with at least one track passing every
    cut so far, as pyutils' CutManager.create_cut_flow does.
    """
    import awkward as ak

    combined = ak.ones_like(ntracks, dtype=bool)
    flow = []
    for name in active:
        combined = combined & masks[name]
        flow.append(int(ak.sum(ak.any(combined, axis=-1))))
    return combined, flow
