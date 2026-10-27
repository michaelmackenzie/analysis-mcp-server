import awkward as ak
from pyutils.pyselect import Select
from pyutils.pymcutil import MC
from pyutils.pylogger import Logger
from pyutils.pycut import CutManager
from pyutils.pyvector import Vector
import matplotlib.pyplot as plt

class Analyze:
    """Class to handle analysis functions
    """
    def __init__(self,  verbosity=1, sign="minus", cut_switch=[]):
        """Initialise the analysis handler
        Args:
            verbosity (int, optional): Level of output detail (0: critical errors only, 1: info, 2: debug, 3: deep debug)
        """
        # Verbosity
        self.verbosity = verbosity
        # Start logger
        self.logger = Logger(
            print_prefix="[Analyse]",
            verbosity=self.verbosity
        )
        # Initialise tools
        self.selector = Select(verbosity=self.verbosity)
        self.mcutil = MC(verbosity=self.verbosity)
        # Analysis configuration
        self.logger.log(f"Initialised", "info")
        self.sign = sign
        self.switch = cut_switch



    def has_trk_front_segment(self, trkfit, surface_name="TT_Front"):
        """Return a track-level boolean mask indicating whether each track
        has at least one segment intersecting the requested surface.

        Args:
            trkfit (ak.Array): The `trkfit` branch (events × tracks × segments)
            surface_name (str): Surface to check (default: "TT_Front")

        Returns:
            ak.Array: boolean array with shape (events, tracks)
        """
        seg_mask = self.selector.select_surface(trkfit, surface_name=surface_name)
        return ak.any(seg_mask, axis=-1)

    def define_cuts(self, data, cut_manager):
        """Define analysis cuts

        Note that all cuts here need to be defined at trk level. 

        Also note that the tracking algorthm produces cut for upstream/downstream muon/electrons and then uses trkqual to guess the right one
        trkqual needs to be good before making a selection 
        this is particulary important for the pileup cut, since it needs to be selected from tracks which are above 90% or whatever 

        Args:
            data (ak.Array): data to apply cuts to
            cut_manager: The CutManager instance to use
        """

    
        selector = self.selector

        # Track segments cuts
        try:
    
    
            # Track segments level definition
            at_trk_front = self.selector.select_surface(data["trkfit"], surface_name="TT_Front") 
            at_trk_mid = self.selector.select_surface(data["trkfit"], surface_name="TT_Mid")
            at_trk_back = self.selector.select_surface(data["trkfit"], surface_name="TT_Back")
            in_trk = (at_trk_front | at_trk_mid | at_trk_back)

            # 1. Electron tracks 
            # Reco track fit is electron
            if (str(self.sign) == "minus"):
                is_reco_electron = selector.is_electron(data["trk"])
                data["is_reco_electron"] = is_reco_electron
        
                cut_manager.add_cut(
                name="is_reco_electron", 
                description="Tracks are assumed to be electrons (trk)", 
                mask=(is_reco_electron ),
                active= self.switch[0]
                )

                # Append track-level definition
                data["is_reco_electron"] = is_reco_electron
                # check for multi electron events
                one_reco_electron_per_event = ak.sum(is_reco_electron, axis=-1) == 1
                # Broadcast to track level
                one_reco_electron, _ = ak.broadcast_arrays(one_reco_electron_per_event, is_reco_electron) # this returns a tuple
                # Add cut 
                cut_manager.add_cut(
                    name="one_reco_electron",
                    description="One reco electron / event",
                    mask=one_reco_electron,
                    active=False#self.switch[0]
                )
                # Append for debugging 
                data["one_reco_electron"] = one_reco_electron
                data["one_reco_electron_per_event"] = one_reco_electron_per_event
    
            if (str(self.sign) == "plus"):
                is_reco_positron = selector.is_positron(data["trk"])
                data["is_reco_positron"] = is_reco_positron
        
                cut_manager.add_cut(
                name="is_reco_positron", 
                description="Tracks are assumed to be positron (trk)", 
                mask=(is_reco_positron ),
                active= self.switch[0]
                )

                data["is_reco_positron"] = is_reco_positron

            
              
            # 2. Downstream tracks only through tracker entrance 
            self.logger.log("Defining downstream tracks cut", "max")
            is_downstream = selector.is_downstream(data['trkfit'])
    
            is_downstream = ak.all( ~at_trk_mid | is_downstream, axis=-1) #~in_trk |
            #has_downstream = ak.any(is_downstream, axis=-1)
    
            cut_manager.add_cut(
                name="has_downstream",
                description="Downstream tracks (p_z > 0 through tracker)",
                mask=is_downstream,
                active= self.switch[1] 
            )

            # trksegs-level definition
            data["is_downstream"] = is_downstream
            # trk-level definition
            #data["has_downstream"] = has_downstream


    
            
            # MC truth electron cut: trkmc.pdg == 11
            # trkmcsim is a nested list per track (sim entries) so reduce to track-level
            """"
            trkmcsim_pdg = data["trkmc"]["trkmcsim"]["pdg"]
            trkmcsim_is_electron = ak.any(trkmcsim_pdg == 11, axis=-1)
            data["truth_electron"] = trkmcsim_is_electron
            active_trkmcsim = False
            
            cut_manager.add_cut(
                name="truth_electron",
                description="MC truth: electron (trkmc.pdg == 11)",
                mask=trkmcsim_is_electron,
                active= active_trkmcsim
            )
            """

            # Track-level: at least one segment intersects the tracker front
            has_trk_front_seg = self.has_trk_front_segment(data['trkfit'], surface_name="TT_Front")
            data["has_trk_front_seg"] = has_trk_front_seg
            try:
                active_has_trk_front = self.switch[2]
            except Exception:
                active_has_trk_front = False
            cut_manager.add_cut(
                name="has_trk_front_seg",
                description="Track has >=1 segment intersecting TT_Front",
                mask=has_trk_front_seg,
                active=active_has_trk_front
            )
        
            # New TrkPID
            good_trkpid = selector.select_trkpid(data["trk"], value=0.638)
            cut_manager.add_cut(
                name="good_trkpid",
                description="Track PID > 0.638",
                mask=good_trkpid,
                active= self.switch[3] 
            )
            data["good_trkpid"] = good_trkpid

            # Track fit quality
    
            good_trkqual = selector.select_trkqual(data["trk"], quality=0.2)
            cut_manager.add_cut(
                name="good_trkqual",
                description="Track quality  > 0.2",
                mask=good_trkqual,
                active= self.switch[4] 
            )
            data["good_trkqual"] = good_trkqual
    
            
    
            # 5. trksegs level
            within_t0 = ((640 < data['trkfit']["trksegs"]["time"]) & 
                            (data['trkfit']["trksegs"]["time"] < 1650))

            # trk-level definition (the actual cut)
            within_t0 = ak.all(~at_trk_front | within_t0, axis=-1)
            cut_manager.add_cut( 
                name="within_t0",
                description="t0 at tracker mid (500 < t_0 < 1650 ns)",
                mask=within_t0,
                active= self.switch[5]
            )
    
            #6. Loop helix track time err
            within_t0err = ((data['trkfit']["trksegpars_lh"]["t0err"])  < 0.9)

            # trk-level definition (the actual cut)
            within_t0err = ak.all(~at_trk_front | within_t0err, axis=-1)
            cut_manager.add_cut(
                name="within_t0err",
                description="t0err < 0.9",
                mask=within_t0err,
                active= self.switch[6]
            )

            # 4. Minimum hits
            has_hits = selector.has_n_hits(data["trk"], n_hits=20)
            cut_manager.add_cut(
                name="has_hits",
                description="Minimum of 20 active hits in the tracker",
                mask=has_hits ,
                active= self.switch[7]
            )


    
            # 7. Loop helix maximum radius
            within_lhr_max = ((450 < data['trkfit']["trksegpars_lh"]["maxr"]) & 
                                (data['trkfit']["trksegpars_lh"]["maxr"] < 680)) # changed from 650

            # trk-level definition (the actual cut)
            within_lhr_max = ak.all(~at_trk_front | within_lhr_max, axis=-1)
            cut_manager.add_cut(
                name="within_lhr_max",
                description="Loop helix maximum radius (450 < R_max < 680 mm)",
                mask=within_lhr_max,
                active= self.switch[8]
            )
    
            # 8. Distance from origin

            within_d0 = (data['trkfit']["trksegpars_lh"]["d0"] < 100)

            # trk-level definition (the actual cut)
            within_d0 = ak.all(~at_trk_front | within_d0, axis=-1) 
            cut_manager.add_cut(
                name="within_d0",
                description="Distance of closest approach (d_0 < 100 mm)",
                mask=within_d0,
                active= self.switch[9] 
        
            )
    
    
            # 9. Pitch angle
            within_pitch_angle = ((0.5577350 < data['trkfit']["trksegpars_lh"]["tanDip"]) & 
                                    (data['trkfit']["trksegpars_lh"]["tanDip"] < 1.0))

            # trk-level definition (the actual cut) 
            within_pitch_angle = ak.all(~at_trk_front | within_pitch_angle, axis=-1)
            cut_manager.add_cut(
                name="within_pitch_angle",
                description="Extrapolated pitch angle (0.5577350 < tan(theta_Dip) < 1.0)",
                mask=within_pitch_angle,
                active= self.switch[10]
            )
    
            
    
            # 11. New ST selection
            has_st  = selector.has_ST(data['trkfit'])
            cut_manager.add_cut(
                name="has_st",
                description="has Nst > 0",
                mask=has_st,
                active= self.switch[11]
            )
    
            # 12. New OPA veto
            no_OPA = selector.has_OPA(data['trkfit'])
            cut_manager.add_cut(
                name="no_opa",
                description="has N_opa == 0",
                mask=no_OPA,
                active= self.switch[12]
            )
    

                
            # 10. CRV veto: |dt| < 150 ns (dt = coinc time - track t0) 
            # Check if EACH track is within 150 ns of ANY coincidence 

            dt_threshold = 150
    
            # Get track and coincidence times
            trk_times = data['trkfit']["trksegs"]["time"][at_trk_front]  # events × tracks × segments
            coinc_times = data["crv"]["crvcoincs.time"]                  # events × coincidences
    
            # Broadcast CRV times to match track structure, so that we can compare element-wise
            # FIXME: should use ak.broadcast
            coinc_broadcast = coinc_times[:, None, None, :]  # Add dimensions for tracks and segments
            trk_broadcast = trk_times[:, :, :, None]         # Add dimension for coincidences

            # Calculate time differences
            dt = abs(trk_broadcast - coinc_broadcast)
    
            # Check if within threshold
            within_threshold = dt < dt_threshold
            """
            fig, (ax1) = plt.subplots(1,1)
            n,bins,patch = plt.hist(ak.flatten(dt, axis=None), bins=50, range=(0,300), histtype='bar', color='red')
            plt.yscale('log')
            ax1.set_xlabel(r'$| T_{trk} - T_{crv}| [ns]$',fontsize=16)
            plt.show()
            """
            # Basic coincidence (used for veto): any coincidence within dt threshold
            any_coinc = ak.any(within_threshold, axis=3)

            # Additionally compute a separate CRV-quality flag (PEs, nHits, span)
            try:
                pe = data["crv"]["crvcoincs.PEs"]
                nh = data["crv"]["crvcoincs.nHits"]
                ts = data["crv"]["crvcoincs.timeStart"]
                te = data["crv"]["crvcoincs.timeEnd"]
                quality = (pe > 25) & (nh >= 15) & ((te - ts) < 175)
                any_coinc_quality = ak.any(within_threshold & quality[:, None, None, :], axis=3)

                # CRV coincidence time-window selection: startTime > 429 and endTime < 1700
                timewindow = (ts > 429) & (te < 1700)
                any_coinc_timewindow = ak.any(within_threshold & timewindow[:, None, None, :], axis=3)

            except Exception:
                any_coinc_quality = ak.zeros_like(any_coinc)
                any_coinc_timewindow = ak.zeros_like(any_coinc)

            
            # Now add a separate CRV quality cut: pass events with NO high-quality coincidences
            quality_veto = ak.any(any_coinc_quality, axis=2)
            data["no_crv_quality"] = ~quality_veto
            try:
                active_quality = self.switch[14]
            except Exception:
                active_quality = False
            cut_manager.add_cut(
                name="no_crv_quality",
                description="No high-quality CRV coincidence (PEs>25, nHits>=15, span<175ns)",
                mask=~quality_veto,
                active= active_quality
            )
            # CRV time-window veto: no coincidences with start>429 and end<1700 within dt threshold
            try:
                timewindow_veto = ak.any(any_coinc_timewindow, axis=2)
            except Exception:
                timewindow_veto = ak.zeros_like(ak.any(any_coinc, axis=2))
            data["no_crv_timewindow"] = ~timewindow_veto
            try:
                active_timewindow = self.switch[15]
            except Exception:
                active_timewindow = False
            cut_manager.add_cut(
                name="no_crv_timewindow",
                description="No CRV coincidence with start>429 and end<1700 within dt threshold",
                mask=~timewindow_veto,
                active= active_timewindow
            )

            # Then reduce over trks (axis=2)
            veto = ak.any(any_coinc, axis=2)
            data["no_crv_veto"] = ~veto
            try:
                active_veto = self.switch[13]
            except Exception:
                active_veto = False
            cut_manager.add_cut(
                name="no_crv_veto",
                description="No crv-trk veto: |dt| >= 150 ns",
                mask=~veto,
                active= active_veto
            )

            # 13. pz/pt cut: compute pz/pt robustly using pyutils.Vector
            try:
                vec = Vector(verbosity=0)
                # restrict to tracker-front segments for vector creation
                trkfit_ent = ak.mask(data['trkfit']["trksegs"], at_trk_front)
                vec3 = vec.get_vector(trkfit_ent, 'mom')
                if vec3 is None:
                    raise Exception("failed to create momentum vector")

                px = vec3.x
                py = vec3.y
                pz = vec3.z
                pt = vec3.rho

                # per-segment ratio (guard against division by zero)
                pz_over_pt = ak.where(pt > 0, pz / pt, ak.zeros_like(pt))

            except Exception:
                # fallback to zeros with same shape as trk segments
                pz_over_pt = ak.zeros_like(data['trkfit']["trksegs"]["time"])

            # Reduce segment-level ratio to a track-level mask: require 0.5 < pz/pt < 1.0
            try:
                mask_seg = (pz_over_pt > 0.5) & (pz_over_pt < 1.0)
                mask_pzpt = ak.all(~at_trk_front | mask_seg, axis=-1)
            except Exception:
                mask_pzpt = ak.zeros_like(ak.any(~at_trk_front, axis=-1))

            # Store numeric per-segment ratio for debugging/inspection
            data["pz_over_pt"] = pz_over_pt

            cut_manager.add_cut(
                name="pz_over_pt",
                description="Track-level cut: 0.5 < pz/pt < 1.0 (pt = transverse mag)",
                mask=mask_pzpt,
                active= self.switch[16]
            )

            # 11. Trigger test
            good_trigger = selector.get_triggers(data["evt"], ["trig_cpr_TrkDe_80m70p","trig_apr_TrkDe_80m70p","trig_tpr_TrkDe_80m70p"])
            cut_manager.add_cut(
                name="good_trigger",
                description="trigger passed",
                mask=good_trigger,
                active= self.switch[17]
            )
            data["good_trigger"] = good_trigger
            
            # momentum selection
            vector = Vector()
            trkfit_ent = ak.mask(data['trkfit']["trksegs"], at_trk_front)
            mom_mag = vector.get_mag(trkfit_ent, 'mom')
            in_mom_range = ((95 < mom_mag) & (mom_mag < 115))
            in_mom_range = ak.all(~at_trk_front | in_mom_range, axis=-1)
            cut_manager.add_cut(
                name="in_mom_range",
                description=" 95 < mom < 115",
                mask=in_mom_range,
                active=self.switch[18]
            )

            within_t0_early = ((0 < data['trkfit']["trksegs"]["time"]) & 
                         (data['trkfit']["trksegs"]["time"] < 700))
        
            # trk-level definition (the actual cut)
            within_t0_early = ak.all(~at_trk_front | within_t0_early, axis=-1)
            cut_manager.add_cut( 
                name="within_t0_early",
                description="t0 at tracker mid (0 < t_0 < 700 ns)",
                mask=within_t0_early,
                active= self.switch[19]
            )

            # Reflection veto cut using pyutils.pyselect
            # Veto events with reflected tracks (has loop helix fit)
            is_reflected = selector.is_reflected(data['trkfit'])
            cut_manager.add_cut(
                name="no_reflected",
                description="Veto tracks with reflections (no loop helix fit)",
                mask=~is_reflected,
                active= self.switch[20]  # Disabled by default, can be enabled via cut_switch
            )
    
            """
            
    
    
            # MC tests
            good_mc = self.mcutil.is_muon(data["trkmc"])
            cut_manager.add_cut(
                name="test",
                description="test",
                mask=good_mc,
                active= True
            )
            data["good_mc"] = good_mc
    
            self.logger.log("All cuts defined", "success")
            """
        except Exception as e:
            self.logger.log(f"Error defining cuts: {e}", "error") 
            return None  


    def apply_cuts(self, data, cut_manager, group=None, active_only=True):

        ## data_cut needs to be an awkward array 

        """Apply all trk-level mask to the data

        Args:
            data: Data to apply cuts to
            mask: Mask to apply 
    
        Returns:
            ak.Array: Data after cuts applied
        """
        self.logger.log("Applying cuts to data", "info")

        try:
            #check mc truth codes before cuts
            #mc_parts = self.mc_pre_cuts(data)
    
            # Copy the array 
            # This is memory intensive but the easiest solution for what I'm trying to do
            data_cut = ak.copy(data) 
    
            # Combine cuts
            self.logger.log(f"Combining cuts", "info") 

            # Track-level mask
            trk_mask = cut_manager.combine_cuts(active_only=active_only)
    
            # Select tracks
            self.logger.log("Selecting tracks", "max")
            data_cut['trk'] = data_cut["trk"][trk_mask]
            data_cut['trkfit'] = data_cut['trkfit'][trk_mask]
            data_cut['trkmc'] = data_cut["trkmc"][trk_mask]

            # Then clean up events with no tracks after cuts
            self.logger.log(f"Cleaning up events with no tracks after cuts", "max") 
            data_cut = data_cut[ak.any(trk_mask, axis=-1)] 
    
            self.logger.log(f"Cuts applied successfully", "success")
    
            return data_cut
    
        except Exception as e:
            self.logger.log(f"Error applying cuts: {e}", "error") 
            return None
    
    # Helper to convert the cut stats into a list 
    def get_stats_list(self, results):
        stats = [] 
        if isinstance(results, list): 
            for result in results: 
                if "cut_stats" in result: 
                    stats.append(result["cut_stats"])
        else: 
            stats.append(results["cut_stats"])
        return stats

    def execute(self, data, file_id,  inactive_cuts=None):
        """Perform complete analysis on an array
        Args:
            data: The data to analyse
            file_id: Identifier for the file
            cut_names: List of cuts to activate/deactivate
            active: activate/deactive cuts
        Returns:
            dict: Complete analysis results
        """
        self.logger.log(f"Beginning analysis execution for file: {file_id}", "info")
        try:

            # Create a unique cut manager for this file
            cut_manager = CutManager(verbosity=self.verbosity)

            self.logger.log("Defining cuts", "max")
            # Define cuts
            self.define_cuts(data, cut_manager)

            # Set activate cuts
            if inactive_cuts: 
                cut_manager.toggle_cut(inactive_cuts, active=False)
    
            # Calculate cut stats
            self.logger.log("Getting cut stats", "max")
            cut_stats = cut_manager.create_cut_flow(data)
        
            # Mark CE-like tracks (useful for debugging 
            data["CE_like"] = cut_manager.combine_cuts(active_only=True)
            # Apply cuts
            data_CE = self.apply_cuts(data, cut_manager) # Just CE-like tracks 
            
            # Compile all results
            self.logger.log("Analysis completed", "success")


            result = {
                "cut_stats": cut_stats,
                "filtered_data": data_CE
            }

            return result
    
        except Exception as e:
            self.logger.log(f"Error during analysis execution: {e}", "error")  
            return None, None
