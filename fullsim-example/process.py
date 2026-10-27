import hist
import gc
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import argparse
import csv
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb

# refana
from compare import Compare
from count import Count
#from count import compute_ses, expected_upper_limit, convert_event_limit_to_branching
from SensitivityAnalyzer import SensitivityAnalyzer

from analyze import Analyze
from pyutils.pycut import CutManager
from pyutils.pylogger import Logger
from MLSelect import MLSelect

#import LikelihoodAnalyzer
#from fits import Fits
from optimize_momentum import optimize_momentum, optimize_momentum_time
from pyutils.pyprocess import Processor, Skeleton
from pyutils.pyplot import Plot
from pyutils.pyprint import Print
from pyutils.pyselect import Select
from pyutils.pyvector import Vector
from optimize_cuts import optimize_on_combine_result, plot_scan, plot_scan_vs_value, save_csv

class AnaProcessor(Skeleton):
    """custom file processor 
    
    This class inherits from the Skeleton defined in pyutils/pyprocess base class, which provides the 
    basic structure and methods withing the Processor framework 
    """
    def __init__(self, file_list_path, jobs=1, sign="minus", cuts=[], location='disk'):
        """Initialise your processor with specific configuration
        
        This method sets up all the parameters needed for this specific analysis.
        """
        # Call the parent class's __init__ method first
        # This ensures we have all the base functionality properly set up
        super().__init__()

        # Now override parameters from the Skeleton with the ones we need
        self.file_list_path = file_list_path

        self.branches = { 
            "evt" : [
                "run",
                "subrun",
                "event",
                "trig_apr_TrkDe_80m70p",
                "trig_cpr_TrkDe_80m70p",
                "trig_tpr_TrkDe_80m70p"
            ],
            "crv" : [
                "crvcoincs.time",
                "crvcoincs.nHits",
                "crvcoincs.PEs",
                "crvcoincs.timeStart",
                "crvcoincs.timeEnd"
            ],
            "trk" : [
                "trk.nactive", 
                "trk.pdg", 
                "trk.status",
                "trkqual.valid",
                "trkqual.result",
                "trkpid.valid",
                "trkpid.result",
            ],
            "trkfit" : [
                "trksegs",
                "trksegsmc",
                "trksegpars_lh"
            ],
            "trkmc" : [
              "trkmcsim",
              "trkmc.valid"
            ]
        }
        self.tree_path = "EventNtuple/ntuple"
        #self.filelist = "filelist.txt"          # text file containing list of files
        self.use_remote = True     # Use remote file via mdh
        if str(location)  == "local":
          self.use_remote = False
        self.location = str(location)     # File location
        self.max_workers = jobs      # Limit the number of workers
        self.verbosity = 2         # Set verbosity 
        self.use_processes = True  # Use processes rather than threads
        #self.schema = "path"
        
        # Now add your own analysis-specific parameters 

        # Init analysis methods
        # Would be good to load an analysis config here 
        self.analyse = Analyze(verbosity=0, sign=sign, cut_switch=cuts)
            
        # Custom prefix for log messages from this processor
        self.logger = Logger(print_prefix="[AnaProcessor]", verbosity=1)
        self.logger.log("Initialised", "info")
    
    # ==========================================
    # Define the core processing logic
    # ==========================================
    # This method overrides the parent class's process_file method
    # It will be called automatically for each file by the execute method
    def process_file(self, file_name): 
        """Process a single ROOT file
        
        This method will be called for each file in our list.
        It extracts data, processes it, and returns a result.
        
        Args:
            file_name: Path to the ROOT file to process
            
        Returns:
            A tuple containing the histogram (counts and bin edges)
        """
        try:
            # Create a local pyprocess Processor to extract data from this file
            # This uses the configuration parameters from our class
            processor = Processor(
                use_remote=self.use_remote,     # Use remote file via mdh
                location=self.location,         # File location
                verbosity= self.verbosity        # Reduce output in worker threads
            )
            
            # Process the files using multithreading
            data = processor.process_data(
                file_name = file_name,
                branches = self.branches
            )
            
            # ---- Analysis ----            
            results = self.analyse.execute(data, file_name)

            # Clean up
            gc.collect()

            return results 
        
        except Exception as e:
            # Handle any errors that occur during processing
            self.logger.log(f"Error processing {file_name}: {e}", "error")
            return None

    def postprocess(self, results):
        """Combine results from all processed files

        Overrides Skeleton.postprocess() to combine filtered arrays
        and cut flows from individual file results.

        Args:
            results: List of dicts from process_file, each with "filtered_data" and "cut_stats"

        Returns:
            dict: Combined data and cut flow
        """
        if not results:
            return None

        # Combine filtered arrays
        arrays_to_combine = []
        cut_flow_list = []
        for result in results:
            if result is None or len(result) == 0:
                continue
            arrays_to_combine.append(result["filtered_data"])
            cut_flow_list.append(result["cut_stats"])

        combined_data = ak.concatenate(arrays_to_combine) if arrays_to_combine else None

        # Combine cut flows using CutManager
        cut_manager = CutManager(verbosity=0)
        combined_cut_flow = cut_manager.combine_cut_flows(cut_flow_list, format_as_df=False)
        df = cut_manager.format_cut_flow(combined_cut_flow)
        print("================== Total Cut Flow =======================")
        print(df.to_string(index=False))
        df.to_csv("cut_stats.csv", index=False)

        return {
            "combined_data": combined_data,
            "combined_cut_flow": combined_cut_flow
        }

def count_particle_types(data, logger=None):
    """
    Counts the occurrences of different particle types based on
    simulation data, leveraging the properties of Awkward Arrays.

    Args:
        data (ak.Array): An Awkward Array containing simulation data,
                         including 'trkmc' with 'trkmcsim' nested field.
        logger: Optional logger instance for output

    Returns:
        list: A list containing particle type identifiers for each event.
    """
    if logger is None:
        logger = Logger(print_prefix="[count_particle_types]", verbosity=1)

    # Check for empty data
    if ak.num(data['trkmc'], axis=0) == 0:
        logger.log("No events found in the data.", "warning")
        return []

    # Vectorized approach for efficiency using Awkward Array operations
    #  This is generally faster than looping through events individually for large datasets.

    # Get startCode for the first track in each event, handling empty lists
    # Use ak.firsts to safely get the first element or None if the list is empty
    proc_codes = ak.firsts(data['trkmc']['trkmcsim', 'startCode'], axis=1) 
    gen_codes = ak.firsts(data['trkmc']['trkmcsim', 'gen'], axis=1)
    vector = Vector()

    #rhos = vector.get_rho(data['trkmc','trkmcsim'],'pos')
    vec = vector.get_vector(branch=data['trkmc','trkmcsim'],vector_name='pos')
    rhos = vec.rho
    position = ak.firsts(rhos, axis=1) 

    #position = ak.firsts(sim_pos_vec.rho, axis = 1)
    # Use vectorized comparisons and selection for counting
    dio_mask = (proc_codes == 166) & (position <= 75) # Create boolean mask for DIO events
    ipa_mask = (proc_codes == 166) & (position > 75) # Create boolean mask for IPA DIO events
    cem_mask = ((proc_codes == 168)  | (proc_codes == 167)  ) # Create boolean mask for CE events
    cep_mask = ((proc_codes == 176) | (proc_codes == 169) )  # Create boolean mask for CE events
    erpc_mask = (proc_codes == 178)  # Create boolean mask for external RPC events
    irpc_mask = (proc_codes == 179)  # Create boolean mask for internal RPC events
    ermc_mask = (proc_codes == 172)  # Create boolean mask for external RMC events
    irmc_mask = (proc_codes == 171)  # Create boolean mask for internal RMC events
    flate_mask = (proc_codes == 173)  # Create boolean mask for internal flate events
    flateplus_mask = (proc_codes == 174)  # Create boolean mask for internal flate events
    cosmic_mask = ((gen_codes == 44) | (gen_codes == 38))  # Create boolean mask for cosmic events
    #combined_rpc_mask = (proc_codes == 178) |  (proc_codes == 179) # Create boolean mask for all RPC events

    # Combine masks to identify 'other' events
    other_mask = ~(dio_mask | cem_mask | erpc_mask | irpc_mask | cosmic_mask | ipa_mask | irmc_mask | ermc_mask | cep_mask)

    # Initialize particle_count with -2 for 'others'
    particle_count = ak.zeros_like(proc_codes, dtype=int) - 2
    
    # Assign particle types based on masks
    particle_count = ak.where(dio_mask, 166, particle_count)
    particle_count = ak.where(ipa_mask, 0, particle_count)
    particle_count = ak.where(cosmic_mask, -1, particle_count)
    particle_count = ak.where(other_mask, -2, particle_count)
    particle_count = ak.where(irpc_mask, 179, particle_count)
    particle_count = ak.where(erpc_mask, 178, particle_count)
    particle_count = ak.where(irmc_mask, 171, particle_count)
    particle_count = ak.where(ermc_mask, 172, particle_count)
    particle_count = ak.where(cem_mask, 168, particle_count)
    particle_count = ak.where(cep_mask, 176, particle_count)
    particle_count = ak.where(flate_mask, 173, particle_count)
    particle_count = ak.where(flateplus_mask, 174, particle_count)
    #particle_count = ak.where(combined_rpc_mask, 999, particle_count)
    particle_count_return = particle_count
    #particle_count = ak.any(dio_mask, axis=1)
    # Count the occurrences of each particle type
    counts = {
        166: (len(particle_count[ak.any(dio_mask, axis=1)==True])),
        0: (len(particle_count[ak.any(ipa_mask, axis=1)==True])),
        168:  (len(particle_count[ak.any(cem_mask, axis=1)==True])),
        176:  (len(particle_count[ak.any(cep_mask, axis=1)==True])),
        178:  (len(particle_count[ak.any(erpc_mask, axis=1)==True])),
        179:  (len(particle_count[ak.any(irpc_mask, axis=1)==True])),
        171:  (len(particle_count[ak.any(irmc_mask, axis=1)==True])),
        172:  (len(particle_count[ak.any(ermc_mask, axis=1)==True])), 
        173:  (len(particle_count[ak.any(flate_mask, axis=1)==True])),
        174:  (len(particle_count[ak.any(flateplus_mask, axis=1)==True])), 
        -1:  (len(particle_count[ak.any(cosmic_mask, axis=1)==True])),
        -2:  (len(particle_count[ak.any(other_mask, axis=1)==True])),
        #999: (len(particle_count[ak.any(combined_rpc_mask, axis=1)==True])),
    }
      
    # Print the yields to terminal for cross-check
    logger.log("===== MC truth yields for full momentum and time range=====", "info")
    logger.log(f"N_DIO: {counts[166]}", "info")
    logger.log(f"N_IPA: {counts[0]}", "info")
    logger.log(f"N_CEM: {counts[168]}", "info")
    logger.log(f"N_CEP: {counts[176]}", "info")
    logger.log(f"N_eRPC: {counts[178]}", "info")
    logger.log(f"N_iRPC: {counts[179]}", "info")
    logger.log(f"N_eRMC: {counts[171]}", "info")
    logger.log(f"N_iRMC: {counts[172]}", "info")
    logger.log(f"N_flateminus: {counts[173]}", "info")
    logger.log(f"N_flateplus: {counts[174]}", "info")
    logger.log(f"N_cosmic: {counts[-1]}", "info")
    logger.log(f"N_others: {counts[-2]}", "info")
    
    # Now return a 1D list with one element per event corresponding to the primary trk
    #particle_count_return = ak.flatten(particle_count_return, axis=None)
    #    The mask will be True for values that are not -2.
    primary_mask = particle_count_return != -2

    # Apply the mask to the flattened array to select desired elements
    particle_count_return = particle_count_return[primary_mask]
    particle_count_return = [[sublist[0]] for sublist in particle_count_return]
    particle_count_return = ak.flatten(particle_count_return, axis=None)
    logger.log(f"returned particle count length {len(particle_count_return)}", "info")
    
    return particle_count_return


def selectML( files, cuts, locations, columns):
    """ creates features lists in useful form for XGBoost cuts """
    logger = Logger(print_prefix="[selectML]", verbosity=1)

    t0err = []
    active = []
    recomom = []
    recotime = []
    mc_count = []
    closest_crv_diff = []
    nST = []
    nOPA = []
    trkqual = []
    trkpid = []
    
    comparison = Compare()
    for i, fil in enumerate(files):
      ana_processor = AnaProcessor(fil, args.jobs, args.sign, cuts[i], locations[i])
      results = ana_processor.execute()
      combine_result = results["combined_data"]

      # run cat
      mc_count.append(count_particle_types(combine_result, logger))

      selector = Select()

      # select only track front to fit to
      trk_front = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front")

      # did the track intersect the ST?
      has_st  = selector.has_ST(combine_result['trkfit'])

      # combined mask
      trkfit_ent = ak.mask(combine_result['trkfit']["trksegs"], trk_front) #& (no_opa) & (has_st)

      # make vector mag branch
      vector = Vector()
      mom_mag = vector.get_mag(trkfit_ent ,'mom')

      time = ak.nan_to_none(trkfit_ent['time'])
      time = ak.drop_none(time)
      
      
      # plot cut distributions
      test_mask = (trk_front) & (has_st) #& (no_opa)& (has_st)
      
     
      # Get track and coincidence times
      trk_times = combine_result['trkfit']["trksegs"]["time"][test_mask]  # events × tracks × segments
      coinc_times = combine_result["crv"]["crvcoincs.time"]                  # events × coincidences

      coinc_broadcast = coinc_times[:, None, None, :]  # Add dimensions for tracks and segments
      trk_broadcast = trk_times[:, :, :, None]         # Add dimension for coincidences

      # Calculate time differences
      dt = abs(trk_broadcast - coinc_broadcast)
      
      
      nST.append(ak.sum(selector.select_surface(combine_result['trkfit'], surface_name="ST_Foils"), axis=-1))
      nOPA.append(ak.sum(selector.select_surface(combine_result['trkfit'], surface_name="OPA"), axis=-1))
      t0err.append(ak.mask(combine_result['trkfit']["trksegpars_lh"], test_mask)['t0err']) 
      closest_crv_diff.append(dt)
      active.append(ak.mask(combine_result['trk'], test_mask)["trk.nactive"])
      trkqual.append(ak.mask(combine_result['trk'], test_mask)["trkqual.result"])
      trkpid.append(ak.mask(combine_result['trk'], test_mask)["trkpid.result"])
      recotime.append(trkfit_ent['time'])
      recomom.append(mom_mag)
    ML = MLSelect()
    ML.runML(recomom, recotime, t0err, active, nST, nOPA,  closest_crv_diff, mc_count, trkpid, trkqual)
 

def WriteFittedData(data, min_v, max_v):
    """ Write data used in fit to csv (i,mom,time) Note: should be in format useful to BAT"""
    flat_mom = ak.flatten(data, axis = None)
    flat_np = np.array(flat_mom)

    # Create a boolean mask where elements are greater than or equal to 85
    mask = (flat_np >= min_v) & (flat_np < max_v)

    # Use the mask to filter the array and keep only the elements where the mask is True
    filtered_array = flat_np[mask]
    file_path = 'output_data.csv'

    with open(file_path , 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for item in filtered_array:
            csv_writer.writerow([item])

def compare_datasets( files, cuts, locations, columns, signs):
    """
    Allows for different types of comparisons:
    
    1) could compare different files same cuts
    2) could compare same file different cut sets
    
    Args:
      files : list of file lists (.txt files)
      cuts : list of cut switches (True/False of each cut)
      locations : list of locations e.g. tape or disk
      columns : labels for the two things you are comparing eg. [dataset 1, dataset 2]
    """
    logger = Logger(print_prefix="[compare_datasets]", verbosity=1)

    rmax = []
    d0 = []
    tanDip = []
    t0err = []
    active = []
    trkqual = []
    recomom = []
    truemom = []
    mc_count = []
    resolutions = []
    nST = []
    nOPA = []
    originmom = []
    losses = []
    times = []
    crv = []
    trkpid = []
    
    comparison = Compare()
    #fit = Fits()
    for i, fil in enumerate(files):
      ana_processor = AnaProcessor(fil, args.jobs, signs[i], cuts[i], locations[i])
      results = ana_processor.execute()
      combine_result = results["combined_data"]

      # run cat
      mc_count.append(count_particle_types(combine_result, logger))

      selector = Select()
      
      # select only track front to fit to
      trk_front = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front")

      # did the track intersect the ST?
      has_st  = selector.has_ST(combine_result['trkfit'])

      # did the track intersect the OPA?
      no_opa  = selector.has_OPA(combine_result['trkfit'])

      # combined mask
      trkfit_ent = ak.mask(combine_result['trkfit']["trksegs"], trk_front) #combine_result['trkfit']["trksegs"].mask[(trk_front) ] #& (no_opa) & (has_st)
          
      trk_front_mc = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front",branch_name="trksegsmc")
      trkfit_ent_mc = ak.mask(combine_result['trkfit']["trksegsmc"], trk_front_mc)#combine_result['trkfit']["trksegsmc"].mask[(trk_front_mc) ]

      # make vector mag branch
      vector = Vector()
      mom_mag = vector.get_mag(trkfit_ent ,'mom')
      
      # save reconstructed momentum magnitudes for this dataset to CSV
      # write fitted-range data using WriteFittedData
      try:
        WriteFittedData(mom_mag, 95, 110)
      except Exception as e:
        print(f"WriteFittedData failed: {e}")

      #mom_mag = ak.nan_to_none(mom_mag)
      #mom_mag = ak.drop_none(mom_mag)

      time = ak.nan_to_none(trkfit_ent['time'])
      time = ak.drop_none(time)
      
      vector = Vector()
      mom_mag_mc = vector.get_mag(trkfit_ent_mc ,'mom')
      
      # get resolution:
      resolution = comparison.compare_resolution(mom_mag,mom_mag_mc)
      
      # for loss studies:
      origin = ak.mask(combine_result['trkmc']["trkmcsim"] , (combine_result['trkmc']["trkmcsim"]["rank"] == 0) & (combine_result['trkmc']["trkmcsim"]["nhits"] > 0))
      originmom.append((vector.get_mag(origin,'mom')))

      # get resolution:
      resolution = comparison.compare_resolution(mom_mag_mc, mom_mag)
      loss  = comparison.compare_resolution( (vector.get_mag(origin,'mom')), mom_mag_mc)
      
      # plot cut distributions
      test_mask = (trk_front) & (has_st) #& (no_opa)& (has_st)
      
      # for CRV:
      # Get track and coincidence times
      trk_times = combine_result['trkfit']["trksegs"]["time"][trk_front]  # events × tracks × segments
      coinc_times = combine_result["crv"]["crvcoincs.time"]                  # events × coincidences

      coinc_broadcast = coinc_times[:, None, None, :]  # Add dimensions for tracks and segments
      trk_broadcast = trk_times[:, :, :, None]         # Add dimension for coincidences

      # Calculate time differences
      dt = abs(trk_broadcast - coinc_broadcast)
      
      nST.append(ak.sum(selector.select_surface(combine_result['trkfit'], surface_name="ST_Foils"), axis=-1))
      nOPA.append(ak.sum(selector.select_surface(combine_result['trkfit'], surface_name="OPA"), axis=-1))
      rmax.append(ak.mask(combine_result['trkfit']["trksegpars_lh"],test_mask)['maxr']) 
      d0.append(ak.mask(combine_result['trkfit']["trksegpars_lh"],test_mask)['d0']) 
      tanDip.append(ak.mask(combine_result['trkfit']["trksegpars_lh"],test_mask)['tanDip']) 
      t0err.append(ak.mask(combine_result['trkfit']["trksegpars_lh"],test_mask)['t0err']) 
      trkqual.append(ak.mask(combine_result['trk'],test_mask)["trkqual.result"])
      trkpid.append(ak.mask(combine_result['trk'],test_mask)["trkpid.result"])
      active.append(ak.mask(combine_result['trk'],test_mask)["trk.nactive"])
      recomom.append(mom_mag)
      times.append(time)
      losses.append(loss)
      truemom.append(mom_mag_mc)
      resolutions.append(resolution)
      crv.append(dt)
      #print(">>>>> Has Trk Front ", len(mom_mag))
      #print(">>>>>>> Is in Fit region ",len(mom_mag.mask[mom_mag < 115 & mom_mag > 95]))
    #fit.fit_momentum(originmom, 85, 105, opt="poly", label = "Origin Momentum [MeV/c]", compare_labels=columns)
    #fit.fit_momentum(losses, 0.1,2,opt="landau",label = "Origin - True at TrkEnt Momentum [MeV/c]", compare_labels=columns)
    #fit.fit_momentum(resolutions, -1,1,opt="dscb", label = "Reco - True Momentum at TrkEnt [MeV/c]",compare_labels=columns)
    comparison.plot_particle_counts(mc_count, columns)
    
    if signs[i] == "minus":
       startmom = 95
       endmom = 115
    else:
        startmom = 85
        endmom = 92
    comparison.plot_variable(crv, "|DT| [ns]","DT",0,300, [150,150],[150,150], mc_count,columns)
    comparison.plot_variable(nST, "nST","nST",0,15, [1,1],[1,1], mc_count,columns, 15)
    comparison.plot_variable(nOPA, "nOPA","nOPA",0,4, [0,0],[0,0], mc_count,columns,4)
    comparison.plot_variable(rmax, "rmax", "rmax",300,750, [450,450],[680,680], mc_count,columns)
    comparison.plot_variable(d0, "d0", "d0",0, 250, [100,100], [100,100], mc_count,columns)
    comparison.plot_variable(tanDip, "tanDip","tanDip",-1,2.5, [0.557,0.557], [1.0,1.0],mc_count,columns)
    comparison.plot_variable(trkqual, "trkqual", "trkqual", 0,1,[0.2,0.2], [0.2, 0.2], mc_count,columns)
    comparison.plot_variable(trkpid, "trkpid", "trkpid", 0,1,[0.6,0.6], [0.6, 0.6], mc_count,columns)
    comparison.plot_variable(t0err, "t0err","t0err", 0,1, [0.9,0.9],[0.9,0.9], mc_count,columns)
    comparison.plot_variable(active, "nactive","nactive", 0,50, [20,20],[0.9,0.9], mc_count,columns)
    comparison.plot_variable(recomom, "Reconstructed Momentum at TrkEnt [MeV/c]","recomom", startmom, endmom, [103.9,103.9],[105.1,105.1], mc_count,columns)
    comparison.plot_variable(times, "Time at TrkEnt [ns]","time", 0, 1700, [640,640],[1650,1650], mc_count,columns)
    comparison.plot_variable(truemom, "True Momentum at TrkEnt [MeV/c]","truemom", startmom, endmom, [103.9,103.9],[105.1,105.1], mc_count,columns)
    #comparison.plot_resolution(resolutions, "True - Reco at TrkEnt [MeV/c]","recomom", -10,3, columns, density=False)
    # Run simple optimizations for a few key variables per dataset
    features = {
      'rmax': rmax,
      'd0': d0,
      'tanDip': tanDip,
      't0err': t0err,
      'recomom': recomom,
      'trkqual': trkqual,
      'trkpid': trkpid,
      'nST': nST,
      'nOPA': nOPA,
      'crv': crv
    }
    for name, arr_list in features.items():
      for idx, arr in enumerate(arr_list):
        try:
          rows, best = optimize_on_combine_result(arr, mc_count[idx], signal_code=168, background_codes=None, direction='greater', n_steps=200, metric='youden')
          # save CSV and plot
          csv_out = f"optimize_{name}_dataset{idx}.csv"
          png_out = f"optimize_{name}_dataset{idx}.png"
          try:
            save_csv(rows, csv_out)
          except Exception as e:
            print(f"Failed to save CSV {csv_out}: {e}")
          try:
            plot_scan(rows, png_out)
          except Exception as e:
            print(f"Failed to create plot {png_out}: {e}")
          try:
            png_vs = png_out.replace('.png','_vsvalue.png')
            plot_scan_vs_value(rows, png_vs)
          except Exception as e:
            print(f"Failed to create plot {png_vs}: {e}")
          if best is not None:
            print(f"Optimize {name} dataset {idx}: Best threshold {best['threshold']:.6g}, metric {best['metric']:.4g}, TPR {best['tpr']:.3f}, BkgRej {best['bkg_rej']:.3f} (CSV: {csv_out}, PNG: {png_out})")
          else:
            print(f"Optimize {name} dataset {idx}: no best found")
        except Exception as e:
          print(f"Optimization failed for {name} dataset {idx}: {e}")
    
def optimize_cuts( files, cuts, locations, columns, signs):
    """
    optimizes cuts on different variables for different datasets, using the combined results from multiple files.
    """
    logger = Logger(print_prefix="[optimize_cuts]", verbosity=1)

    rmax = []
    d0 = []
    tanDip = []
    t0err = []
    active = []
    trkqual = []
    recomom = []
    truemom = []
    mc_count = []
    resolutions = []
    nST = []
    nOPA = []
    originmom = []
    losses = []
    times = []
    crv = []
    trkpid = []
    
    comparison = Compare()
    #fit = Fits()
    for i, fil in enumerate(files):
      ana_processor = AnaProcessor(fil, args.jobs, signs[i], cuts[i], locations[i])
      results = ana_processor.execute()
      combine_result = results["combined_data"]

      # run cat
      mc_count.append(count_particle_types(combine_result, logger))

      selector = Select()
      
      # select only track front to fit to
      trk_front = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front")

      # did the track intersect the ST?
      has_st  = selector.has_ST(combine_result['trkfit'])

      # did the track intersect the OPA?
      no_opa  = selector.has_OPA(combine_result['trkfit'])

      # combined mask
      trkfit_ent = ak.mask(combine_result['trkfit']["trksegs"], trk_front) #combine_result['trkfit']["trksegs"].mask[(trk_front) ] #& (no_opa) & (has_st)
          
      trk_front_mc = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front",branch_name="trksegsmc")
      trkfit_ent_mc = ak.mask(combine_result['trkfit']["trksegsmc"], trk_front_mc)#combine_result['trkfit']["trksegsmc"].mask[(trk_front_mc) ]

      # make vector mag branch
      vector = Vector()
      mom_mag = vector.get_mag(trkfit_ent ,'mom')

      time = ak.nan_to_none(trkfit_ent['time'])
      time = ak.drop_none(time)
      
      vector = Vector()
      mom_mag_mc = vector.get_mag(trkfit_ent_mc ,'mom')
      
      # get resolution:
      resolution = comparison.compare_resolution(mom_mag,mom_mag_mc)
      
      # for loss studies:
      origin = ak.mask(combine_result['trkmc']["trkmcsim"] , (combine_result['trkmc']["trkmcsim"]["rank"] == 0) & (combine_result['trkmc']["trkmcsim"]["nhits"] > 0))
      originmom.append((vector.get_mag(origin,'mom')))

      # get resolution:
      resolution = comparison.compare_resolution(mom_mag_mc, mom_mag)
      loss  = comparison.compare_resolution( (vector.get_mag(origin,'mom')), mom_mag_mc)
      
      # plot cut distributions
      test_mask = (trk_front) & (has_st) #& (no_opa)& (has_st)
      
      # for CRV:
      # Get track and coincidence times
      trk_times = combine_result['trkfit']["trksegs"]["time"][trk_front]  # events × tracks × segments
      coinc_times = combine_result["crv"]["crvcoincs.time"]                  # events × coincidences

      coinc_broadcast = coinc_times[:, None, None, :]  # Add dimensions for tracks and segments
      trk_broadcast = trk_times[:, :, :, None]         # Add dimension for coincidences

      # Calculate time differences
      dt = abs(trk_broadcast - coinc_broadcast)
      
      nST.append(ak.sum(selector.select_surface(combine_result['trkfit'], surface_name="ST_Foils"), axis=-1))
      nOPA.append(ak.sum(selector.select_surface(combine_result['trkfit'], surface_name="OPA"), axis=-1))
      rmax.append(ak.mask(combine_result['trkfit']["trksegpars_lh"],test_mask)['maxr']) 
      d0.append(ak.mask(combine_result['trkfit']["trksegpars_lh"],test_mask)['d0']) 
      tanDip.append(ak.mask(combine_result['trkfit']["trksegpars_lh"],test_mask)['tanDip']) 
      t0err.append(ak.mask(combine_result['trkfit']["trksegpars_lh"],test_mask)['t0err']) 
      trkqual.append(ak.mask(combine_result['trk'],test_mask)["trkqual.result"])
      trkpid.append(ak.mask(combine_result['trk'],test_mask)["trkpid.result"])
      active.append(ak.mask(combine_result['trk'],test_mask)["trk.nactive"])
      recomom.append(mom_mag)
      times.append(time)
      losses.append(loss)
      truemom.append(mom_mag_mc)
      resolutions.append(resolution)
      crv.append(dt)

    # Run simple optimizations for a few key variables per dataset
    features = {
      'rmax': rmax,
      'd0': d0,
      'tanDip': tanDip,
      't0err': t0err,
      'recomom': recomom,
      'trkqual': trkqual,
      'trkpid': trkpid,
      'nST': nST,
      'nOPA': nOPA,
      'crv': crv
    }
    for name, arr_list in features.items():
      for idx, arr in enumerate(arr_list):
        try:
          rows, best = optimize_on_combine_result(arr, mc_count[idx], signal_code=168, background_codes=None, direction='greater', n_steps=200, metric='youden')
          # save CSV and plot
          csv_out = f"optimize_{name}_dataset{idx}.csv"
          png_out = f"optimize_{name}_dataset{idx}.png"
          try:
            save_csv(rows, csv_out)
          except Exception as e:
            print(f"Failed to save CSV {csv_out}: {e}")
          try:
            plot_scan(rows, png_out)
          except Exception as e:
            print(f"Failed to create plot {png_out}: {e}")
          try:
            png_vs = png_out.replace('.png','_vsvalue.png')
            plot_scan_vs_value(rows, png_vs)
          except Exception as e:
            print(f"Failed to create plot {png_vs}: {e}")
          if best is not None:
            print(f"Optimize {name} dataset {idx}: Best threshold {best['threshold']:.6g}, metric {best['metric']:.4g}, TPR {best['tpr']:.3f}, BkgRej {best['bkg_rej']:.3f} (CSV: {csv_out}, PNG: {png_out})")
          else:
            print(f"Optimize {name} dataset {idx}: no best found")
        except Exception as e:
          print(f"Optimization failed for {name} dataset {idx}: {e}")
    
    
def run_count(args, cuts):
  """ runs count analysis """
  logger = Logger(print_prefix="[run_count]", verbosity=1)
  
  ana_processor = AnaProcessor(args.file, args.jobs, args.sign, cuts, args.loc)
  results = ana_processor.execute()
  combine_result = results["combined_data"]

  # run cat
  # make vector mag branch
  mc_count = count_particle_types(combine_result, logger)

  selector = Select()
  
  # select only track front to fit to
  trk_front = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front")

  # did the track intersect the ST?
  has_st  = selector.has_ST(combine_result['trkfit'])

  # did the track intersect the OPA?
  no_opa  = selector.has_OPA(combine_result['trkfit'])

  # combined mask
  trkfit_ent = ak.mask(combine_result['trkfit']["trksegs"], trk_front) #combine_result['trkfit']["trksegs"].mask[(trk_front)]#ak.mask(combine_result['trkfit']["trksegs"], trk_front)& (no_opa)& (has_st)] 
  
  # Print run/subrun/event for events passing cuts
  test_mask = (trk_front) & (has_st)
  print_passing_events(combine_result, test_mask, output_file="passing_events_count.txt")
  
  # make vector mag branch
  vector = Vector()
  mom_mag = vector.get_mag(trkfit_ent ,'mom')

  mom_mag = ak.nan_to_none(mom_mag)
  mom_mag = ak.drop_none(mom_mag)

  time = ak.nan_to_none(trkfit_ent['time'])
  time = ak.drop_none(time)
  
  # run count
  signal_mom = []
  signal_time = []
  if (str(args.sign) == "minus"):
    # try to find an optimal momentum+time window automatically
    try:
      opt = optimize_momentum_time(
        mom_mag,
        mc_count,
        time,
        mom_start_range=(103.0,105.5),
        mom_step=0.1,
        mom_width=1.0,
        time_start_range=(500.0,700.0),
        time_step=10.0,
        time_end=1650.0,
        signal_code=168,
        plot=True,
        heatmap_out='opt_mom_time_heatmap.png'
      )

      best = opt.get('best')
      if best is not None:
        signal_mom = [best['mom_start'], best['mom_end']]
        signal_time = [best['time_start'], best['time_end']]
        print(f"Optimized windows: mom={signal_mom}, time={signal_time}, S={best['n_signal']}, B={best['n_bkg']}, S/sqrt(B)={best['s_over_sqrtb']:.3f}")
      else:
        signal_mom = [103.9, 105.1]
        signal_time = [640., 1650.]
    except Exception as e:
      print(f"Momentum+time optimization failed: {e}; falling back to default windows")
      signal_mom = [103.9, 105.1]
      signal_time = [640., 1650.]
  elif (str(args.sign) == "plus"):
    signal_mom = [90.0,92.0]
    signal_time = [640., 1650.]
 
  counter = Count(mom_mag, time, mc_count,signal_mom,signal_time, args.sign)
  signal, background = counter.ExtractReco()
  
  signal_SR, background_SR = counter.CheckMCTruth()
  
  exposure = 3.40E+15   # effective stopped muons (or equivalent)
  eff = 0.12    # signal efficiency
  bkg = background_SR        # expected background events in SR
  cl = 0.9            # 90% CL 
  print("Calculating sensitivity with bkg =", bkg, "exposure =", exposure, "efficiency =", eff, "CL =", cl, " assumes 1 month of 3,84 kW beam time and 0.15 signal efficiency")
  get_sensitivity(bkg, exposure, eff, cl)


  
  print("signal SR", signal_SR, "background SR",background_SR)


def print_passing_events(combine_result, cut_mask, output_file="passing_events.txt"):
    """
    Extract and print run/subrun/event for tracks that pass all cuts
    
    Args:
        combine_result: Full combined data from AnaProcessor
        cut_mask: Boolean mask indicating which events/tracks pass all cuts (can be jagged)
        output_file: Output filename for event list
    """
    # Reduce the jagged mask to event level systematically
    # Keep applying ak.any() until we reach 1D (event level)
    event_mask = ak.Array(cut_mask)
    
    # Reduce all dimensions except the first (events)
    while event_mask.ndim > 1:
        event_mask = ak.any(event_mask, axis=-1)
    
    # Now use awkward indexing directly (no numpy conversion)
    runs = combine_result['evt']['run'][event_mask]
    subruns = combine_result['evt']['subrun'][event_mask]
    events = combine_result['evt']['event'][event_mask]
    
    # Flatten to 1D for writing
    runs_flat = ak.flatten(runs, axis=None)
    subruns_flat = ak.flatten(subruns, axis=None)
    events_flat = ak.flatten(events, axis=None)
    
    # Convert to numpy for file writing
    runs_np = np.asarray(runs_flat)
    subruns_np = np.asarray(subruns_flat)
    events_np = np.asarray(events_flat)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("run,subrun,event\n")
        for run, subrun, event in zip(runs_np, subruns_np, events_np):
            f.write(f"{int(run)},{int(subrun)},{int(event)}\n")
    
    print(f"Wrote {len(runs_np)} passing events to {output_file}")
    return runs_np, subruns_np, events_np

def getrecomom(combine_result):
    """Extract reconstructed momentum information
    
    This function extracts and returns the reconstructed momentum
    magnitude from the combined analysis results.
    
    Args:
        combine_result: Combined results from AnaProcessor
        
    Returns:
        mom_mag: Array of reconstructed momentum magnitudes
    """
    selector = Select()
    vector = Vector()
    
    # select only track front to fit to
    trk_front = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front")
    
    # combined mask
    trkfit_ent = ak.mask(combine_result['trkfit']["trksegs"], trk_front)
    
    # make vector mag branch
    mom_mag = vector.get_mag(trkfit_ent, 'mom')
    
    # Plot with data points and error bars
    mom_mag_flat = ak.drop_none(mom_mag)
    mom_mag_flat = np.array(ak.flatten(mom_mag_flat, axis=None))
    
    # Create histogram to get bin contents
    counts, bins = np.histogram(mom_mag_flat, bins=50, range=(95,115))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    errors = np.sqrt(counts)  # SQRT(N) error bars
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(bin_centers, counts, yerr=errors, fmt='o', color='black', ecolor='black', 
                 markersize=6, capsize=4, capthick=1.5, label='Data')
    plt.xlabel('Reconstructed Momentum [MeV/c]')
    plt.ylabel('Counts')
    plt.title('Reconstructed Momentum Distribution')
    plt.xlim(95, 115)
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('recomom_histogram.pdf')
    plt.show()
    
    return mom_mag

def get_sensitivity(bkg, exposure, efficiency, cl):
    """ test sensitivity analyzer works as expected with some dummy inputs """

    import numpy as np
    from SensitivityAnalyzer import SensitivityAnalyzer

    # load table (b, s90)
    data = np.loadtxt("FC.csv", delimiter=",")
    b_tab, s90_tab = data[:,0], data[:,1]

    # example inputs
    #bkg = 114.38
    #exposure = 2.35e15
    #efficiency = 1.0

    # analyzer numeric expected UL (events)
    an = SensitivityAnalyzer(0.0, bkg, exposure=exposure, efficiency=efficiency)
    n = int(round(bkg))            # or use observed SR count
    low_br, high_br = an.run_analysis(n_observed=n, method="FC", cl=0.9)
    print('FC interval (BR):', low_br, high_br)
    an = SensitivityAnalyzer(s_expected=0.0, b_expected=bkg)
    s_up_numeric = an.expected_upper_limit(bkg=bkg, cl=0.9)

    # table (interpolated)
    s_up_table = np.interp(bkg, b_tab, s90_tab)

    # convert to BR
    br_numeric = s_up_numeric / (exposure * efficiency)
    br_table = s_up_table / (exposure * efficiency)

    print("b:", bkg)
    print("s_up (numeric):", s_up_numeric)
    print("s_up (table):  ", s_up_table)
    print("divide this by (exposure * efficiency) to get BR upper limit")
    #print("BR_numeric:", br_numeric)
    #print("BR_table:  ", br_table)

    req = an.required_signal_for_significances(5)
    print("5 sigma required signal:", req[5])#, "BF is ", req[5]/(exposure*efficiency))

# Create an instance of our custom processor
def  main(args):
  """ main driver function to run analysis
  """
  
 

  new = [
    True,  # 0 is_reco_electron
    True,  # 1 has_downstream
    True, # 2 has trk front
    True,  # 3 good_trkqual
    True,  # 4 good_trkpid
    False, # 5 within_t0
    True,  # 6 within_t0err
    True,  # 7 has_hits
    False, # 8 within_lhr_maxl
    False, # 9 within_d0
    False, # 10 within_pitch_angle
    True,  #11 has_st
    True,  #12 no_opa
    True,  #13 no_crv_veto
    True,  #14 no_crv_quality
    True,  #15 no_crv_timewindow
    True,  #16 pz/pt
    True,  #17 triggers
    False,  #18 within_mom_time
    False, #19 early time
    False #20 reflected
  ]

  old = [
    True,  # 0 is_reco_electron
    True,  # 1 has_downstream
    True, # 2 has trk front
    True,  # 3 good_trkqual
    True,  # 4 good_trkpid
    False, # 5 within_t0
    True,  # 6 within_t0err
    True,  # 7 has_hits
    True, # 8 within_lhr_maxl
    True, # 9 within_d0
    True, # 10 within_pitch_angle
    False,  #11 has_st
    False,  #12 no_opa
    True,  #13 no_crv_veto
    True,  #14 no_crv_quality
    True,  #15 no_crv_timewindow
    False,  #16 pz/pt
    True,  #17 triggers
    False,  #18 within_mom_time
    False #19 early time
  ]
  
   # Check if nomc flag is set
  if args.nomc:
    print("Running getrecomom analysis...")
    # Create processor and run analysis
    ana_processor = AnaProcessor(args.file, args.jobs, args.sign, new, args.loc)
    results = ana_processor.execute()
    combine_result = results["combined_data"]

    # Call getrecomom function
    mom_mag = getrecomom(combine_result)
    print(f"Reconstructed momentum data extracted. Shape: {ak.count(mom_mag)}")
    return
  
  if not args.nomc:
    files = [args.file]
    signs = [args.sign]
    locations = [args.loc]
    columns = ["Run1A"]
    #compare_datasets(files, [new], locations, columns, signs)
    run_count(args, new)
    return
  """
  files = [args.file, args.file]
  signs = [args.sign, args.sign]
  locations = [args.loc, args.loc]
  columns = ["SU2020", "New"]
  compare_datasets(files, [old,new], locations, columns, signs)
  """
  
  """
  # run ML test
  files = ["CE_full.txt","MDS3a_1e-13_v2.txt"]
  locations = ["tape", "local"]
  columns = ["nocuts","nocuts"]
  selectML( files, [new,new], locations, columns)
  """
  
def PrintArgs(args):
  """
  prints users input parameters
  """
  print("========= [RefAna/pyCount/process.py]✅  Analyzing with user opts: ===========")
  print("file:", args.file)
  print("number of processes (njobs - optimal is 1 per file):", args.jobs)
  print("verbose: ", args.verbose)

if __name__ == "__main__":
    # list of input arguments, defaults should be overridden
    parser = argparse.ArgumentParser(description='command arguments', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--file", type=str, required=True, help="filename or file list name (text file list,fullpaths)")
    parser.add_argument("--loc", type=str, required=False, default='disk', help="location of files")
    parser.add_argument("--sign", type=str, required="minus", help="sign of the signal being sought in words")
    parser.add_argument("--jobs", type=int, required=False, default=1,help="use if more than one file, should be nfiles")
    parser.add_argument("--nomc", action='store_true', help="run getrecomom function to extract reconstructed momentum")
    parser.add_argument("--verbose", default=1, help="verbose")
    args = parser.parse_args()
    (args) = parser.parse_args()

    # if verbose print the user input
    if(args.verbose > 0):
      PrintArgs(args)
    
    # run main function
    main(args)




