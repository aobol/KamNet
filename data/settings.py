#=====================================================================================
#  Author: Aobo Li
#  Contact: liaobo77@gmail.com
#  
#  Last Modified: Sep. 5, 2021
#  
#  * setting files for data batch processing scripts
#=====================================================================================
import os
TAIL = "_trial1"                                                                                     # The use-defined suffix of pickle list, used to distinguish different input pickle lists
OUT_DIR = os.path.join("/projectnb2/snoplus/KLZ_NEWFINAL/machine_learning/", "KamLAND" + TAIL + "/") # Location to store the .pickle files
OUT_PICKLE_DIR = "/projectnb/snoplus/KLZ_NEWFINAL/machine_learning_plist/"                           # Location of the .dat pickle list
INPUT_DIR = "/projectnb/snoplus/KLZ_NEWFINAL/new_ml_data/data-root-KamNET/"                          # Location of the input .root   files
MACRO_DIR = "/projectnb/snoplus/machine_learning/data/shell"                                         # A place to store all the generated shell scripts
TIME = "0:5:00"                                                                                      # Processing time of each shell script
PROCESSOR = "/project/snoplus/ml2/preprocessing/processing_kamland_new.py"                           # The processor we'd like to use
ISOTOPE = ["Solar", "C10p"]                                                                        # Name of the isotopes we'd like to process, this name must be part of the file name of the .root file

# Define the size of each hit maps
COLS = 38
ROWS = COLS

# Define the fiducial volume cutting threshold for event selection
FV_CUT_LOW = 0.0
FV_CUT_HI = 167.0

# Define the energy range for event selection
ELOW = 2.0
EHI  = 3.0

good_hit = True        # If true, use only good PMT hits, otherwise use all PMT hits
only_17inch = False    # If true, use only 17-inch PMTs, otherwise use both 17 and 20 inch pmts
use_charge = False     # If true, register the corresponding charge of each PMT to hit map for each hit, otherwise register 1.0 for each hit
PLOT_HITMAP=False      # Plot flag. If true, plot hit maps of a input event. Note that if this flag is set to True, then the processing script won't process any file.


