#=====================================================================================
#        Author: Aobo Li
#        Contact: liaobo77@gmail.com
#        
#        Last Modified: Aug. 29, 2021
#        
#        * This code generates the .dat list of all .picke files.
#        * After running processing_kamland_new_mc.py or processing_sparse_time.py
#          run this code to generate the pickle list. The pickle list is the input to 
#          KamNet.
#=====================================================================================
#!/usr/bin/python
import json
import time
import datetime
import sys
import argparse
import os
import re
import string
import signal
import subprocess
from settings import OUT_DIR, OUT_PICKLE_DIR, TAIL, ROWS, COLS
from tools import cd, append_file

def main():
    # Setting the output Directory if it does not exist.
    if not os.path.exists(OUT_PICKLE_DIR):
            os.mkdir(OUT_PICKLE_DIR )

    '''
    Training combo is a python dict containing types of isotopes to generate pickle list
    Each entry of the python dict takes the form of:
        map[sig] = [bkg1, bkg2, bkg3,...]
    Note that "sig" and every "bkg" string has to be part of the .pickle filename
    '''
    training_combo = {}
    training_combo['Solar'] = ['Bi214m']

    # Reads out all .picke file addresses
    inputfiles = [(ifile) for ifile in os.listdir(OUT_DIR) if ".pickle" in ifile]
    inputfiles.sort()

    filename_array = {}
    # Categorize .picke file addresses into corresponding types of isotopes (sig or bkg)
    for npyfile in inputfiles:
        for sig, bkg in training_combo.iteritems():
            if sig in npyfile:
                filename_array = append_file(sig, str(OUT_DIR + npyfile), filename_array)
            else:
                for single_bkg in bkg:
                    if single_bkg in npyfile:
                        filename_array = append_file(single_bkg, str(OUT_DIR + npyfile), filename_array)
    # Generate the .dat pickle list
    for key in filename_array.keys():
        with cd(OUT_PICKLE_DIR):
            writefile = open(str(key + TAIL +'.dat'),"w")
            for filename in filename_array[key]:
                if os.stat(filename).st_size == 0:
                    # Skip file with 0 size
                    continue
                writefile.write(filename + '\n')
            writefile.close()

if __name__=="__main__":
     main()