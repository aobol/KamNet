#=====================================================================================
#  Author: Aobo Li
#  Contact: liaobo77@gmail.com
#  
#  Last Modified: Sep. 5, 2021
#  
#  * Batch processing script for KamNet
#  * Converting the 
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
import shutil
import numpy as np
from settings import OUT_DIR, INPUT_DIR, MACRO_DIR, TIME, PROCESSOR, ISOTOPE

def main(argv):
   refresh = False

   if refresh:
      '''
      If refresh is true, this will delete all files in OUT_DIR, then start re-generating files.
      '''
      print("Delete all files in 5 second!")
      time.sleep(5)

   # Setting the output directory if it does not exist.
   if not os.path.exists(OUT_DIR):
      os.mkdir(OUT_DIR)
   else: 
      if refresh:
         shutil.rmtree(OUT_DIR)
         os.mkdir(OUT_DIR)

   if not os.path.exists(MACRO_DIR):
      os.mkdir(MACRO_DIR)
   else:
      if refresh:
         shutil.rmtree(MACRO_DIR)
         os.mkdir(MACRO_DIR)

   inputfiles = []
   # Reading out isotopes from the input directory, and add their addresses into a list
   for SIG in ISOTOPE:
      inputfiles += [(ifile) for ifile in os.listdir(INPUT_DIR) if SIG in ifile and ".root" in ifile]

   for rootfile in inputfiles:
      # Loading the template to run on the Boston University SCC cluster batch queue
      # In order to run on your cluster batch system, please modify this .sh template and its inputs
      macrotemplate = string.Template(open('process_kamland.sh', 'r').read())
      with cd(MACRO_DIR):
         outputstring = str(OUT_DIR)
         timestring = str(TIME)
         inputstring = str(INPUT_DIR + rootfile)
         macrostring = macrotemplate.substitute(TIME=timestring, INPUT=inputstring, OUTPUT=outputstring, PROCESSOR=PROCESSOR, PROCESSING_UNIT=-1)
         macrofilename = 'shell_%s.sh' % (str(rootfile))
         macro = open(macrofilename,'w')
         macro.write(macrostring)
         macro.close()
         # print(os.path.join(MACRO_DIR, macrofilename))
         try:
            # os.system("source " + os.path.join(MACRO_DIR, macrofilename))
            command = ['qsub', macrofilename]
            process = subprocess.call(command)
         except Exception as error:
            return 0
   return 1


class cd:
   '''
   Context manager for changing the current working directory
   '''
   def __init__(self, newPath):
      self.newPath = newPath

   def __enter__(self):
      self.savedPath = os.getcwd()
      os.chdir(self.newPath)

   def __exit__(self, etype, value, traceback):
      os.chdir(self.savedPath)

if __name__=="__main__":

   print(sys.exit(main(sys.argv[1:])))

