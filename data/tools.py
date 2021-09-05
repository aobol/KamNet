import math
import numpy as np
import json
import re
from settings import *

def PMT_setup(pmt_file_with_index):
  '''
  Reads in PMT positional information according to PMT location file.
  This file is internal to KamLAND collaboration thus cannot be provided here,
  it should follow the format of:

  PMTID  PMT_X[cm]  PMT_Y[cm]  PMT_Z[cm]

  Each line represents a PMT and each field is separated by blank space
  '''
  PMT_POSITION = {}
  for pmt in np.loadtxt(pmt_file_with_index):
    current_pmt_pos = pmt[1:] / 100.0
    PMT_POSITION[int(pmt[0])] = current_pmt_pos
  return PMT_POSITION


def xyz_to_phi_theta(x, y, z):
   phi = math.atan2(y, x)
   r = (x**2 + y**2 + z**2)**.5
   theta = math.acos(z / r)
   return phi, theta

#change directory
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

def tof(pmt_position, vertex_position):

  # ceff = 16.95 #cm/ns From KatLTVertex.cc source code in Kat

  return np.linalg.norm(pmt_position-vertex_position,2) * 100.0/16.95




# Convert the phi theta information to row and column index in 2D grid
def phi_theta_to_row_col(phi, theta, rows=ROWS, cols=COLS):
   # phi is in [-pi, pi], theta is in [0, pi]
   row = min(rows/2 + (math.floor((rows/2)*phi/math.pi)), rows-1)
   row = max(row, 0)
   col = min(math.floor(cols*theta/math.pi), cols-1);
   col = max(col, 0)
   return int(row), int(col)

# Calculating the angle between two input vectors
def calculate_angle(vec1, vec2):
  x1,y1,z1 = vec1
  x2,y2,z2 = vec2
  inner_product = x1*x2 + y1*y2 + z1*z2
  len1 = (x1**2 + y1**2 + z1**2)**0.5
  len2 = (x2**2 + y2**2 + z2**2)**0.5
  return math.acos(float(inner_product)/float(len1*len2))

# Converting Cartesian position to 2D Grid
def xyz_to_row_col(pmt_index, PMT_POSITION,rows=ROWS, cols=COLS):
   x, y, z = tuple(PMT_POSITION[pmt_index])
   return phi_theta_to_row_col(*xyz_to_phi_theta(x, y, z), rows=rows, cols=cols)


# Set up the clocl to start ticking on the first incoming photon of a given events.
def set_clock(tree, evt):
  tree.GetEntry(evt)
  time_array = []
  for i in range(tree.N_phot):
    time_array.append(tree.PE_time[i])
  return clock(np.array(time_array).min())

# Save input file as a .json file.
def savefile(saved_file, appendix, filename, pathname):
    if not os.path.exists(pathname):
     os.mkdir(pathname)
    with cd(pathname):
        with open(filename, 'w') as datafile:
          json.dump(saved_file, datafile)

def calculate_tzero(t, tof, charge):
  return np.sum((t - tof) * charge)/ np.sum(charge)

def append_file(key, val, filename_array):
    if key not in filename_array.keys():
        filename_array[key] = [val]
    else:
        filename_array[key].append(val)
    return filename_array

def get_name(file):
     '''
     Retrieve run number from file name
     '''
     zdab_regex = re.compile(r"^eventfile_sph_out_(.+)_[a-z].*_1k_(\d+)\.\d+\.\d+.pickle$")
     matches = zdab_regex.match(file)
     if matches:
            return str(matches.group(1)), str(matches.group(2))
     else:
            return 0