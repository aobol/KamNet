###########################
# Author: Aobo Li
############################
# History:
# Jan.22, 2019 - First Version
#################################
# Purpose:
# This code is used to convert MC simulated .root file into a 2D square grid,
# then it saves the code as a CSR sparse matrix in .pickle format.
#############################################################
import argparse
import math
import os
import json
import pickle
from scipy import sparse
from scipy import constants as const
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from random import *
import numpy as np
import time
from ROOT import TFile
from datetime import datetime
from tqdm import tqdm
import kent_distribution as kt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
colormap_normal = cm.get_cmap("OrRd")

#-20,40,3
class clock:

  def __init__(self, initial_time):
    clock.initiated=True
    self.tick_interval = 1.5
    self.final_time = 22
    self.initiated=False
    self.clock_array = np.arange(-20,self.final_time, self.tick_interval)
    self.clock_array = self.clock_array + initial_time
    # init_time = -20
    # final_time = 40
    # tick_interval = 1.5
    # self.clock_array = np.arange(init_time, final_time, tick_interval)
    # # ticks = 40
    # # self.clock_array = np.logspace(np.log10(1) , np.log10(abs(final_time - init_time)) , num=ticks) + init_time

  def tick(self, time):
    if (time <= self.clock_array[0]):
      return 0
    return self.clock_array[self.clock_array <= time].argmax()

  def clock_size(self):
    return (len(self.clock_array))

  def get_range_from_tick(self, tick):
    if tick == 0:
      return -9999, self.clock_array[0]
    return self.clock_array[tick-1], self.clock_array[tick]
