'''
Author: Aobo Li
History:
June 10, 2022 - First Version

Purpose:
This code defines the clock of liquid scintillator detector data.
Simulation file with [pmt_position, hittime, hitcharge] info are 
stored as the spatiotemporal [t, theta, phi] grid, the clock
controls which t index should each hit be stored at.
'''
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
colormap_normal = cm.get_cmap("OrRd")

class clock:

  def __init__(self, initial_time):

    clock.initiated=True
    self.tick_interval = 1.5
    self.final_time = 22
    self.initiated=False
    self.clock_array = np.arange(-20,self.final_time, self.tick_interval)
    self.clock_array = self.clock_array + initial_time

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
