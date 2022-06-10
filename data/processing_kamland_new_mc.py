#=====================================================================================
#    Author: Aobo Li
#    Contact: liaobo77@gmail.com
#    
#    Last Modified: Aug. 29, 2021
#    
#    * This code is used to convert MC simulated .root file into a 2D square grid
#    * Save each event and other variables as a CSR sparse matrix in .pickle format.
#    * Only applicable to the KLGSim simulation by the KamLAND-Zen group. To use this on your
#      own experiment, please modify this code to adapt to your own MC data structures.
#=====================================================================================
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
import matPLOT_HITMAPlib.gridspec as gridspec
from clock import clock
from tools import *
import matPLOT_HITMAPlib
matPLOT_HITMAPlib.use("Agg")
import matPLOT_HITMAPlib.pyPLOT_HITMAP as plt
from matPLOT_HITMAPlib import cm
from settings import COLS, FV_CUT_LOW, FV_CUT_HI, good_hit, only_17inch, use_charge, ELOW, EHI, PLOT_HITMAP
colormap_normal = cm.get_cmap("cool")
FSIZE = 20
plt.rcParams['font.size'] = FSIZE

# Transcribe hits to a 4D tensor
def transcribe_hits(input, outputdir, PMT_POSITION, elow, ehi):
    current_clock = clock(0)
    f1 = TFile(input)
    tree = f1.Get("nt")  # Read the ROOT tree
    start_evt = 0
    end_evt = tree.GetEntries()
    tz=[]
    if PLOT_HITMAP:
        end_evt = 10000
    input_name = os.path.basename(input).split('.')[0]


    event_map = []
    for evt_index in tqdm(range(start_evt,end_evt)):
        tree.GetEntry(evt_index)
        #FV/ROI cut
        try:
            energy = tree.EnergyA2.   # These are the 
            position = tree.r
            hit = tree.NhitID
            if (energy < ELOW) or (energy > EHI) or (position > FV_CUT_HI) or (position < FV_CUT_LOW):
                continue
        except:
            print("error")
            continue

        if good_hit:
            good_pmt_list = np.array(tree.hitlist_array_good)
            good_pmt_time_list = np.array(tree.time_array_good) 
            good_pmt_charge_list = np.array(tree.charge_array_good)
            event = np.zeros((current_clock.clock_size(),ROWS,COLS))
        else:
            good_pmt_list = np.array(tree.hitlist_array)
            good_pmt_time_list = np.array(tree.time_array)
            good_pmt_charge_list = np.array(tree.charge_array)
            event = np.zeros((current_clock.clock_size(),ROWS,COLS))

        if only_17inch:
            good_index = good_pmt_list<1325
            good_pmt_list = good_pmt_list[good_index]
            good_pmt_time_list = good_pmt_time_list[good_index]
            good_pmt_charge_list = good_pmt_charge_list[good_index]

        vertex = np.array([tree.x/100.0,tree.y/100.0,tree.z/100.0])
        # Calculate time of flight
        tof_array = []
        for pmtid in good_pmt_list:
            tof_array.append(0)
        good_pmt_tof = np.array(tof_array)

        tzero = tree.T0
        total_charge = np.sum(good_pmt_charge_list)

        stacked_pmt_info = np.dstack((good_pmt_list, good_pmt_time_list, good_pmt_charge_list, good_pmt_tof))[0]

        timea = []

        for pmtinfo in stacked_pmt_info:
            if pmtinfo[-2] == 0.0:
                # Skip PMT with 0 charge
                continue
            col, row = xyz_to_row_col(pmtinfo[0], PMT_POSITION)
            t_center = pmtinfo[1] -    tzero
            tz.append(t_center)
            time = current_clock.tick(t_center)
            if use_charge:
                event[time][row][col] += pmtinfo[-2]
            else:
                event[time][row][col] += 1.0

        event_dic = {}
        event_dic['id'] = tree.EventNumber
        event_dic['run'] = tree.run
        event_dic['Nhit'] = np.count_nonzero(event)
        event_dic['energy'] = energy
        event_dic['vertex'] = tree.r
        event_dic['zpos'] = tree.z
        event_dic['event'] = event

        event_map.append(event_dic)
    if PLOT_HITMAP:
        '''
        This is the plot method for given dataset, it plots a few selected hit maps for
        demonstration purpose
        '''
        plt.figure(figsize=(15,15))
        spec = gridspec.GridSpec(ncols=4, nrows=2, height_ratios=[1,2])
        plt.subPLOT_HITMAP(spec[1,:])
        idx_pool = [5,11,14,18]
        plt.hist(tz,bins=np.arange(-20,40,1.5),density=True,color=colormap_normal(0.2))
        plt.axvline(x=-20,color="red",label="KamNet Window")
        plt.axvline(x=22,color="red")
        for idxc in idx_pool:
            begin,end = current_clock.get_range_from_tick(idxc)
            plt.axvspan(xmin=begin,xmax=end,color=colormap_normal(0.7),alpha=0.5)
        plt.ylim(0,0.08)
        plt.legend(frameon=False)
        plt.xlabel("Proper Hit Time [ns]",fontsize=25,labelpad=20)
        plt.ylabel("Normalized Amplitude",fontsize=25,labelpad=20)
        # plt.savefig("th.png",dpi=600)

    with open(os.path.join(outputdir, "eventfile_%s_%.2f_%.2f.pickle" % (input_name, elow, ehi)), 'wb') as handle:
        numev = 0
        print(len(event_map))
        for eventd in event_map:
            evnt = eventd['event']
            eventd['nhit'] = np.count_nonzero(evnt)
            numev += 1
            time_sequence = []
            subPLOT_HITMAP_index = 0
            for idx, maps in enumerate(evnt):
                if PLOT_HITMAP and (idx in idx_pool):
                        ax = plt.subPLOT_HITMAP(spec[0,subPLOT_HITMAP_index ])
                        begin,end = current_clock.get_range_from_tick(idx)
                        if begin == -9999:
                            plt.title("(Past, %.1f ns)"%(end),fontsize=FSIZE)
                        else:
                            plt.title("(%s ns, %.1f ns)"%(begin,end),fontsize=FSIZE)
                        subPLOT_HITMAP_index += 1
                        ax.axes.get_xaxis().set_visible(False)
                        ax.axes.get_yaxis().set_visible(False)
                        ax.imshow(maps,cmap=colormap_normal, norm=matPLOT_HITMAPlib.colors.LogNorm(vmin=0.3, vmax=10.0))
                        # plt.colorbar()
                        if subPLOT_HITMAP_index > 49:
                            break
                time_sequence.append(sparse.csr_matrix(maps)) # Save each event as a CSR sparse matrix
            if PLOT_HITMAP:
                plt.tight_layout()
                plt.savefig("event.png",dpi=600)
                plt.show()
                assert 1==0
            eventd['event'] = time_sequence
            pickle.dump(eventd, handle, protocol=pickle.HIGHEST_PROTOCOL) # dump event into .pickle file
        print("Number of Events: ", numev)
    return 0




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/projectnb2/snoplus/KLZ_NEWFINAL/new_ml_data/data-root-KamNET/root-XeLS-0nu_Xe136run016532-000-000001.root")
    parser.add_argument("--outputdir", default="/projectnb/snoplus/sphere_data/c10_2MeV")
    parser.add_argument("--pmt_file_index", default="/project/snoplus/KamLAND-Zen/base-root-analysis/pmt_xyz.dat")
    parser.add_argument("--pmt_file_size", default="/projectnb/snoplus/machine_learning/prototype/pmt.txt")
    parser.add_argument("--process_index", type=int, default=-1)
    args = parser.parse_args()

    position = PMT_setup(args.pmt_file_index)

    fmc = transcribe_hits(input=args.input, outputdir=args.outputdir, PMT_POSITION = position)





main()

