import argparse
import os
import sys
import time
import json
import pickle
from scipy import sparse
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve

import numpy as np
# from keras.callbacks import LearningRateScheduler
from tqdm import tqdm


DIM1 = 50
DIM2 = 25
DIM3 = 34


def get_roc(sig,bkg):
    testY = np.array([1]*len(sig) + [0]*len(bkg))
    predY = np.array(sig+bkg)
    print(testY,predY)
    auc = roc_auc_score(testY, predY)
    fpr, tpr, thr = roc_curve(testY, predY)
    return fpr,tpr,thr,auc


def label_data(signal_images, background_images):
  labels = np.array([1] * len(signal_images) + [0] * len(background_images))
  data = np.concatenate((signal_images, background_images))
  return data, labels


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


def shrink_image(input_image):
  shrink_list = []
  for index, image in enumerate(input_image,0):
    if (np.count_nonzero(image.flatten()) == 0):
      shrink_list.append(index)
  output_image = np.delete(input_image, shrink_list ,0)
  return output_image

def plot_loss(history, save_prefix=''):
  # Loss Curves
  plt.figure(figsize=[8,6])
  plt.plot(history.history['loss'],'r',linewidth=3.0)
  plt.plot(history.history['val_loss'],'b',linewidth=3.0)
  plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss',fontsize=16)
  plt.title('Loss Curves',fontsize=16)
  plt.savefig(save_prefix + "acc.png")
 
def plot_accuracy(history, save_prefix=''):
  # Accuracy Curves
  plt.figure(figsize=[8,6])
  plt.plot(history.history['acc'],'r',linewidth=3.0)
  plt.plot(history.history['val_acc'],'b',linewidth=3.0)
  plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Accuracy',fontsize=16)
  plt.title('Accuracy Curves',fontsize=16)
  plt.savefig(save_prefix + "acc.png")


def plot_roc(my_network, testX, testY, save_prefix):
  predY = my_network.predict_proba(testX)
  print('\npredY.shape = ',predY.shape)
  print(predY[0:10])
  print(testY[0:10])
  auc = roc_auc_score(testY, predY)
  print('\nauc:', auc)
  fpr, tpr, thr = roc_curve(testY, predY)
  plt.plot(fpr, tpr, label = 'auc = ' + str(auc) )
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc="lower right")
  print('False positive rate:',fpr[1], '\nTrue positive rate:',tpr[1])
  plt.savefig(save_prefix + "roc.png")

def load_data(npy_filename):
  startTime = datetime.now()
  with open(npy_filename) as json_data:
    data = pd.read_json(json_data)
    #print datetime.now() - startTime
  return data.values.tolist()

def create_table(file_list, load_strings, dense=False):
  event_dict = {el:[] for el in load_strings}
  for file in tqdm(file_list):
    # print(file)
    try:
      with open(file, 'rb') as f:
        while True:
          try:
            event = pickle.load(f, encoding='latin1')
            for load in load_strings:
              event_dict[load].append(event[load])
          except:
          #except:
            break
    except:
      '''
      do nothing
      '''
  return event_dict

def create_table_zpos(file_list, load_strings, upper=True):
  event_dict = {el:[] for el in load_strings}
  for file in tqdm(file_list):
    try:
      with open(file, 'rb') as f:
        while True:
          try:
            event = pickle.load(f, encoding='latin1')
            if (upper and event["zpos"] < 0) or ((not upper) and event["zpos"] >= 0):
              continue
            for load in load_strings:
              event_dict[load].append(event[load])
          except:
          #except:
            break
    except:
      '''
      do nothing
      '''
  return event_dict

def create_table_energy(file_list, load_strings, low=2.0,high=3.0):
  event_dict = {el:[] for el in load_strings}
  for file in tqdm(file_list):
    try:
      with open(file, 'rb') as f:
        while True:
          try:
            event = pickle.load(f, encoding='latin1')
            if (event["energy"] > high) or (event["energy"] < low):
              continue
            for load in load_strings:
              event_dict[load].append(event[load])
          except EOFerror:
          #except:
            break
    except:
      '''
      do nothing
      '''
  return event_dict

def look_table(file_list, load_strings):
  event_dict = {el:[] for el in load_strings}
  for file in tqdm(file_list):
    with open(file, 'rb') as f:
      event = pickle.load(f, encoding='latin1')
      print(event["event"])
      assert 0
      break
  return event_dict

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

def get_rej(sig,bkg):
    testY = np.array([1]*len(sig) + [0]*len(bkg))
    predY = np.concatenate([np.array(sig).flatten(),np.array(bkg).flatten()],axis=0)
    auc = roc_auc_score(testY, predY)
    fpr, tpr, thr = roc_curve(testY, predY)
    effindex = np.abs(tpr-0.9).argmin()

    return 1 - fpr[effindex]

def get_roc(sig,bkg):
    testY = np.array([1]*len(sig) + [0]*len(bkg))
    predY = np.array(sig+bkg)
    print(testY,predY)
    auc = roc_auc_score(testY, predY)
    fpr, tpr, thr = roc_curve(testY, predY)
    return fpr,tpr,thr,auc

def roc_nhit(nhits, nhitb):
    nhit_tot = nhits + nhitb
    nhits = np.array(nhits)
    nhitb = np.array(nhitb)
    fpr = []
    tpr = []
    for nhit_cut in range(min(nhit_tot), max(nhit_tot)):
        if np.average(nhits) > np.average(nhitb):
            tpr.append(len(nhits[nhits>nhit_cut])/len(nhits))
            fpr.append(len(nhitb[nhitb>nhit_cut])/len(nhitb))
        else:
            tpr.append(len(nhits[nhits<nhit_cut])/len(nhits))
            fpr.append(len(nhitb[nhitb<nhit_cut])/len(nhitb))
    return fpr, tpr


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

