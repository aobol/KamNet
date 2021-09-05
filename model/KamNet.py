#=====================================================================================
#  Author: Aobo Li
#  Contact: liaobo77@gmail.com
#  
#  Last Modified: Aug. 29, 2021
#  
#  * KamNet is a deep learning model developed for KamLAND-Zen and 
#    other spherical liquid scintillator detectors.
#  * It attempts to harness all of the inherent symmetries to produce a
#    state-of-the-art algorithms for a spherical liquid scintillator detector.
#=====================================================================================
# pylint: disable=E1101,R,C
import numpy as np
import os
import argparse
import time
import math
import random
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
from s2cnn import so3_near_identity_grid, so3_equatorial_grid
from s2cnn import s2_near_identity_grid, s2_equatorial_grid
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pickle
import numpy as np
import copy
from torch.autograd import Variable
from torchsummary import summary
from scipy import sparse
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchsnooper
import pytorch_warmup as warmup
from matplotlib import cm
colormap_normal = cm.get_cmap("cool")
from torch.cuda.amp import autocast
from tqdm import tqdm

from AttentionConvLSTM import ConvLSTM
from KamNetDataset import DetectorDataset, DetectorDataset_Nhit, DetectorDataset_NonUniform, DetectorDatasetRep
from settings import SEED, NUM_EPOCHS, BATCH_SIZE, FILE_UPPERLIM, KAMNET_PARAMS, LEARNING_RATE, EV_SUFFIX, DSIZE
from tool import get_roc, get_rej, roc_nhit, cd

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if SEED:
    '''
    Setting reproducability. If SEED=True, then training the neural network with
    the same configuration will result in exactly the same output
    '''
    manualSeed = 7

    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are using GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)


    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(batch_size):
  '''
  Load datasets from various pickle list
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument("--signallist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/kamdata/XeLS-Solar.pickle%s.dat"%(EV_SUFFIX))
  parser.add_argument("--bglist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/kamdata/Bi214_210320.pickle%s.dat"%(EV_SUFFIX))
  parser.add_argument("--datalist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/kamdata/XeLS-Solar.pickle%s.dat"%(EV_SUFFIX))
  parser.add_argument("--datablist", type = str, default = "/projectnb/snoplus/machine_learning/data/training_log/kamdata/XeLS-I132.pickle%s.dat"%(EV_SUFFIX))
  parser.add_argument("--signal", type = str, default = "Te130")
  parser.add_argument("--bg", type = str, default = "C10")
  parser.add_argument("--outdir", type = str, default = "/projectnb/snoplus/sphere_data/Xe136_C10_torch_new/")
  parser.add_argument("--time_index", type = int, default = 8)
  parser.add_argument("--qe_index", type = int, default = 10)
  parser.add_argument("--elow", type = float, default = 2.0)
  parser.add_argument("--ehi", type = float, default = 3.0)

  args = parser.parse_args()

  save_prefix = "/project/snoplus/ml2/network/"

  # This is used when we perform pressume map study
  time_index = args.time_index
  qe_index = args.qe_index
  json_name = str(time_index) + '_' + str(qe_index)
  # This is used when we train KamNet with MC
  json_name = "event"

  # Read out each pickle list as a list of address
  signal_images_list = [str(filename.strip()) for filename in list(open(args.signallist, 'r')) if filename != '']
  bkg_image_list = [str(filename.strip()) for filename in list(open(args.bglist, 'r')) if filename != '']
  data_list = [str(filename.strip()) for filename in list(open(args.datalist, 'r')) if filename != '']
  datab_list = [str(filename.strip()) for filename in list(open(args.datablist, 'r')) if filename != '']
  signal_images_list = signal_images_list[:FILE_UPPERLIM]
  bkg_image_list = bkg_image_list[:FILE_UPPERLIM]
  data_list = data_list[:FILE_UPPERLIM]
  datab_list = datab_list[:FILE_UPPERLIM]

  # Add different types of backgrounds to the bkg_image_dict, used for verifying KamNet on other background events
  bkg_image_dict = {
    "Sb118":[str(filename.strip()) for filename in list(open("/projectnb/snoplus/machine_learning/data/training_log/kamdata/XeLS-Sb118.pickle%s.dat"%(EV_SUFFIX), 'r'))][:FILE_UPPERLIM],
    "I122":[str(filename.strip()) for filename in list(open("/projectnb/snoplus/machine_learning/data/training_log/kamdata/XeLS-I122.pickle%s.dat"%(EV_SUFFIX), 'r'))][:FILE_UPPERLIM],
    "I124":[str(filename.strip()) for filename in list(open("/projectnb/snoplus/machine_learning/data/training_log/kamdata/XeLS-I124.pickle%s.dat"%(EV_SUFFIX), 'r'))][:FILE_UPPERLIM],
    "I130":[str(filename.strip()) for filename in list(open("/projectnb/snoplus/machine_learning/data/training_log/kamdata/XeLS-I130.pickle%s.dat"%(EV_SUFFIX), 'r'))][:FILE_UPPERLIM],
    "I132":[str(filename.strip()) for filename in list(open("/projectnb/snoplus/machine_learning/data/training_log/kamdata/XeLS-I132.pickle%s.dat"%(EV_SUFFIX), 'r'))][:FILE_UPPERLIM],
    "Bi214-MC":[str(filename.strip()) for filename in list(open("/projectnb/snoplus/machine_learning/data/training_log/kamdata/XeLS-Bi214m.pickle%s.dat"%(EV_SUFFIX), 'r'))][:FILE_UPPERLIM],
    "Bi214-film":[str(filename.strip()) for filename in list(open("/projectnb/snoplus/machine_learning/data/training_log/kamdata/film-Bi214m.pickle%s.dat"%(EV_SUFFIX), 'r'))][:FILE_UPPERLIM],
    # "C10p":[str(filename.strip()) for filename in list(open("/projectnb/snoplus/machine_learning/data/training_log/kamdata/XeLS-C10p.pickle%s.dat"%(EV_SUFFIX), 'r'))][:FILE_UPPERLIM],
    # "C10OP":[str(filename.strip()) for filename in list(open("/projectnb/snoplus/machine_learning/data/training_log/kamdata/XeLS-C10OP.pickle%s.dat"%(EV_SUFFIX), 'r'))][:FILE_UPPERLIM],
    }

  # Read out detector events to verify KamNet's performance
  event_list = [str(filename.strip()) for filename in list(open("/projectnb/snoplus/machine_learning/data/training_log/kamdata/DB_untagged.pickle%s.dat"%(EV_SUFFIX), 'r')) if filename != '']
  dataset = DetectorDataset_Nhit(data_list[:FILE_UPPERLIM], datab_list[:FILE_UPPERLIM], str(json_name),dsize=DSIZE,bootstrap=False)
  validation_split = .3
  shuffle_dataset = True
  random_seed= 42222


  division = 2
  dataset_size = int(len(dataset)/division)
  indices = list(range(dataset_size))
  split = int(np.floor(validation_split * dataset_size))
  if shuffle_dataset :
    # Shuffle the dataset
    np.random.seed(random_seed)
    np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]

  train_indices += list(division*dataset_size - 1-np.array(train_indices))
  val_indices += list(division*dataset_size- 1-np.array(val_indices))

  np.random.shuffle(train_indices)
  np.random.shuffle(val_indices)

  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)

  rtq_dataset = DetectorDatasetRep(signal_images_list[:FILE_UPPERLIM], bkg_image_dict, str(json_name))
  test_dataset = DetectorDataset_NonUniform(event_list[:FILE_UPPERLIM], bkg_image_list[:FILE_UPPERLIM], str(json_name))

  # Convert dataset to data loader
  train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,  drop_last=True)
  eval_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,  drop_last=True)
  test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
  data_loader = data_utils.DataLoader(rtq_dataset, batch_size=batch_size, drop_last=False)

  return train_loader,eval_loader, test_loader,data_loader, dataset.return_time_channel(), save_prefix, args.outdir

class KamNet(nn.Module):

    def __init__(self, time_channel):
        super(KamNet, self).__init__()

        param_dict = KAMNET_PARAMS  # Store the hyperparameters for KamNet

        # Initialize the grid for spherical CNN
        grid_dict = {'s2_eq': s2_equatorial_grid, 's2_ni': s2_near_identity_grid, "so3_eq":so3_equatorial_grid, 'so3_ni':so3_near_identity_grid}
        s2_grid_type = param_dict["s2gridtype"]
        so3_grid_type = param_dict["so3gridtype"]
        grid_s2 = grid_dict[s2_grid_type]()
        grid_so3 = grid_dict[so3_grid_type]()

        self.ftype = param_dict["ftype"]

        # Number of neurons in spherical CNN
        s2_1  = param_dict["s2_1"]
        so3_2 = param_dict["so3_2"]
        so3_3 = param_dict["so3_3"]
        so3_4 = param_dict["so3_4"]

        # Number of neurons in fully connected NN
        fc1 = int(param_dict["fc_max"])
        fc2 = int(param_dict["fc_max"] * 0.8)
        fc3 = int(param_dict["fc_max"] * 0.4)
        fc4 = int(param_dict["fc_max"] * 0.2)
        fc5 = int(param_dict["fc_max"] * 0.05)

        do1r = param_dict["do"]
        do2r = param_dict["do"]
        do3r = param_dict["do"]
        do4r = param_dict["do"]
        do5r = param_dict["do"]

        do1r = min(max(do1r,0.0),1.0)
        do2r = min(max(do2r,0.0),1.0)
        do3r = min(max(do3r,0.0),1.0)
        do4r = min(max(do4r,0.0),1.0)
        do5r = min(max(do5r,0.0),1.0)

        # Number of neurons in AttentionConvLSTM
        s1 = param_dict["s1"]
        s2 = param_dict["s2"]

        # Last output of spherical CNN
        last_entry = so3_4

        # Last output of fully connected NN
        last_fc_entry = fc5

        # The spherical CNN bandwidth
        last_bw = int(param_dict["last_bw"])
        bw = np.linspace(int(time_channel[1]/2), last_bw, 5).astype(int)

        #. Spherical CNN part of KamNet
        self.conv1 = S2Convolution(
            nfeature_in=s2,
            nfeature_out=s2_1,
            b_in=bw[0],
            b_out=bw[1],
            grid=grid_s2)

        self.conv2 = SO3Convolution(
            nfeature_in=s2_1,
            nfeature_out=so3_2,
            b_in=bw[1],
            b_out=bw[2],
            grid=grid_so3)

        self.conv3 = SO3Convolution(
            nfeature_in=so3_2,
            nfeature_out=so3_3,
            b_in=bw[2],
            b_out=bw[3],
            grid=grid_so3)

        self.conv4 = SO3Convolution(
            nfeature_in=so3_3,
            nfeature_out=so3_4,
            b_in=bw[3],
            b_out=bw[4],
            grid=grid_so3)

        #. AttentionConvLSTM part of KamNet
        self.convlstm1=ConvLSTM(1, [s1,s2], [(param_dict["first_filter"],param_dict["first_filter"]),(param_dict["second_filter"],param_dict["second_filter"])],2, time_channel,batch_first=True,fill_value=0.1)

        if self.ftype == "SO3I":
            # This means integrating the last spherical CNN output using the Haar measure as provided in the paper
            self.fc_layer = nn.Linear(so3_4, fc1)
        else:
            # This means flattening the last spherical CNN output into a 1D vector (batch_size,flattened_dimension)
            self.fc_layer = nn.Linear(so3_4*(2*last_bw)**3, fc1)

        # Fully connected part of KamNet
        self.fc_layer_2 = nn.Linear(fc1, fc2)
        self.fc_layer_3 = nn.Linear(fc2, fc3)
        self.fc_layer_4 = nn.Linear(fc3, fc4)
        self.fc_layer_5 = nn.Linear(fc4, fc5)

        self.norm_layer_3d_1 = nn.BatchNorm3d(s2_1)
        self.norm_layer_3d_2 = nn.BatchNorm3d(so3_2)
        self.norm_layer_3d_3 = nn.BatchNorm3d(so3_3)
        self.norm_layer_3d_4 = nn.BatchNorm3d(so3_4)


        self.norm_1d_1 = nn.BatchNorm1d(fc1)
        self.norm_1d_2 = nn.BatchNorm1d(fc2)
        self.norm_1d_3 = nn.BatchNorm1d(fc3)
        self.norm_1d_4 = nn.BatchNorm1d(fc4)
        self.norm_1d_5 = nn.BatchNorm1d(fc5)
        self.norm_1d_6 = nn.BatchNorm1d(1)

        self.fc_layer_6 = nn.Linear(fc5, 1)

        self.do1 = nn.Dropout(do1r)
        self.do2 = nn.Dropout(do2r)
        self.do3 = nn.Dropout(do3r)
        self.do4 = nn.Dropout(do4r)
        self.do5 = nn.Dropout(do5r)

        self.sdo1 = nn.Dropout(param_dict["sdo"])
        self.sdo2 = nn.Dropout(param_dict["sdo"])
        self.sdo3 = nn.Dropout(param_dict["sdo"])
        self.sdo4 = nn.Dropout(param_dict["sdo"])


    def forward(self, x):
        x = x.unsqueeze(2)
        with autocast():
          x = self.convlstm1(x)

        x = self.conv1(x)
        x = self.norm_layer_3d_1(x)
        x = torch.relu(x)
        x = self.sdo1(x)

        x = self.conv2(x)
        x = self.norm_layer_3d_2(x)
        x = torch.relu(x)
        x = self.sdo2(x)

        x = self.conv3(x)
        x = self.norm_layer_3d_3(x)
        x = torch.relu(x)
        x = self.sdo3(x)


        x = self.conv4(x)
        x = self.norm_layer_3d_4(x)
        x = torch.relu(x)
        x = self.sdo4(x)

        if self.ftype == "SO3I":
            x = so3_integrate(x)
        else:
            x = x.view(x.size(0),-1)
        with autocast():
          x = self.fc_layer(x)
          x = self.norm_1d_1(x)
          x = torch.relu(x)
          x = self.do1(x)

          x = self.fc_layer_2(x)
          x = self.norm_1d_2(x)
          x = torch.relu(x)
          x = self.do2(x)

          x = self.fc_layer_3(x)
          x = self.norm_1d_3(x)
          x = torch.relu(x)
          x = self.do3(x)

          x = self.fc_layer_4(x)
          x = self.norm_1d_4(x)
          x = torch.relu(x)
          x = self.do4(x)

          x = self.fc_layer_5(x)
          x = self.norm_1d_5(x)
          x = torch.relu(x)
          x = self.do5(x)

          x = self.fc_layer_6(x)

        return x

def plot_result(test_loader, data_loader, classifier,suffix=EV_SUFFIX):
  '''
  Plot the training results of KamNet
  '''
  sigmoid_signal = []
  sigmoid_bkg = []
  energy_signal = []
  energy_bkg = []
  for images, labels, energies in test_loader:

      classifier.eval()

      with torch.no_grad():
          energy_data = energies.cpu().data.numpy().flatten()

          images = images.to(DEVICE)
          labels = labels.view(-1,1)
          labels = labels.to(DEVICE).float()

          outputs  = classifier(images).view(-1,1)

          image_data = images.cpu().data.numpy().reshape(BATCH_SIZE,-1)
          lb_data = labels.cpu().data.numpy().flatten()
          outpt_data = outputs.cpu().data.numpy().flatten()
          energy_data = energies.cpu().data.numpy().flatten()


          signal = np.argwhere(lb_data == 1)
          bkg = np.argwhere(lb_data == 0)
          sigmoid_signal += list(outpt_data[signal].flatten())
          sigmoid_bkg += list(outpt_data[bkg].flatten())
          energy_signal += list(energy_data[signal].flatten())
          energy_bkg += list(energy_data[bkg].flatten())


  sigmoid_s = []
  sigmoid_b_dict = {}
  for images, labels in tqdm(data_loader):

      classifier.eval()

      with torch.no_grad():
          images = images.to(DEVICE)

          outputs  = classifier(images)
          outputs = outputs.view(-1,1)

          lb_data = np.array(labels)
          outpt_data = outputs.cpu().data.numpy().flatten()

          signal = np.argwhere(lb_data == "Xe136")
          sigmoid_s += list(outpt_data[signal].flatten())

          bkg_name_list = np.unique(lb_data[lb_data != "Xe136"])
          for bkg_name in bkg_name_list:
              if bkg_name not in sigmoid_b_dict:
                   sigmoid_b_dict[bkg_name] = []
              sigmoid_b_dict[bkg_name] += list(outpt_data[lb_data == bkg_name].flatten())

  # Plot KamNet output spectrums with various backgrounds
  # Calculate rejection % based on the signal acceptance threshold defined by thresh variable
  thresh = 0.9
  metric_list = []
  # rg=np.linspace(0.0,1.0,100)
  rg=np.linspace(min(sigmoid_s),max(sigmoid_s),100)
  plt.hist(sigmoid_s, label ="Xe136", bins=rg, color="magenta", zorder=0,normed=True,alpha=0.3)
  for bkgname in sigmoid_b_dict.keys():
      fpr,tpr,thr,auc = get_roc(sigmoid_s, sigmoid_b_dict[bkgname])
      effindex = np.abs(tpr-thresh).argmin()
      effpurity = 1.-fpr[effindex]
      plt.hist(sigmoid_b_dict[bkgname], label = "%s(%.3f)"%(bkgname,effpurity), bins=rg, histtype="step",normed=True, linewidth=1)
      metric_list.append(auc)
  # plt.ylim(0.0,8)
  plt.ylabel("% of event/0.02 bins")
  plt.xlabel('KamNet output')
  plt.legend()
  plt.savefig("llhist_%s.png"%(suffix),dpi=200)
  plt.cla()
  plt.clf()
  plt.close()

  # Plot the data energy spectrum and spectrum removed by KamNet
  energy_signal = np.array(energy_signal)
  sigmoid_signal = np.array(sigmoid_signal)
  effindex = np.abs(tpr-0.9).argmin()
  effthr = thr[effindex]
  rg2 = np.arange(2.0,3.0,0.05)
  plt.yscale("log")
  plt.hist(energy_signal, bins=rg2,histtype = "step",label="All Data")
  plt.hist(energy_signal[sigmoid_signal<=effthr], bins=rg2,histtype = "step",label="Cut Data")
  plt.xlabel('Energy[MeV]')
  plt.legend()
  plt.savefig("cutevent_%s.png"%(suffix),dpi=200)
  plt.cla()
  plt.clf()
  plt.close()

  # Plot the KamNet spectrum of Bi214 MC and Bi214 data to check for data/MC agreement
  # The factor 0.85/0.15 corresponds to the XeLS/film Bi214 fractions in the tagged Bi214 dataset
  sigmoid_mcbi = np.concatenate([np.random.permutation(sigmoid_b_dict["Bi214-MC"])[:int(len(sigmoid_b_dict["Bi214-MC"])*0.85)],np.random.permutation(sigmoid_b_dict["Bi214-film"])[:int(len(sigmoid_b_dict["Bi214-film"])*0.15)]])
  plt.hist(sigmoid_signal, label = r'DB_untagged', histtype='step',bins=rg, color=colormap_normal(0.9), density=True)
  plt.hist(sigmoid_bkg, label = r'$^{214}Bi$ Data(%.3f)'%(get_rej(sigmoid_s, sigmoid_bkg)), histtype='step',bins=rg,color=colormap_normal(0.1),density=True)
  plt.hist(sigmoid_mcbi, label = r'$^{214}Bi$ MC(%.3f)'%(get_rej(sigmoid_s, sigmoid_mcbi)), histtype='step',bins=rg,color=colormap_normal(0.4),density=True)
  plt.xlabel('Sigmoid Ouptut')
  plt.ylabel('Counts')
  plt.legend(loc='upper center')
  plt.savefig('test_log_%s.png'%(suffix))
  plt.cla()
  plt.clf()
  plt.close()

def main():
    '''
    Training KamNet
    '''

    train_loader,eval_loader, test_loader, data_loader, time_channel, save_prefix, outdir = load_data(BATCH_SIZE)

    classifier = KamNet(time_channel)

    #=====================================================================================
    '''
    This part allows the loading of previously trained of KamNet using '.pt' model
    '''
    # pretrained_dict = torch.load('pretrain_data.pt')
    # model_dict = classifier.state_dict()
    # model_dict.update(pretrained_dict) 
    # classifier.load_state_dict(pretrained_dict)
    #=====================================================================================

    classifier.to(DEVICE)

    print("#params", sum(x.numel() for x in classifier.parameters()))

    '''
    Define the loss function
    '''
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(DEVICE)

    param_dict = KAMNET_PARAMS

    #=====================================================================================
    '''
    Set up optimizer with varying learning rate:
      Ramp Up   : Gradually ramp up learning rate in the first 5 epochs, this allows the attention mechanism to learn proper attention score
      Flat      : Fix the learning rate at the nominal value
      Ramp Down : Ramp down the learning rate to 10% of nominal value in the last 10th - 5th epochs
      Flat      : Fix the learning rate at 10% of the nominal value for the last 5 epochs
    '''
    step_length = len(train_loader)
    total_step = int(NUM_EPOCHS * step_length)
    ramp_up = np.linspace(1e-4, 1.0, 5*step_length)
    ramp_down = list(np.linspace(1.0, 0.1, 5*step_length).flatten()) + [0.1]* 5*step_length
    ramp_down_start = total_step - len(ramp_down)
    lmbda = lambda epoch: ramp_up[epoch] if epoch<len(ramp_up) else ramp_down[epoch-ramp_down_start-1] if epoch > ramp_down_start else 1.0
    optimizer = torch.optim.RMSprop(classifier.parameters(),lr=param_dict["lr"], momentum=param_dict["momentum"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    #=====================================================================================

    for epoch in tqdm(range(NUM_EPOCHS)):
        print(scheduler.get_lr())
        for i, (images, labels) in enumerate(train_loader):
            classifier.train()
            images = images.to(DEVICE)
            labels = labels.view(-1,1)
            labels = labels.to(DEVICE).float()

            outputs  = classifier(images)
            loss = criterion(outputs,labels)

            loss.backward()         # optimizer the net
            optimizer.step()        # update parameters of net
            optimizer.zero_grad()   # reset gradient
            scheduler.step()
            print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
                epoch+1, NUM_EPOCHS, i+1, len(train_loader),
                loss.item(), end=""))

        del images
        torch.cuda.empty_cache()
        plot_result(test_loader, data_loader, classifier)
        torch.save(classifier.state_dict(), 'KamNet%s.pt'%(EV_SUFFIX))     # Save KamNet parameters in KamNet.pt file

main()
