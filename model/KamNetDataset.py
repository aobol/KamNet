#=====================================================================================
#  Author: Aobo Li
#  Contact: liaobo77@gmail.com
#  
#  Last Modified: Aug. 29, 2021
#  
#  * The PyTorch dataset classes for KamNet
#=====================================================================================
import numpy as np
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tool import label_data, create_table, create_table_zpos, get_roc, create_table_energy, look_table
from settings import FILE_UPPERLIM 
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DetectorDataset(Dataset):

    def __init__(self, json_name):
        """
        Base class for all KamNet datasets
        """
        self.json_name = json_name

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        image = np.zeros(self.image_shape, dtype=np.float32)
        for time_index, time in enumerate(self.trainX[idx]):
            image[time_index] = time.todense()
        return image, self.trainY[idx]

    def return_time_channel(self):
        '''
        This method returns the time channel and one hit map dimension of input
        E.g. If it returns (28,38), this means the input has 28 time channel, where
        each channel contains a 38*38 hitmap
        '''
        return (self.__getitem__(0)[0].shape[0], self.image_shape[1])

    def cap_resample(self,input,cap=5000):
        '''
        This method randomly resamples part of the dataset
        '''
        if input.shape[0] < cap:
            return input
        signal_samples = np.random.choice(np.arange(input.shape[0]), cap, replace=False)
        return input[signal_samples]

    def get_sparse_nhit(self, sparse_dict):
        '''
        This method get the nhit as a list of given event dict
        It reads out the Nhit directly if Nhit is stored in the dict
        Otherwise it calculate Nhit from the sparce matrices
        '''
        if "Nhit" in sparse_dict.keys():
            return np.array(sparse_dict["Nhit"], dtype=int).flatten()
        else:
            sparsem = np.array(sparse_dict["event"], dtype=object)
            sparse_nhit = []
            for i in tqdm(range(len(sparsem))):
                sparse_nhit.append(np.sum([len(slice.nonzero()[0]) for slice in sparsem[i]]))
            return np.array(sparse_nhit)

    def match_nhit(self, signal_dict, background_dict, multiplier=1.0):
        '''
        Perform Nhit matching between input signal and output background
        '''
        signal_images = np.array(signal_dict[self.json_name], dtype=object)
        background_images = np.array(background_dict[self.json_name], dtype=object)
        nhit_range = np.arange(0,2000,1)
        signal_nhit = np.array(self.get_sparse_nhit(signal_dict))
        bkg_nhit = np.array(self.get_sparse_nhit(background_dict))

        signal_list = []
        bkg_list = []

        for (nlow, nhi) in tqdm(zip(list(nhit_range[:-1]), list(nhit_range[1:])),0):
            signal_index = np.where((signal_nhit >= nlow) & (signal_nhit <nhi))[0]
            bkg_index = np.where((bkg_nhit >= nlow) & (bkg_nhit <nhi))[0]
            if (len(signal_index) != 0) and (len(bkg_index) != 0):
                sampled_amount = min(len(signal_index), len(bkg_index))
                signal_list += list(np.random.choice(list(signal_index), sampled_amount, replace=False))
                bkg_list += list(np.random.choice(list(bkg_index), min(len(bkg_index), int(sampled_amount*multiplier)), replace=False))
        rg = np.arange(0,1000,1)
        plt.hist(signal_nhit[signal_list],label="Signal",histtype="step",bins=rg)
        plt.hist(bkg_nhit[bkg_list],label="Bkg",histtype="step",bins=rg)
        plt.legend()
        plt.savefig("Nhit.png")
        plt.cla()
        plt.clf()
        plt.close()

        return signal_images[signal_list], background_images[bkg_list]

    def match_nhit_bootstrap(self, signal_dict, background_dict, multiplier=1.0):
        '''
        Perform Nhit matching between input signal and output background with bootstrap allowed
        Bootstrap: sample with replacement in each dataset
        '''
        signal_images = np.array(signal_dict["event"], dtype=object)
        background_images = np.array(background_dict["event"], dtype=object)
        nhit_range = np.arange(0,38**2,1)
        signal_nhit = np.array(self.get_sparse_nhit(signal_dict))
        bkg_nhit = np.array(self.get_sparse_nhit(background_dict))

        signal_samples = np.random.choice(signal_images.shape[0], signal_images.shape[0], replace=True)##NO RESAMPLE
        signal_images = signal_images[signal_samples]
        signal_nhit = signal_nhit[signal_samples]

        bkg_samples = np.random.choice(background_images.shape[0], background_images.shape[0], replace=True)
        background_images = background_images[bkg_samples]
        bkg_nhit = bkg_nhit[bkg_samples]


        signal_list = []
        bkg_list = []

        for (nlow, nhi) in tqdm(zip(list(nhit_range[:-1]), list(nhit_range[1:])),0):
            signal_index = np.where((signal_nhit >= nlow) & (signal_nhit <nhi))[0]
            bkg_index = np.where((bkg_nhit >= nlow) & (bkg_nhit <nhi))[0]
            if (len(signal_index) != 0) and (len(bkg_index) != 0):
                sampled_amount = min(len(signal_index), len(bkg_index))
                signal_list += list(np.random.choice(list(signal_index), sampled_amount, replace=False))
                bkg_list += list(np.random.choice(list(bkg_index), min(len(bkg_index), int(sampled_amount*multiplier)), replace=False))

        return signal_images[signal_list], background_images[bkg_list]

    def label_data(self, signal_images, background_images):
        signal_labels = np.ones(len(signal_images), dtype=np.float32)
        background_labels = np.zeros(len(background_images), dtype=np.float32)
        size = len(signal_images) + len(background_images)
        trainX = np.concatenate((signal_images, background_images), axis=0)
        trainY = np.concatenate((signal_labels, background_labels), axis=0)
        image_shape = (trainX.shape[-1], *trainX[0,0].shape)

        return trainX, trainY, image_shape, size

class DetectorDataset_Nhit(DetectorDataset):

    def __init__(self, signal_images_list, bkg_image_list, json_name, bootstrap=False, dsize=-1, elow=2.0,ehi=3.0):
        super(DetectorDataset_Nhit, self).__init__(json_name)
        """
        KamNet dataset with Nhit matching. Nhit matching removes Nhit dependency of signal/background events
        Used for training the neural network
        elow and ehi indicates the min/max energy of events we'd like to read out
        """
        signal_dict = create_table_energy(signal_images_list, (json_name, 'Nhit','energy','zpos'), low=elow, high=ehi)
        background_dict = create_table_energy(bkg_image_list, (json_name, 'Nhit','energy','zpos'), low=elow, high=ehi)
        if bootstrap:
          signal_images, background_images = self.match_nhit_bootstrap(signal_dict, background_dict)
        else:
          signal_images, background_images = self.match_nhit(signal_dict, background_dict)

        if dsize != -1:
          signal_images = self.cap_resample(signal_images,dsize)
          background_images = self.cap_resample(background_images,dsize)

        self.trainX, self.trainY, self.image_shape, self.size = self.label_data(signal_images, background_images)

class DetectorDataset_NonUniform(DetectorDataset):

    def __init__(self, signal_images_list, bkg_image_list, json_name, elow=2.0,ehi=3.0):
        super(DetectorDataset_NonUniform, self).__init__(json_name)
        """
        KamNet dataset which do not require the signal/bkg dataset to follow the same size
        """
        signal_dict = create_table_energy(signal_images_list, (json_name, 'Nhit','energy','zpos'), low=elow, high=ehi)
        background_dict = create_table_energy(bkg_image_list, (json_name, 'Nhit','energy','zpos'), low=elow, high=ehi)
        signal_images = np.array(signal_dict[json_name], dtype=object)
        background_images = np.array(background_dict[json_name], dtype=object)
        signal_images = self.cap_resample(signal_images)
        background_images = self.cap_resample(background_images)
        signal_labels = np.ones(len(signal_images), dtype=np.float32)
        background_labels = np.zeros(len(background_images), dtype=np.float32)
        print(signal_images.shape, background_images.shape)
        self.trainX = np.concatenate((signal_images, background_images), axis=0)
        print(self.trainX.shape)
        self.size = self.trainX.shape[0]
        self.trainY = np.concatenate((signal_labels, background_labels), axis=0)
        self.image_shape = (self.trainX.shape[-1], *self.trainX[0,0].shape)

        signal_ene = np.array(signal_dict["energy"]).flatten()
        background_ene = np.array(background_dict["energy"]).flatten()
        self.energy = np.concatenate((signal_ene, background_ene), axis=0)



    def __getitem__(self, idx):

        image = np.ndarray(self.image_shape, dtype=np.float32)
        for time_index, time in enumerate(self.trainX[idx]):
            image[time_index] = time.todense()
        return image, self.trainY[idx], self.energy[idx]

class DetectorDatasetRep(DetectorDataset):

    def __init__(self, signal_images_list, bkg_image_dict, json_name, dsize = -1, elow=2.0,ehi=3.0):
        super(DetectorDatasetRep, self).__init__(json_name)
        """
        KamNet dataset outputing multiple isotopes for validation purpose
        """

        self.trainX = []
        self.trainY = []

        signal_dict = create_table_energy(signal_images_list[:FILE_UPPERLIM], (json_name,"Nhit"), low=elow, high=ehi)
        sigim = np.array(signal_dict[json_name], dtype=object)
        sigim = self.cap_resample(sigim, 2000)
        self.trainX.append(sigim)
        self.trainY += ["Xe136"] * len(sigim)

        for bkgn,bkglist in bkg_image_dict.items():
            bkgev = create_table_energy(bkglist[:FILE_UPPERLIM], (json_name, 'id'), low=elow, high=ehi)
            sigim = np.array(bkgev[json_name], dtype=object)
            if len(sigim) == 0:
                continue
            sigim = self.cap_resample(sigim, 2000)
            print(bkgn)
            self.trainX.append(sigim)
            self.trainY += [bkgn] * len(sigim)

        self.trainX = np.concatenate(self.trainX,axis=0)
        self.trainY = np.array(self.trainY)
        self.image_shape = (self.trainX.shape[-1], *self.trainX[0,0].shape)
        self.size = len(self.trainY)
