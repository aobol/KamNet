#=====================================================================================
#  Author: Aobo Li
#  Contact: liaobo77@gmail.com
#  
#  Last Modified: Aug. 29, 2021
#  
#  * constants and file addresses for KamNet
#=====================================================================================
SEED = False                                    # Reproducibility of KamNet
NUM_EPOCHS = 30                                 # Number of training epochs
BATCH_SIZE = 32                                 # Batch size
FILE_UPPERLIM = 5                               # Number of files in the pickle list, setting this to a small value allows faster training of model.
												# Setting FILE_UPPERLIM to a very large number (a.k.a. 999999) will allow us to use the entire dataset
EV_SUFFIX = "_17and20good_nocharge"             # Suffix of pickle list
DSIZE = 20000                                   # Number of signal/background events in training dataset. The final training size is DSIZE*2
KAMNET_PARAMS = {"momentum": 0.7806697572271865,# Hyperparameters of KamNet
				"lr": 7.729560386535045e-05, "first_filter": 5,
				"second_filter": 3,
				"do": 0.08623589261579744,
				"s2_1": 36, "so3_2": 63,
				"so3_3": 74, 
				"so3_4": 124,
				"fc_max": 1080,
				"s1": 12, "s2": 22,
				"sdo": 0.406085565931351,
				"optimizer": "RMSprop",
				"s2gridtype": "s2_ni",
				"so3gridtype": "so3_eq",
				"ftype": "SO3I",
				"BATCH_SIZE": 32,
				"last_bw": 2,
				"do2": 0.0}
LEARNING_RATE =0.000018675460538381732          # Learning rate