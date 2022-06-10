# KamNet
KamNet is the state-of-the-art neural network model for spherical liquid scintillator detector.
------------------------
# Dependencies
All of KamNet's prerequisite comes from the [s2cnn](https://github.com/jonkhler/s2cnn) packages, we recommend the users to first install that package, and then KamNet will be ready to use. The instruction below are copied from the [s2cnn](https://github.com/jonkhler/s2cnn) github page:

:warning: :warning: This code is old and does not support the last versions of pytorch! Especially since the change in the fft interface. :warning: :warning: 

* __PyTorch__: http://pytorch.org/ (>= 0.4.0)
* __cupy__: https://github.com/cupy/cupy
* __lie_learn__: https://github.com/AMLab-Amsterdam/lie_learn
* __pynvrtc__: https://github.com/NVIDIA/pynvrtc

(commands to install all the dependencies on a new conda environment)
```bash
conda create --name cuda9 python=3.6 
conda activate cuda9

# s2cnn deps
#conda install pytorch torchvision cuda90 -c pytorch # get correct command line at http://pytorch.org/
conda install -c anaconda cupy  
pip install pynvrtc joblib

# lie_learn deps
conda install -c anaconda cython  
conda install -c anaconda requests  
```

## Installation

To install, run

```bash
$ python setup.py install
```
------------------------
# Data and Training
- The source code of KamNet is stored in the `model` folder.
- The `data` folder contains the pre-processing code to produce the input spatiotemporal data. Pre-processing code is written to be submitted as batch job to the [Boston University Shared Computing Cluster](https://www.bu.edu/tech/support/research/computing-resources/scc/) system with `qsub`. Using it on other common batch system (such as slurm) needs additional modifications.
- The spatiotemporal data will be stored in `.pickle` file as one python dictionary per event, each event contains a list of 2D hit maps following temporal order. Each 2D hit map is stored as CSR sparse matrix to save memory and disk space. Input to KamNet is a `.dat` list with addresses to all the `.pickle` files.
- We plan to open source our training data in a stepwise manner, including:
- - `Benchmarking dataset`: referred to as `sim-FAST` in the paper, available soon
- - `Decay to Excited States dataset`: referred to as `sim-RAT` in the paper, available soon
- - `KamLAND-Zen 800 officical MC simulation`: referred to as `sim-KLZ800` in the paper, need approval from the KamLAND-Zen collaboration
---------------------------
# Acknowledgement
If you used this model in your work, please cite this paper:
```
@article{Li:2022frp,
    author = "Li, A. and Fu, Z. and Winslow, L. and Grant, C. and Song, H. and Ozaki, H. and Shimizu, I. and Takeuchi, A.",
    title = "{KamNet: An Integrated Spatiotemporal Deep Neural Network for Rare Event Search in KamLAND-Zen}",
    eprint = "2203.01870",
    archivePrefix = "arXiv",
    primaryClass = "physics.ins-det",
    month = "3",
    year = "2022"
}
```

