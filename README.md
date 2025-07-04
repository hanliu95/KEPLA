# KEPLA
# KEPLA: A Knowledge-Enhanced Deep Learning Framework for Accurate Protein-Ligand Binding Affinity Prediction | [Paper](https://arxiv.org/abs/2506.13196)


<div align="left">

[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/hanliu95/KEPLA/tree/main/LICENSE)
</div>

## Introduction
This repository contains the PyTorch implementation of **KEPLA** framework. KEPLA is an end-to-end knowledge-enhanced deep learning framework for protein-ligand binding affinity (PLA) prediction, designed to address the performance bottlenecks and limited interpretability of existing methods.  It works on target protein sequences and ligand molecular graphs to perform prediction.
## Framework
![KEPLA](image/framework.jpeg)
## System Requirements
The source code is developed in Python 3.10 using PyTorch 2.1.2. The required python dependencies are given below. KEPLA is supported for any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run. There are no additional non-standard hardware requirements.

```
torch>=2.1.2
dgl>=2.4.0
dgllife>=0.3.2
numpy>=1.24.3
scikit-learn>=1.5.2
pandas>=2.2.2
prettytable>=3.12.0
rdkit~=2024.03.5
yacs~=0.1.8
```
## Installation Guide
Clone this Github repo and set up a new conda environment. It normally takes about 10 minutes to install on a normal desktop computer.
```
# create a new conda environment
$ conda create --name kepla python=3.10
$ conda activate kepla

# install requried python dependencies
$ conda install pytorch==2.1.2 torchvision==0.16.2 cudatoolkit=12.1 -c pytorch
$ conda install -c dglteam dgl-cuda12.1==2.4.0
$ conda install -c conda-forge rdkit==2024.03.5
$ pip install dgllife==0.3.2
$ pip install -U scikit-learn
$ pip install yacs
$ pip install prettytable

# clone the source code of KEPLA
$ git clone https://github.com/hanliu95/KEPLA.git
$ cd KEPLA
```


## Datasets
The PDBbind dataset [1] can be downloaded [here](http://pdbbind-cn.org).

The CSAR-HiQ dataset [2] can be downloaded [here](http://www.csardock.org).

You may need to use the [UCSF Chimera tool](https://www.cgl.ucsf.edu/chimera/) to convert the PDB-format files into MOL2-format files for feature extraction at first.

Alternatively, we also provided a [dropbox link](https://www.dropbox.com/sh/2uih3c6fq37qfli/AAD-LHXSWMLAuGWzcQLk5WI3a) for downloading PDBbind and CSAR-HiQ datasets.


## Run KEPLA on Our Data to Reproduce Results

To train KEPLA, where we provide the basic configurations for all hyperparameters in `config.py`. For different in-domain and cross-domain tasks, the customized task configurations can be found in respective `configs/KEPLA.yaml` files.

For the in-domain experiments with KEPLA, you can directly run the following command. `${dataset}` could either be `pdbbind` or `csar`.
```
$ python main.py --cfg "configs/KEPLA.yaml" --data ${dataset} --split "random"
```

For the cross-domain experiments with KEPLA, you can directly run the following command. `${dataset}` could be `pdbbind`.
```
$ python main.py --cfg "configs/KEPLA.yaml" --data ${dataset} --split "cluster"
```
For the cold experiments with KEPLA, you can directly run the following command. `${dataset}` could be `pdbbind`.
```
$ python main.py --cfg "configs/KEPLA.yaml" --data ${dataset} --split "cold"
```

## Acknowledgements
This implementation is inspired and partially based on earlier work [3].

## References
    [1] Liu, Zhihai, et al. "PDB-wide collection of binding data: current status of the PDBbind database." Bioinformatics 31.3 (2015): 405-412.
    [2] Smith, Richard D., et al. "CSAR benchmark exercise of 2010: combined evaluation across all submitted scoring functions." Journal of Chemical Information and Modeling 51.9 (2011): 2115-2131.
    [3] Bai, Peizhen, et al. "Interpretable bilinear attention network with domain adaptation improves drug–target prediction." Nature Machine Intelligence 5.2 (2023): 126-136.
