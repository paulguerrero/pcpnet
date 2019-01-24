# PCPNet
This is our implementation of [PCPNet](http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/),
a network that estimates local geometric properties such as normals and curvature from point clouds.

![PCPNet estimates local point cloud properties](https://raw.githubusercontent.com/paulguerrero/pcpnet/master/images/teaser.png "PCPNet")

The architecture is similar to [PointNet](http://stanford.edu/~rqi/pointnet/) (with a few smaller modifications),
but features are computed from local patches instead of of the entire point cloud,
which makes estimated local properties more accurate.

This code was written by [Paul Guerrero](https://paulguerrero.github.io) and [Yanir Kleiman](https://www.cs.tau.ac.il/~yanirk/),
based on the excellent PyTorch implementation of PointNet by [Fei Xia](https://github.com/fxia22/pointnet.pytorch).

This work was presented at [Eurographics 2018](https://www.eurographics2018.nl/).

**Update 18/Oct/2018:** The code has been updated to pytorch 0.4, and an option for choosing the GPU or using CPU only has been added. The old version that is compatible with pytorch 0.3 is still available in the branch [`pytorch_0.3`](https://github.com/paulguerrero/pcpnet/tree/pytorch_0.3).

**Update 21/Jun/2018:** The test dataset has been updated to include one shape that was missing to exactly reproduce the results in our paper. Thanks to Itzik Ben Shabat for pointing this out! Also note that the `--sparse_patches` option needs to be activated when running eval_pcpnet.py to exactly reproduce the results in our paper.

## Prerequisites
* Python 3.6
* PyTorch ≥ 0.4
* CUDA and CuDNN if training on the GPU

## Setup
Install required python packages, if they are not already installed ([tensorboardX](https://github.com/lanpa/tensorboard-pytorch) is only required for training):
``` bash
pip install numpy
pip install scipy
pip install tensorboardX
```

Clone this repository:
``` bash
git clone https://github.com/paulguerrero/pcpnet.git
cd pcpnet
```

Download dataset and pre-trained models:
``` bash
python pclouds/download_pclouds.py
python models/download_models.py
```

## Data Format
The point cloud and its properties are stored in different files with the same name, but different extensions:
`.xyz` for point clouds, `.normals` for normals and `.curv` for curvature values.

These files have similar formats. Each line describes one point, containing (x, y, z) coordinates separated by spaces for points and normals and (max. curvature, min. curvature) values for curvatures.

A dataset is given by a text file containing the file name (without extension) of one point cloud per line. The file name is given relative to the location of the text file.

## Estimating Point Cloud Properties
To estimate point cloud properties using default settings:
``` bash
python eval_pcpnet.py
```
This outputs unoriented normals using the single-scale normal estimation model described in the paper
for the test set used in the paper. To use alternative models and data sets, either edit the default arguments
defined in the first few lines of `eval_pcpnet.py`, or run `eval_pcpnet.py` with additional arguments:
``` bash
python eval_pcpnet.py --indir "path/to/dataset" --dataset "dataset.txt" --models "/path/to/model/model_name"
```
Where dataset.txt is a dataset as described above.
The model is given without the `_model.pth` suffix. For example the model file `models/single_scale_normal_model.pth`
would be specified as `--models "models/single_scale_normal"`. In addition to the model file,
a file containing model hyperparameters and training parameters has to be available at the same location and with the same name,
but with suffix `_params.pth`. Both the model and the parameters file are available for all pre-trained models and
are generated when training with `train_pcpnet.py`.

## Training
To train PCPNet with the default settings:
``` bash
python train_pcpnet.py
```
This trains the single-scale normal estimation model described in the paper on the training set used in the paper.
To train on a different dataset:
``` bash
python train_pcpnet.py --indir "path/to/dataset" --trainset "dataset.txt"
```
The dataset is given in the format described above. To change model settings or train a multi-scale model, see the description of the input arguments in the first few lines of `train_pcpnet.py`.

## Citation
If you use our work, please cite our paper:
```
@article{GuerreroEtAl:PCPNet:EG:2018,
  title   = {{PCPNet}: Learning Local Shape Properties from Raw Point Clouds}, 
  author  = {Paul Guerrero and Yanir Kleiman and Maks Ovsjanikov and Niloy J. Mitra},
  year    = {2018},
  journal = {Computer Graphics Forum},
  volume = {37},
  number = {2},
  pages = {75-85},
  doi = {10.1111/cgf.13343},
}
```
