# PCPNet
This is our implementation for [PCPNet](http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/),
a network that estimates local geometric properties such as normals and curvature from point clouds.

The architecture is similar to [PointNet](http://stanford.edu/~rqi/pointnet/) (with a few smaller modifications),
but features are computed from local patches instead of of the entire point cloud,
which makes estimated local properties more accurate.

This code was written by [Paul Guerrero](https://paulguerrero.github.io) and [Yanir Kleiman](https://www.cs.tau.ac.il/~yanirk/),
based on the excellent PyTorch implementation of PointNet by [Fei Xia](https://github.com/fxia22/pointnet.pytorch).

This work was presented at Eurographics 2018.
```
@article{GuerreroEtAl:PCPNet:2016,
  title   = {{PCPNet}: Learning Local Shape Properties from Raw Point Clouds}, 
  author  = {Paul Guerrero and Yanir Kleiman and Maks Ovsjanikov and Niloy J. Mitra},
  year    = {2018},
  journal = {Eurographics},
  volume = {},
  number = {},
  issn = {},
  pages = {},
  numpages = {},
  doi = {},
}
```
