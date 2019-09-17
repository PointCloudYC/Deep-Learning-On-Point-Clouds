# review
## abstract
- bg;
  - data format; point cloud(PC), an important geometric data type w. apps in computer vision,robotics, and computer graphics
  - availability of large dataset; widely available point cloud data(lots of inexpensive 3d sensors)
  - hype; automomous driving highlighted the importance of reliable, efficient point cloud processing 

- gap; due the iregularity of the PC, current deep learning methods can not be  used directly with PCs.

- previous papers; convert PCs to images or voxels, this results in voluminous, quantization and other issues.

- proposal--PointNet and PointNet++; 
  - PointNet: a unified nn achitecture for 3d classi and segmentation in point clouds, respecting the permuatation invariance of input points.
  - PointNet++: recursively applying pointnet on the metric space, settling the issue that pointnet does not capture the local context. it tries to address the challenge of non-uniform sampling density in common 3d scans and design new layers to adapt to varying sampling densities.

- evaluation; pioneer nn framework, achive the S.T.O.A performance on several benchmark datasets.

- features of the proposed nn;
  - 2 invariance; permuatation invariance and tranformation invar. to some extent.
  - even for **input perturbations and data corruption**, it is still robust and efficient.
  - it can rougly learn the skelton of the object as the insteresting features to classify the object.

- open doors to new 3d-centric approaches to 3d scene understading.
  - 2 appliactions; 1) frustum-pointnet for 3d obj detection; 2)FlowNet3d; recover 3d motion flow from 2 frames of point clouds.
  - more other apps;   

## intro

## architecture

## code

## refs
PointNet
PointNet++
Charles QI's Ph.D. thesis