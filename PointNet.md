# summary of PointNet

this a **pioneering and key** paper on applying deep learning(DL) on point clouds since it **firstly opens doors to novel 3d-centric approaches to 3D scene understanding** with the perspertive of DL. The proposed net named PointNet enables feature learning directly on point cloud data. Surprisingly, most of the later papers in this area since 2017 are largely influenced by it and many of them even design new nets directly based on the pointnet, for example, Charles Qi's later papers including PointNet++, F-PointNet, VoteNet.

## background

- point cloud data(PCD) is an important geometric data structure with numerous apps in robotics, autonomous driving, AR/VR, AEC/FME, surveying, etc; **However, PCD is quite different from other formats(e.g: mesh, volumentric, multi-view images), it has some particular characteristics:** 1)irregular; While pixels in images or voxels in volumentric are regular which distribute evenly in the space, PCD is a irregular format which has no fixed distribution pattern in the space. 2) orderless(point member permutation invariance), rigid tranformation invariance and interaction among points; PCD is orderless, and has rigid tranformation invariance and each point member is not isolated, instead neighboring points form a meanningful subset.

- geometric data structures comparison; mesh, volumetric, multi-view images, and point cloud. refer to Charles's thesis ch01,02 for comparision details.
  
- gap; For leveraging DL, PCD is often converted to other formats(eg: volumetric, multi-view imgs, mesh) since **typical convolutional architectures require highly regular input data formats like image grids, 3d voxels**. However, this will result in problems and issues.
  >volumetric,unncessarily voluminous since most lidar point cloud only has surface points and also computation inefficent when dealing with 3d cnn; mesh, need to decide mesh structures,eg: triangules, quad,etc; multi-view images, need to decide from which angles to generate the images so that the model can have a good performance, info loss.

## proposal

- proposal; to fill the gap, a novel deep neural network that directly consumes point cloud dubbed PointNet is proposed; This net is designed with respecting properties of PCD which are the permutation invariance of points and rigid transformation invariances of the object.
  >Note: PointNet does not capture local structure of PCD; Instead, it either processes on 1 point(MLP operation) or all points(max pooling operation).As a result, this results in its main limitation: no learning on local context. PointNet++ is proposed to overcome this.) 
  >It provides a **unified and lightweighted** approach to a number of 3D recognition tasks including object classification, part segmentation and semantic segmentation

- PointNet unique characteristics;1)consume PCD directly as the input; 2)respect permutation invariance of points and rigid tranformation invariance(this actually is not important). 3)robust to data corruption and pertubation. 4)can obtain S.T.O.A performance(2016). 5) various PCP tasks(classi, semantic segmentation).

- limiataion; not tailored to the property: interaction between points, not be able to capture local context.

## key components and its architecture


## [optional] methodology

## results

### qualitative results

### quantitative results

## code analysis

## potential use

## what is next to read?

## critiques and comments

## FAQ

### Among all popular geometric data structures, why point cloud data is favored?

- (TODO) The popular geometric data structures involve mesh, volumetric, multi-view images, and PCD; Except PCD, all those formats have their limitations. 1)for mesh, it is hard to define a particular TIN and quadrangles for DL task. 2)for volumetric, on the one hand, it is computationally expensive for apply 3d cnn(O(n^3)); On the other hand, there will 'a hole' representation when converting PCD to volumentric data meanning that most of the voxels are on the surface of the objects. Evidently, this is not suitable for PCD data. 3) regarding multi-view image representation, it is difficult to define the directions for project PCD into images. 
    >volumetric,unncessarily voluminous since most lidar point cloud only has surface points and also computation inefficent when dealing with 3d cnn; mesh, need to decide mesh structures,eg: triangules, quad,etc; multi-view images, need to decide from which angles to generate the images so that the model can have a good performance, info loss.

- PCD is representationally simple; close to raw data which enables end-to-end learning.

## references
