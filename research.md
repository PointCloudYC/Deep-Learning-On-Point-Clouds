# research focus

Research question: "**Can we achieve automated BIM LOD2 model generation from laser scan data using deep learning?**"

Justification of significance and orginiality for your question:

- data availability; quite convenient to obtain data from lidar devices and other sensors; also many benchmark datasets.

- data quality; the point cloud data is of high precision.

- significant values for numerous applications, 1) Facility management; 2) vsisualization, e.g.: establishment of smart city. 3) defection detection, quality asessment. 4) VR/AR, facility their modeling work. 5) autonomous driving.

## research pipeline

The proposed pipeline is consisted of 5 modules.

- PCD preprocessing: registration and filtering
  
- PCD semantic segmentation(pointnet,pointnet++,pointcnn,dgcnn,splatnet,...), instance segmentation(GSPN,...), detection(VoteNet,...). **research problem**
  - semantic segmentation;
  - instance segmentation;
  - detection;
  
- Geometry extraction and model retrieval (iterative cross fitting, PCA, SAC-ICP, ...)

- BIM generation using dynamo

- evaluation the generated model

** The pipeline can be applied on MEP, indoor room, outdoor road scene, so try to propose new nets or add new tricks on the existing nets to solve a particular problem based on your observation and properties of a particular kind of PCD. **

## abbreviation

DL, deep learning
PCD, point cloud processing


## references
### 3d point cloud processing

1.classification and segmentation
pointnet
pointnet++
pointcnn
splatnet
dgcnn
octnet
so-net
KP-conv

2.instance segmenation
* GSPN
* SGPN
* [[1906.01140] Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds](https://arxiv.org/abs/1906.01140)
* [[1904.00699] JSIS3D: Joint Semantic-Instance Segmentation of 3D Point Clouds with Multi-Task Pointwise Networks and Multi-Value Conditional Random Fields](https://arxiv.org/abs/1904.00699)
Instance Segmentation of Point Clouds using Deep Learning, msc thesis.


3.detection
frustum-pointnet
VoteNet

4.other tasks(registration,filtering, completion,compression,...)

PCN
3DFeatNet

### 2d image processing
1.image classification and segmentation
paper
* [A review of semantic segmentation using deep neural networks | SpringerLink](https://link.springer.com/article/10.1007/s13735-017-0141-z#Sec8)
>

core papers
- fcn
- pspnet
- deeplabv1,..,v3+

reviews
* [Review of Deep Learning Algorithms for Image Semantic Segmentation](https://medium.com/@arthur_ouaknine/review-of-deep-learning-algorithms-for-image-semantic-segmentation-509a600f7b57)
>

* [Semantic Segmentation — U-Net - Kerem Turgutlu - Medium](https://medium.com/@keremturgutlu/semantic-segmentation-u-net-part-1-d8d6f6005066), awesome article with focus on u-net.


2.instance seg
core papers
- deepmask
- mask rcnn

code
* [matterport/Mask_RCNN: Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow](https://github.com/matterport/Mask_RCNN)
  
* [instance-segmentation · GitHub Topics](https://github.com/topics/instance-segmentation)
* [JackieZhangdx/InstanceSegmentationList: This repository contains lists of state-or-art semantic instance segmentation works](https://github.com/JackieZhangdx/InstanceSegmentationList)
  

reviews
* [A Simple Guide to Semantic Segmentation | TOPBOTS](https://www.topbots.com/semantic-segmentation-guide/), comp semantic and instance segmentation ,summarize over the deep learning models on the semantic segmentaiton.
* [Review: DeepMask (Instance Segmentation) - Towards Data Science](https://towardsdatascience.com/review-deepmask-instance-segmentation-30327a072339)
* [Instance segmentation using Mask R-CNN - Towards Data Science](https://towardsdatascience.com/instance-segmentation-using-mask-r-cnn-7f77bdd46abd)


* [Open Images 2019 - Instance Segmentation | Kaggle](https://www.kaggle.com/c/open-images-2019-instance-segmentation)