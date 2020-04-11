[TOC]

# Deep-Learning-On-Point-Clouds

A curated list of **primary sources** involving papers, books, blogs and other sources on the research topic *applying deep learning on point cloud data*. Moreover, I will try to summarize on these primary sources with a note. 

For papers, each note will try to summarize the basic background, main proposals, key components of the proposals, architecture, code implementation, methodology part, potential use of the paper and etc. For books, briefly list out which parts are important to the research topic.

## Papers for Point clouds

### Classification, semantic segmentation and instance segmentation for point clouds

**1. classification and semantic segmentation**
**1.1 primary sources**
- [PointNet (CVPR 2017 oral)](https://github.com/charlesq34/pointnet) | [my note](PointNet.md)
- [PointNet++ (NIPS 2017)](https://github.com/charlesq34/pointnet2) | [my note](PointNet++.md)
- [RandLA-Net (CVPR 2020 oral)](https://github.com/QingyongHu/RandLA-Net) | [my note](RandLA-net.md)
- [PointCNN (NIPS2018)](https://github.com/yangyanli/PointCNN)
- [DGCNN (ACM Transactions on Graphics 2019)](https://github.com/WangYueFt/dgcnn)
- [PointConv (CVPR2018)](https://github.com/DylanWusee/pointconv)
- [superpoint graph (CVPR2018 & CVPR2019)](https://github.com/loicland/superpoint_graph)
- [KpConv (ICCV 2019)](https://github.com/HuguesTHOMAS/KPConv)
- [Relation-CNN (CVPR 2019 Oral & Best paper finalist))](https://github.com/Yochengliu/Relation-Shape-CNN) | [my note](RS-CNN.md)

**1.2 other source**
- [Volumetric and multi-view CNNs (CVPR 2016 spotlight)](https://github.com/charlesq34/3dcnn.torch)
- [Render for cnn (ICCV 2015)](https://github.com/ShapeNet/RenderForCNN)
- [SPLATNet (CVPR2018)](https://github.com/NVlabs/splatnet)
- [So-Net (CVPR 2018)](https://github.com/lijx10/SO-Net)
- [OctNet (CVPR 2017)](https://github.com/griegler/octnet)
- [PointRCNN (CVPR 2019)](https://github.com/sshaoshuai/PointRCNN)
- [Scan2CAD (CVPR 2019)](https://github.com/skanti/Scan2CAD)
- [PartNet (CVPR 2019)](https://github.com/daerduoCarey/partnet_dataset)
- [GeoNet (CVPR2018)](https://github.com/yzcjtr/GeoNet)
- [SPFN (CVPR 2019 Oral)](https://github.com/lingxiaoli94/SPFN)
- [VoxelNet (CVPR 2017)](https://github.com/tsinghua-rll/VoxelNet-tensorflow)
- [Tangent Convolutions (CVPR 2018)](https://github.com/tatarchm/tangent_conv)
- [PCN (3DV 2018 (Oral))](https://github.com/wentaoyuan/pcn)
- [3DMatch (CVPR 2017 Oral)](https://github.com/andyzeng/3dmatch-toolbox)
- 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, ScanNet No.1 on the leatherboard(26th,sep,19)
- 3D Semantic Segmentation with Submanifold Sparse Convolutional Networks and Submanifold Sparse Convolutional Networks, ScanNet No.2 on the leatherboard(26th,sep,19)

**2.instance segmentation**
**1. primary sources**
- [SGPN (CVPR 2018)](https://github.com/laughtervv/SGPN)
- [ASIS](https://paperswithcode.com/paper/associatively-segmenting-instances-and), CVPR19
- [3D-SIS (CVPR 2019)](https://github.com/Sekunde/3D-SIS)
- [3D-BoNet](https://paperswithcode.com/paper/learning-object-bounding-boxes-for-3d)
- [JSIS3D](https://paperswithcode.com/paper/jsis3d-joint-semantic-instance-segmentation)
- [GSPN (CVPR 2019)](https://github.com/ericyi/GSPN)

**2. other sources**
todo

### generation/synthesis for point clouds
**1. primary sources**
- [r-GAN/l-GAN/GMM-AE (ICML 2018)](https://github.com/optas/latent_3d_points)
>Achlioptas, Panos, Olga Diamanti, Ioannis Mitliagkas, and Leonidas Guibas. “Learning Representations and Generative Models for 3D Point Clouds(ICML-2018).” ArXiv:1707.02392 [Cs], June 12, 2018. http://arxiv.org/abs/1707.02392.

- [PointFlow (ICCV 2019 Oral)](https://github.com/stevenygd/PointFlow)
>Yang, Guandao, Xun Huang, Zekun Hao, Ming-Yu Liu, Serge Belongie, and Bharath Hariharan. “PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows,” June 28, 2019. https://arxiv.org/abs/1906.12320v3.

**2. other sources**
- [3D adversarial point clouds (CVPR 2018)](https://github.com/xiangchong1/3d-adv-pc)
- [3D-point-cloud-generation (AAAI 2018 oral)](https://github.com/chenhsuanlin/3D-point-cloud-generation)

### detection and flow anlysis for point clouds
**1. detection**
**1.1 primary sources**
- [F-PointNet (CVPR 2018)](https://github.com/charlesq34/frustum-pointnets)
- [VoteNet (ICCV 2019)](https://github.com/facebookresearch/votenet)
- [3D Fully Convolutional Network for Vehicle Detection in Point Cloud](https://github.com/yukitsuji/3D_CNN_tensorflow)

2.scene flow analysis
- [FlowNet3D (CVPR 2019)](https://github.com/xingyul/flownet3d)

### reconstruction and model retrieval for point clouds

**1. reconstruction**
**1.1 primary sources**
**1.2 other sources**

**2. model retrieval**
**2.1 primary sources**
**2.2 other sources**


## Papers for images, sequential data
### review papers
todo

### classic papers
todo

### other papers
todo

## Books
### Deep learning

- DEEP LEARNING WITH PYTHON, Francis Chollet.

>an extremely classic and concise book on deep learning using keras.

- HANDS-ON MACHINE LEARNING WITH SCIKIT-LEARN, KERAS AND TENSORFLOW, 2nd Edition, Aurélien Géron.

>a practical DL book with lots of examples and exercises using Scikit-learn, TF 2.x (keras included)

- DEEP LEARNING, Ian Goodfellow.

>a classical book on DL

- MACHINE LEARNING YEARNING, Andrew Ng.
  
>a collection of practical machine learning techniques to build yr intelligent system.

### Point clouds

- Topographic Laser Ranging and Scanning: Principles and Processing, 2nd Edition: Jie Shan, Charles K. Toth.

- Airborne and Terrestrial Laser Scanning, Vosselman George.

## Todos

- notes for each key paper
  - pointnet
  - pointnet++
  - votenet
  - pointcnn
