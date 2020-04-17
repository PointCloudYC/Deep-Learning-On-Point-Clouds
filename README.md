[TOC]

# Deep-Learning-On-Point-Clouds

A curated list of **primary sources** involving papers, books, blogs on the research theme *applying deep learning on point cloud data*. Moreover, I will try to summarize these primary sources with a note. 

For papers, each note will try to summarize the basic background, main proposals, key components of the proposals, architecture, code implementation, methodology part, potential use of the paper and etc. For books, briefly list out which parts are important to the research topic.

## Papers for Point clouds

### classification, semantic segmentation and instance segmentation for point clouds

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
- [PU-Net (CVPR 2018)](https://github.com/yulequan/PU-Net)
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

### detection and flow analysis for point clouds
**1. detection**
**1.1 primary sources**
- [F-PointNet (CVPR 2018)](https://github.com/charlesq34/frustum-pointnets)
- [VoteNet (ICCV 2019)](https://github.com/facebookresearch/votenet)
- [3D Fully Convolutional Network for Vehicle Detection in Point Cloud](https://github.com/yukitsuji/3D_CNN_tensorflow)

2.scene flow analysis
- [FlowNet3D (CVPR 2019)](https://github.com/xingyul/flownet3d)

### generation/synthesis for point clouds
**1. primary sources**
- [r-GAN/l-GAN/GMM-AE (ICML 2018)](https://github.com/optas/latent_3d_points)
>Achlioptas, Panos, Olga Diamanti, Ioannis Mitliagkas, and Leonidas Guibas. “Learning Representations and Generative Models for 3D Point Clouds(ICML-2018).” ArXiv:1707.02392 [Cs], June 12, 2018. http://arxiv.org/abs/1707.02392.

- [PointFlow (ICCV 2019 Oral)](https://github.com/stevenygd/PointFlow)
>Yang, Guandao, Xun Huang, Zekun Hao, Ming-Yu Liu, Serge Belongie, and Bharath Hariharan. “PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows,” June 28, 2019. https://arxiv.org/abs/1906.12320v3.

- Learning Efficient Point Cloud Generation for Dense 3D Object Reconstruction – Lin et al. (AAAI 2018 )

**2. other sources**
- [PointSetGeneration (CVPR 2017)](https://github.com/fanhqme/PointSetGeneration)
- [3D adversarial point clouds (CVPR 2018)](https://github.com/xiangchong1/3d-adv-pc)
- [3D-point-cloud-generation (AAAI 2018 oral)](https://github.com/chenhsuanlin/3D-point-cloud-generation)

### reconstruction and model retrieval for point clouds

**1. reconstruction**

**1.1 primary sources**
* [3D-point-cloud-generation (AAAI 2018)](https://github.com/chenhsuanlin/3D-point-cloud-generation)

**1.2 other sources**

**2. model retrieval**
**2.1 primary sources**
**2.2 other sources**


## Papers for images, sequential data

### review papers
- DL nature Yann Lecun
- A review of ML

### classic nets
- Lenet, ImageNet, Googlenet,VGGnet, Restnet
- xception
- LSTM, RNN, and some variants

### image semantic segmentation
- FCN
- PSPNet
- Deeplab v3+

### image detection
- SSD
- Yolo
- Fast R-CNN
- Faster R-CNN

### other papers
todo

## books
### deep learning and machine learning

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition](http://shop.oreilly.com/product/0636920142874.do), Aurélien Géron, Sep, 2019. ⭐️⭐️⭐️⭐️⭐️️
>a practical DL book with lots of examples and exercises using Scikit-learn, TF 2.x (keras included)

- [Deep learning with python](https://www.manning.com/books/deep-learning-with-python), François Chollet, Dec, 2017. ⭐️⭐️⭐️⭐️⭐️
>an extremely classic and concise book on deep learning using keras.

- [Deep Learning](http://www.deeplearningbook.org/), Ian Goodfellow, etc, 2017 ⭐️⭐️⭐️⭐️⭐️
>a classical book on DL

- [Machine Learning Yearning](https://www.deeplearning.ai/machine-learning-yearning/), Andrew Ng, 2018. ⭐️⭐️⭐️⭐️
>a collection of practical machine learning techniques to build yr intelligent system.

- [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/), Michael Nielsen, around 2015. ⭐⭐️⭐️⭐️
  

### point clouds

- Topographic Laser Ranging and Scanning: Principles and Processing, 2nd Edition: Jie Shan, Charles K. Toth.

- Airborne and Terrestrial Laser Scanning, Vosselman George.

## todos

### notes for sem. seg.

- ~~pointnet~~
- ~~pointnet++~~
- ~~RandLA-Net~~
- RS-CNN
- PointCNN
- KPConv

### brief summary for 2d images
- classical nets, e.g.: AlexNet
- nets for img sem. seg.
- nets for img det.

### notes for det.
- votenet