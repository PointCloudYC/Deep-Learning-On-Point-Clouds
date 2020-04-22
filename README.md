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
- [Relation-CNN (CVPR 2019 Oral))](https://github.com/Yochengliu/Relation-Shape-CNN) | [my note](RS-CNN.md)
- [RandLA-Net (CVPR 2020 oral)](https://github.com/QingyongHu/RandLA-Net) | [my note](RandLA-net.md)
- [PointCNN (NIPS2018)](https://github.com/yangyanli/PointCNN)
- [PointConv (CVPR2018)](https://github.com/DylanWusee/pointconv)
- [DGCNN (ACM Transactions on Graphics 2019)](https://github.com/WangYueFt/dgcnn)
- [KpConv (ICCV 2019)](https://github.com/HuguesTHOMAS/KPConv)
- [superpoint graph (CVPR2018 & CVPR2019)](https://github.com/loicland/superpoint_graph)

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

-  LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. "Deep learning." Nature 521.7553 (2015): 436-444. [pdf](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) (Three Giants' Survey) ⭐⭐⭐⭐⭐
- Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "A fast learning algorithm for deep belief nets." Neural computation 18.7 (2006): 1527-1554. [pdf](http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf)(Deep Learning Eve) ⭐⭐⭐
- Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. "Reducing the dimensionality of data with neural networks." Science 313.5786 (2006): 504-507. [pdf](http://www.cs.toronto.edu/~hinton/science.pdf) (Milestone, Show the promise of deep learning) ⭐⭐⭐
- Domingos, Pedro. “A Few Useful Things to Know about Machine Learning.” Communications of the ACM 55, no. 10 (October 1, 2012): 78. [pdf](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) ⭐⭐
- Jordan, M. I., and T. M. Mitchell. “Machine Learning: Trends, Perspectives, and Prospects.” Science 349, no. 6245 (July 17, 2015): 255–60. [pdf](https://pdfs.semanticscholar.org/b36e/cb4dd52969b391a072425816792d05108f39.pdf) ⭐⭐

### ImageNet Evolution

- Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012. [pdf](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (AlexNet, Deep Learning Breakthrough) ⭐️⭐️⭐️⭐️⭐️

- Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014). [pdf](https://arxiv.org/pdf/1409.1556.pdf) (VGGNet,Neural Networks become very deep!) ⭐️⭐️⭐️

- Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015. [pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) (GoogLeNet) ⭐️⭐️⭐️

- He, Kaiming, et al. "Deep residual learning for image recognition." arXiv preprint arXiv:1512.03385 (2015). [pdf](https://arxiv.org/pdf/1512.03385.pdf) (ResNet, Very very deep networks, CVPR best paper) ⭐️⭐️⭐️⭐️⭐️

- xception

- densenet

- LSTM, RNN, and some variants

### image semantic segmentation

**[1]** J. Long, E. Shelhamer, and T. Darrell, “**Fully convolutional networks for semantic segmentation**.” in CVPR, 2015. [[pdf]](https://arxiv.org/pdf/1411.4038v2.pdf) (FCN) :star::star::star::star::star:

**[2]** L.C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. "**Semantic image segmentation with deep convolutional nets and fully connected crfs**." In ICLR, 2015. [[pdf]](https://arxiv.org/pdf/1606.00915v1.pdf) () :star::star::star::star::star:

**[3]** Pinheiro, P.O., Collobert, R., Dollar, P. "**Learning to segment object candidates.**" In: NIPS. 2015. [[pdf]](https://arxiv.org/pdf/1506.06204v2.pdf) :star::star::star::star:

- PSPNet
- Deeplab v3+

### image detection

**[1]** Szegedy, Christian, Alexander Toshev, and Dumitru Erhan. "**Deep neural networks for object detection**." Advances in Neural Information Processing Systems. 2013. [[pdf]](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf) :star::star::star:

**[2]** Girshick, Ross, et al. "**Rich feature hierarchies for accurate object detection and semantic segmentation**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) **(RCNN)** :star::star::star::star::star:

**[3]** He, Kaiming, et al. "**Spatial pyramid pooling in deep convolutional networks for visual recognition**." European Conference on Computer Vision. Springer International Publishing, 2014. [[pdf]](http://arxiv.org/pdf/1406.4729) **(SPPNet)** :star::star::star::star:

**[4]** Girshick, Ross. "**Fast r-cnn**." Proceedings of the IEEE International Conference on Computer Vision. 2015. [[pdf]](https://pdfs.semanticscholar.org/8f67/64a59f0d17081f2a2a9d06f4ed1cdea1a0ad.pdf) :star::star::star::star:

**[5]** Ren, Shaoqing, et al. "**Faster R-CNN: Towards real-time object detection with region proposal networks**." Advances in neural information processing systems. 2015. [[pdf]](https://arxiv.org/pdf/1506.01497.pdf) :star::star::star::star:

**[6]** Redmon, Joseph, et al. "**You only look once: Unified, real-time object detection**." arXiv preprint arXiv:1506.02640 (2015). [[pdf]](http://homes.cs.washington.edu/~ali/papers/YOLO.pdf) **(YOLO,Oustanding Work, really practical, currently already YOLO v4)** :star::star::star::star::star:

**[7]** Liu, Wei, et al. "**SSD: Single Shot MultiBox Detector**." arXiv preprint arXiv:1512.02325 (2015). [[pdf]](http://arxiv.org/pdf/1512.02325) :star::star::star:

**[8]** Dai, Jifeng, et al. "**R-FCN: Object Detection via
Region-based Fully Convolutional Networks**." arXiv preprint arXiv:1605.06409 (2016). [[pdf]](https://arxiv.org/abs/1605.06409) :star::star::star::star:

**[9]** He, Gkioxari, et al. "**Mask R-CNN**" arXiv preprint arXiv:1703.06870 (2017). [[pdf]](https://arxiv.org/abs/1703.06870) :star::star::star::star:

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
>Deep Learning Bible

- [Machine Learning Yearning](https://www.deeplearning.ai/machine-learning-yearning/), Andrew Ng, 2018. ⭐️⭐️⭐️⭐️
>a collection of practical machine learning techniques to build yr intelligent system.

- [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/), Michael Nielsen, around 2015. ⭐⭐️⭐️⭐️

- [An introduction to statistical learning : with applications in R](http://faculty.marshall.usc.edu/gareth-james/ISL/). Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, 2013. ⭐️⭐️⭐️

  

### point clouds

- Topographic Laser Ranging and Scanning: Principles and Processing, 2nd Edition: Jie Shan, Charles K. Toth.

- Airborne and Terrestrial Laser Scanning, Vosselman George.

## Acknowledgment
- Part of sources refer to [floodsung's  Deep-Learning-Papers-Reading-Roadmap](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap)

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