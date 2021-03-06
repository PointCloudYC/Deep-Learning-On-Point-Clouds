# summary of PointNet

this is a **pioneering and key** paper on applying deep learning(DL) on point clouds since it **firstly opens doors to novel 3d-centric approaches to 3D scene understanding** with the perspertive of DL. The proposed net named PointNet enables feature learning directly on point cloud data. Surprisingly, most of the later papers in this area since 2017 are largely influenced by it and many of them even design new nets directly based on the pointnet, for example, Charles Qi's later papers including PointNet++, F-PointNet, VoteNet.

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
### key components

There are mainly 3 key modules inclduing max pooling module, conctenation structure and T-Net.; 1) max pooling module to aggragate info from all the points. (ensure `point member permuatation invariance`.) 2) concatenation structure combining local and global info (enable semantic segmentation). 3) 2 joint networks named T-Net.(ensure `rigid tranformation invariance`)

### achitecture

- input and output; 1) For classification, NxD(1 object N points, each with D dims) --> label (1 class,eg: table). 2) for part seg, NxD(1 object N points, each with D dims and each obj has many parts) --> labels (each point have a label,eg: table leg, table plane,etc.). 3)for semantic seg, NxD(1 sub-volumn sampled from a scene,e.g: 1x1 block from a scene) --> labels (each point have a label,eg: table, chair, sofa,etc.).

- architecture;
![architecture of pointnet](images/pointnet/architecture-pointnet.png)
  - max pooling;
  - T-Net;PointNet(vanilla) does not have the ability to be invariant to the rigid tranformation. To overcome this issue, the author use `T-Net` to learn the rotation so that the input(Nx3) and features(Nx64) can be stardardized before performing classi or segmentation task.P.S.: for feature tranformation, it add a regularizer loss so that the learned rotation matrix can be approximated as a orthonormal matrix.
  ![](images/pointnet/t-net-3.png)
  - concatenation of local and global info;
  - Tensor shape envolvement in 4d format(for verification, you can visualize it in tensorboard); 
  
  - The architecture has lots of similarities with typical convolutional NNs but it **has a special preference for point/depth convolution** namely 2d convolution with 1x1 kernels; ( *TODO: 2d conv mainly abstract features across space, while 2d conv w. 1x1 kernels abstract features across channels. Check the intuition of point conv from `Network in network` and `Xception` paper*)

## results
omitted, check the paper.

## [optional] methodology

## results
omitted, check the paper.

## conclusion

- PointNet is a novel deep neural network that directly consumes point cloud, respecting permutation and geometric invariances of the points, while being light-weight and robust to various data corruptions.

- It provides **a unified approach** to a number of 3D recognition tasks including object classification, part segmentation and semantic segmentation.

## code analysis

code structure is:
![](images/pointnet/code-structure.png)

- `data`, stores the benchmark datasets.
- `doc`
- `log`, stores classification logs including training log, tensorboard events, checkpoints
- `models`, stores models for 3 tasks(classi, part seg, and semantic seg)
- `part_seg`, stores training and test files for part seg.
- `sem_seg`, stores training and test files for part seg.
- `utils`, utility files for this project.
- `train.py`, training file for classification task.
- `evaluate.py`, evaluation file for classification task
- `provider.py`, for preprocessing inputs and handle i/o for h5 format data.

### quick use for ModelNet40, ShapeNet and S3DIS

### classification analysis
1.related files(root mean root folder)
root/train.py, 
root/evaluate.py, 
root/models/pointnet_cls.py,root/models/pointnet_cls_basic.py

2.model: `pointnet_cls.py`. Note: `pointnet_cls_basic.py` does not add T-NET to ensure rigid tranformation invariance.

- input(Batch * Height * Width) and output(Batch)
```
# X(Batch-Height-Width-Channel,eg: 32*1024*3*1), y(Batch-Label,eg: 32*40)
def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl
```

- classification achitecture code;
  >Symmetric function: max pooling
KEY PART: after the CNN, we get the redundant info for each point(N*1024),Using pooling(max,avg) we can hopefully obtain the intesrsting pts(salient representations--global discriptor) which proves to correspond to the skeleton of the shape. Also,here we can find the limitation of pointnent framework which does not capture local context/structure net = tf_util.max_pool2d(net, [num_point,1],..)

```
def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {} # store the tranformation matrix


    # input tranf. for ensuring the tranformation invariance
    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1) #4d tensor

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)                                                    
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    # feature tranf. for ensuring the tranformation invariance
    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    # KEY PART: after the CNN, we get the redundant info for each point(N*1024)
    # Using pooling(max,avg) we can hopefully obtain the intesrsting pts(salient representations--global discriptor) which proves to correspond to the skeleton of the shape.
    # Also,here we can find the limitation of pointnent framework which does not capture local context/structure
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool') # Bx1x1x1024

                             
    net = tf.reshape(net, [batch_size, -1]) # Bx1024
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points
```

- loss/objective function; joint loss, pay attention to the softmax cross entropy type. Check the comment in the code.

```
def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES, the one-hot encoding format
        label: B, not the one-hot encoding format, label is 0 to K-1 --yc
        So use the sparse_softmax_cross_entropy_with_logits, for details check:
        https://stackoverflow.com/questions/37312421/whats-the-difference-between-sparse-softmax-cross-entropy-with-logits-and-softm
    """
    
    # loss shape 32*1
    # cross-entropy vs softmax cross-entropy,check a good blog: https://gombru.github.io/2018/05/23/cross_entropy_loss/
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Add a regularization term to the training loss w.r.t feature transf.
    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.summary.scalar('mat loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight
```

3.train.py;

- create argparse object and config its settings for our training, including epochs,learning rates, etc.

- train function;below is the my understanding in pseudo code.
```pseudo-code
On the default graph
    on the GPU
        get the training set(X,y) in placeholder, construct the computation graph,namely relate pred,loss,train_op(min cost fucntion) with X,y
        add scalars(loss,bn_decay,...) to tf.summary 
    set the config, and create a session
    init the varaibles
    get all tf summaries(`merged` var)
    create train/test writer to write to file

    for epochs
        for batches
            train one epoch: load training set, learn from the mini-batch data, namely,backprop, compute grads, update the weights and biases
            eval one epoch: load test set, evaluate on test set
        save the model every 10 times.
```

4.evaluate.py; used for predicting new data, most of the code is similar to train.py but no learning for weights and biases since no backprop is executed when `session run method` does not involve `optimizer` variable,e.g: adam optimizer which is named `ops['train_op']`.

### FAQ

- how tensorboard is used in this project?

- what are the eval metrics for classification and segmentation tasks?

- how to prepare your own datasets?

- how the blocks in segmentation input data are generated?


- why normalized location is added to form as a 9-dim input in segmentation task?



### semantic segmentation(S3DIS)
**have much in common with classification tasks.**

- model.py is the segmentation model file, similar to classification model but with feature concatenation and more MLP for segmentation.
![](images/pointnet/code-sem_seg.png)

- `train.py`'s ideas are similar to above classi model.

- `batch_inference.py` to pred test set.

- `eval_iou_accuracy.py` to compute mean IoU metric.

- if preparing your own datasets, remember to use `collect_indoor3d_data.py` to generate npy and `gen_indoor3d_h5.py` to h5 files for your own datasets.


## Research questions
### related works on point clouds
- 3d CNN;
- projected images, then apply CNN to classify.
- hand-crafted features;
  - normal
  - intensity;激光雷达的采样的时候一种特性强度信息的获取是激光扫描仪接受装置采集到的回波强度，此强度信息与目标 的表面材质、粗糙度、入射角方向，以及仪器的发射能量，激光波长有关
  - local density, curvature
  - linearity; check [Dimensionality based scale selection in 3D lidar point clouds](https://www.researchgate.net/publication/236846179_Dimensionality_based_scale_selection_in_3D_lidar_point_clouds)
  - vertical feature; check [Weakly supervised segmentation-aided classification of urban scenes from 3d LiDAR point clouds](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-1-W1/151/2017/)

### Among all popular geometric data structures, why point cloud data is favored?

- (TODO) The popular geometric data structures involve mesh, volumetric, multi-view images, and PCD; Except PCD, all those formats have their limitations. 1)for mesh, it is hard to define a particular TIN and quadrangles for DL task. 2)for volumetric, on the one hand, it is computationally expensive for apply 3d cnn(O(n^3)); On the other hand, there will 'a hole' representation when converting PCD to volumentric data meanning that most of the voxels are on the surface of the objects. Evidently, this is not suitable for PCD data. 3) regarding multi-view image representation, it is difficult to define the directions for project PCD into images.
    >volumetric,unncessarily voluminous since most lidar point cloud only has surface points and also computation inefficent when dealing with 3d cnn; mesh, need to decide mesh structures,eg: triangules, quad,etc; multi-view images, need to decide from which angles to generate the images so that the model can have a good performance, info loss.

- PCD is representationally simple; close to raw data which enables end-to-end learning.

### what are learned by PointNet?

Interestingly, PointNet learns a discriminative feature/representation/embedding for each input which is a set of critical interesting points, rougly corresponding to the skeleton of the input. Specifically, this representation is quite informative and robust for representing the input PCD. For example, table points(Nx3) --PointNet--> 1024 vector,later on this representation(1024 vector) can be used to perform classificaiton and segmentation task.

### how does PointNet respect the point cloud data permutation invariance?

PointNet smartly uses a symmetry function(Pointnet(vanilla)) to realize;

- The designed net needs to be invariant to N! permutations.
 ![](images/pointnet/orderless-pointnet.png) 

- Luckily, a symmetry function like add, sum and pooling(max,avg) operation and PointNet(vanilla) can achieve this effect.

-  In PointNet, the author exquistitely construct a symmetry function named PointNet(vanilla) which is composed of MLP, max-pooling and another MLP. Specially, each point in the input will use MLP to be convolved into a high-dimensional point, then pooling is adopted over all points to obtain a global descriptor. finally, use another MLP to digest the global descriptor to perform classifification task. **Obviously, the permutation invariance can be achieved when using a pooling operation.**
![pointnet-vanilla](images/pointnet/pointnet-vannila.png)

### how the PointNet(vanilla) is designed?

![pointnet-vanilla](images/pointnet/pointnet-vannila.png)

above img is the pontnet(vannila) structure.

- if using the simplest form(eg:max or avg pooling), evidently the resulted info is not a good representation of the whole PC data; either a fartheest point or a point roughly in the centroid of the PC data.

- use the point vanilla comprised of 3 parts, h, g, gamma; This is the key part of this paper, how the author manages to propose such an tricky framework is based on this prototype.
  - 1)use MLP(h)-CNN for each point to generate a high-dim redundant info since **the following aggregation step in the (redundant) high-dim space can preserve interesting properties of the geometry**; `BN1C`
  - 2)then aggragate all pts using max/avg pooling, this can still preserve a discrimintive representation and insteresting info for all pts/the geo;`B11C`

  - 3)then use another mlp(r)-fully connected layers to digest the info, hopefully we can do the classi and seg applications.  
  - 4)PointNet vanilla is just a special case of the symmetric function set;you can use a therem to prove that the point vanilla can approximate any functions.
![](images/pointnet/theory-pointnet.png)


### how to respect the data tranformation invariance?

use the STN(spatial tranformer network)/T-Net；
![](images/pointnet/t-net-1.png)
![](images/pointnet/t-net-2.png)

### why pointnet is so robust to data corruption incluiding data insertions and outliers?

- pointnet learns to pick perceptually interesting/critical pts.
![](images/pointnet/robustness-1.png)

- critical pts are those which activate the neuro j in the max pooling process. P.S： the visualization is based on an examples.
![](images/pointnet/robustness-2.png)

## ideas

- can I propose a special net tailored for the seg ?
- can I use kervolutional ideas on point cloud data?

## references

- PointNet
- PointNet++
- Charles QI's Ph.D. thesis.

code
[pointnet tensorflow](https://github.com/charlesq34/pointnet) | [pointnet pytorch](https://github.com/fxia22/pointnet.pytorch) | [pointnet keras](https://github.com/garyli1019/pointnet-keras)
