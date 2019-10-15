# summary


## abstract

- To overcome the limitation of pointnet which does not capture local context within the metric space, PointNet++ is proposaed.

- PointNet++ is **a hierarchical NN which recursively applying pointnet on a nested partitioning of input point set**. Moreover, to tackle the degraded performance on non-uniform point cloud data which has varied densities, a novel set of learning layers(multi scale grouping(MSG) and multi resolution group(MRG)) are proposed to **adaptively combine features from multiple scales.**

- Achieved S.T.O.A performance on challenging benchmarks(e.g: ScanNet, ShapeNet, ModelNet40).

- Experiments show that PointNet++ can learn deep point set features robustly and efficiently.

## introduction

### why introduce hierarchical ideas into the proposal?

The tremendous success of CNNs inspires us that extracting local structure is extremely important for feature learning which will allow better generalization ability to unseen cases.

### general ideas of POintnet++

- partition the input into overlapping local regions by the distance metric.
  
- on the small neighouborhood, extract features capturing fine geometric structures.
  
- such local features are further grouped into larger units and processed to produce higher level features.

- reapeat until obtaining the featues of the whole input.

### 2 key issues

- backbone for feature extraction; use pointnet
- generate overlapping partitions of a point set; use FPS algorithm.

### main contribution

PointNet++ leverages neighouborhood at multiple scales to achieve both robustness and detail capture.
>**an extension of pointnet with heirarchical structure added.**

## Architecture
- review of pointnet;

- Hierarchical point set feature learning; the new achitecture **builds a hierarchical grouping of points and progressively abstract larger and larger regions along the hierarchy**.
  - set abstraction(SA) levels; each SA is composed of sampling and grouping;1) sampling; random sampling and FPS(farthest point sampling) can be used. Considering better coverage of the entire input and generating receptive fields dependent on data, FPS is favored to generate the centroid points. 2)grouping;ball query(radius search) and k-nn can be used to group points. It turns out that ball query is better since it guarantees a fixed region scale thus making local region feature more generalizable across space.
  
  - PointNet layer; the input is N'xKx(d+C), output is N'x(d+C'). 1)Each local region in the output will be abstracted by its centroid point and local features that encode the centroid's neighouborhood.2) Specifically, each local region represented by Kx(d+C)(need coordinates tranlation) are mapped into d+C' using pointnet. therefore, N' local regions, we can get the output N'x(d+C').
  
  - MRG and MSG; To tackle the non-uniform point sets, adaptive layers MRG and MSG which learn to combine features from regions of diff scales are proposed. check the figure that how MRG and MSG are trying to combine features.(Note: this can be clearly understood using the code.) For more robust understanding, check the code analysis section.
  ![](achitecture-MSG-MRG)

  - point feature propagation/upsampling for segmentation task; For semantic segmentation, point features for all the original points should be obtained, but after several set abstraction levels(SAs), we have less points than original points but with higher features. So how to overcome this? Due to the computation cost, **propagating features from subsampled points to the orignal points** is favored rather than sampling all points as centroid points in all SAs. Specifically, the inputs is N_lX(d+C2) and N_(l-1)x(d+C1);1)interpolation;the SA l-1 points is linked directely to the output(N_(l-1)x(d+C1)) but w.o C2 feature info, here **inverse distance weighted average based on KNN** is used to compute the C2 features w.r.t SA N_l points.(check the formula).  2)concatenation; then concatenate SA N_(l-1) points(N_(l-1)x(d+C1)) with newly computed features(N_(l-1)x(C2)), we can get the ouput(N_(l-1)x(d+C1+C2)). 3)unit pointnet;similar to point convolutions(1x1 conv in cnn). 4)repeat this process until propogating features to the original set of points.


## FAQ

### metric space

metric space means a small space within a radius, e.g: 0.5m along a point in euclidean space forms a ball space.

### compared with 3d cnn, what characteristics do the local receptive fields have?

Although the local receptive fields of 3d cnn scan the space w. fixed strides, in pointnet++ they are dependent on both input and the metric, thus are more efficient.

### 