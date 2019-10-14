# summary


## abstract

- To overcome the limitation of pointnet which does not capture local context within the metric space, PointNet++ is proposaed.

- PointNet++ is **a hierarchical NN which recursively applying pointnet on a nested partitioning of input point set**. Moreover, to tackle the degraded performance on non-uniform point cloud data which has varied densities, a novel set of learning layers(multi scale grouping(MSG) and multi resolution group(MRG)) are proposed to **adaptively combine features from multiple scales.**

- Achieved S.T.O.A performance on challenging benchmarks(e.g: ScanNet, ShapeNet, ModelNet40).

- Experiments show that PointNet++ can learn deep point set features robustly and efficiently.