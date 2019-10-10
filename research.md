# research focus

Research question: "**Can we achieve automated BIM LOD2 model generation from laser scan data using deep learning?**"

Justification of significance and orginiality for your question:

- data availability; quite convenient to obtain data from lidar devices and other sensors; also many benchmark datasets.

- data quality; the point cloud data is of high precision.

- significant values for numerous applications, 1) Facility management; 2) establishment of smart city. 3) defection detection, quality asessment. 4) VR/AR, facility their modeling work. 5) autonomous driving.

## research pipeline

The proposed pipeline is consisted of 5 modules.

- PCD preprocessing: registration and filtering
  
- PCD semantic segmentation(pointnet,pointnet++,pointcnn,dgcnn,splatnet,...), instance segmentation(GSPN,...), detection(VoteNet,...).
  
- Geometry extraction and model retrieval (iterative cross fitting, PCA, SAC-ICP, ...) 

- BIM generation using dynamo

- evaluation the generated model

** The pipeline can be applied on MEP, indoor room, outdoor road scene, so try to propose new nets or add new tricks on the existing nets to solve a particular problem based on your observation and properties of a particular kind of PCD. **

## abbreviation

DL, deep learning
PCD, point cloud processing