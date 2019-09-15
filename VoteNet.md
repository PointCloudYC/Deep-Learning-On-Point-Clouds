# Deep Hough Voting for 3D Object Detection in Point Clouds
## bg
- 3d obj detection(localization + recognition or say bounding box + semantic classification).
  
- 3d detection apps: vr, ar, autonomous driving, robotics.
  
- 2d obj detection pop works(faster rcnn, mask rcnn)
  
- 3d data representations (mesh,volumetric,point cloud, bird's ey view images)
  
- current status of 3d detection: heavily rely on 2d-based detectors in various aspects
  - voxelize point clouds and apply 3d cnn detector;(high comp cost and fails to leverage sparity in the data)
  -  images plus 2d image detectors(sacrifice geo details)
  -  f-pointnet(strictly dependent on the 2d detector, will miss the object entirely if not detected in 2d)

## summary


## 4 questions
- what did the author try to accomplish?

- what were the key elements of this approach?

- what can u use yourself?

- imp refs you'd like to cite?

## hard 


## resources
[paper](https://arxiv.org/pdf/1904.09664.pdf) / [code](https://github.com/facebookresearch/votenet) / [slides](https://orlitany.github.io/OL_files/talks/deep_hough_voting.pdf)