
# Hyperspectral-LiDAR-processing
Functions for Hyperspectral LiDAR processing, such as point cloud supervoxel segmentation, HSI and LiDAR fusion, etc.

To use this code, you need to install open3d and opencv, and complie the c++ codes with pybind11. We have precomplied the c++ codes under ubuntu 18.04 and MacOS Big Sur enviroment. 

# usage

### LiDAR segmentation and visualazation

```python
>>import My_HSI_LiDAR as HSIL
##generate random points
>>pcd = HSIL.geometry.PointCloud()
>>pcd.cube_cloud(1000,1,1,1,1.5)

##supervoxel seg
>>FHSup = HSIL.segmentation.FH_supervoxel()
>>FHSup.setCloud(pcd)
>>FH.FindNeighbor1(50,2)
>>FH.compute_edge()
>>FH.graph_seg(0.6,50)

##visualization
>>sup_map = FHSup.Generate_supmap(HSIL.geometry.PointCloud()
```
### LiDAR to DSM
```python
>>import My_HSI_LiDAR as HSIL
>>import numpy as np
##generate random points
>>pcd = HSIL.geometry.PointCloud()
>>np_points = np.random.rand(100,3)
>>pcd.points = HSIL.utility.Vector3dVector(np_points)

##generate dsm
>>CTI = HSIL.resample.CloudToImg(pcd)
>>CTI.initializa()CTI.setGeodos(0.01)
>>CTI.compute_idn()
>>CTI.compute_point_in_grid()
>>dsm = CTI.compute_dsm()
```

### LiDAR and HSI fusion
see example.py
