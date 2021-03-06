
# Hyperspectral-LiDAR-processing
Functions for Hyperspectral images and LiDAR point cloud processing, such as point cloud supervoxel segmentation, HSI and LiDAR fusion, etc.

To use this code, you need to install open3d and opencv, and complie the c++ codes with pybind11. We have precomplied the c++ codes under ubuntu 18.04 and MacOS Big Sur enviroment. 

# usage

### LiDAR segmentation and visualazation
Generate supervoxels and visualize the supervoxel map.
```python
>>import My_HSI_LiDAR as HSIL
>>import numpy as np
##load LiDAR
>>LiDAR = np.load("./testData/testLiDAR.npy", allow_pickle = True)
>>cloud = HSIL.geometry.PointCloud()
>>cloud.points = HSIL.utility.Vector3dVector(LiDAR.item()['points'])
>>colorIII = np.asarray(LiDAR.item()['III'], np.float64)/5000
>>cloud.colors = HSIL.utility.Vector3dVector(colorIII)

##supervoxel seg
>>FHSup = HSIL.segmentation.FH_supvoxel()
>>FHSup.setCloud(cloud)
>>FHSup.FindNeighbor(50)
>>FHSup.compute_edge()
>>FHSup.graph_seg(0.6,50)

##visualization
>>sup_map = FHSup.Generate_supmap(HSIL.geometry.PointCloud())
```
### LiDAR to DSM
Generate Digital Surface Model (DSM) from a given LiDAR point cloud
```python
>>import My_HSI_LiDAR as HSIL
>>import numpy as np
##load LiDAR
>>LiDAR = np.load("./testData/testLiDAR.npy", allow_pickle = True)
>>cloud = HSIL.geometry.PointCloud()
>>cloud.points = HSIL.utility.Vector3dVector(LiDAR.item()['points'])

##generate dsm
>>CTI = HSIL.resample.CloudToImg(cloud)
>>CTI.initialize()
## Specify the resolution of DSM
>>CTI.setGeodis(1.)
>>CTI.compute_idn()
>>CTI.compute_point_in_grid()
>>dsm = CTI.compute_dsm()

##visualization
Iutility = HSIL.Iutility.Img_utility()
mask = np.ones(dsm.shape)
dsm_smooth = Iutility.inpaintZ(dsm, mask, 1, 1,  -1000)
dsm_vis = Iutility.visualizeDEM(dsm_smooth, np.min(dsm_smooth), 0.1)
```

### LiDAR and HSI fusion
Generate illumination-invariant hyperspectral point cloud. 
see example.py
