#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'notebook')
import cv2
import matplotlib.pyplot as plt
import numpy as np

from osgeo import gdal, osr
import open3d as o3d
import scipy.io as sio

import MyFunc.ImgF as ImgF
import MyFunc.CloudF as CloudF
import MyFunc.IHID_class as IHID_Class
import MyFunc.IHSPC_class as IHSPC_class
import MyFunc.My_HSI_LiDAR as HSIL


# ### load data
foldname = '/boot/jxd/Ndata/UH_2018/synthetic/'
savefold = '/boot/jxd/Ndata/UH_2018/synthetic/'

filename = "synthe_scene_1"

HSI = sio.loadmat(foldname + filename + "_HSI_Ln_0.mat")
HSI.pop('__header__')
HSI.pop('__version__')
HSI.pop('__globals__')

LiDAR = sio.loadmat(foldname + filename + "_data.mat")
LiDAR.pop('__header__')
LiDAR.pop('__version__')
LiDAR.pop('__globals__')

HSI['gt'] = HSI['gt'].ravel()
HSI['Ln'] = HSI['Ln'].ravel()

HSI_ori = ImgF.loadData(foldname + "synthe_scene_1_origin_hsi" + ".tiff")

vis0 = ImgF.ImTransform(HSI['data'],[20, 27, 17], 1)
colors = LiDAR['reflectance'][:,[20, 27, 17]]
colors = colors/np.max(colors)
cloud_hsil = o3d.geometry.PointCloud()
cloud_hsil.points = o3d.utility.Vector3dVector(LiDAR['points'])
cloud_hsil.colors = o3d.utility.Vector3dVector(colors)


o3d.visualization.draw_geometries([cloud_hsil])
HSI_cube = ImgF.view_cube(HSI['data'], [20, 27, 17], 40, 1, 1, 'jet', 0.03)



# ### main
Iutility = HSIL.Iutility.Img_utility()

Intensity = np.asarray(LiDAR['III'][:,1], np.float)
Intensity[Intensity>5000] = 5000
Intensity = Intensity/np.max(Intensity)
IHSPC = IHSPC_class.IHSPC_class()
IHSPC.setPoints(LiDAR['points'])
colorIII = np.asarray(LiDAR['III'], np.float64)[:, [0, 1, 2]]/5000
# IHSPC.setColors(colorIII)
IHSPC.setColors(colors)
IHSPC.setNumReturn(LiDAR['num_return'])
IHSPC.setHSI(HSI)
IHSPC.setIntensity(Intensity)
cloud_hsi_ori = IHSPC.visI() 

# ### DSM_N
IHSPC.Compute_DSM()
dsm_smooth_vis = Iutility.visualizeDEM(IHSPC.dsm, np.min(IHSPC.dsm), 0.1)
IHSPC.Compute_N()
IHSPC.Compute_Intensity()



# ### shadow_remove
IHSPC.Supixel_HSI(0.06, 0.02, 1)
IHSPC.Supixel_DI(0.3, 0.3, 0.2, 5)
IHSPC.remove_shadow(0.2)
IHSPC.SegBuilding(1., 100, 0.15, -13.)
IHSPC.LightEstimate()

# ### cloud_seg

IHSPC.cloud_seg(0.04 , 30, 0.025, 200, 140, 100)
cloud_sup = IHSPC.vis_sup()
IHSPC.Ref_prepare()

IHSPC.solve_R(15, 0.2)
visR = IHSPC.visR()

