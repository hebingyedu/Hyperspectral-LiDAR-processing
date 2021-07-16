#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 18:01:27 2020

@author: jxd
"""

import numpy as np



from laspy.file import File

import open3d as o3d
import cv2


def load_LASXYZI(pathname, filename):
    cloud = File(pathname + filename + '.las', mode='r')
    cloud_x = np.float32(cloud.x)[:, np.newaxis]
    cloud_y = np.float32(cloud.y)[:, np.newaxis]
    cloud_z = np.float32(cloud.z)[:, np.newaxis]
    cloud_I = np.float32(cloud.intensity)[:, np.newaxis]
    
    cloud_xyz = np.concatenate([cloud_x, cloud_y, cloud_z], 1)
    cloud_xyz = cloud_xyz
    
    return np.float32(cloud_xyz), cloud_I

def cloud_show(points):
    pcd_c1 = o3d.geometry.PointCloud()
    pcd_c1.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd_c1])

def reFineCloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=50,std_ratio=2.0)
    return ind


def load_LAS2(pathname, filename):
    cloud = File(pathname + filename + '.las', mode='r')
    
    cloud_x = np.float32(cloud.x)[:, np.newaxis]
    cloud_y = np.float32(cloud.y)[:, np.newaxis]
    cloud_z = np.float32(cloud.z)[:, np.newaxis]
    cloud_I = np.float32(cloud.intensity)[:, np.newaxis]
    classification = cloud.classification[:, np.newaxis]
    num_returns = cloud.get_num_returns()[:, np.newaxis]
    get_gps_time = cloud.get_gps_time()[:, np.newaxis]

    cloud_xyzI = np.concatenate([cloud_x, cloud_y, cloud_z, cloud_I, 
                                 classification, num_returns, get_gps_time], 1)

    return cloud_xyzI

def combine(cloud_c1, cloud_c2, cloud_c3 ):
    features = np.concatenate([cloud_c1[:, 4:], cloud_c2[:, 4:], cloud_c3[:, 4:]], 0)
    cloud_all = np.concatenate([cloud_c1[:, :3], cloud_c2[:, :3], cloud_c3[:, :3]], 0)
    
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(cloud_all)
    
    pcd_c1 = o3d.geometry.PointCloud()
    pcd_c1.points = o3d.utility.Vector3dVector(cloud_c1[:, :3])
    pcd1_tree = o3d.geometry.KDTreeFlann(pcd_c1)
    
    indices1 = []
    dis_list1 = []
    for point in cloud_all:
        [k, idx, idis] = pcd1_tree.search_knn_vector_3d(point, 1)
        indices1.append(idx[0])
        dis_list1.append(idis[0])
        
    pcd_c2 = o3d.geometry.PointCloud()
    pcd_c2.points = o3d.utility.Vector3dVector(cloud_c2[:, :3])
    pcd2_tree = o3d.geometry.KDTreeFlann(pcd_c2)

    indices2 = []
    dis_list2 = []
    for point in cloud_all:
        [k, idx, idis] = pcd2_tree.search_knn_vector_3d(point, 1)
        indices2.append(idx[0])
        dis_list2.append(idis[0])
        
    pcd_c3 = o3d.geometry.PointCloud()
    pcd_c3.points = o3d.utility.Vector3dVector(cloud_c3[:, :3])
    pcd3_tree = o3d.geometry.KDTreeFlann(pcd_c3)

    indices3 = []
    dis_list3 = []
    for point in cloud_all:
        [k, idx, idis] = pcd3_tree.search_knn_vector_3d(point, 1)
        indices3.append(idx[0])
        dis_list3.append(idis[0])
        
    Intensity_c1 = cloud_c1[:,3:4][indices1]
    Intensity_c2 = cloud_c2[:,3:4][indices2]
    Intensity_c3 = cloud_c3[:,3:4][indices3]
    
    Intensity_rgb = np.concatenate([Intensity_c1,Intensity_c2,Intensity_c3],1)

    sqr_indice = (np.asarray(dis_list1)<1) & (np.asarray(dis_list2)<1) & (np.asarray(dis_list3)<1) 
    
    return cloud_all[sqr_indice.ravel(), :], Intensity_rgb[sqr_indice.ravel(), :], features[sqr_indice.ravel(), :]


def ClTransform(spectra, bands, delta1 = 0.02, delta2 = 0.02, indices = []):
    Trans = []
    for band in bands:
        if(len(indices)==0):
            im = cv2.normalize(spectra[:,band], np.zeros(0), 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
            low_b = np.min(spectra[:,band])
            top_b = np.max(spectra[:,band])
        else:
            im = cv2.normalize(spectra[:,band][indices], np.zeros(0), 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
            low_b = np.min(spectra[:,band][indices])
            top_b = np.max(spectra[:,band][indices])
            
        hist = cv2.calcHist([im],[0],None,[256],[0,256])
        for i in range(1, hist.shape[0]):
            hist[i,0] = hist[i,0] + hist[i-1,0]
            
        hist = hist/im.shape[0]/im.shape[1]
        
        dif_low = np.abs(hist - delta1)
        dif_top = np.abs(hist - 1 + delta2)
        low = np.where( dif_low == np.min(dif_low) )[0][0]
        top = np.where( dif_top == np.min(dif_top) )[0][0]
        
        im = (spectra[:,band] - low_b)*255/(top_b - low_b)
        im = np.asarray(im, np.float32)[:, np.newaxis]
        imT = (im - low)*255/(top - low)
        imT[imT < 0] = 0
        imT[imT > 255] = 255
        
        Trans.append(imT/255)
        
    Trans = np.concatenate([band for band in Trans], 1)
    return Trans


# +
# def ClTransform(spectra, bands, delta = 0.02):
#     Trans = []
#     for band in bands:
#         im = cv2.normalize(spectra[:,band], np.zeros(0), 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
#         hist = cv2.calcHist([im],[0],None,[256],[0,256])
#         for i in range(1, hist.shape[0]):
#             hist[i,0] = hist[i,0] + hist[i-1,0]
            
#         hist = hist/im.shape[0]/im.shape[1]
            
#         low = np.where(np.abs(hist - delta) == np.min(np.abs(hist-delta) ))[0][0]
#         top = np.where(np.abs(hist - 1 + delta) == np.min(np.abs(hist - 1 + delta) ))[0][0]
        
#         im = np.asarray(im, np.float32)
#         imT = (im - low)*255/(top - low)
#         imT[imT < 0] = 0
#         imT[imT > 255] = 255
        
#         Trans.append(imT/255)
        
#     Trans = np.concatenate([band for band in Trans], 1)
#     return Trans

def imshowmypoint(mypcd):
    cloud_vis = o3d.geometry.PointCloud()
    cloud_vis.points = o3d.utility.Vector3dVector(np.asarray(mypcd.points) )
    cloud_vis.colors = o3d.utility.Vector3dVector(np.asarray(mypcd.colors) ) 
    o3d.visualization.draw_geometries([cloud_vis])
# -


