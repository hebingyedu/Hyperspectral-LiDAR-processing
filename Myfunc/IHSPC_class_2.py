#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Sat Jun 20 19:28:13 2020

@author: jxd
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os,glob
import math
from collections import defaultdict

from osgeo import gdal, osr
import subprocess
import json

import matplotlib.patches as mpatches

from laspy.file import File
import numpy as np
import open3d as o3d

from scipy.sparse import lil_matrix,csr_matrix
import scipy.linalg as slina
import scipy.sparse.linalg as splinalg

from skimage.segmentation import mark_boundaries

import MyFunc.My_HSI_LiDAR as HSIL
from MyFunc.ImgF import ImTransform
import MyFunc.CloudF as CloudF
from MyFunc.CloudF import imshowmypoint
import ismember.ismember as ismember
import MyFunc.IHID_class as IHID_Class
import MyFunc.My_suppixel as My_suppixel

# %%
class IHSPC_class:
    def __init__(self):
        self.cloud_hsil_ = HSIL.geometry.PointCloud()
    
    def setPoints(self, points):
        self.points_ = points
        self.cloud_hsil_.points = HSIL.utility.Vector3dVector(points)
        self.cloud_hsil_.ComputeNeighbor(50)
        
        cloud_normal = o3d.geometry.PointCloud()
        cloud_normal.points = o3d.utility.Vector3dVector(points)
        cloud_normal.estimate_normals()
        normals = np.asarray(cloud_normal.normals)
        flag = 2*np.asarray(normals[:,2] > 0, np.float) - 1
        normals = normals * flag[:,np.newaxis]
        self.cloud_hsil_.normals = HSIL.utility.Vector3dVector(normals)
        
#         self.cloud_hsil_.NormalEstimate()
        self.normals_ = normals
        
    def setColors(self, colors):
        self.cloud_hsil_.colors = HSIL.utility.Vector3dVector(colors)
        
    def setNumReturn(self, numreturn):
        self.num_returns_ = numreturn
        
    def setHSI(self, sub_HSI):
        self.m_height_ = sub_HSI['data'].shape[0]
        self.m_width_  = sub_HSI['data'].shape[1]
        self.m_channels_ = sub_HSI['data'].shape[2]
        self.gt_ = sub_HSI['gt']
        
#         log_Im = np.log(sub_HSI['data'])

#         log_Im_mean  = np.mean(log_Im, 2)
#         self.log_Im_1 = log_Im - log_Im_mean[:,:, np.newaxis]
#         self.log_Im_1.resize(self.m_height_*self.m_width_, self.m_channels_)

#         Im_HSI = sub_HSI['data'].reshape(self.m_height_*self.m_width_, self.m_channels_)
        Im_HSI = np.asarray(sub_HSI['data'], np.float64)
        Im_HSI = Im_HSI/np.max(Im_HSI) + 0.0000000001

        self.Im_ = Im_HSI
        
        self.vis0 = ImTransform(sub_HSI['data'],[20, 27, 17], 1)
        
        self.CTI = HSIL.resample.CloudToImg()

        self.CTI.setCloud(self.cloud_hsil_)
        self.CTI.initialize()

        self.CTI.setGeoPrj(self.gt_, self.m_height_, self.m_width_)
        self.CTI.compute_idn()
        self.CTI.compute_point_in_grid()
        
    def setIntensity(self, Intensity):
        self.Intensity = Intensity
        
        
    def FHSeg(self, cloud, ratio, min_size, neig_num = 50):
        cloud.ClearNeighbor()
        FHSup = HSIL.segmentation.FH_supvoxel()
        FHSup.setCloud(cloud)
        FHSup.FindNeighbor(neig_num)
        FHSup.compute_edge()
        
        FHSup.graph_seg(ratio, min_size)
        sup_map = FHSup.Generate_supmap(HSIL.geometry.PointCloud())
        
        return np.asarray(FHSup.labels), sup_map
    
    def FHSeg1(self, cloud, ratio, min_size, radius, neig_num = 50):
        cloud.ClearNeighbor()
        FHSup = HSIL.segmentation.FH_supvoxel()
        cloud.ComputeNeighbor1(neig_num, radius)
        FHSup.setCloud(cloud)
#         FHSup.FindNeighbor1(neig_num, radius)
        FHSup.compute_edge()
        
        FHSup.graph_seg(ratio, min_size)
        sup_map = FHSup.Generate_supmap(HSIL.geometry.PointCloud())
        
        return np.asarray(FHSup.labels), sup_map
    
    def SegBuilding(self, ratio1, min_size, ratio2, z_th): 
        self.tree_indices = np.where(self.num_returns_ > 1)[0]
        self.tree_cloud = self.cloud_hsil_.SelectDownSample(HSIL.geometry.PointCloud(), 
                                                       self.tree_indices)
        
        not_trees_indices = np.where(self.num_returns_ == 1)[0]
        
        not_trees_indices = np.sort(not_trees_indices)
        not_tree_cloud = self.cloud_hsil_.SelectDownSample(HSIL.geometry.PointCloud(), 
                                                       not_trees_indices)
        
        points = np.asarray(not_tree_cloud.points)
        self.non_ground_indices = np.where(points[:,2] > z_th)[0]
        self.ground_indices = np.where(points[:,2] <= z_th)[0]
        
        self.non_ground_indices = not_trees_indices[self.non_ground_indices]
        self.ground_indices = not_trees_indices[self.ground_indices]
        
        self.non_ground_indices = np.sort(self.non_ground_indices)
        self.ground_indices = np.sort(self.ground_indices)

        self.non_ground_cloud = self.cloud_hsil_.SelectDownSample(HSIL.geometry.PointCloud(), 
                                                       self.non_ground_indices)
        
        self.ground_cloud = self.cloud_hsil_.SelectDownSample(HSIL.geometry.PointCloud(), 
                                                       self.ground_indices)

        
        
        label_supvox, sup_map = self.FHSeg1(self.non_ground_cloud , ratio1, min_size, 1.)
        imshowmypoint(sup_map)
        
        ss = HSIL.segmentation.supervoxel_structure()
        ss.setCloud(self.non_ground_cloud)
        ss.setVoxLabel(label_supvox)
        ss.compute_voxBoundary(10)
        
        building = []
        tree = []
        buildingS = []
        for i in range(ss.n_supervoxels_):
            cluster = ss.GetIndice(i)
            if(len(cluster) > ratio2):
                building.extend(cluster)
                buildingS.append(i)
            else:
                tree.extend(cluster)
                    
        building_cloud = self.non_ground_cloud.SelectDownSample(HSIL.geometry.PointCloud(),building)
        self.build_indices = self.non_ground_indices[np.sort(building)]
        
        imshowmypoint(building_cloud)
        
        self.building_indices_ = []
        for index in buildingS:
            indices_k = ss.GetIndice(index)
            self.building_indices_.append(indices_k)
#         self.buildingS_ = buildingS

############################################################
              ######################################################3#####
    
    def Compute_DSM(self):
        dsm = self.CTI.compute_dsm()
        
        label = np.ones(dsm.shape)
        Iutility = HSIL.Iutility.Img_utility()
        dsm_smooth = Iutility.inpaintZ(dsm, label, 1, 1,  -1000)
        
        self.dsm = dsm_smooth
        
    def Compute_Intensity(self):
        indptr = [0]
        indices = []
        data = []
        
        n_pixels = len(self.CTI.cloud_idn_u)
        
        for i in range(n_pixels):
            n_points_in_grid = len(self.CTI.get_points_in_grid(i))
            if(n_points_in_grid != 0):
                g = np.ones(n_points_in_grid)
                g = g/np.sum(g)

            data.append(g.ravel())
            indices.extend(self.CTI.get_points_in_grid(i))
            indptr.append(len(indices))
        
        data = np.concatenate(data)
        W = csr_matrix((data, indices, indptr), (n_pixels, self.Intensity.shape[0]))
        img = W.dot(self.Intensity)
        
        Img = np.zeros((self.m_height_*self.m_width_))
        Img[self.CTI.cloud_idn_u] = img
        Img = Img.reshape(self.m_height_, self.m_width_)
        
        Iutility = HSIL.Iutility.Img_utility()
        label = np.ones(Img.shape)
        Img_smooth = Iutility.inpaintZ(Img, label, 1, 1,  0)
        
        self.Img = Img_smooth/np.max(Img_smooth)
        
    def Compute_Intensity1(self, geodis):
        CTI = HSIL.resample.CloudToImg()
        CTI.setCloud(cloud_hsil)
        CTI.initialize()

        CTI.setGeodis(geodis)
        # CTI.setGeoPrj(Im['gt'], Im['data'].shape[0], Im['data'].shape[1])
        CTI.compute_idn()
        CTI.compute_point_in_grid()
        
        indptr = [0]
        indices = []
        data = []
        
        n_pixels = len(CTI.cloud_idn_u)
        
        for i in range(n_pixels):
            n_points_in_grid = len(CTI.get_points_in_grid(i))
            if(n_points_in_grid != 0):
                g = np.ones(n_points_in_grid)
                g = g/np.sum(g)

            data.append(g.ravel())
            indices.extend(CTI.get_points_in_grid(i))
            indptr.append(len(indices))
        
        data = np.concatenate(data)
        W = csr_matrix((data, indices, indptr), (n_pixels, self.Intensity.shape[0]))
        img = W.dot(self.Intensity)
        
        Img = np.zeros((self.m_height_*self.m_width_))
        Img[self.CTI.cloud_idn_u] = img
        Img = Img.reshape(self.m_height_, self.m_width_)
        
        Iutility = HSIL.Iutility.Img_utility()
        label = np.ones(Img.shape)
        Img_smooth = Iutility.inpaintZ(Img, label, 1, 1,  0)
        
        self.Img1 = Img_smooth/np.max(Img_smooth)
        
    def Compute_N(self):
        Mnormals = self.normals_*self.normals_[:,2:3]
        M = self.CTI.compute_M([])
        N = M.dot(Mnormals)/M.dot(self.normals_[:,2:3])
        
        DN = np.ones((self.m_height_*self.m_width_,3))*-1
        DN[self.CTI.cloud_idn_u] = N
        Nimg = DN.reshape(self.m_height_, self.m_width_, 3)
        
        bands = []
        for i in range(Nimg.shape[2]):
            band = Nimg[:,:,i]
            Iutility = HSIL.Iutility.Img_utility()
            label = np.ones(band.shape)
            band_smooth = Iutility.inpaintZ(band, label, 1, 1,  -1)
            bands.append(band_smooth[:,:,np.newaxis])
        self.DN = np.concatenate(bands, 2)
        
        vis = ImTransform(self.DN, [0,1,2], 1)
############################################################
              ######################################################3#####
    

    
  

        

        
############################################################
              ######################################################3#####
    
    def LightEstimate(self):
        Lns = []
        for indices_k in self.building_indices_:
            if(len(indices_k) > 1000):
                junk = HSIL.geometry.PointCloud()
                cloud_segK = self.non_ground_cloud.SelectDownSample(junk, indices_k, False)
                
                CTI = HSIL.resample.CloudToImg()
                CTI.setCloud(cloud_segK)
                CTI.initialize()
                CTI.setGeoPrj(self.gt_, self.m_height_, self.m_width_)
                CTI.compute_idn()
                CTI.compute_point_in_grid()
                
#                 a = self.superpixel_HSI1.ravel()[CTI.cloud_idn_u]
#                 b = np.argmax(np.bincount(a[a>0]))
#                 idn = np.where(self.superpixel_HSI1.ravel() == b)[0]
                idn = CTI.cloud_idn_u
                
                M_L = self.DN.reshape(self.m_height_*self.m_width_, 3)[idn]
                I_k = self.Im_.reshape(self.m_height_*self.m_width_, self.m_channels_)[idn]
                
                Ln = slina.lstsq(M_L,np.mean(I_k, 1))[0]
                Ln = Ln/np.sqrt(np.sum(Ln**2))
                Lns.append(Ln)
                
        zhf = np.sum(Lns, 0)
        Ln = np.sqrt(np.mean(np.multiply(Lns, Lns), 0))
        self.Ln_ = np.where(zhf < 0, -1, 1) * Ln
        
    def FilterCloud(self, cloud, Intensityk, geodis):
        CTI = HSIL.resample.CloudToImg()

        CTI.setCloud(cloud)
        CTI.initialize()

        CTI.setGeodis(geodis)
        # CTI.setGeoPrj(Im['gt'], Im['data'].shape[0], Im['data'].shape[1])
        CTI.compute_idn()
        CTI.compute_point_in_grid()

        indptr = [0]
        indices = []
        data = []

        z = np.asarray(cloud.points)[:,2]

        n_pixels = len(CTI.cloud_idn_u)

        for i in range(n_pixels):
            n_points_in_grid = len(CTI.get_points_in_grid(i))
            if(n_points_in_grid != 0):
                g = np.ones(n_points_in_grid)
                g = g/np.sum(g)

            data.append(g.ravel())
            indices.extend(CTI.get_points_in_grid(i))
            indptr.append(len(indices))

        data = np.concatenate(data)
        W = csr_matrix((data, indices, indptr), (n_pixels, Intensityk.shape[0]))

        img = W.dot(Intensityk)
        Img = np.zeros((CTI.resample_height_*CTI.resample_width))
        Img[CTI.cloud_idn_u] = img
        Img = Img.reshape(CTI.resample_height_, CTI.resample_width)

        Z = W.dot(z)
        dsm = np.zeros((CTI.resample_height_*CTI.resample_width))
        dsm[CTI.cloud_idn_u] = Z
        dsm = dsm.reshape(CTI.resample_height_, CTI.resample_width)

        return Img, dsm, CTI.cloud_idn_
    
    
    def IHID_cloud_seg1(self, indices, cloud_0, sig1, sig2):
        Intensityk = self.Intensity[indices]
        Img, dsm, idn = self.FilterCloud(cloud_0, Intensityk, 0.3)
        
        dsm[Img > 0] = dsm[Img > 0] - np.min(dsm[Img > 0])
        label = np.zeros(Img.shape)
        label[Img > 0] = 1
        
        IH = IHID_Class.IHID_single(Img)
        IH.compute_idx(label)
        IH.compute_diff(sig1)
        IH.get_A()
        IH.solve_R(sig2)
        
        IH.R = IH.R.reshape((IH.R.shape[0]*IH.R.shape[1], 1))
        dsm = dsm.reshape((dsm.shape[0]*dsm.shape[1], 1))
        dsm = dsm/np.max(dsm)
        IH.R = IH.R/np.max(IH.R)
        
        colors = np.concatenate([IH.R[idn],
                                 IH.R[idn],
                                 dsm[idn]],1)
        
        return colors
    
    def IHID_cloud_seg2(self, indices, cloud_0, sig1, sig2, sig3, sig4):
        Intensityk = self.Intensity[indices]
        Img, dsm, idn = self.FilterCloud(cloud_0, Intensityk, 0.4)
        dsm[Img > 0] = dsm[Img > 0] - np.min(dsm[Img > 0])
        
        label = np.zeros(Img.shape)
        label[Img > 0] = 1
        
        IH = IHID_Class.IHID_single(Img)
        IH.compute_idx(label)
        IH.compute_diff(sig1)
        IH.get_A()
        IH.solve_R(sig2)
        RI = IH.R/np.max(IH.R)
        
        IH = IHID_Class.IHID_single(dsm)
        IH.compute_idx(label)
        IH.compute_diff(sig3)
        IH.get_A()
        IH.solve_R(sig4)
        RD = IH.R/np.max(IH.R)
        
        RI = RI.reshape((RI.shape[0]*RI.shape[1], 1))
        RD = RD.reshape((RD.shape[0]*RD.shape[1], 1))

        colors = np.concatenate([RI[idn],
                                 RI[idn],
                                 RD[idn]],1)
        
        return colors
        
    
    def cloud_seg_00(self):
        colors = self.IHID_cloud_seg1(self.ground_indices, self.ground_cloud, 0.08, 0.5)
        self.ground_cloud.colors = HSIL.utility.Vector3dVector(colors)
        
        colors = self.IHID_cloud_seg2(self.non_ground_indices, self.non_ground_cloud, 0.35, 0.5, 0.6, 0.2 )
        self.non_ground_cloud.colors = HSIL.utility.Vector3dVector(colors)
        
        colors = self.IHID_cloud_seg2(self.tree_indices, self.tree_cloud, 0.35, 0.5, 0.6, 0.2)
        self.tree_cloud.colors = HSIL.utility.Vector3dVector(colors)
        
    def cloud_seg_01(self, sig1, min_sz1, radius1, neig_num1 = 50):
        self.label_supvox1, sup_map1 = self.FHSeg1(self.ground_cloud , sig1, min_sz1, radius1, neig_num1)
        imshowmypoint(sup_map1)
        
    def cloud_seg_02(self, sig2, min_sz2, radius2, neig_num2 = 50):
        self.label_supvox2, sup_map2 = self.FHSeg1(self.non_ground_cloud , sig2, min_sz2, radius2, neig_num2)
        imshowmypoint(sup_map2)
        
    def cloud_seg_03(self, sig3, min_sz3, radius3, neig_num3 = 50):
        self.label_supvox3, sup_map3 = self.FHSeg1(self.tree_cloud , sig3, min_sz3, radius3, neig_num3)
        imshowmypoint(sup_map3)
        
    def cloud_seg_04(self):
        num_points = np.asarray(self.cloud_hsil_.points).shape[0]
        self.label_supvox_ = np.zeros(num_points, np.int)
        self.label_supvox_[self.ground_indices] = self.label_supvox1
        self.label_supvox_[self.non_ground_indices] = self.label_supvox2+np.max(self.label_supvox1)+1
        self.label_supvox_[self.tree_indices] = self.label_supvox3+np.max(self.label_supvox1)+np.max(self.label_supvox2)+2
        self.vis_sup()
    
    def cloud_seg(self, sig1, min_sz1, radius1, sig2, min_sz2,radius2, sig3, min_sz3,radius3, neig_num1 = 50, neig_num2 = 50, neig_num3 = 50):
        colors = self.IHID_cloud_seg1(self.ground_indices, self.ground_cloud, 0.08, 0.5)
        self.ground_cloud.colors = HSIL.utility.Vector3dVector(colors)
        label_supvox1, sup_map1 = self.FHSeg1(self.ground_cloud , sig1, min_sz1, radius1, neig_num1)
        
        colors = self.IHID_cloud_seg2(self.non_ground_indices, self.non_ground_cloud, 0.35, 0.5, 0.6, 0.2 )
        self.non_ground_cloud.colors = HSIL.utility.Vector3dVector(colors)
        label_supvox2, sup_map2 = self.FHSeg1(self.non_ground_cloud , sig2, min_sz2, radius2, neig_num2)
        
        
        colors = self.IHID_cloud_seg2(self.tree_indices, self.tree_cloud, 0.35, 0.5, 0.6, 0.2)
        self.tree_cloud.colors = HSIL.utility.Vector3dVector(colors)
        label_supvox3, sup_map3 = self.FHSeg1(self.tree_cloud , sig3, min_sz3,radius3, neig_num3)
        

        num_points = np.asarray(self.cloud_hsil_.points).shape[0]
        self.label_supvox_ = np.zeros(num_points, np.int)
        self.label_supvox_[self.ground_indices] = label_supvox1
        self.label_supvox_[self.non_ground_indices] = label_supvox2+np.max(label_supvox1)+1
        self.label_supvox_[self.tree_indices] = label_supvox3+np.max(label_supvox1)+np.max(label_supvox2)+2
        self.vis_sup()
        
    def vis_sup(self):
        cloud_visR = HSIL.geometry.PointCloud()
        
        color = np.random.random((np.max(self.label_supvox_)+1, 3))

        cloud_visR.points = HSIL.utility.Vector3dVector(self.points_)
        cloud_visR.colors = HSIL.utility.Vector3dVector(color[self.label_supvox_])

        imshowmypoint(cloud_visR)
        return cloud_visR

    
    def segR(self, ratio1, minsize1, ratio2, minsize2, ratio3, minsize3):
        label_supvox1, sup_map1 = self.FHSeg(self.building_cloud , ratio1, minsize1)
        label_supvox2, sup_map2 = self.FHSeg(self.tree_cloud , ratio2, minsize2)
        label_supvox3, sup_map3 = self.FHSeg(self.ground_cloud , ratio3, minsize3)
        
        imshowmypoint(sup_map1+sup_map2+sup_map3)
        
        num_points = np.asarray(self.cloud_hsil_.points).shape[0]
        self.label_supvox_ = np.zeros(num_points, np.int)
        self.label_supvox_[self.build_indices] = label_supvox1
        self.label_supvox_[self.tree_indices] = label_supvox2+np.max(label_supvox1)+1
        self.label_supvox_[self.ground_indices] = label_supvox3+np.max(label_supvox1)+np.max(label_supvox2)+2
        
    def sefR2(self, num):
        TBB = HSIL.segmentation.TBBSupervoxel()
        TBB.setCloud(self.cloud_hsil_)
        TBB.set_n_sup1(num)
        TBB.set_z_scale(10.)
        TBB.FindNeighbor(50)
        TBB.StartSegmentation()
        
        self.label_supvox_ = np.asarray(TBB.labels)
        sup_map = TBB.Generate_supmap(HSIL.geometry.PointCloud())
        imshowmypoint(sup_map)
        
        
        
    def solve_R2(self, sig):
    
        self.label_supvox_ = np.arange(self.points_.shape[0])
        CTI = HSIL.resample.CloudToImg()

        CTI.setCloud(self.cloud_hsil_)
        CTI.initialize()

        CTI.setGeoPrj(self.gt_, self.m_height_, self.m_width_)
        CTI.compute_idn()
        CTI.compute_point_in_grid()

        M = CTI.compute_M(self.shadow_idn_)
        
        self.Ms_ = CTI.compute_Sm(self.label_supvox_, self.Ln_, self.shadow_idn_)
        self.M_Nz_ = M.dot(self.normals_[:,2])[:, np.newaxis]
        
        idn_noshadow = CTI.compute_idn_noshadow(self.shadow_idn_)
        self.ImG_ = self.Im_[idn_noshadow,:]
        
        indptr = [0]
        indices = []
        data = []

        colors = np.asarray(self.cloud_hsil_.colors)
        for i in range(self.points_.shape[0]):
            neigbor =  self.cloud_hsil_.GetNeighbor(i)

            dif = np.sum((colors[neigbor] - colors[i])**2, 1)

            c1_car = np.mean(dif)
            if(c1_car == 0):
                c1_car = 1

            g = np.exp(-(dif/c1_car ))
            g = np.ones(len(neigbor))
            g = g/np.sum(g)

            data.append(g.ravel())
            indices.extend(neigbor)
            indptr.append(len(indices))

        data = np.concatenate(data)
        m = self.points_.shape[0]
        W = csr_matrix((data, indices, indptr), (m,m))

        G = lil_matrix((m, m))
        G.setdiag(sig)
        G = G + self.Ms_.T.dot(self.Ms_) + W.T.dot(W)*sig - W.T*sig - W*sig

        self.R_ = spr.linalg.spsolve(G.tocsc(), self.Ms_.T.dot(self.ImG_*self.M_Nz_))
        
    def Ref_prepare(self):
        self.ss_ = HSIL.segmentation.supervoxel_structure()
        self.ss_.setCloud(self.cloud_hsil_)
        self.ss_.setVoxLabel(self.label_supvox_)

#         self.ss_.compute_voxBoundary(10)
        
        self.centroid_ = self.ss_.GetCentroid(HSIL.geometry.PointCloud() )
#         self.centroid_.ComputeNeighbor(5)
        self.ss_.compute_voxBoundary(50)
        
        CTI = HSIL.resample.CloudToImg()

        CTI.setCloud(self.cloud_hsil_)
        CTI.initialize()

        CTI.setGeoPrj(self.gt_, self.m_height_, self.m_width_)
        CTI.compute_idn()
        CTI.compute_point_in_grid()

        M = CTI.compute_M([])
        
        self.Ms_ = CTI.compute_Sm(self.label_supvox_, self.Ln_, [])
        self.M_Nz_ = M.dot(self.normals_[:,2])[:, np.newaxis]
        
#         idn_noshadow = CTI.compute_idn_noshadow([])
        self.ImG_ = self.Im_.reshape(self.m_height_*self.m_width_, self.m_channels_)[CTI.cloud_idn_u,:]
        
        
        
    def solve_R(self, sig, threshold):
        MeanSup = self.ss_.GetCentroid(HSIL.geometry.PointCloud())
        meanColors = np.asarray(MeanSup.colors)
        ###
        CTI = HSIL.resample.CloudToImg()

        CTI.setCloud(MeanSup)
        CTI.initialize()

        CTI.setGeoPrj(self.gt_, self.m_height_, self.m_width_)
        CTI.compute_idn()
        voxel_idn = CTI.cloud_idn_
        
#         IsShadow = ismember(voxel_idn, self.shadow_idn_)[0]

        n_supvox = self.ss_.n_supervoxels_

        indptr = [0]
        indices = []
        data = []

        for i in range(n_supvox):
            neigbor = self.ss_.GetNeighbors(i)
            
#             c_feature = self.CloudHSIMean_[i]
#             w_feature = self.CloudHSIMean_[neigbor]

            c_feature = meanColors[i]
            w_feature = meanColors[neigbor]
            
            diff = c_feature - w_feature
            diff = np.sqrt(np.mean(diff**2, 1))

            idn_c = np.asarray(neigbor)[diff<threshold]
            
            g = np.ones(len(idn_c))
            g = g/np.sum(g)
            
            data.append(np.append(g, -1.))
            indices.extend(np.append(idn_c, i).tolist())
            indptr.append(len(indices))
            
        data = np.concatenate(data)
        m = n_supvox
        W = csr_matrix((data, indices, indptr), (m,m))

#         G = lil_matrix((m, m))
#         diag = np.ones(m)
# #         diag[IsShadow] = sig2**2
#         G.setdiag(sig*diag)
        G = self.Ms_.T.dot(self.Ms_) + sig*W.T.dot(W)

        self.Rs_ = splinalg.spsolve(G.tocsc(), self.Ms_.T.dot(self.ImG_*self.M_Nz_))
        
#         for i in range(n_supvox):
#             neigbor = self.ss_.GetNeighbors(i)

#             if((np.mean(self.Rs_[i]) < 0.1) and (len(neigbor)>0)):
#                 c_feature = meanColors[i]
#                 w_feature = meanColors[neigbor]

#                 diff = c_feature - w_feature
#                 diff = np.sqrt(np.mean(diff**2, 1))
#                 self.Rs_[i] = self.Rs_[neigbor[np.argmin(diff)]]
        
        self.R_ = self.Rs_[self.label_supvox_]
        self.G_ = G
        self.W_ = W
    
            
        
        
    def visR(self):
        cloud_visR = HSIL.geometry.PointCloud()
        colors = np.concatenate([self.R_[:, 20][:,np.newaxis],
                         self.R_[:, 27][:,np.newaxis],
                         self.R_[:, 17][:,np.newaxis]],1)

        cloud_visR.points = HSIL.utility.Vector3dVector(self.points_)
        cloud_visR.colors = HSIL.utility.Vector3dVector(colors)

        imshowmypoint(cloud_visR)
        return cloud_visR
        
    def visI(self):
        CTI = HSIL.resample.CloudToImg()

        CTI.setCloud(self.cloud_hsil_)
        CTI.initialize()

        CTI.setGeoPrj(self.gt_, self.m_height_, self.m_width_)
        CTI.compute_idn()
        CTI.compute_point_in_grid()
        
        cloud_visR = HSIL.geometry.PointCloud()
        colors = np.concatenate([self.Im_.reshape(self.m_height_*self.m_width_, self.m_channels_)[CTI.cloud_idn_,20][:,np.newaxis],
                         self.Im_.reshape(self.m_height_*self.m_width_, self.m_channels_)[CTI.cloud_idn_, 27][:,np.newaxis],
                         self.Im_.reshape(self.m_height_*self.m_width_, self.m_channels_)[CTI.cloud_idn_, 17][:,np.newaxis]],1)\
        
        colors = CloudF.ClTransform(colors, [0,1,2])

        cloud_visR.points = HSIL.utility.Vector3dVector(self.points_)
        cloud_visR.colors = HSIL.utility.Vector3dVector(colors)
        
        imshowmypoint(cloud_visR)
        return cloud_visR
        
    def shadow_test(self, grid_size, max_th, min_th):
        junk = HSIL.geometry.PointCloud()
        cloud_prject = self.cloud_hsil_.project(junk, self.Ln_)
        
        CTI = HSIL.resample.CloudToImg()

        CTI.setCloud(cloud_prject)
        CTI.initialize()
        CTI.setGeodis(grid_size)
        # CTI.setGeoPrj(HSIHID.gt_, HSIHID.m_height_, HSIHID.m_width_)
        CTI.compute_idn()
        CTI.compute_point_in_grid()

        shadow_index = CTI.SelectByMinMax(max_th, min_th)

        junk = HSIL.geometry.PointCloud()
        shadow_cloud = self.cloud_hsil_.SelectDownSample(junk, shadow_index, False)
        
        imshowmypoint(shadow_cloud)
        
        CTI = HSIL.resample.CloudToImg()

        CTI.setCloud(shadow_cloud)
        CTI.initialize()
        # CTI.setGeodis(0.5)
        CTI.setGeoPrj(self.gt_, self.m_height_, self.m_width_)
        CTI.compute_idn()
        CTI.compute_point_in_grid()
        
        self.shadow_idn_ = CTI.cloud_idn_u
        
        SH = np.zeros(self.Im_.shape)
        SH[self.shadow_idn_] = self.Im_[self.shadow_idn_]

        SH.resize(self.m_height_, self.m_width_, self.m_channels_)

        vis = ImTransform(SH, [20,27,17], 1)

# %%

# %%
