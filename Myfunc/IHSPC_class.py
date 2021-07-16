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
    
    def SegBuilding(self, ratio1, min_size, ratio2, z_th):   
        points = np.asarray(self.cloud_hsil_.points)
        self.non_ground_indices = np.where(points[:,2] > z_th)[0]
        self.ground_indices = np.where(points[:,2] <= z_th)[0]
        
        self.non_ground_indices = np.sort(self.non_ground_indices)
        self.ground_indices = np.sort(self.ground_indices)

        self.non_ground_cloud = self.cloud_hsil_.SelectDownSample(HSIL.geometry.PointCloud(), 
                                                       self.non_ground_indices)
        self.ground_cloud = self.cloud_hsil_.SelectDownSample(HSIL.geometry.PointCloud(), 
                                                       self.ground_indices)
        
        
        label_supvox, sup_map = self.FHSeg(self.non_ground_cloud , ratio1, min_size)
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
            num_return_sum = np.sum( self.num_returns_[self.non_ground_indices][cluster] > 1)
            if(num_return_sum/len(cluster) < ratio2):
                building.extend(cluster)
                buildingS.append(i)
            else:
                tree.extend(cluster)
                    
        self.building_cloud = self.non_ground_cloud.SelectDownSample(HSIL.geometry.PointCloud(),building)
        self.build_indices = self.non_ground_indices[np.sort(building)]
                    
        self.tree_cloud = self.non_ground_cloud.SelectDownSample(HSIL.geometry.PointCloud(),tree)
        self.tree_indices = self.non_ground_indices[np.sort(tree)]
        imshowmypoint(self.building_cloud)
        
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
    
    def Supixel_HSI(self, sig1, sig2, size1):
        mask = np.ones((self.m_height_, self.m_width_))

        # mask[:100, :100] = 1
        IH = IHID_Class.IHID(self.Im_)
        IH.compute_idx(mask)
        IH.compute_diff(sig1)
        IH.get_A()
        IH.solve_Rk(0.01)
        
        self.superpixel_HSI, mask = self.HSI_FH(IH.R, sig2, size1)
        supMask1 = mark_boundaries(self.vis0, self.superpixel_HSI)
        
        plt.figure()
        plt.imshow(supMask1)
        plt.show()
        
        self.superpixel_HSI1 = self.superpixel_HSI.copy()
        self.superpixel_HSI1[mask == 1] = -1
    
    def Supixel_DI(self, sig1, sig2, sig3, size1):
        dsm_t = (self.dsm-np.min(self.dsm))/(np.max(self.dsm) - np.min(self.dsm))
        self.RD = self.IHID(dsm_t, sig1, 0.01)
        self.RI = self.IHID(self.Img, sig2, 0.01)
        self.DImg = np.concatenate([self.RD[:,:,np.newaxis], self.RI[:,:,np.newaxis], self.RI[:,:,np.newaxis]], 2)
        
        self.superpixel_DI, mask = self.HSI_FH(self.DImg, sig3, size1)
        supMask1 = mark_boundaries(self.vis0, self.superpixel_DI)
        
        plt.figure()
        plt.imshow(supMask1)
        plt.show()
        
        
    def remove_shadow(self, sig):
        Im_mean = np.mean(self.vis0/255, 2)
        self.Im_ = self.Im_.reshape(self.m_height_*self.m_width_, self.m_channels_)
        label = np.asarray(self.superpixel_DI, np.float64)
        for i in range(int(np.max(label)+1)):
            l0 = np.where(label.ravel() == i)[0]
            k0 = Im_mean.ravel()[l0]
            k1 = l0[k0 < sig]
            k2 = l0[k0 > sig]

            if((len(k1)>0) and (len(k2)>0)):
                self.Im_[k1] = np.mean(self.Im_[k2], 0)
        self.Im_ = self.Im_.reshape(self.m_height_, self.m_width_, self.m_channels_)
        vis = ImTransform(self.Im_,[20, 27, 17], 1)
#         self.Im_[Im_mean < sig] = -1
        
        
#         label = np.asarray(self.superpixel_DI, np.float64)
#         bands = []
#         for i in range(self.Im_.shape[2]):
#             band = self.Im_[:,:,i]
#             Iutility = HSIL.Iutility.Img_utility()
#             band_smooth = Iutility.inpaintZ(band, label, 1, 0,  -1)
#             bands.append(band_smooth[:,:,np.newaxis])
#         self.Im_ = np.concatenate(bands, 2)
        
#         vis = ImTransform(self.Im_,[20, 27, 17], 1)
        
    def HSI_FH(self, img, sig1, sig2):
        HSI_FH = My_suppixel.segmentation.HSI_FH()
        HSI_FH.setHSI(img)
        HSI_FH.compute_edge()
        HSI_FH.graph_seg(sig1, sig2)
        
        return HSI_FH.GetSuppixelLabel(), HSI_FH.Generate_supmask(False)
    
    def IHID(self, img, sig1, sig2):
        mask = np.ones((img.shape[0], img.shape[1]))

        IH = IHID_Class.IHID_single(img)
        IH.compute_idx(mask)
        IH.compute_diff(sig1)
        IH.get_A()
        IH.solve_R(sig2)
        
        return IH.R/np.max(IH.R)
        
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
    
    def cloud_seg(self, sig1, min_sz1, sig2, min_sz2, neig_num1 = 50, neig_num2 = 50):
        Intensityk = self.Intensity[self.ground_indices]
        Img, dsm, idn = self.FilterCloud(self.ground_cloud, Intensityk, 0.3)
        
        dsm[Img > 0] = dsm[Img > 0] - np.min(dsm[Img > 0])
        label = np.zeros(Img.shape)
        label[Img > 0] = 1
        
        IH = IHID_Class.IHID_single(Img)
        IH.compute_idx(label)
        IH.compute_diff(0.08)
        IH.get_A()
        IH.solve_R(0.5)
        
        IH.R = IH.R.reshape((IH.R.shape[0]*IH.R.shape[1], 1))
        dsm = dsm.reshape((dsm.shape[0]*dsm.shape[1], 1))
        dsm = dsm/np.max(dsm)
        IH.R = IH.R/np.max(IH.R)

        colors = np.concatenate([IH.R[idn],
                                 IH.R[idn],
                                 dsm[idn]],1)

        self.ground_cloud.colors = HSIL.utility.Vector3dVector(colors)
        
        label_supvox1, sup_map1 = self.FHSeg(self.ground_cloud , sig1, min_sz1, neig_num1)
        
        Intensityk = self.Intensity[self.non_ground_indices]
        Img, dsm, idn = self.FilterCloud(self.non_ground_cloud, Intensityk, 0.4)
        dsm[Img > 0] = dsm[Img > 0] - np.min(dsm[Img > 0])
        
        label = np.zeros(Img.shape)
        label[Img > 0] = 1
        
        IH = IHID_Class.IHID_single(Img)
        IH.compute_idx(label)
        IH.compute_diff(0.35)
        IH.get_A()
        IH.solve_R(0.5)
        RI = IH.R/np.max(IH.R)
        
        IH = IHID_Class.IHID_single(dsm)
        IH.compute_idx(label)
        IH.compute_diff(0.6)
        IH.get_A()
        IH.solve_R(0.2)
        RD = IH.R/np.max(IH.R)
        
        RI = RI.reshape((RI.shape[0]*RI.shape[1], 1))
        RD = RD.reshape((RD.shape[0]*RD.shape[1], 1))

        colors = np.concatenate([RI[idn],
                                 RI[idn],
                                 RD[idn]],1)

        self.non_ground_cloud.colors = HSIL.utility.Vector3dVector(colors)
        label_supvox2, sup_map2 = self.FHSeg(self.non_ground_cloud , sig2, min_sz2, neig_num2)
        
        num_points = np.asarray(self.cloud_hsil_.points).shape[0]
        self.label_supvox_ = np.zeros(num_points, np.int)
        self.label_supvox_[self.ground_indices] = label_supvox1
        self.label_supvox_[self.non_ground_indices] = label_supvox2+np.max(label_supvox1)+1
        self.vis_sup()
        
    def vis_sup(self):
        cloud_visR = HSIL.geometry.PointCloud()
        
        color = np.random.random((np.max(self.label_supvox_)+1, 3))

        cloud_visR.points = HSIL.utility.Vector3dVector(self.points_)
        cloud_visR.colors = HSIL.utility.Vector3dVector(color[self.label_supvox_])

        imshowmypoint(cloud_visR)
        return cloud_visR

#     def LightEstimate(self):
#         ML2 = self.compute_M_L_2()
#         MLL = ML2.T.dot(ML2)
#         eig, vec = np.linalg.eig(MLL)
#         i = np.where(eig == np.min(eig))[0][0]
#         Ln = vec[:,i]
#         if(Ln[2]>0):
#             self.Ln_ = Ln
#         else:
#             self.Ln_ = -Ln
        
#     def compute_M_L_2(self):
#         ML2 = []
#         for indices_k in self.building_indices_:
#             junk = HSIL.geometry.PointCloud()
#             cloud_segK = self.non_ground_cloud.SelectDownSample(junk, indices_k, False)
#             normal_k = np.asarray(cloud_segK.normals)
#             ML1 = self.compute_M_L_1(cloud_segK, normal_k)
#             ML2.append(ML1)
#         return np.concatenate(ML2, 0)
    
#     def compute_M_L_1(self, cloud_segK, normal_k):
#         M_k, idn = self.compute_M_seg(cloud_segK)

#         N_up = M_k.dot(normal_k*normal_k[:, 2:3])
#         N_down = M_k.dot(normal_k[:, 2:3])

#         M_L = N_up / N_down
#         I_k = self.Im_[idn]

#         M_L_mean = np.mean(M_L, 0)
#         I_mean = np.mean(I_k, 0)

#         ML1 = []
#         for i in range(I_mean.shape[0]):
#             ML1.append(M_L*I_mean[i] - I_k[:, i:i+1]*M_L_mean)
#         return np.concatenate(ML1, 0)
    
#     def compute_M_seg(self, cloud_seg):
#         CTI = HSIL.resample.CloudToImg()

#         CTI.setCloud(cloud_seg)
#         CTI.initialize()

#         CTI.setGeoPrj(self.gt_, self.m_height_, self.m_width_)
#         CTI.compute_idn()
#         CTI.compute_point_in_grid()

#         M = CTI.compute_M([])

#         return M, CTI.cloud_idn_u
    
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
        
#         n_supvox = self.ss_.n_supervoxels_
#         CloudHSIMean = []
#         for i in range(n_supvox):
#             indices = self.ss_.GetIndice(i)
#             idn_u = np.unique(np.asarray(CTI.cloud_idn_)[indices])
#             CloudHSIMean.append(np.mean(self.log_Im_1[idn_u], 0)[:, np.newaxis])
            
#         self.CloudHSIMean_ = np.concatenate(CloudHSIMean, 0)  
        
        
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
        
        for i in range(n_supvox):
            neigbor = self.ss_.GetNeighbors(i)

            if((np.mean(self.Rs_[i]) < 0.1) and (len(neigbor)>0)):
                c_feature = meanColors[i]
                w_feature = meanColors[neigbor]

                diff = c_feature - w_feature
                diff = np.sqrt(np.mean(diff**2, 1))
                self.Rs_[i] = self.Rs_[neigbor[np.argmin(diff)]]
        
        self.R_ = self.Rs_[self.label_supvox_]
        self.G_ = G
        self.W_ = W
        
        
#     def Ref_prepare(self):
#         self.ss_ = HSIL.segmentation.supervoxel_structure()
#         self.ss_.setCloud(self.cloud_hsil_)
#         self.ss_.setVoxLabel(self.label_supvox_)

# #         self.ss_.compute_voxBoundary(10)
        
#         self.centroid_ = self.ss_.GetCentroid(HSIL.geometry.PointCloud() )
#         self.centroid_.ComputeNeighbor(5)
        
#         CTI = HSIL.resample.CloudToImg()

#         CTI.setCloud(self.cloud_hsil_)
#         CTI.initialize()

#         CTI.setGeoPrj(self.gt_, self.m_height_, self.m_width_)
#         CTI.compute_idn()
#         CTI.compute_point_in_grid()

#         M = CTI.compute_M(self.shadow_idn_)
        
#         self.Ms_ = CTI.compute_Sm(self.label_supvox_, self.Ln_, self.shadow_idn_)
#         self.M_Nz_ = M.dot(self.normals_[:,2])[:, np.newaxis]
        
#         idn_noshadow = CTI.compute_idn_noshadow(self.shadow_idn_)
#         self.ImG_ = self.Im_[idn_noshadow,:]
        
#         n_supvox = self.ss_.n_supervoxels_
#         CloudHSIMean = []
#         for i in range(n_supvox):
#             indices = self.ss_.GetIndice(i)
#             idn_u = np.unique(np.asarray(CTI.cloud_idn_)[indices])
#             CloudHSIMean.append(np.mean(self.log_Im_1[idn_u], 0)[:, np.newaxis])
            
#         self.CloudHSIMean_ = np.concatenate(CloudHSIMean, 0)
    
        
#     def solve_R(self, sig, threshold):
#         MeanSup = self.ss_.GetCentroid(HSIL.geometry.PointCloud())
#         meanColors = np.asarray(MeanSup.colors)
#         ###
#         CTI = HSIL.resample.CloudToImg()

#         CTI.setCloud(MeanSup)
#         CTI.initialize()

#         CTI.setGeoPrj(self.gt_, self.m_height_, self.m_width_)
#         CTI.compute_idn()
#         voxel_idn = CTI.cloud_idn_
        
#         IsShadow = ismember(voxel_idn, self.shadow_idn_)[0]
        
        
#         n_supvox = self.ss_.n_supervoxels_

#         indptr = [0]
#         indices = []
#         data = []

#         for i in range(n_supvox):
#             neigbor = self.centroid_.GetNeighbor(i)
            
#             c_feature = self.CloudHSIMean_[i]
#             w_feature = self.CloudHSIMean_[neigbor]
            
#             diff = c_feature - w_feature
#             diff = np.sqrt(np.mean(diff**2, 1))

#             idn_c = np.asarray(neigbor)[diff<threshold]
            
#             g = np.ones(len(idn_c))
#             g = g/np.sum(g)
            
#             data.append(np.append(g, -1.))
#             indices.extend(np.append(idn_c, i).tolist())
#             indptr.append(len(indices))
            
#         data = np.concatenate(data)
#         m = n_supvox
#         W = csr_matrix((data, indices, indptr), (m,m))

# #         G = lil_matrix((m, m))
# #         diag = np.ones(m)
# # #         diag[IsShadow] = sig2**2
# #         G.setdiag(sig*diag)
#         G = self.Ms_.T.dot(self.Ms_) + sig*W.T.dot(W)

#         self.Rs_ = splinalg.spsolve(G.tocsc(), self.Ms_.T.dot(self.ImG_*self.M_Nz_))
#         self.R_ = self.Rs_[self.label_supvox_]
#         self.G_ = G
#         self.W_ = W
            
        
        
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
