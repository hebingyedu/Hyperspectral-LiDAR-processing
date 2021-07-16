#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# + {}
"""
Created on Fri Jun 19 15:43:30 2020

functions to deal with geotiff and PointCloud data
list:
    loadData: read a geotiff as 3-D cube
    cropTiff: crop tiff to a sub image
    cropTiff_P: crop tiff to a sub image

@author: xudong jin
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, osr
import open3d as o3d
import matplotlib.cm as cmx
import matplotlib.colors as colors

from scipy.signal import convolve2d as conv2

# +
from matplotlib.colors import hsv_to_rgb,rgb_to_hsv

def mat2array(x):
    conv = []
    if len(x.size)==2:
        m_height = x.size[0]
        m_width = x.size[1]
        m_size = m_height*m_width
        
        for i in range(m_height):
            lip = x._data[i::m_height].tolist()
            conv.append(lip)
    elif len(x.size)==3:
        m_height = x.size[0]
        m_width = x.size[1]
        m_channel = x.size[2]
        m_size = m_height*m_width 
        for i in range(m_height):
            c=[]
            for j in range(m_width):
                lip = x._data[(j*m_height+i)::m_size].tolist()
                c.append(lip)
            conv.append(c)
        
    return np.asarray(conv)

    
    
def getBorderNormals(V):
    d = 5

    B =( conv2(np.double(~V),np.array([[0.,1.,0.],[1.,1.,1.],
                                      [0.,1.,0.]]),mode='same')>0) & V

    Bi,Bj = np.where(B)

    P = np.c_[Bi,Bj]

    x, y = np.meshgrid(np.arange(-d,d+1),np.arange(-d,d+1))

    gaussian = np.exp(-5*(x**2+y**2)/(d**2))

    P = P[np.all((P+d < V.shape) & (P+d >= 0),1),:]

    N = np.nan* np.zeros((P.shape[0], 2))

    for i in range(P.shape[0]):  
        patch = V[-d+P[i,0]:d+1+P[i,0],-d+P[i,1]:d+1+P[i,1]]

        ii, jj = np.where(patch)

        # a = np.zeros((patch.size, patch.size))

        # a[patch.ravel()[:,np.newaxis] & patch.ravel()[:,np.newaxis].T]=((( ii[:,np.newaxis] - ii[:,np.newaxis].T)**2+
        # ( jj[:,np.newaxis] - jj[:,np.newaxis].T)**2) <= 2).ravel()

        patch = patch*gaussian

        patch_i, patch_j = np.where(patch)

        v = patch[patch_i, patch_j]

        patch_i = patch_i - (d)
        patch_j = patch_j - (d)

        n = -np.array([np.mean(patch_i*v), np.mean(patch_j*v)])

        n = n/np.sqrt(np.sum(n**2))

        N[i,:] = n

    T = np.c_[-N[:,1], N[:,0]]
    B = {}
    B['idx'] = np.ravel_multi_index([P[:,0],P[:,1]], V.shape)
    B['position'] = P
    B['normal'] = N
    B['tangent'] = T
    
    return B

def conv3(X,f):
    """ 
    输入数据X和卷积模板f，先对X的周边用重叠数据进行扩展，在进行卷积 
    输入：X: m*n 矩阵
         f: 3*3 卷积模板
    输出: Xc: m*n 矩阵
    """  
    Xc = conv2(X, f, 'same')

    Xc[-1,:] = (Xc[-1,:][:,np.newaxis]+
               conv2(X[-1,:][:,np.newaxis],f[0,:][:,np.newaxis], 'same')).ravel()

    Xc[0,:] = (Xc[0,:][:,np.newaxis]+
               conv2(X[0,:][:,np.newaxis],f[-1,:][:,np.newaxis], 'same')).ravel()

    Xc[:,-1] = (Xc[:,-1][:,np.newaxis]+
               conv2(X[:,-1][:,np.newaxis],f[:,0][:,np.newaxis], 'same')).ravel()

    Xc[:,0] = (Xc[:,0][:,np.newaxis]+
               conv2(X[:,0][:,np.newaxis],f[:,-1][:,np.newaxis], 'same')).ravel()

    Xc[0,0] = Xc[0,0] + X[0,0]*f[-1,-1]
    Xc[0,-1] = Xc[0,-1] + X[0,-1]*f[-1,0]
    Xc[-1,0] = Xc[-1,0] + X[-1,0]*f[0,-1]
    Xc[-1,-1] = Xc[-1,-1] + X[-1,-1]*f[0,0]
    return Xc

def getNormals_filters():
    f1 = np.array([[1,2,1], [0,0,0], [-1,-2,-1]], np.float)/8
    f2 = np.array([[1,0,-1], [2,0,-2], [1,0,-1]], np.float)/8
    f1m = f1.ravel()[::-1].reshape(3,3)
    f2m = f2.ravel()[::-1].reshape(3,3)
    return f1,f2,f1m,f2m

def getNormals_conv(Z):    
    f1,f2,f1m,f2m = getNormals_filters() 

    n1 = conv3(Z, f1m)
    n2 = conv3(Z, f2m)
    N3 = 1./np.sqrt(n1**2+n2**2+1)
    N1 = n1*N3
    N2 = n2*N3

    N = np.concatenate((N2[:,:,np.newaxis],N1[:,:,np.newaxis],N3[:,:,np.newaxis]), axis=2)

    N123 = -(N1*N2*N3)

    N3sq = N3**2

    dN_Z = {}
    dN_Z['F1_1'] = (1-N1*N1)*N3
    dN_Z['F1_2'] = N123
    dN_Z['F1_3'] = -N1*N3sq

    dN_Z['F2_1'] = N123
    dN_Z['F2_2'] = (1- N2*N2)*N3
    dN_Z['F2_3'] = -N2*N3sq

    dN_Z['f1'] = f1
    dN_Z['f2'] = f2
    
    return N, dN_Z, n1, n2

def visualizeZ(Z, contrast = 0.75, mask = None):
    if mask is None:
        mask = np.ones(Z.shape, np.uint8)
    N = getNormals_conv(Z)[0]

    idns = np.ravel_multi_index(np.where(mask>0), mask.shape)
    m_height, m_width = Z.shape

    Z_valid = Z.ravel()[idns]
    max_value = np.max(Z_valid)
    min_value = np.min(Z_valid)

    if(max_value == min_value):
        Z_valid = Z_valid - min_value;
    else:
        Z_valid = (Z_valid - min_value)/(max_value -min_value );

    Z_valid = Z_valid*0.75

    Z_valid_ = np.zeros((m_height*m_width, 1))
    Z_valid_[idns,0] = Z_valid

    Z_valid_ = Z_valid_.reshape(Z.shape)


    N3 = N[:,:,2]

    max_value = np.max(N3)
    min_value = np.min(N3)

    if(max_value == min_value):
        N3 = N3 - min_value;
    else:
        N3 = (N3 - min_value)/(max_value -min_value );

    N3 = N3*contrast + (1-contrast)


    Z_valid_ = np.asarray(Z_valid_*180, np.uint8)
    N3 = np.asarray(N3*255, np.uint8)

    V2 = np.asarray(np.ones((Z.shape[0],Z.shape[1],1))*0.75*255, np.uint8)

    vis = np.concatenate((Z_valid_[:,:,np.newaxis], V2,
                         N3[:,:,np.newaxis]),axis=2)
    vis = cv2.cvtColor(vis, cv2.COLOR_HSV2BGR)

    plt.figure()
    plt.imshow(vis)
    plt.show()
#     return vis
def yuv_to_rgb(yuv):
    rgb = np.zeros(yuv.shape,np.float)
    rgb[:,:,1] = yuv[:,:,0]-0.25*(yuv[:,:,1]+yuv[:,:,2])
    rgb[:,:,0] = yuv[:,:,2] + rgb[:,:,1]
    rgb[:,:,2] = yuv[:,:,1] + rgb[:,:,1]
    return rgb

def visualizeNormals(Z):
    N = getNormals_conv(Z)[0]
    N = N[:,:,[2,0,1]]
    V = yuv_to_rgb(N)
    V[V>1]=1
    V[V<0]=0
    
    hsv = rgb_to_hsv(V)
    hsv[:,:,2]=N[:,:,0]
    V=hsv_to_rgb(hsv)
    
    V[np.isnan(V)]=1
    plt.figure()
    plt.imshow(V)
    plt.show()
#     return V

def visualizeNormals_color(N):
    N = N[:,:,[2,0,1]]
    V = yuv_to_rgb(N)
    V[V>1]=1
    V[V<0]=0
    
    hsv = rgb_to_hsv(V)
    hsv[:,:,2]=N[:,:,0]
    V=hsv_to_rgb(hsv)
    
    V[np.isnan(V)]=1
    plt.figure()
    plt.imshow(V)
    plt.show()
# -



def loadData(file):
    dataset = gdal.Open(file, gdal.GA_ReadOnly)
    gt_ref = dataset.GetGeoTransform()
    prj_ref = dataset.GetProjection()
    n_channel = dataset.RasterCount
    HSI = {}
    HSI["gt"] = gt_ref
    HSI["prj"] = prj_ref
    if(n_channel == 1):
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        dataset = None
        HSI["data"] = data
        return HSI
    if(n_channel > 1):
        datas = []
        for bandk in range(n_channel):
            band = dataset.GetRasterBand(bandk+1)
            data = band.ReadAsArray()
            datas.append(data[:,:,np.newaxis] )
            HSI["data"] = np.concatenate(datas, 2)
        return HSI

def crop_cloud(lidar_data, indices):
    sub_data = {}
    for name in lidar_data:
        sub_data[name] = lidar_data[name][indices]
    return sub_data

def cropTiff(HSI, x_start, x_end, y_start, y_end):
    TiffImage = HSI["data"]
    prj_ref = HSI["prj"]
    gt_ref = HSI["gt"]


    sub_HSI = {}
    sub_HSI['data'] = TiffImage[y_start:y_end, x_start:x_end, :]
    sub_HSI['prj'] = prj_ref
    sub_HSI["gt"] = [gt_ref[0] + gt_ref[1]*x_start, gt_ref[1], gt_ref[2],
               gt_ref[3] + gt_ref[5]*y_start, gt_ref[4], gt_ref[5]]

    return sub_HSI


def cropTiff_P(HSI, Px_start, Px_end, Py_start, Py_end):
    TiffImage = HSI["data"]
    prj_ref = HSI["prj"]
    gt_ref = HSI["gt"]

    m_height = TiffImage.shape[0]
    m_width = TiffImage.shape[1]

    x_start = max(int((Px_start - gt_ref[0])/gt_ref[1]), 0)
    x_end = int((Px_end - gt_ref[0])/gt_ref[1])
    y_start = max(-int((gt_ref[3] - Py_start)/gt_ref[5]), 0)
    y_end = -int((gt_ref[3] - Py_end)/gt_ref[5])


    sub_HSI = {}
    if(TiffImage.ndim == 3):
        sub_HSI['data'] = TiffImage[y_start:y_end, x_start:x_end, :]
    else:
        sub_HSI['data'] = TiffImage[y_start:y_end, x_start:x_end]
    sub_HSI['prj'] = prj_ref
    sub_HSI["gt"] = [gt_ref[0] + gt_ref[1]*x_start, gt_ref[1], gt_ref[2],
               gt_ref[3] + gt_ref[5]*y_start, gt_ref[4], gt_ref[5]]

    return sub_HSI


def PixelToGeo(x_start, x_end, y_start, y_end, gt_ref):
    Px_start = x_start*gt_ref[1] + gt_ref[0]
    Px_end = x_end*gt_ref[1] + gt_ref[0]
    Py_start = gt_ref[3] + gt_ref[5]*y_start
    Py_end = gt_ref[3] + gt_ref[5]*y_end
    return Px_start, Px_end, Py_start, Py_end


def GeoToPixel(Px_start, Px_end, Py_start, Py_end, gt_ref):
    x_start = int((Px_start - gt_ref[0])/gt_ref[1])
    x_end = int((Px_end - gt_ref[0])/gt_ref[1])
    y_start = int((gt_ref[3] - Py_start)/gt_ref[5])
    y_end = int((gt_ref[3] - Py_end)/gt_ref[5])
    return x_start, x_end, y_start, y_end


def GdalSaveTiff(saveName, HSI):
    TiffImage = HSI["data"]
    prj_ref = HSI["prj"]
    geotransform = HSI["gt"]
    if(TiffImage.dtype == 'uint8'):
        dataType = gdal.GDT_Byte
    elif(TiffImage.dtype == 'uint16'):
        dataType = gdal.GDT_UInt16
    elif(TiffImage.dtype == 'float32'):
        dataType = gdal.GDT_Float32
    elif(TiffImage.dtype == 'float64'):
        dataType = gdal.GDT_Float64
    else:
        dataType = gdal.GDT_Float32

    '''
    Function to save geotiff
    dataType: gdal.GDT_Byte, gdal.GDT_UINT16, gdal.GDT_Float32
    '''
    if(TiffImage.ndim == 2):
        # Write output
        nrows, ncols = TiffImage.shape
        driver = gdal.GetDriverByName('Gtiff')
        Newdataset = driver.Create(saveName, ncols, nrows, 1, dataType)
        Newdataset.SetGeoTransform(geotransform)
        srs = osr.SpatialReference(wkt=prj_ref)
        Newdataset.SetProjection(srs.ExportToWkt())
        Newdataset.GetRasterBand(1).WriteArray(TiffImage)
        Newdataset = None
    if(TiffImage.ndim == 3):
        # Write output
        nrows, ncols, ndim = TiffImage.shape
        driver = gdal.GetDriverByName('Gtiff')
        Newdataset = driver.Create(saveName, ncols, nrows, ndim, dataType)
        Newdataset.SetGeoTransform(geotransform)
        srs = osr.SpatialReference(wkt=prj_ref)
        Newdataset.SetProjection(srs.ExportToWkt())
        for i in range(ndim):
            Newdataset.GetRasterBand(i+1).WriteArray(TiffImage[:,:,i])
        Newdataset = None

def view_cube(Im, band, size4, mode, mode2 = 0, camp = 'jet', delta = 0.02):
    size1, size2, size3 = Im.shape
    
    vis = ImTransform(Im, band, mode2, None, delta, delta)
    vis = np.asarray(vis, np.float32)/255
    
    Im = (Im - np.min(Im))/(np.max(Im) - np.min(Im))
    
    x = np.arange(size2)
    y = np.arange(size1-1, -1, -1)
    z = np.arange(size3-1, -1, -1)
    xv, yv = np.meshgrid(x, y)
    yv = np.asarray(yv.ravel(), np.float32)
    xv = np.asarray(xv.ravel(), np.float32)
    zv = np.zeros(xv.shape) + 0.1
    
    xv1, zv1 = np.meshgrid(x, z)
    xv1 = np.asarray(xv1.ravel(), np.float32)
    zv1 = np.asarray(zv1.ravel(), np.float32)
    yv1 = np.zeros(xv1.shape) + 1
    yv1_2 = np.ones(xv1.shape)*(size1-1) +1
    
    yv2, zv2 = np.meshgrid(y, z)
    yv2 = np.asarray(yv2.ravel(), np.float32)
    zv2 = np.asarray(zv2.ravel(), np.float32)
    xv2 = np.zeros(yv2.shape)
    xv2_2 = np.ones(yv2.shape)*(size2-1)
    
    color = [vis[j,i, :][np.newaxis, :] for j in y for i in x]
    HSIpoints = np.concatenate([xv[:, np.newaxis], size1 - yv[:, np.newaxis], zv[:, np.newaxis]], 1)
    HSIcolor = np.concatenate(color, 0)

    cloud_hsil = o3d.geometry.PointCloud()
    # colors = CloudF.ClTransform(HSIcolor, [0,1,2])
    cloud_hsil.points = o3d.utility.Vector3dVector(HSIpoints)
    cloud_hsil.colors = o3d.utility.Vector3dVector(HSIcolor)
    
    color1 = [(np.log(i+1)+1)*Im[0,j, i] for i in z for j in x]
    color1_2 = [(np.log(i+1)+1)*Im[size1 - 1,j, i] for i in z for j in x]
    color2 = [(np.log(i+1)+1)*Im[j,0, i] for i in z for j in y]
    color2_2 = [(np.log(i+1)+1)*Im[j,size2-1, i] for i in z for j in y]
    
    if(mode == 0):

        xv_all = np.concatenate([xv1, xv1, xv2, xv2_2], 0)
        yv_all = np.concatenate([yv1, yv1_2, size1-yv2, size1-yv2], 0)
        zv_all = np.concatenate([zv1, zv1, zv2, zv2],0)
        zv_all = size4 - (zv_all - np.min(zv_all))/(np.max(zv_all) - np.min(zv_all))*size4
        
        up = np.max([np.max(c) for c in [color1, color1_2, color2, color2_2]])
        color_norm = colors.Normalize(vmin=0, vmax=up)
        scalar_map = cmx.ScalarMappable(norm=color_norm, cmap = camp) 
        
        HSIpoints1 = np.concatenate([xv_all[:, np.newaxis], yv_all[:, np.newaxis], -zv_all[:, np.newaxis]], 1)
        HSIcolor1 = np.concatenate([scalar_map.to_rgba(np.asarray(color))[:,:3] for color in [color1_2, color1, color2, color2_2]], 0)
        
        cloud_hsil1 = o3d.geometry.PointCloud()
        cloud_hsil1.points = o3d.utility.Vector3dVector(HSIpoints1)
        cloud_hsil1.colors = o3d.utility.Vector3dVector(HSIcolor1)
        
        o3d.visualization.draw_geometries([cloud_hsil, cloud_hsil1])
        return cloud_hsil + cloud_hsil1
        
    if(mode == 1):
        up = np.max([np.max(c) for c in [color1, color2_2]])
        color_norm = colors.Normalize(vmin=0, vmax=up)
        scalar_map = cmx.ScalarMappable(norm=color_norm, cmap = camp) 
        
        zv1 = (zv1 - np.min(zv1))/(np.max(zv1) - np.min(zv1))*size4
        zv2 = (zv2 - np.min(zv2))/(np.max(zv2) - np.min(zv2))*size4
        
        xv1_3 = xv1 - zv1*0.5 + np.max(zv1)*0.5 - 0.5
        zv1_3 = zv1 - zv1*0.5 

        HSIpoints2 = np.concatenate([xv1_3[:, np.newaxis] +1, size1+np.max(zv1_3)-zv1_3[:, np.newaxis] + 0.5, np.zeros(xv1.shape)[:, np.newaxis]], 1)
        HSIcolor2 = scalar_map.to_rgba(np.asarray(color1))[:,:3]

        cloud_hsil2 = o3d.geometry.PointCloud()
        # colors = CloudF.ClTransform(HSIcolor, [0,1,2])
        cloud_hsil2.points = o3d.utility.Vector3dVector(HSIpoints2)
        cloud_hsil2.colors = o3d.utility.Vector3dVector(HSIcolor2)



        yv2_3 = yv2 + zv2*0.5 - np.max(zv2)*0.5 - 0.5
        zv2_3 = zv2 - zv2*0.5 

        HSIpoints3 = np.concatenate([size2+np.max(zv2_3)-zv2_3[:, np.newaxis],size1-yv2_3[:, np.newaxis], np.zeros(yv2.shape)[:, np.newaxis]], 1)
        HSIcolor3 = scalar_map.to_rgba(np.asarray(color2_2))[:,:3]

        cloud_hsil3 = o3d.geometry.PointCloud()
        # colors = CloudF.ClTransform(HSIcolor, [0,1,2])
        cloud_hsil3.points = o3d.utility.Vector3dVector(HSIpoints3)
        cloud_hsil3.colors = o3d.utility.Vector3dVector(HSIcolor3)


        o3d.visualization.draw_geometries([cloud_hsil, cloud_hsil2, cloud_hsil3])
        return cloud_hsil + cloud_hsil2 + cloud_hsil3


def HSIShow(img, k, mask = []):
    if(mask != []):
        mask = np.ones((img.shape[0], img.shape[1]), np.uint8)
        
    RGB_vis = np.zeros(0)
    RGB_vis = cv2.normalize(img, RGB_vis, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3, mask = mask)
    RGB_visg = np.asarray(RGB_vis, np.float32)
    RGB_visg = RGB_visg/np.max(RGB_visg)
    RGB_visg = np.power(RGB_visg, k)
    RGB_visg = cv2.normalize(RGB_visg, RGB_visg, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3, mask=mask)

    
    plt.figure()
    plt.imshow(RGB_visg)
    plt.show()
    return RGB_visg

def ImTransform(img, bands, methods, mask = None, delta1 = 0.02, delta2 = 0.02):
    if(img.ndim == 2):
        if mask is None:
            mask = np.ones((img.shape[0], img.shape[1]), np.uint8)
            
        im = cv2.normalize(img[:,:], np.zeros(0), 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3, mask = mask)
        idns = np.ravel_multi_index(np.where(mask>0), mask.shape)
        im_roi = im.ravel()[idns]

        hist = cv2.calcHist([im_roi],[0],None,[256],[0,256])
        hist = hist/idns.shape[0]

        for i in range(1, hist.shape[0]):
            hist[i,0] = hist[i,0] + hist[i-1,0]

        low = np.where(np.abs(hist - delta1) == np.min(np.abs(hist - delta1) ))[0][0]
        top = np.where(np.abs(hist - 1 + delta2) == np.min(np.abs(hist - 1 + delta2) ))[0][0]

        im_roi = np.asarray(im_roi, np.float32)

        imT = (im_roi - low)*255/(top - low)

        imT[imT < 0] = 0
        imT[imT > 255] = 255
        imT = np.asarray(imT, np.uint8)

        imT_T = np.zeros((im.shape[0] * im.shape[1], 1), np.uint8)
        imT_T[idns, 0] = imT

        imT_T = imT_T.reshape((im.shape[0],  im.shape[1]))
        
        plt.figure()
        plt.imshow(imT_T, cmap = 'gray')
        plt.show()
    else:
        if(methods == 0):
            Trans = []
            if mask is None:
                mask = np.ones((img.shape[0], img.shape[1]), np.uint8)
            for band in bands:
                Trans.append(cv2.normalize(img[:,:,band], np.zeros(0), 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3, mask = mask))
            Trans = np.concatenate([band[:,:, np.newaxis] for band in Trans], 2)
            plt.figure()
            plt.imshow(Trans)
            plt.show()
            return Trans
        elif(methods ==1):
            Trans = []
            if mask is None:
                mask = np.ones((img.shape[0], img.shape[1]), np.uint8)
            for band in bands:
                im = cv2.normalize(img[:,:,band], np.zeros(0), 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3, mask = mask)
                idns = np.ravel_multi_index(np.where(mask>0), mask.shape)
                im_roi = im.ravel()[idns]

                hist = cv2.calcHist([im_roi],[0],None,[256],[0,256])
                hist = hist/idns.shape[0]

                for i in range(1, hist.shape[0]):
                    hist[i,0] = hist[i,0] + hist[i-1,0]

                low = np.where(np.abs(hist - delta1) == np.min(np.abs(hist - delta1) ))[0][0]
                top = np.where(np.abs(hist - 1 + delta2) == np.min(np.abs(hist - 1 + delta2) ))[0][0]

                im_roi = np.asarray(im_roi, np.float32)

                imT = (im_roi - low)*255/(top - low)

                imT[imT < 0] = 0
                imT[imT > 255] = 255
                imT = np.asarray(imT, np.uint8)

                imT_T = np.zeros((im.shape[0] * im.shape[1], 1), np.uint8)
                imT_T[idns, 0] = imT

                Trans.append(imT_T.reshape((im.shape[0],  im.shape[1])))

            Trans = np.concatenate([band[:,:, np.newaxis] for band in Trans], 2)
            plt.figure()
            plt.imshow(Trans)
            plt.show()
            return Trans



