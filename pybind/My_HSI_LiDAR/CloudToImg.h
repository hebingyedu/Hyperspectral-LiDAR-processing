#ifndef CLOUDTOIMG_H
#define CLOUDTOIMG_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <memory>
#include <thread>
#include <cstdio>
#include <vector>
#include <queue>
#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "PointCloud.h"
#include "utility"
#include "ndarray_converter.h"

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;
typedef Eigen::LeastSquaresConjugateGradient<SpMat> Solve;

namespace PPP{
namespace resample{

class CloudToImg{
public:
    CloudToImg(){}
    CloudToImg(const geometry::PointCloud &cloud){
        cloud_ = cloud;
    }
    ~CloudToImg(){}

    void setCloud(const geometry::PointCloud &cloud){
        cloud_ = cloud;
    }

    std::vector<int> get_points_in_grid(int i){
        return points_in_grid[i];
    }

    void initialize(){
        box_ = geometry::BoundingBox(cloud_.points_);
    }

    void setGeodis(double geo_dis){
        geo_dis_ = geo_dis;

        geo_prj_.resize(6);

        geo_prj_[0] = box_.min_values(0);
        geo_prj_[1] = geo_dis_;
        geo_prj_[2] = 0;
        geo_prj_[3] = box_.max_values(1);
        geo_prj_[4] = 0;
        geo_prj_[5] = -geo_dis_;

        resample_width_ = static_cast<int>((box_.max_values(0) - box_.min_values(0))/geo_dis_);
        resample_height_ = static_cast<int>((box_.max_values(1) - box_.min_values(1))/geo_dis_);
    }

    void setGeoPrj(vector<double> geo_prj, int m_height, int m_width){
        geo_prj_ = geo_prj;
        resample_width_ = m_width;
        resample_height_ = m_height;
    }

    void compute_idn(){
        cloud_idn_.clear();
        cloud_idn_u_.clear();
        u_reverse_indices_.clear();
        cloud_idn_.resize(cloud_.points_.size() );
        for(size_t i = 0; i < cloud_.points_.size(); ++i){
            int idx = static_cast<int>( (cloud_.points_[i](0) - geo_prj_[0])/geo_prj_[1]);
            int idy = static_cast<int>( (cloud_.points_[i](1) - geo_prj_[3])/geo_prj_[5]);

            if(idx < 0) idx = 0;
            if(idx >= resample_width_) idx = resample_width_ - 1;
            if(idy < 0) idy = 0;
            if(idy >= resample_height_) idy = resample_height_ - 1;

            cloud_idn_[i] = idy * resample_width_ + idx;
        }

        geometry::myunique(cloud_idn_, cloud_idn_u_, u_reverse_indices_);
    }

    void compute_point_in_grid(){
        points_in_grid.clear();
        points_in_grid.resize(cloud_idn_u_.size() );
        for(int i = 0; i < u_reverse_indices_.size(); ++i){
            int index = u_reverse_indices_[i];
            points_in_grid[index].push_back(i);
        }
    }

    vector<int> SelectByMinMax(double max_th, double min_th){
        vector<int> selected;
        for(int i = 0; i < points_in_grid.size(); ++i){
            vector<double> Z;
            double maxF = -FLT_MAX;
            double minF = FLT_MAX;
            for(int j : points_in_grid[i]){
                if(cloud_.points_[j](2) > maxF){
                    maxF = cloud_.points_[j](2);
                }
                if(cloud_.points_[j](2) < minF){
                    minF = cloud_.points_[j](2);
                }
            }

            if(maxF - minF > max_th){
                for(int j : points_in_grid[i]){
                    if(cloud_.points_[j](2) - minF < min_th){
                        selected.push_back(j);
                    }
                }
            }
        }
        return selected;
    }

    vector<int> SelectByPixel(vector<int> pixels){
        vector<int> selected;
        for(int i : pixels){
            int index;
            for(int j = 0; j < points_in_grid.size(); ++j){
                if(i == cloud_idn_u_[j])
                    index = j;
            }
            for(int j : points_in_grid[index]){
                selected.push_back(j);
            }
        }
        return selected;
    }

    vector<int> compute_idn_noshadow(vector<int> &shadow_idn){
        vector<int> idn_noshadow;
        for(int i = 0; i < cloud_idn_u_.size(); ++i){
            int idn = cloud_idn_u_[i];
            if(!is_element_in_vector(shadow_idn, idn)){
                idn_noshadow.push_back(idn);
            }
        }
        return idn_noshadow;
    }


    SpMat compute_M(vector<int> &shadow_idn){
        std::vector<T> triple;
        int row_M = 0;
        for(int i = 0; i < points_in_grid.size(); ++i){
            int idn = cloud_idn_u_[i];
            if(!is_element_in_vector(shadow_idn, idn)){
                for(int j : points_in_grid[i]){
                    triple.push_back(T(row_M, j, 1.));
                }
                row_M++;
            }
        }

        int col_M;
//        row_M = static_cast<int>(cloud_idn_u_.size() );
        col_M = static_cast<int>(cloud_.points_.size() );
        SpMat M(row_M, col_M);
        M.setFromTriplets(triple.begin(), triple.end() );
        return M;
    }

    bool is_element_in_vector(vector<int> &v,int element){
        vector<int>::iterator it;
        it=find(v.begin(),v.end(),element);
        if (it!=v.end()){
            return true;
        }
        else{
            return false;
        }
    }


    SpMat compute_Sm(vector<int> vox_label, vector<double> Ln, vector<int> &shadow_idn){
        std::vector<T> triple;
        Eigen::Vector3d LnE(Ln[0], Ln[1], Ln[2]);

        int row_M = 0;
        for(int i = 0; i < points_in_grid.size(); ++i){
            int idn = cloud_idn_u_[i];
            if(!is_element_in_vector(shadow_idn, idn)){
                for(int j : points_in_grid[i]){
                    double Mdata = LnE.dot(cloud_.normals_[j])*cloud_.normals_[j](2);
                    triple.push_back(T(row_M, vox_label[j], Mdata));
                }
                row_M++;
            }

        }

        int col_M;
//        row_M = static_cast<int>(cloud_idn_u_.size() );
        col_M = *(std::max_element(vox_label.begin(), vox_label.end() )) + 1;
        SpMat M(row_M, col_M);
        M.setFromTriplets(triple.begin(), triple.end() );
        return M;
    }

    cv::Mat compute_dsm(){
        std::vector<double> dsm_ravel;
        dsm_ravel.resize(resample_width_*resample_height_);
        for(int i = 0; i < dsm_ravel.size(); ++i)
            dsm_ravel[i] = -1000;
        for(int i = 0; i < points_in_grid.size(); ++i){
            int index = cloud_idn_u_[i];
            double elavation = -1000 ;
            for(int j : points_in_grid[i]){
                if(cloud_.points_[j](2) > elavation)
                    elavation = cloud_.points_[j](2);
            }
            dsm_ravel[index] = elavation;
        }
//        cv::Mat DSM  = cv::Mat::zeros(6, 3, CV_64F);

        cv::Mat DSM = cv::Mat(dsm_ravel).reshape(0, resample_height_);
        cv::Mat dsm1;
        DSM.convertTo(dsm1, CV_64F);
        return dsm1;
    }


    geometry::BoundingBox box_;
    geometry::PointCloud cloud_;//点云

    int resample_width_;
    int resample_height_;
    double geo_dis_;//重采样空间分辨率
    std::vector<double> geo_prj_;//重采样后地理映射参数
    std::vector<int> cloud_idn_;//点云重采样后标签
    std::vector<int> cloud_idn_u_;
    std::vector<int> u_reverse_indices_;


    std::vector<std::vector<int> > points_in_grid;
};


}

}

#endif // CLOUDTOIMG_H
