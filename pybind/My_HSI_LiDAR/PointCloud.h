#ifndef POINTCLOUD_H
#define POINTCLOUD_H
#include <iostream>
#include <memory>
#include <thread>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>

#include <Eigen/Core>
#include <omp.h>

#include "utility.h"
#include "kdtree.h"

namespace PPP{
namespace geometry{

class PointCloud{
public:
    PointCloud(){}
    PointCloud(const std::vector<Eigen::Vector3d> &points){
        points_ = points;
    }
    ~PointCloud(){}
public:

//example=================================================================
    std::string example(){
        return std::string("import My_HSI_LiDAR as HSIL\n ")+
                "import numpy as np\n"+
                "pcd = HSIL.geometry.PointCloud()\n"+
                "np_points = np.random.rand(100, 3)\n"+
                "pcd.points = HSIL.utility.Vector3dVector(np_points)\n"+
                "pcd.NormalEstimate(5)\n"+
                "np.asarray(pcd.normals)";
    }


//test cube cloud============================================================
    void cube_cloud(int size, double longx, double longy,
                    double longz,double offx = 0.,double offy = 0.,double offz = 0.){
        Eigen::Vector3d offvec(offx, offy, offz);
        for(int i = 0; i < size; ++i)
        {
            Eigen::Vector3d dot1(rand()/double(RAND_MAX)*longx,0.,
                                 rand()/double(RAND_MAX)*longz);
            Eigen::Vector3d color1(0.,0.,
                                 1.);
            points_.push_back(dot1+offvec);
            colors_.push_back(color1);
        }

        for(int i = 0; i < size; ++i)
        {
            Eigen::Vector3d dot1(rand()/double(RAND_MAX)*longx,longy,
                                 rand()/double(RAND_MAX)*longz);
            Eigen::Vector3d color1(1.,0.,
                                 0.);
            points_.push_back(dot1+offvec);
            colors_.push_back(color1);
        }

        for(int i = 0; i < size; ++i)
        {
            Eigen::Vector3d dot1(0. ,rand()/double(RAND_MAX)*longy,
                                 rand()/double(RAND_MAX)*longz);
            Eigen::Vector3d color1(0.,1.,
                                 0.);
            points_.push_back(dot1+offvec);
            colors_.push_back(color1);
        }

        for(int i = 0; i < size; ++i)
        {
            Eigen::Vector3d dot1(longx ,rand()/double(RAND_MAX)*longy,
                                 rand()/double(RAND_MAX)*longz);
            Eigen::Vector3d color1(1.,1.,
                                 0.);
            points_.push_back(dot1+offvec);
            colors_.push_back(color1);
        }

        for(int i = 0; i < size; ++i)
        {
            Eigen::Vector3d dot1(rand()/double(RAND_MAX)*longx ,rand()/double(RAND_MAX)*longy,
                                 0.);
            Eigen::Vector3d color1(1.,0.,
                                 1.);
            points_.push_back(dot1+offvec);
            colors_.push_back(color1);
        }

        for(int i = 0; i < size; ++i)
        {
            Eigen::Vector3d dot1(rand()/double(RAND_MAX)*longx ,rand()/double(RAND_MAX)*longy,
                                 longz);
            Eigen::Vector3d color1(0.5,0.5,
                                 1.);
            points_.push_back(dot1+offvec);
            colors_.push_back(color1);
        }
    }





//清除点云===================================================================
    PointCloud &Clear(){
        points_.clear();
        normals_.clear();
        colors_.clear();
        curvatures_.clear();
        neighbors_.clear();
        return *this;
    }
//==========================================================================
    bool HasPoints() const { return points_.size() > 0; }

    bool HasNormals() const {
        return points_.size() > 0 && normals_.size() == points_.size();
    }
    bool HasCurvatures() const {
        return curvatures_.size() > 0 && curvatures_.size() == points_.size();
    }

    bool HasColors() const {
        return points_.size() > 0 && colors_.size() == points_.size();
    }

    bool IsEmpty() const { return !HasPoints(); }

    vector<int> GetNeighbor(int i){
        return neighbors_[i];
    }

    PointCloud &operator+=(const PointCloud &cloud) {
        // We do not use std::vector::insert to combine std::vector because it will
        // crash if the pointcloud is added to itself.
        if (cloud.IsEmpty()) return (*this);
        size_t old_vert_num = points_.size();
        size_t add_vert_num = cloud.points_.size();
        size_t new_vert_num = old_vert_num + add_vert_num;
        if ((!HasPoints() || HasNormals()) && cloud.HasNormals()) {
            normals_.resize(new_vert_num);
            for (size_t i = 0; i < add_vert_num; i++)
                normals_[old_vert_num + i] = cloud.normals_[i];
        } else {
            normals_.clear();
        }
        if ((!HasPoints() || HasColors()) && cloud.HasColors()) {
            colors_.resize(new_vert_num);
            for (size_t i = 0; i < add_vert_num; i++)
                colors_[old_vert_num + i] = cloud.colors_[i];
        } else {
            colors_.clear();
        }
        points_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            points_[old_vert_num + i] = cloud.points_[i];
        return (*this);
    }

    PointCloud operator+(const PointCloud &cloud) const {
        return (PointCloud(*this) += cloud);
    }

    //===================================================================
    geometry::PointCloud& SelectDownSample(
            geometry::PointCloud &output,
            std::vector<int> indices,
            bool invert /* = false */){
        output.Clear();
        bool has_normals = HasNormals();
        bool has_colors = HasColors();
        bool has_curvature = HasCurvatures();

        std::vector<bool> mask = std::vector<bool>(points_.size(), invert);
        for (size_t i : indices) {
            mask[i] = !invert;
        }

        for (size_t i = 0; i < points_.size(); i++) {
            if (mask[i]) {
                output.points_.push_back(points_[i]);
                if (has_normals) output.normals_.push_back(normals_[i]);
                if (has_colors) output.colors_.push_back(colors_[i]);
                if (has_curvature) output.curvatures_.push_back(curvatures_[i]);
            }
        }
        return output;
    }

    //////////////////////===========================
    ///

    geometry::PointCloud project(geometry::PointCloud &project_point_, vector<double> Ln){
        double Lx = Ln[0];
        double Ly = Ln[1];
        double Lz = Ln[2];

        project_point_.points_.resize(points_.size() );
        for(int i = 0; i < points_.size(); ++i)
        {
            double x0 = points_[i](0);
            double y0 = points_[i](1);
            double z0 = points_[i](2);

            double lambda = z0/Lz;

            project_point_.points_[i](0) = x0 - lambda * Lx;
            project_point_.points_[i](1) = y0 - lambda * Ly;
            project_point_.points_[i](2) = lambda;
        }

        return project_point_;
    }

    //kdtree======================================================================
    void ComputeNeighbor(int nn){
        geometry::KDTreeFlann kdtree;
        kdtree.setCloud(points_);
        neighbors_.clear();
        neighbors_.resize(points_.size());
        for(size_t i = 0; i < points_.size(); ++i){
            std::vector<int> new_indices_vec(nn);
            std::vector<double> new_dists_vec(nn);
            kdtree.SearchKNN(points_[i], nn, new_indices_vec,
                             new_dists_vec);
            neighbors_[i] = new_indices_vec;
        }
    }

    void ClearNeighbor(){
        neighbors_.clear();
    }

    bool HasNeighbors(){
        if(neighbors_.size() == 0){
            return false;
        }else{
            return true;
        }
    }


    int NeighborsSize(){
        if(HasNeighbors()){
            return neighbors_[0].size();
        }else{
            return 0;
        }
    }


//计算法线=========================================================================
    void NormalEstimate(){
        normals_.resize(points_.size() );
        curvatures_.resize(points_.size() );

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif

        for (size_t i = 0; i < points_.size(); ++i) {
            Eigen::Vector4d normal0(0., 0., 0., 0.);
            std::vector<Eigen::Vector3d > neighbor_point;

            int number_each_neighbour = static_cast<int>(neighbors_[i].size() );
            for (int j = 0; j < number_each_neighbour; ++j) {
                int point_index = neighbors_[i][j];
                neighbor_point.push_back(points_[point_index] );
            }

            std::vector<double> weights(std::distance(neighbor_point.begin(), neighbor_point.end()), 1.0);
            PCAEstimateNormal(neighbor_point.begin(), neighbor_point.end(), weights, normal0);

            if(normal0(2) >= 0){
                normals_[i](0) = normal0(0);
                normals_[i](1)  = normal0(1);
                normals_[i](2)  = normal0(2);
            }else {
                normals_[i](0)  = -normal0(0);
                normals_[i](1)  = -normal0(1);
                normals_[i](2)  = -normal0(2);
            }

            curvatures_[i] = normal0(3);
        }
    }

    void NormalEstimatek(int nn, bool fast_normal_computation){
        geometry::KDTreeFlann kdtree;
        kdtree.setCloud(points_);
        normals_.resize(points_.size() );
        curvatures_.resize(points_.size() );

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < points_.size(); ++i) {
            std::vector<int> new_indices_vec(nn);
            std::vector<double> new_dists_vec(nn);
            kdtree.SearchKNN(points_[i], nn, new_indices_vec,
                             new_dists_vec);
            Eigen::Vector4d vec = ComputeNormal(*this, new_indices_vec, fast_normal_computation);
            normals_[i](0) = vec(0);
            normals_[i](1) = vec(1);
            normals_[i](2) = vec(2);

            curvatures_[i] = vec(3);
        }
    }



    Eigen::Vector4d ComputeNormal(const PointCloud &cloud,
                                  const std::vector<int> &indices,
                                  bool fast_normal_computation) {
        if (indices.size() == 0) {
            return Eigen::Vector4d::Zero();
        }
        Eigen::Matrix3d covariance;
        Eigen::Matrix<double, 9, 1> cumulants;
        cumulants.setZero();
        for (size_t i = 0; i < indices.size(); i++) {
            const Eigen::Vector3d &point = cloud.points_[indices[i]];
            cumulants(0) += point(0);
            cumulants(1) += point(1);
            cumulants(2) += point(2);
            cumulants(3) += point(0) * point(0);
            cumulants(4) += point(0) * point(1);
            cumulants(5) += point(0) * point(2);
            cumulants(6) += point(1) * point(1);
            cumulants(7) += point(1) * point(2);
            cumulants(8) += point(2) * point(2);
        }
        cumulants /= (double)indices.size();
        covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
        covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
        covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
        covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
        covariance(1, 0) = covariance(0, 1);
        covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
        covariance(2, 0) = covariance(0, 2);
        covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
        covariance(2, 1) = covariance(1, 2);

        if (fast_normal_computation) {
            return FastEigen3x3(covariance);
        } else {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
            solver.compute(covariance, Eigen::ComputeEigenvectors);
            Eigen::Vector3d normal = solver.eigenvectors().col(0);
//            double curvature = solver.eigenvalues().minCoeff() / solver.eigenvalues().sum();
            return Eigen::Vector4d(normal(0), normal(1), normal(2), 0.);
        }
    }
//==========================================================================


    Eigen::Vector3d GetMinBound() const{
        BoundingBox box(points_);
        return box.min_values;
    }

    Eigen::Vector3d GetMaxBound() const{
        BoundingBox box(points_);
        return box.max_values;
    }



public:
    std::vector<Eigen::Vector3d> points_;
    std::vector<Eigen::Vector3d> colors_;
    std::vector<Eigen::Vector3d> normals_;
    std::vector<double> curvatures_;
    std::vector<std::vector<int> > neighbors_;
};

}//namespace geometry

}//namespace PPP

#endif //POINTCLOUD_H
