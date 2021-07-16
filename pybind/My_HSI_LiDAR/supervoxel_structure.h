#ifndef SUPERVOXEL_STRUCTURE_H
#define SUPERVOXEL_STRUCTURE_H
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
#include <numeric>
#include<ctime>

using namespace std;
#include "PointCloud.h"
#include "kdtree.h"
#include "utility.h"

namespace PPP{
namespace segmentation{
class supervoxel_structure
{
public:
    supervoxel_structure(){};
    ~supervoxel_structure(){};
/* Input:  points (point cloud), labels ( labels of supervoxel)
 *
 * output:
 */

    void setCloud(const geometry::PointCloud &cloud )
    {    
        cloud_ = cloud;
        n_points_ = cloud_.points_.size();
    }

    void setVoxLabel(vector<int> & labels){
        labels_ = labels;

        auto itr_max = std::max_element(labels_.begin(), labels_.end());
        n_supervoxels_ = *(itr_max) +1;

        supervoxels_.clear();

        supervoxels_.resize(n_supervoxels_);

        for(int i = 0; i < n_points_; ++i){
            int indice = labels_[i];
            supervoxels_[indice].indices_.push_back(i);
//            supervoxels_[indice].point_.push_back(points_[i]);
        }

        for(int i = 0; i < n_supervoxels_; ++i) supervoxels_[i].intialize(cloud_);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    //计算超体素的边界的点
    void compute_voxBoundary(int num_neigh){

        if(cloud_.neighbors_.size() == 0){
            cloud_.ComputeNeighbor(num_neigh);
        }else if(cloud_.neighbors_[0].size() != num_neigh){
            cloud_.ComputeNeighbor(num_neigh);
        }

        for (vector<voxel>::iterator voxIter = supervoxels_.begin(); voxIter != supervoxels_.end(); voxIter++) {

            voxIter->boundaries_.clear();


            for (int i = 0; i < voxIter->size_ ; ++i) {
                int index = voxIter->indices_[i];
                int number_each_neighbour = num_neigh;
                for (int j = 0; j < number_each_neighbour; ++j) {
                    int point_index = cloud_.neighbors_[index][j];
                    if(labels_[point_index ] != labels_[index ] ){
                        voxIter->boundaries_.push_back(index );
                        break;
                    }
                }
            }

            voxIter->neighbors_.clear();
            for (int i = 0; i < voxIter->boundaries_.size(); ++i) {
                int index = voxIter->boundaries_[i];
                int number_each_neighbour = num_neigh;
                for (int j = 0; j < number_each_neighbour; ++j) {
                    int point_index = cloud_.neighbors_[index][j];
                    if(labels_[point_index ] != labels_[index ] ){
                        voxIter->neighbors_.push_back(labels_[point_index ]);
                    }
                }
            }

            sort(voxIter->neighbors_.begin(), voxIter->neighbors_.end() );
            voxIter->neighbors_.erase(unique(voxIter->neighbors_.begin(), voxIter->neighbors_.end() ), voxIter->neighbors_.end() );
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////

    vector<int> GetNeighbors(int i){
       return supervoxels_[i].neighbors_;
    }

    ///

    geometry::PointCloud& GetCentroid(geometry::PointCloud& output){
        for(int i = 0; i < n_supervoxels_; ++i){
            output.points_.push_back(supervoxels_[i].centroid_);
            output.normals_.push_back(supervoxels_[i].centroid_normal_);
            output.curvatures_.push_back(supervoxels_[i].centroid_curvature_);
            if(cloud_.HasColors()){
                output.colors_.push_back(supervoxels_[i].centroid_colors_);
            }
        }
        return output;
    }

    vector<int> GetIndice(int i){
       return supervoxels_[i].indices_;
    }

    geometry::PointCloud& GetMeanColor(geometry::PointCloud& output){
        output.Clear();
        bool has_normals = cloud_.HasNormals();
        bool has_colors = cloud_.HasColors();
        bool has_curvature = cloud_.HasCurvatures();

        std::vector<Eigen::Vector3d>::iterator pointItr = cloud_.points_.begin();
        std::vector<Eigen::Vector3d>::iterator normalItr = cloud_.normals_.begin();
//        std::vector<Eigen::Vector3d>::iterator colorItr = cloud_.colors_.begin();
        std::vector<double>::iterator curvatureItr = cloud_.curvatures_.begin();

        for (int i = 0; i < cloud_.points_.size(); i++) {
            output.points_.push_back(pointItr[i]);
            if (has_normals) output.normals_.push_back(normalItr[i]);
            if (has_colors) output.colors_.push_back(supervoxels_[labels_[i]].centroid_colors_);
            if (has_curvature) output.curvatures_.push_back(curvatureItr[i]);
        }
        return output;
    }

    void project(vector<double> Ln){
        for(int i = 0; i < n_supervoxels_; ++i)
            supervoxels_[i].project(cloud_, Ln[0], Ln[1], Ln[2]);
    }
    ////////////////////////
    geometry::PointCloud& GetProjectCentroid(geometry::PointCloud& output){
        for(int i = 0; i < n_supervoxels_; ++i){
            output.points_.push_back(supervoxels_[i].project_centroid_);
        }
        return output;
    }
    vector<int> light_project(double margin, double z_th){
        // 计算超像素投影到xy平面后,超像素外接矩形斜边的最大长度
        double max_radius = 0;
        for(int i = 0; i < n_supervoxels_; ++i){
            Eigen::Vector3d min_values, max_values;
            min_values = supervoxels_[i].project_box_.min_values;
            max_values = supervoxels_[i].project_box_.max_values;
            double bevel = 0;
            bevel += (max_values(0) - min_values(0))*(max_values(0) - min_values(0));
            bevel += (max_values(1) - min_values(1))*(max_values(1) - min_values(1));
            bevel = sqrt(bevel);
            if(max_radius < bevel){
                max_radius = bevel;
            }
        }


        geometry::PointCloud centroid_project;
        centroid_project = GetProjectCentroid(centroid_project);
        centroid_project.ComputeNeighbor(50);


        vector<bool> is_shadow;
        for(int i = 0; i < n_supervoxels_; ++i){
            double centroid_x =  centroid_project.points_[i](0);
            double centroid_y =  centroid_project.points_[i](1);
    //        double height = ss.supervoxels_[i].project_height_;
            double height = supervoxels_[i].project_box_.max_values(2);

            bool is_shadow_ = false;

            for(int j = 0; j < centroid_project.neighbors_[i].size(); ++j){
                int k = centroid_project.neighbors_[i][j];
                double min_x = supervoxels_[k].project_box_.min_values(0);
                double min_y = supervoxels_[k].project_box_.min_values(1);
                double max_x = supervoxels_[k].project_box_.max_values(0);
                double max_y = supervoxels_[k].project_box_.max_values(1);

                min_x = min_x-margin;
                min_y = min_y-margin;
                max_x = max_x+margin;
                max_y = max_y+margin;

                if((centroid_x >= min_x) && (centroid_x <= max_x) &&
                        (centroid_y >= min_y) && (centroid_y <= max_y)){
                    if(height <=  supervoxels_[k].project_box_.min_values(2) - z_th){
                        is_shadow_ = true;
                        break;
                    }
                }
            }
            is_shadow.push_back(is_shadow_);
        }

        vector<int> shadow_point;
        for(int i = 0; i < n_supervoxels_; ++i){
            for(int j = 0; j < supervoxels_[i].size_; ++j){
                int index = supervoxels_[i].indices_[j];
                if(is_shadow[i] == true){
                    shadow_point.push_back(index);
                }
            }
        }

        return shadow_point;
    }


    bool test(){
        return cloud_.HasColors();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    void Clear()
    {
        cloud_.Clear();
//        point_neighbours_.clear();
        supervoxels_.clear();
        labels_.clear();

    }


    struct voxel{
        voxel(){}

        void intialize(geometry::PointCloud &cloud){
            vector<Eigen::Vector3d> point_;
            std::vector<Eigen::Vector3d>::iterator pointItr = cloud.points_.begin();
            for(int i = 0; i < indices_.size() ; ++i){
                point_.push_back(pointItr[indices_[i]]);
            }

            size_ = indices_.size();
            box_ = geometry::BoundingBox(point_);

            // 计算超体素法线
            std::vector<double> weights(std::distance(point_.begin(), point_.end() ), 1.);
            centroid_ = geometry::Centroid3D(point_.begin(), point_.end(), weights);
            Eigen::Vector4d normal0(0., 0., 0., 0.);
            geometry::PCAEstimateNormal(point_.begin(), point_.end(), weights, normal0 );

            centroid_normal_(0) = normal0(0);
            centroid_normal_(1) = normal0(1);
            centroid_normal_(2) = normal0(2);

            if(centroid_normal_(2) < 0) centroid_normal_ *= -1;

            centroid_curvature_ = normal0(3);

            if(cloud.HasColors()){
                vector<Eigen::Vector3d> color_;
                std::vector<Eigen::Vector3d>::iterator colorItr = cloud.colors_.begin();
                for(int i = 0; i < indices_.size() ; ++i){
                    color_.push_back(colorItr[indices_[i]]);
                }
                centroid_colors_ = geometry::Centroid3D(color_.begin(), color_.end(), weights);
            }
        }

        void project(geometry::PointCloud &cloud, double Lx, double Ly, double Lz){
            vector<Eigen::Vector3d> point_;
            std::vector<Eigen::Vector3d>::iterator pointItr = cloud.points_.begin();
            for(int i = 0; i < indices_.size() ; ++i){
                point_.push_back(pointItr[indices_[i]]);
            }

            vector<Eigen::Vector3d> project_point_;
            project_point_.resize(point_.size() );
            for(int i = 0; i < point_.size(); ++i)
            {
                double x0 = point_[i](0);
                double y0 = point_[i](1);
                double z0 = point_[i](2);

                double lambda = z0/Lz;

                project_point_[i](0) = x0 - lambda * Lx;
                project_point_[i](1) = y0 - lambda * Ly;
                project_point_[i](2) = lambda;
            }


            geometry::BoundingBox project_box(project_point_);
            project_box_ = project_box;

            double x0 = centroid_(0);
            double y0 = centroid_(1);
            double z0 = centroid_(2);

            double lambda = z0/Lz;

            project_centroid_(0) = x0 - lambda * Lx;
            project_centroid_(1) = y0 - lambda * Ly;
            project_centroid_(2) = 0;
            project_height_ = lambda;
        }

        //超体素每个点在原始点云中的序号
        vector<int > indices_;
        //超体素每个点的（x,y,z)值
//        vector<Eigen::Vector3d> point_;
        //超体素的边框
        geometry::BoundingBox box_;

        //中心
        Eigen::Vector3d centroid_;
        //整体估计的法线
        Eigen::Vector3d centroid_normal_;
        //整体估计的曲率
        double centroid_curvature_;
        //整体估计的法线
        Eigen::Vector3d centroid_colors_;

        //在超体素边界的点
        vector<int> boundaries_;
        //邻近的超体素序号
        vector<int> neighbors_;
        //超体素点的个数

        int size_;

        //超体素投影到xy平面后的边框
        geometry::BoundingBox project_box_;
        //投影后的中心点
        Eigen::Vector3d project_centroid_;
        //投影后的相对高度
        double project_height_;
    };



    //超体素的基本结构
    vector<voxel> supervoxels_;
    //超体素的标签
    vector<int> labels_;
    //原始点云
    geometry::PointCloud cloud_;

    //超体素个数
    int n_supervoxels_;
    //点云点的个数
    int n_points_;

};

}//namespace PPP
}//namespace geomotry
#endif // SUPERVOXEL_STRUCTURE_H
