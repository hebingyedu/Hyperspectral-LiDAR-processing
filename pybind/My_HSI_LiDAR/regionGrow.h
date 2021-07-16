#ifndef REGIONGROW_H
#define REGIONGROW_H

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <cstdio>
#include <queue>
#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <numeric>

#include "kdtree.h"
#include "PointCloud.h"

namespace py = pybind11;

namespace PPP{
namespace segmentation{

//template <typename Eigen::Vector3d, typename Eigen::Vector3d>
class RegionGrowing
{
public:
    /////////////////////////////////////////////////////////////////////////////////////////////
    RegionGrowing(){}

    /////////////////////////////////////////////////////////////////////////////////////////////
    ~RegionGrowing(){}

    /////////////////////////////////////////////////////////////////////////////////////////////
    void setCloud(geometry::PointCloud &cloud)
    {
        min_pts_per_cluster_ = 1;
        max_pts_per_cluster_ = std::numeric_limits<int>::max() ,
        theta_threshold_ = 30.0f / 180.0f * static_cast<float>(M_PI) ;
        residual_threshold_ = 0.05f;
        curvature_threshold_ = 0.05f;
        neighbour_number_ = 50;
//        point_neighbours_.clear();
        point_labels_.clear();
        clusters_.clear();
        num_pts_in_segment_.clear();
        number_of_segments_ = 0;
        is_compute_nghbr_ = (true);
        cloud_ = cloud;
    }


    /////////////////////////////////////////////////////////////////////////////////////////////
    //返回分割标签
    std::vector<int> getPoint_Labels(){
        return point_labels_;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////
    //设定最小分割尺寸
    void setMinclusterSize(int min_cluster_size){
        min_pts_per_cluster_ = min_cluster_size;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    //设定最大分割尺寸
    void setMaxClusterSize(int max_cluster_size){
        max_pts_per_cluster_ = max_cluster_size;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    //设定表面平滑阈值
    void setSmoothThreshold(float theta){
        theta_threshold_ = theta;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    //设定半径阈值
    void setResidualThreshold(float residual){
        residual_threshold_ = residual;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    //设定曲率阈值
    void setCurvatureThreshold (float curvature){
        curvature_threshold_ = curvature;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    //设定近邻个数
    void setNumberOfNeighbours(unsigned int neighbour_number){
        neighbour_number_ = neighbour_number;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    //设定需要进行分割的部分点的标签
    void setIndices(std::vector<int> &indices){
        indices_ = indices;
    }

//    //输入已经计算好的邻域
//    void setNghbr(const std::vector<std::vector<int> > &point_nghbr){
//        is_compute_nghbr_ = false;
//        point_neighbours_ = point_nghbr;
//    }


    /////////////////////////////////////////////////////////////////////////////////////////////
    //启动分割算法并返回标签
    void extract(){

        std::vector<std::vector<int> > clusters;
        clusters_.clear();
//        point_neighbours_.clear();
        point_labels_.clear();
        num_pts_in_segment_.clear();
        number_of_segments_ = 0;

        //如果没有设定感兴趣点集，就把所有点设定为感兴趣点
        if(indices_.empty() ){
            int point_number = static_cast<int>(cloud_.points_.size() );
            for(int i = 0; i < point_number; ++i){
                indices_.push_back(i);
            }
        }


        //寻找每个点云的近邻
        if(cloud_.neighbors_.size() == 0){
            cloud_.ComputeNeighbor(neighbour_number_);
        }else if(cloud_.neighbors_[0].size() != neighbour_number_){
            cloud_.ComputeNeighbor(neighbour_number_);
        }
        //计算法线
        if(cloud_.normals_.size() != cloud_.points_.size()){
            cloud_.NormalEstimate();
        }

        //进行区域生长分割
        applySmoothRegionGrowingAlgorithm();
        assembleRegions();

        //将大小超出或低于限值的分割设置为-1
        clusters.resize(clusters_.size() );
        std::vector<std::vector<int> >::iterator cluster_iter_input = clusters.begin();
        for (std::vector<std::vector<int> >::const_iterator cluster_iter = clusters_.begin();
             cluster_iter != clusters_.end(); cluster_iter++) {
            if( (static_cast<int>(cluster_iter->size() ) >= min_pts_per_cluster_) &&
                    (static_cast<int>(cluster_iter->size()) <= max_pts_per_cluster_) ){
                *cluster_iter_input = *cluster_iter;
                cluster_iter_input++;
            }
        }

        clusters_ = std::vector<std::vector<int> >(clusters.begin(), cluster_iter_input );
        clusters.resize(clusters_.size() );

//        point_labels_.resize(point_labels_.size(), -1);
        for (int i = 0; i < point_labels_.size(); ++i) {
            point_labels_[i] = -1;
        }

        for (int seg_index = 0; seg_index < clusters.size(); ++seg_index) {
            for(int pts_inex_seg = 0; pts_inex_seg < clusters[seg_index].size(); ++pts_inex_seg){
                int index = clusters[seg_index][pts_inex_seg];
                point_labels_[index] = seg_index;
            }
        }

    }

    // sort using a custom function object
    struct {
        bool operator()(std::pair<float, int> paira, std::pair<float, int> pairb) const
        {
            return paira.first < paira.first;
        }
    } comparePair;

    void applySmoothRegionGrowingAlgorithm (){
        int num_of_pts = static_cast<int>(indices_.size() );
        point_labels_.resize(cloud_.points_.size(), -1 );
        for (int i = 0; i < point_labels_.size(); ++i) {
            point_labels_[i] = -1;
        }

        //对曲率进行排序
        std::vector<std::pair<float, int> > point_residual;
        std::pair<float, int> pair;
        point_residual.resize(num_of_pts, pair);

        for (int i_point = 0 ;i_point < num_of_pts ;i_point++ ) {
            int point_index = indices_[i_point];
            point_residual[i_point].first = cloud_.curvatures_[point_index];
            point_residual[i_point].second = point_index;
        }
        std::sort(point_residual.begin(), point_residual.end(), comparePair);

        int seed_counter = 0;
        int seed = point_residual[seed_counter].second;

        int segmented_pts_num = 0;
        int number_of_segments = 0;
        while(segmented_pts_num < num_of_pts ){
            int pts_in_segment;
            pts_in_segment = growRegion(seed, number_of_segments);
//            std::cout<<"c"<<" "<<seed<<" "<<number_of_segments<<std::endl;
            segmented_pts_num += pts_in_segment;
            num_pts_in_segment_.push_back(pts_in_segment );
            number_of_segments++;

            //寻找下一个未分割的点
            for (int i_seed = seed_counter + 1; i_seed < num_of_pts; i_seed++) {
                int index = point_residual[i_seed ].second;
                if(point_labels_[index ] == -1 ){
                    seed = index;
                    seed_counter = i_seed;
                    break;
                }
            }
        }

    }
    // ///////////////////////////////////////////////////////////////////////////////////////

    int growRegion(int initial_seed, int segment_number){
        std::queue<int> seeds;
        seeds.push(initial_seed );
        point_labels_[initial_seed ] = segment_number;

        int num_pts_in_segment = 1;

        while (!seeds.empty() ) {
            int curr_seed = seeds.front();
            seeds.pop();

            size_t i_nghbr = 0;
//            while (i_nghbr < neighbour_number_ && i_nghbr < point_neighbours_[curr_seed].size() ) {
            while (i_nghbr < cloud_.neighbors_[curr_seed].size() ) {
                int index = cloud_.neighbors_[curr_seed][i_nghbr];
                if(point_labels_[index] != -1){
                    i_nghbr++;
                    continue;
                }

                bool is_a_seed = false;
                bool belongs_to_segment = validatePoint(initial_seed, curr_seed, index, is_a_seed);

                if(belongs_to_segment == false){
                    i_nghbr++;
                    continue;
                }

                point_labels_[index] = segment_number;
                num_pts_in_segment++;

                if(is_a_seed){
                    seeds.push(index);
                }

                i_nghbr++;
            }//next neighbour
        }//next seed

        return (num_pts_in_segment);
    }

    // ///////////////////////////////////////////////////////////////////////////////////////
    // 判定当前点point与邻近点nghbr之间法线的点积是否小于阈值，邻近点的曲率是否小于阈值
    bool validatePoint(int initial_seed, int point, int nghbr, bool& is_a_seed) const{
        is_a_seed = true;

        float cosine_threshold = cosf(theta_threshold_);
        float data[4];

        Eigen::Vector3d initial_point = cloud_.points_[point];
        Eigen::Vector3d initial_normal = cloud_.normals_[point];

        Eigen::Vector3d nghbr_normal = cloud_.normals_[nghbr];
        float dot_product = std::abs(nghbr_normal.dot(initial_normal) );
        if(dot_product < cosine_threshold){
            return false;
        }

        if(cloud_.curvatures_[nghbr] > curvature_threshold_){
            is_a_seed = false;
        }

        return true;
    }

     // ///////////////////////////////////////////////////////////////////////////////////////
    void assembleRegions(){
        int number_of_segments = static_cast<int>(num_pts_in_segment_.size() );
        int number_of_points = static_cast<int>(cloud_.points_.size() );

        std::vector<int > segment;
        clusters_.resize(number_of_segments, segment);

        for (int i_seg = 0; i_seg < number_of_segments; i_seg++) {
            clusters_[i_seg].resize( num_pts_in_segment_[i_seg], 0);
        }

        std::vector<int> counter;
        counter.resize(number_of_segments, 0);

        for(int i_point = 0; i_point < number_of_points; i_point++){
            int segment_index = point_labels_[i_point ];
            if(segment_index != -1){
                int point_index = counter[segment_index ];
                clusters_[segment_index ][point_index] = i_point;
                counter[segment_index ] = point_index + 1;
            }
        }

        number_of_segments_ = number_of_segments;
    }

    // ///////////////////////////////////////////////////////////////////////////////////////

    geometry::PointCloud& Generate_supmap(geometry::PointCloud& output){
        output.points_ = cloud_.points_;
        output.colors_.resize(cloud_.points_.size());

        std::vector<Eigen::Vector3d> color_map;
        color_map.resize(number_of_segments_);
        for(size_t i = 0; i < color_map.size(); ++i){
            Eigen::Vector3d color(rand()/double(RAND_MAX) ,rand()/double(RAND_MAX),
                                 rand()/double(RAND_MAX));
            color_map[i] = color;
        }

        for(size_t i = 0 ; i < output.colors_.size(); i++){
            if(point_labels_[i]>=0){
                output.colors_[i] = color_map[point_labels_[i]];
            }else{
                output.colors_[i] = Eigen::Vector3d(1., 0.,0.);
            }
        }
        return output;

    }


public:
    //输入的点云
    geometry::PointCloud cloud_;

    //需要进行分割的那部分点的标签
    std::vector<int> indices_;

    //每个簇允许的最小的像素数
    int min_pts_per_cluster_;

    //每个簇允许的最大的像素数
    int max_pts_per_cluster_;

    //测试点与点之间平滑性的阈值
    float theta_threshold_;

    //最大距离的阈值
    float residual_threshold_;

    //曲率的阈值
    float curvature_threshold_;

    //近邻数量
    unsigned int neighbour_number_;


    //每个点的近邻点
//    std::vector<std::vector<int> > point_neighbours_;

    //每个点的分割标签
    std::vector<int> point_labels_;

    //每个分割有多少各点，用于节省内存
    std::vector<int> num_pts_in_segment_;

    //分割的数量
    int number_of_segments_;

    //每个分割所包含的点的集合
    std::vector<std::vector<int> > clusters_;

    //  确定是否计算领域
    bool is_compute_nghbr_;

};


}//namesapce PPP
}//namespace geometry

#endif //REGIONGROW_H




