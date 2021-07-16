#ifndef SUPERVOXMERGING_H
#define SUPERVOXMERGING_H

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


namespace PPP {
namespace segmentation {

class SupervoxMerging
{
public:
    /////////////////////////////////////////////////////////////////////////////////////////////
    SupervoxMerging(){}

    /////////////////////////////////////////////////////////////////////////////////////////////
    ~SupervoxMerging(){}

    //////////////////////////////////////////////////////////////////////////////////////////////
    void setCloud(const geometry::PointCloud &cloud,
                  std::vector<int> & point_labels, std::vector<int> & point_supvox_labels){
        cloud_ = cloud;
        //labels via point-based region growing segmentation,
        //where point i belongs to seg point_labels[i]
        point_labels_ = point_labels;
        //labels via point-based supervox segmentation,
        //where point i belongs to seg point_supvox_labels[i]
        point_supvox_labels_ = point_supvox_labels;

        n_points_ = cloud_.points_.size();
        //how many segments we have
        auto itr_max =  std::max_element(point_labels_.begin(), point_labels_.end() );
        n_segments_ = *(itr_max) + 1;
        //how many supervoxers we have
        itr_max = std::max_element(point_supvox_labels_.begin(), point_supvox_labels_.end() );
        n_supvoxs_ = *(itr_max) + 1;
        //set distance_thre_
        distance_thre_ = 2.;
        nn_neiber_ = 60;
    }


    /////////////////////////////////////////////////////////////////////////////////////////////
    void setTreeSegPrameter(double distance_thre, int nn_neiber){
        distance_thre_ = distance_thre;
        nn_neiber_ = nn_neiber;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    //返回分割标签
    std::vector<int> getPoint_Labels(){
        return point_labels_;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    //返回类别标签
    std::vector<int> getPoint_Classes(){
        return point_classes_;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    void point_label2voxel(){
        //point_clusters[j] collects all points belong to seg j.
        std::vector<int> cluster;
        point_clusters_.clear();
        point_clusters_.resize(n_segments_, cluster);
        for (int i = 0; i < n_points_; ++i) {
            int index = point_labels_[i];
            if(index != -1 ){
                point_clusters_[index].push_back(i);
            }
        }

        //number_each_segments[j] recorde how many points in seg [j]
        std::vector<int> number_each_segments;
        number_each_segments.resize(n_segments_ );
        for (int i = 0; i < n_segments_; ++i) {
            number_each_segments[i] = point_clusters_[i].size();
        }

        //seg which has the largest number_each_segments is ground
        auto itr_max = std::max_element(number_each_segments.begin(), number_each_segments.end() );
        //the ground points are in cluster[ground_position]
        //or point i is ground such that point_labels[i] == ground_position
        int ground_position = std::distance(number_each_segments.begin(), itr_max );

        //每个点的类别，0:地面，1:建筑，2:树木，3:车，-1:其他
        point_classes_.clear();
        //地面点类别设为0
        point_classes_.resize(n_points_, -1);
        for (int i = 0; i < n_points_ ; ++i) {
            int index = point_labels_[i];
            if(index == ground_position){
                point_classes_[i] = 0;
            }else if (index != -1) {
                point_classes_[i] = 1;
            }
        }


        //voxel_label_index里存的是超体素每个点的标签，用于投票表决超体素的标签
        std::vector<std::vector<int> > voxel_label_index;
        std::vector<int> vec;
        voxel_label_index.clear();
        voxel_label_index.resize(n_supvoxs_, vec);
        for (int i = 0; i < n_points_; ++i) {
            int index_seg = point_labels_[i];
            int index_vox = point_supvox_labels_[i];
            if(index_seg != -1){
                voxel_label_index[index_vox ].push_back(index_seg);
            }
        }

        //把超体素的标签设置为超体素中每个点标签的众数
        voxel_labels_.clear();
        voxel_labels_.resize(n_supvoxs_, -1);
        for (int i = 0; i < n_supvoxs_; ++i) {
            if(voxel_label_index[i].size() != 0){
                int label = findMajorityElement(voxel_label_index[i]);
                voxel_labels_[i] = label;
            }
        }

        //point_voxel_clusters_存的是超体素每个点的序号
        point_voxel_clusters_.clear();
        point_voxel_clusters_.resize(n_supvoxs_, cluster);
        for (int i = 0; i < n_points_; ++i) {
            int index_vox = point_supvox_labels_[i];
            if(index_vox != -1 ){
                point_voxel_clusters_[index_vox].push_back(i);
            }
        }

        //根据超体素的标签更新每个点的标签
        for (int i = 0; i < point_voxel_clusters_.size(); ++i) {
            for (int index : point_voxel_clusters_[i]) {
                point_labels_[index] = voxel_labels_[i];
            }
        }

        //把树状点的标签设为-1

        //更新point_clusters标签
        auto max_itr = std::max_element(point_labels_.begin(), point_labels_.end() );
        n_segments_ = *(max_itr) + 1;
        point_clusters_.clear();
        cluster.clear();
        point_clusters_.resize(n_segments_, cluster);
        for (int i = 0; i < n_points_; ++i) {
            int index = point_labels_[i];
            if(index > -1 ){
                point_clusters_[index].push_back(i);
            }
        }


        //标签为地面的超体素集合
        std::vector<int > ground_voxel;
        for (int i = 0; i < n_supvoxs_; ++i){
            int index_vox = voxel_labels_[i];
            if(index_vox == ground_position){
                ground_voxel.push_back(i);
            }
        }
        //地面超体素的边框
        std::vector<BoundingBox > ground_voxel_box;
        BoundingBox box;
        ground_voxel_box.resize(ground_voxel.size(), box);
        for (int i = 0; i < ground_voxel.size(); ++i) {
            int index_vox = ground_voxel[i];
            std::vector<Eigen::Vector3d > vec_point;
            for (int j : point_voxel_clusters_[index_vox]) {
                vec_point.push_back(cloud_.points_[j]);
            }
            ground_voxel_box[i] = BoundingBox(vec_point);
        }

        //对于每个建筑候选分割，计算中心点，投影到地面，
        //如果中心在某个地面超像素框中，就认为是散状点，把标签设为-1
        std::vector<std::vector<int> > clusters(0);
        clusters.resize(point_clusters_.size() );
        std::vector<std::vector<int> >::iterator cluster_iter_input = clusters.begin();
        for (std::vector<std::vector<int> >::iterator cluster_iter = point_clusters_.begin();
             cluster_iter != point_clusters_.end(); cluster_iter++) {
            int count = std::distance(point_clusters_.begin(), cluster_iter);
            if(count == ground_position){
                *cluster_iter_input = *cluster_iter;
                ground_position = static_cast<int>(std::distance(clusters.begin(), cluster_iter_input ));
                cluster_iter_input++;
            }else{
                std::vector<Eigen::Vector3d > vec_point;
                if(cluster_iter->size() > 0){
                    for (int j : *cluster_iter) {
                        vec_point.push_back(cloud_.points_[j]);
                    }
                    BoundingBox box1(vec_point);

                    double center_x = box1.max_values(0) + box1.min_values(0);
                    center_x = 0.5 * center_x;
                    double center_y = box1.max_values(1) + box1.min_values(1);
                    center_y = 0.5 * center_y;

                    double margin = 0.5;
                    bool is_tree = false;
                    for (BoundingBox gbox : ground_voxel_box) {
                        if(center_x > gbox.min_values(0) - margin && center_x < gbox.max_values(0) + margin&&
                                center_y > gbox.min_values(1) - margin && center_y < gbox.max_values(1) + margin ){
                            is_tree = true;
                            break;
                        }
                    }

                    if(!is_tree){
                        *cluster_iter_input = *cluster_iter;
                        cluster_iter_input++;
                    }
                }
            }

        }

        //更新point_clusters_，并根据point_clusters_重新设定point_labels_，使得point_labels_编号连续
        point_clusters_ = std::vector<std::vector<int> >(clusters.begin(), cluster_iter_input );
        clusters.resize(point_clusters_.size() );

        for (int i = 0; i < point_labels_.size(); ++i) {
            point_labels_[i] = -1;
        }

        for (int seg_index = 0; seg_index < point_clusters_.size(); ++seg_index) {
            for( int index : point_clusters_[seg_index]){
                point_labels_[index] = seg_index;
            }
        }

        point_classes_.clear();
        point_classes_.resize(n_points_, -1);
        for (int i = 0; i < n_points_ ; ++i) {
            int index = point_labels_[i];
            if(index == ground_position){
                point_classes_[i] = 0;
            }else if (index != -1) {
                point_classes_[i] = 1;
            }
        }


        //把标签为-1的散状点按空间距离分割成独立的小块
        std::vector<int> indices_;
        for (int i = 0; i < n_points_; i++) {
            int label = point_labels_[i];
            if(label == -1){
                indices_.push_back(i);
            }
        }

        geometry::PointCloud subcloud;
        for (int index : indices_) {
            subcloud.points_.push_back(cloud_.points_[index]);
        }
        subcloud.ComputeNeighbor(nn_neiber_);

        itr_max =  std::max_element(point_labels_.begin(), point_labels_.end() );
        n_segments_ = *(itr_max) + 1;
        std::vector<std::vector<int> > tree_clusters(0);
        std::queue<int> curr_Q;
        std::vector<bool> is_added(indices_.size(), false);
        for (int i = 0; i < indices_.size(); ++i) {
            if(!is_added[i]){
                std::vector<int> AddedQ(0);
                AddedQ.push_back(i);
                curr_Q.push(i);
                while (!curr_Q.empty() ) {
                    int curr_index = curr_Q.front();
                    AddedQ.push_back(curr_index);
                    curr_Q.pop();
                    for(int j = 0; j < 10; ++j){
                        int negbr = subcloud.neighbors_[curr_index][j];
                        if(!is_added[negbr] ){
                            double dis = compute_dis(indices_[negbr], indices_[curr_index]);
                            if(dis < distance_thre_){
                                curr_Q.push(negbr);
                                is_added[negbr] = true;
                            }
                        }
                    }
                }
                tree_clusters.push_back(AddedQ);
            }
        }

        for (std::vector<int> cluster : tree_clusters) {
            if(cluster.size() > 2000){
                for(int index : cluster){
                    point_labels_[indices_[index]] = n_segments_;
                    point_classes_[indices_[index]] = 2;
                }
                n_segments_++;
            }

            if(cluster.size() <= 2000){
                bool is_isolate = true;
                int belong_index;
                for(int curr_index : cluster){
                    for(int negbr : subcloud.neighbors_[curr_index]){
                        int negbr_index = indices_[negbr];
                        if(point_labels_[negbr_index] != point_labels_[indices_[curr_index]]){
                            is_isolate = false;
                            belong_index = point_labels_[negbr_index];
                            break;
                        }
                    }
                }

                if(is_isolate){
                    for(int index : cluster){
                        point_labels_[indices_[index]] = n_segments_;
                        point_classes_[indices_[index]] = 3;
                    }
                    n_segments_++;
                }else{
                    for(int index : cluster){
                        point_labels_[indices_[index]] = belong_index;
                        point_classes_[indices_[index]] = 2;
                    }
                }

            }
        }

        //更新point_clusters标签;
        point_clusters_.clear();
        cluster.clear();
        point_clusters_.resize(n_segments_, cluster);
        for (int i = 0; i < n_points_; ++i) {
            int index = point_labels_[i];
            if(index > -1 ){
                point_clusters_[index].push_back(i);
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    double compute_dis(int i, int j){
        double dis = 0;
        dis = (cloud_.points_[i](0) - cloud_.points_[j](0) ) *
               (cloud_.points_[i](0) - cloud_.points_[j](0) );
        dis += (cloud_.points_[i](1) - cloud_.points_[j](1) ) *
                (cloud_.points_[i](1) - cloud_.points_[j](1) );
        dis += (cloud_.points_[i](2) - cloud_.points_[j](2) ) *
                (cloud_.points_[i](2) - cloud_.points_[j](2) );
        return  std::sqrt(dis);
    }


    ///////////////////////////////////////////////////////////////////////////////
    struct BoundingBox {
        BoundingBox() {}

        BoundingBox(vector<Eigen::Vector3d> points_) {
            if (points_.begin() == points_.end()) return;

            auto itr_x = std::min_element(
                    points_.begin(), points_.end(),
                    [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                        return a(0) < b(0);
                    });
            auto itr_y = std::min_element(
                    points_.begin(), points_.end(),
                    [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                        return a(1) < b(1);
                    });
            auto itr_z = std::min_element(
                    points_.begin(), points_.end(),
                    [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                        return a(2) < b(2);
                    });
            min_values = Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));

            auto itr_x1 = std::max_element(
                    points_.begin(), points_.end(),
                    [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                        return a(0) < b(0);
                    });
            auto itr_y1 = std::max_element(
                    points_.begin(), points_.end(),
                    [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                        return a(1) < b(1);
                    });
            auto itr_z1 = std::max_element(
                    points_.begin(), points_.end(),
                    [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                        return a(2) < b(2);
                    });
            max_values = Eigen::Vector3d((*itr_x1)(0), (*itr_y1)(1), (*itr_z1)(2));
        }

        Eigen::Vector3d min_values, max_values;
    };

    // ////////////////////////////////////////////////////////////////////////////////////////
    //求众数
    int findMajorityElement(std::vector<int> vec){
        std::sort(vec.begin(), vec.end() );
        int i = 0;
        int MaxCount = 1;
        int index = 0;

        while (i < vec.size() - 1)//遍历
        {
            int count = 1;
            int j ;
            for (j = i; j < vec.size() - 1; j++)
            {
                if (vec[j] == vec[j + 1])//存在连续两个数相等，则众数+1
                {
                    count++;
                }
                else
                {
                    break;
                }
            }
            if (MaxCount < count)
            {
                MaxCount = count;//当前最大众数
                index = j;//当前众数标记位置
            }
            ++j;
            i = j;//位置后移到下一个未出现的数字
        }
        return vec[index];
    }

    // sort using a custom function object
    struct {
        bool operator()(std::pair<int, int> paira, std::pair<int, int> pairb) const
        {
            return paira.second < paira.second;
        }
    } comparePair;


    // ///////////////////////////////////////////////////////////////////////////////////////

    geometry::PointCloud& Generate_supmap(geometry::PointCloud& output){
        output.points_ = cloud_.points_;
        output.colors_.resize(cloud_.points_.size());

        std::vector<Eigen::Vector3d> color_map;
        color_map.resize(n_segments_);
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

    int n_points_;
    int n_segments_;
    int n_supvoxs_;
    int n_supvox_segments_;

    geometry::PointCloud cloud_;
    //每个点根据区域生长得到的标签
    std::vector<int> point_labels_;
    //每个点根据超体素分割得到的标签
    std::vector<int> point_supvox_labels_;
    //每个点的类别，0:地面，1:建筑，2:树木，3:车，-1:其他
    std::vector<int> point_classes_;
    //每个分割对应的点
    std::vector<std::vector<int> > point_clusters_;
    //每个超体素对应的点
    std::vector<std::vector<int> > point_voxel_clusters_;
    //每个超体素根据区域生长得到的标签
    std::vector<int> voxel_labels_;
    //超体素集合
    std::vector<int> voxel_clusters_;

    //地面位置
    int ground_position_;
    //在分割树状点时候点参数，nn_neiber_,distance_thre_越大则分割越大
    double distance_thre_;
    int nn_neiber_ = 60;

};

}

}


#endif //SUPERVOXMERGING_H




