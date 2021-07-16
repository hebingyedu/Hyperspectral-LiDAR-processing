#ifndef FH_SUPERVOXEL_H
#define FH_SUPERVOXEL_H

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
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

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "segment-graph.h"

using namespace std;
#include "PointCloud.h"
#include "kdtree.h"
#include "utility.h"

namespace PPP {
namespace segmentation {

class FH_supvoxel{
public:
    FH_supvoxel(){}
    ~FH_supvoxel(){}

    void setCloud(const geometry::PointCloud &cloud){
        cloud_ = cloud ;
        n_points_ = static_cast<int>(cloud_.points_.size() );
        HasHSI = false;
    }

    void setHSIFeature(const cv::Mat &HSIFeature){
        HSIFeature_ = HSIFeature;
        for(int i = 0; i < HSIFeature_.rows; ++i){
            double sum = HSIFeature_.row(i).dot(HSIFeature_.row(i));
            sum = std::sqrt(sum );
            HSIFeature_.row(i) = HSIFeature_.row(i)/sum;
        }
        HasHSI = true;
    }
    ///////////////////////////////////////////////////////////////
    //compute egde
    void compute_edge(){
        edget_.clear();
        for(int i = 0; i < n_points_; ++i){
            vector<int> neigh = cloud_.neighbors_[i];
            for(int j : neigh){
                if(j > i){
                    edge L;
                    L.a = i;
                    L.b = j;
                    L.w =  metric(i, j) ;
                    edget_.push_back(L);
                }
            }
        }
    }
    /////////////////////////////////////////////////////////////////
    double metric(int i, int j){
        double dis1 = 0, dis2 = 0, dis3 = 0;
        Eigen::Vector3d vec1 = cloud_.points_[i];
        Eigen::Vector3d vec2 = cloud_.points_[j];
        Eigen::Vector3d nor1 = cloud_.normals_[i];
        Eigen::Vector3d nor2 = cloud_.normals_[j];
        Eigen::Vector3d cor1 = cloud_.colors_[i];
        Eigen::Vector3d cor2 = cloud_.colors_[j];

//        dis1 = (vec1(2) - vec2(2))*(vec1(2) - vec2(2));
//        dis2 = (nor1 - nor2).dot(nor1 - nor2);

        dis3  = (cor1 - cor2).dot(cor1 - cor2);
        return dis3;
//        dis3  = (cor1(2) - cor2(2)) * (cor1(2) - cor2(2)) ;

//        if(HasHSI){
//            dis3 = HSIFeature_.row(i).dot(HSIFeature_.row(j));
//            dis3  = std::acos( dis3 );
//        }
//        return dis1 + dis2 + 3*dis3;
    }
    //////////////////////////////////////////////////////////////////////
    void FindNeighbor(int nn){
        if(cloud_.neighbors_.size() == 0){
            cloud_.ComputeNeighbor(nn);
        }else if(cloud_.neighbors_[0].size() != nn){
            cloud_.ComputeNeighbor(nn);
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //graph-based
    void graph_seg(double ratio, int min_size){
//        vector<edge> edget;
        int nr_lables =  n_points_;
        int pl = edget_.size();
//cout<<"b1"<<endl;
        universe *u = segment_graph(nr_lables, pl, edget_, ratio );
//cout<<"b2"<<endl;
        for (int i = 0; i < pl; i++) {
            int a = u->find(edget_[i].a);
            int b = u->find(edget_[i].b);
            if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
            u->join(a, b);
        }
//cout<<"b3"<<endl;
        n_supervoxels_ = u->num_sets();


        std::vector<int> label_uncontinue;
        for(int i = 0; i < nr_lables; i++ )
        {
            label_uncontinue.push_back(u->find(i));
        }

        std::vector<int> junk;
        labels_.clear();
        geometry::myunique(label_uncontinue, junk, labels_);
    }

    ///////////////////////////////////////////////////////////
    geometry::PointCloud& Generate_supmap(geometry::PointCloud& output){
        output.points_ = cloud_.points_;
        output.colors_.resize(cloud_.points_.size());

        std::vector<Eigen::Vector3d> color_map;
        color_map.resize(n_supervoxels_);
        for(size_t i = 0; i < color_map.size(); ++i){
            Eigen::Vector3d color(rand()/double(RAND_MAX) ,rand()/double(RAND_MAX),
                                 rand()/double(RAND_MAX));
            color_map[i] = color;
        }

        for(size_t i = 0 ; i < output.colors_.size(); i++){
            output.colors_[i] = color_map[labels_[i]];
        }
        return output;

    }



public:
    int n_supervoxels_;
    int n_points_;

    vector<edge> edget_;

    // The size of supervoxel.
    std::vector<int> sizes_;

    std::vector<int> labels_;
    std::vector<int> supervoxels_;
    //K neareast neighbor points of cloud

    geometry::PointCloud cloud_;
    cv::Mat HSIFeature_;
    bool HasHSI;
};

}//namespace segmentation
}//namespace PPP

#endif // FH_SUPERVOXEL_H
