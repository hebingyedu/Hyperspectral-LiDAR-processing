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

#include "disjointset.h"
#include "PointCloud.h"
#include "kdtree.h"
#include "octree.h"


namespace py = pybind11;
namespace PPP{
namespace segmentation{

class TBBSupervoxel {
public:
    TBBSupervoxel(){}

    ~TBBSupervoxel(){}
//    Eigen::MatrixXd big_mat = Eigen::MatrixXd::Zero(10000, 10000);
public:
    void setCloud(const geometry::PointCloud &cloud){
        cloud_ = cloud ;
        n_points_ = static_cast<int>(cloud_.points_.size() );
        //设置初始的z_scale,越大则超像素在水平方向拉伸，小则超像素在竖直方向拉伸
        z_scale_ = 1.;
        // At first, each point is a supervoxel.
        set_.Reset(n_points_);
        supervoxels_.resize(n_points_);
        for (int i = 0; i < n_points_; ++i) {
                supervoxels_[i] = i;
        }

        // The size of supervoxel.
        sizes_.resize(n_points_, 1);

        // Minimization the energe function by fusion.
        number_of_supervoxels_ = n_points_;


    }

    void FindNeighbor(int nn){
        if(cloud_.neighbors_.size() == 0){
            cloud_.ComputeNeighbor(nn);
        }else if(cloud_.neighbors_[0].size() != nn){
            cloud_.ComputeNeighbor(nn);
        }

        adjacents_ = cloud_.neighbors_;

        // Compute the minimum value of lambda.
        vector<double> dis(n_points_, DBL_MAX);
        for (int i = 0; i < n_points_; ++i) {
            for (int j : adjacents_[i]) {
                if (i != j) {
                    dis[i] = std::min(dis[i], metric(i, j));
                }
            }
        }
        lambda_ = std::max(DBL_EPSILON, Median(dis.begin(), dis.end()));
    }

    double metric(int i, int j){
        double dis1 = 0;
        Eigen::Vector3d vec1 = cloud_.points_[i];
        Eigen::Vector3d vec2 = cloud_.points_[j];

        dis1 = 1.*(vec1[0] - vec2[0])*(vec1[0] - vec2[0]) + 1.*(vec1[1] - vec2[1])*(vec1[1] - vec2[1])
                + z_scale_*(vec1[2] - vec2[2])*(vec1[2] - vec2[2]);

        return dis1;
    }

    /**
     * Get the median in range [first, last).
     */
    template <typename Iterator>
    const typename std::iterator_traits<Iterator>::value_type
    Median(Iterator first, Iterator last) {
        assert(first != last);

        typedef typename std::iterator_traits<Iterator>::value_type T;

        vector<T> values(first, last);
        std::nth_element(values.begin(), values.begin() + values.size() / 2,
                         values.end());
        return values[values.size() / 2];
    }

    void set_n_sup1(int n_supervoxels){
        n_supervoxels_ = n_supervoxels;
    }

    void set_z_scale(double z_scale){
        z_scale_ = z_scale;
    }
// calculate n_supervoxels through resolution
    void set_n_sup2(double resolution){
        if(cloud_.colors_.size() != cloud_.points_.size()){
            cloud_.colors_ = cloud_.points_;
        }
        Eigen::Array3d min_bound = cloud_.GetMinBound();
        Eigen::Array3d max_bound = cloud_.GetMaxBound();
        Eigen::Array3d sizes = max_bound - min_bound;
        double max_size = sizes.maxCoeff();
        max_size = max_size * (1 + 0.01);

        int depth = 0;
        while(max_size > 4.*resolution/3.){
            max_size = max_size/2.;
            depth++;
        }

        geometry::Octree octree;
        octree.max_depth_ = depth;
        octree.ConvertFromPointCloud(cloud_, 0.01);
        int n_leaf_node = 0;
        countLeafNode(octree.root_node_, &n_leaf_node);

        n_supervoxels_ = n_leaf_node;

    }

    void StartSegmentation(){
        Find_superpixels();
        Refine_boundaries();
        Relabel();
    }

    std::vector<int> getLabels(){
        return labels_;
    }


    void countLeafNode(const std::shared_ptr<geometry::OctreeNode>& node, int * n_leaf_node){
        if (node == nullptr) {
                return;
            } else if (auto internal_node =
                               std::dynamic_pointer_cast<geometry::OctreeInternalNode>(node)) {
                for (size_t child_index = 0; child_index < 8; ++child_index) {
                    auto child_node = internal_node->children_[child_index];
                    countLeafNode(child_node, n_leaf_node);
                }
        }else if(auto leaf_node =
                 std::dynamic_pointer_cast<geometry::OctreeLeafNode>(node)){
            (*n_leaf_node)++;
        }else {
            throw std::runtime_error("Internal error: unknown node type");
        }
    }


    //step 1: Find superpixels
    void Find_superpixels( ){


        // Queue for region growing.
        vector<int> queue1(n_points_);
        vector<bool> visited(n_points_, false);
        for(; ; lambda_ *= 2.0){
            if(supervoxels_.size() <= 1 ) break;

            for(int i : supervoxels_){
                if(adjacents_[i].empty() ) continue;

                visited[i] = true;
                int front = 0, back = 1;
                queue1[front++] = i;
                for(int j : adjacents_[i]){
                    j = set_.Find(j);
                    if(!visited[j]){
                        visited[j] = true;
                        queue1[back++] = j;
                    }
                }

                vector<int> adjacent;
                while(front < back){
                    int j = queue1[front++];

                    double loss = sizes_[j] * metric(i, j);
                    double improvement = lambda_ - loss;
                    if(improvement > 0.0){
                        set_.Link(j, i);

                        sizes_[i] += sizes_[j];

                        for(int k : adjacents_[j]){
                            k = set_.Find(k);
                            if(!visited[k]){
                                visited[k] = true;
                                queue1[back++] = k;
                            }
                        }

                        adjacents_[j].clear();
                        if(--number_of_supervoxels_ == n_supervoxels_) break;
                    }else{
                        adjacent.push_back(j);
                    }

                }
                adjacents_[i].swap(adjacent);

                for(int j = 0; j < back; ++j){
                    visited[queue1[j]] = false;
                }
                if(number_of_supervoxels_ == n_supervoxels_) break;
            }

            number_of_supervoxels_ = 0;
            for(int i : supervoxels_){
                if(set_.Find(i) == i){
                    supervoxels_[number_of_supervoxels_++] = i;
                }
            }
            supervoxels_.resize(number_of_supervoxels_);

            if(number_of_supervoxels_ == n_supervoxels_) break;
        }

        //Assign the label to each point according to supervoxel ID
        labels_.resize(n_points_);
        for(int i = 0; i < n_points_; ++i){
            labels_[i] = set_.Find(i);
        }
    }

    //-----------Step2: refine the boundaries----------------
    void Refine_boundaries(){
        vector<double> dis(n_points_, DBL_MAX);
        for(int i = 0; i < n_points_; ++i){
            int j = labels_[i];
            dis[i] = metric(i, j);
        }

        queue<int> q;
        vector<bool> in_q(n_points_, false);

        for(int i = 0; i < n_points_; ++i){
            for(int j : cloud_.neighbors_[i]){
                if(labels_[i] != labels_[j]){
                    if(!in_q[i]){
                        q.push(i);
                        in_q[i] = true;
                    }
                    if(!in_q[j]){
                        q.push(j);
                        in_q[j] = true;
                    }
                }
            }
        }

        while(!q.empty()){
            int i = q.front();
            q.pop();
            in_q[i] = false;

            bool change = false;
            for (int j : cloud_.neighbors_[i]){
                int a = labels_[i];
                int b = labels_[j];
                if(a == b) continue;
                double d = metric(i, b);
                if(d < dis[i]){
                    labels_[i] = b;
                    dis[i] = d;
                    change = true;
                }
            }

            if(change){
                for(int j : cloud_.neighbors_[i]){
                    if(labels_[i] != labels_[j]){
                        if(!in_q[j]){
                            q.push(j);
                            in_q[j] = true;
                        }
                    }
                }
            }
        }

    }

    void Relabel(){
        vector<int> map(n_points_);
        for(int i = 0; i < supervoxels_.size(); ++i){
            map[supervoxels_[i]] = i;
        }
        for(int i = 0; i < n_points_; ++i){
            labels_[i] = map[labels_[i]];
        }
    }

//    Eigen::MatrixXd &getMatrix() { return big_mat; }
//    const Eigen::MatrixXd &viewMatrix() { return big_mat; }
//    void CopyMat(Eigen::MatrixXd M){
//        big_mat = M;
//    }

//    Eigen::VectorXd scale_by_2(Eigen::Ref<Eigen::VectorXd> v) {
//        v *= 2;
//        return v;
//    }
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


    int n_supervoxels_;
    int number_of_supervoxels_;
    int n_points_;

    double z_scale_;

    // The size of supervoxel.
    std::vector<int> sizes_;

    std::vector<int> labels_;
    geometry::DisjointSet set_;
    std::vector<int> supervoxels_;
    //K neareast neighbor points of cloud
    std::vector<std::vector<int> > adjacents_;
//    std::vector<std::vector<int> > neighbors_;
    geometry::PointCloud cloud_;
    double lambda_;

};

}//namespace geometry

}//namespace PPP
