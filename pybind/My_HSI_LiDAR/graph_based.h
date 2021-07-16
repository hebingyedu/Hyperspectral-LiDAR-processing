#ifndef GRAPH_BASED_H
#define GRAPH_BASED_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "segment-graph.h"

#include <time.h>
#include <stdlib.h>

#include <fstream>
#include<iostream>
#include <cmath>

#include "kdtree.h"
#include "PointCloud.h"
#include "supervoxel_structure.h"

#include <vector>
using namespace cv;
using namespace std;

typedef unsigned char uchar;// 用uchar表示unsigned char

typedef struct { uchar r, g, b; } rgb;//可用rgb定义结构体变量

// 随机颜色
//============================================================================================
rgb random_rgb(){
  rgb c;
 c.r = (uchar)rand();
 c.g = (uchar)rand();
 c.b = (uchar)rand();

  return c;
}
typedef struct {

  int p;
  int k;

} uni_elt1;
bool operator<(const uni_elt1 &a, const uni_elt1 &b) {
  return a.p < b.p;
}


namespace PPP {
namespace segmentation {

class Graph_based
{
public:
    /////////////////////////////////////////////////////////////////////////////////////////////
    Graph_based(){}

    /////////////////////////////////////////////////////////////////////////////////////////////
    ~Graph_based(){}


public:
    int n_points_;
    int n_segments_;
    int n_supvoxs_;
    int n_supvox_segments_;

    geometry::PointCloud cloud_;
    supervoxel_structure voxel;

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

#endif // GRAPH_BASED_H
