#ifndef HSI_FH_H
#define HSI_FH_H

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
using namespace cv;

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
namespace HSI_supixel {

class HSI_FH{
public:
    HSI_FH(){}
    ~HSI_FH(){}

    void setHSI(const cv::Mat &HSI){
        m_height_ = HSI.rows;
        m_width_ = HSI.cols;
        m_channels_ = HSI.channels();
        n_pixels_ = m_height_ * m_width_;

        cv::Mat HSId;
        HSI.convertTo(HSId, CV_64F);

        Im_.create(n_pixels_, m_channels_, CV_64F);
        for( int x = 0; x < m_width_; x++ )
        {
            for( int y = 0; y < m_height_; y++ )
            {
                int idn = y*m_width_+x;
                double *p = Im_.ptr<double>(idn);
                double *q = HSId.ptr<double>(y, x);
                for( int j = 0; j < m_channels_; j++ )
                {
                    p[j] = q[j];
                }
            }
        }

        for(int i = 0; i < n_pixels_; ++i){
            double sum = Im_.row(i).dot(Im_.row(i));
            sum = std::sqrt(sum );
            Im_.row(i) = Im_.row(i)/sum;
        }

    }
    ///////////////////////////////////////////////////////////////
    //compute egde
    void compute_edge(){
        edget_.clear();
        for (int y = 0; y < m_height_; y++) {
            for (int x = 0; x < m_width_; x++) {
                edge L;
                if (x < m_width_-1) {
                    L.a = y * m_width_ + x;
                    L.b = y * m_width_ + (x+1);
                    L.w =   metric(L.a, L.b) ;
                    edget_.push_back(L);
                }

                if (y < m_height_-1) {
                    L.a = y * m_width_ + x;
                    L.b = (y+1) * m_width_ + x;
                    L.w =   metric(L.a, L.b) ;
                    edget_.push_back(L);
                }

                if ((x < m_width_-1) && (y < m_height_-1)) {
                    L.a = y * m_width_ + x;
                    L.b = (y+1) * m_width_ + (x+1);
                    L.w =   metric(L.a, L.b) ;
                    edget_.push_back(L);
                }

                if ((x < m_width_-1) && (y > 0)) {
                    L.a = y * m_width_ + x;
                    L.b = (y-1) * m_width_ + (x+1);
                    L.w =   metric(L.a, L.b) ;
                    edget_.push_back(L);
                }
            }
          }
    }
    /////////////////////////////////////////////////////////////////
    double metric(int i, int j){
        double dis1 = 0;
        dis1  = std::acos( Im_.row(i).dot(Im_.row(j) ) );
        return dis1;
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    //graph-based
    void graph_seg(double ratio, int min_size){
        int nr_lables =  n_pixels_;
        int pl = edget_.size();

        universe *u = segment_graph(nr_lables, pl, edget_, ratio );

        for (int i = 0; i < pl; i++) {
            int a = u->find(edget_[i].a);
            int b = u->find(edget_[i].b);
            if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
            u->join(a, b);
        }

        n_supixels = u->num_sets();


        std::vector<int> label_uncontinue;
        for(int i = 0; i < nr_lables; i++ )
        {
            label_uncontinue.push_back(u->find(i));
        }

        std::vector<int> junk;
        labels_.clear();
        myunique(label_uncontinue, junk, labels_);

        label_T = cv::Mat(labels_).reshape(0, m_height_);
    }

    ///////////////////////////////////////////////////////////
    cv::Mat Generate_supmap(cv::Mat RGBimg){
        int Dx[2] = {-1, 1};
        int Dy[2] = {-1, 1};
        for (int y = 0; y < m_height_; y++) {
            for (int x = 0; x < m_width_; x++) {
                int count = 0;
                int L = label_T.at<int>(y, x);
                for(int dx : Dx){
                    for(int dy : Dy){
                        int nei_x = std::min(x+dx, m_width_-1);
                        int nei_y = std::min(y+dy, m_height_-1);
                        nei_x = std::max(nei_x, 0);
                        nei_y = std::max(nei_y, 0);

                        if(label_T.at<int>(nei_y, nei_x) != L){
                            count++;
                        }
                    }
                }

                if(count > 1){
                    auto *p = RGBimg.ptr<uchar>(y,x);
                    p[0] = 255;
                    p[1] = 0;
                    p[2] = 0;
                }
            }
        }

        return  RGBimg;
    }

    ///////////////////////////////////////////////////////////
    cv::Mat Generate_supmask(bool _thick_line){
        // default width
        int line_width = 2;

        if ( !_thick_line ) line_width = 1;

        cv::Mat mask;
        mask.create( m_height_, m_width_, CV_8UC1 );

        mask.setTo(0);

        const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
        const int dy8[8] = {  0, -1, -1, -1, 0, 1, 1,  1 };

        int sz = m_width_*m_height_;

        vector<bool> istaken(sz, false);

        int mainindex = 0;
        for( int j = 0; j < m_height_; j++ )
        {
          for( int k = 0; k < m_width_; k++ )
          {
            int np = 0;
            for( int i = 0; i < 8; i++ )
            {
              int x = k + dx8[i];
              int y = j + dy8[i];

              if( (x >= 0 && x < m_width_) && (y >= 0 && y < m_height_) )
              {
                int index = y*m_width_ + x;

                if( false == istaken[index] )
                {
                  if( label_T.at<int>(j,k) != label_T.at<int>(y,x) ) np++;
                }
              }
            }
            if( np > line_width )
            {
               mask.at<char>(j,k) = (uchar)255;
               istaken[mainindex] = true;
            }
            mainindex++;
          }
        }
        return  mask;
    }


    cv::Mat GetSuppixelLabel(){
        return label_T;
    }

    struct num{
        int a;
        int b;
        int c;
    };

    void myunique(const std::vector<int> &idn, std::vector<int> &idn_u, std::vector<int> &reverse_indices){
        idn_u.clear();
        reverse_indices.clear();

        std::vector<num> numf(idn.size() );
        for(size_t i = 0; i < numf.size(); ++i){
            numf[i].a = idn[i];
            numf[i].b = static_cast<int>(i);
        }

        sort(numf.begin(), numf.end(), [](const num &odd1,const num &odd2){return odd1.a < odd2.a;});

        idn_u.push_back(numf[0].a);
        numf[0].c = 0;

        for(size_t i = 1; i < numf.size(); ++i){
            if(numf[i].a == numf[i-1].a){
                numf[i].c = numf[i-1].c;
            }else{
                idn_u.push_back(numf[i].a);
                numf[i].c = numf[i-1].c + 1;
            }
        }

        sort(numf.begin(), numf.end(), [](const num &odd1,const num &odd2){return odd1.b < odd2.b;});
        reverse_indices.resize(numf.size() );
        for(int i = 0; i < numf.size(); ++i){
            reverse_indices[i] = numf[i].c;
        }
    }
    //模仿python numpy unique函数====================================================

public:
    int n_supixels;
    int n_pixels_;

    vector<edge> edget_;


    std::vector<int> sizes_;

    cv::Mat label_T;

    std::vector<int> labels_;
    cv::Mat Im_;
    int m_height_;
    int m_width_;
    int m_channels_;

};

}//namespace segmentation
}//namespace PPP

#endif // HSI_FH_H
