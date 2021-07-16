#ifndef IMG_UTILITY_H
#define IMG_UTILITY_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <Eigen/Sparse>
#include <iostream>
#include <memory>
#include <thread>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "ndarray_converter.h"

namespace py = pybind11;
using namespace std;
using namespace cv;

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;
typedef Eigen::LeastSquaresConjugateGradient<SpMat> Solve;

namespace PPP{
namespace Iutility {


class Img_utility{
public:
    Img_utility(){}
    ~Img_utility(){}

    struct Pidx{
        int x;
        int y;
        uchar valid;
        int idn;
        int label;
    };

    // DEM可视化 =========================================================

    cv::Mat visualizeDEM(cv::Mat Z, float valid_min = 0, double contrast = 0.75){
        double f1[] = {1., 0., -1., 2., 0., -2., 1., 0., -1.};
        double f2[] = {1., 2., 1., 0., 0., 0., -1., -2., -1.};
        cv::Mat f1m = cv::Mat(3, 3, CV_64F, f1)/8.;
        cv::Mat f2m = cv::Mat(3, 3, CV_64F, f2)/8.;

        cv::Mat N1, N2, N3;

        cv::Mat mask = Mat::zeros(Z.size(), CV_8UC1);
        for(int i = 0; i < mask.rows; ++i){
            uchar* maski = mask.ptr<uchar>(i);
            double* Zi = Z.ptr<double>(i);
            for(int j = 0; j < mask.cols; ++j){
                if(Zi[j] > valid_min){
                    maski[j] = 1.;
                }
            }
        }

        cv::Mat Z_valid;
        Z.copyTo(Z_valid, mask);

        cv::filter2D(Z_valid, N1, -1, f1m, Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(Z_valid, N2, -1, f2m, Point(-1, -1), 0, cv::BORDER_REPLICATE);

        double max_value, min_value;


        pow(N1, 2, N1);
        pow(N2, 2, N2);
        sqrt(N1+N2+1., N3);
        N3 = 1./N3;

        cv::minMaxLoc(N3, &min_value, &max_value);
        if(max_value == min_value){
            N3 = N3 - min_value;
        }else{
            N3 = (N3 - min_value)/(max_value -min_value );
        }

        N3 = N3*contrast + (1-contrast);


        cv::minMaxLoc(Z_valid, &min_value, &max_value);
        if(max_value == min_value){
            Z_valid = Z_valid - min_value;
        }else{
            Z_valid = (Z_valid - min_value)/(max_value -min_value );
        }

        Z_valid = Z_valid*0.75;

        Z_valid.convertTo(Z_valid, CV_8U, 180);
        N3.convertTo(N3, CV_8U, 255);

        vector<cv::Mat> hsv;
        cv::Mat hsv_m, vis;
        hsv.push_back(Z_valid);
        hsv.push_back(cv::Mat::ones(Z_valid.size(), CV_8U)*0.75 * 255);
        hsv.push_back(N3);

        merge(hsv, hsv_m );
        cvtColor(hsv_m, vis, COLOR_HSV2RGB);

        std::vector<Mat> chs;
        split(vis, chs);

        std::swap(chs[0], chs[2]);
        merge(chs, vis);


        return vis;
    }
     // DEM可视化 =========================================================

     // Z填充 ======================================================   ===
    cv::Mat inpaintZ(cv::Mat Z, cv::Mat label, double lambda_grad, double lambda_curve, float valid_min = 0){
        int m_height = Z.rows;
        int m_width = Z.cols;

        double lambda_constrain = 10^3;

        Mat dst, kernel;
        kernel = Mat::ones(5, 5, CV_64F);

        cv::Mat mask = Mat::zeros(Z.size(), CV_64F);

        for(int i = 0; i < mask.rows; ++i){
            double* maski = mask.ptr<double>(i);
            double* Zi = Z.ptr<double>(i);
            for(int j = 0; j < mask.cols; ++j){
                if(Zi[j] > valid_min){
                    maski[j] = 1.;
                }
            }
        }

        cv::filter2D(mask, dst, -1, kernel, Point(-1, -1), 0, cv::BORDER_REPLICATE);


        vector<Pidx> fidx;

        double kelnel_size = kernel.rows*kernel.cols;

        for(int i = 0; i < dst.rows; ++i){
            const double* Mi = dst.ptr<double>(i);
            for(int j = 0; j < dst.cols; j++){
                if(Mi[j] < kelnel_size){
                    Pidx pidx;
                    pidx.y = i;
                    pidx.x = j;
                    pidx.valid = mask.at<double>(i, j);
                    pidx.idn = i*mask.cols + j;
                    pidx.label = label.at<double>(i,j);
                    fidx.push_back(pidx);
                }
            }
        }

        vector<T> triple;
        generate_triple(triple, fidx, m_width, m_height, lambda_grad,lambda_curve);

        int triple_size_1 = triple.size();
        int triple_size_2 = triple_size_1;

        vector<double> valid_Z;

        for(int i = 0; i < fidx.size(); ++i){
            if(fidx[i].valid != 0){
                triple.push_back(T(triple_size_2, i, lambda_constrain));
                triple_size_2++;

                double z_valid = Z.at<double>(fidx[i].y, fidx[i].x);
                valid_Z.push_back(z_valid*lambda_constrain);
            }
        }



        int row_A = static_cast<int>(triple.size() );
        int col_A = static_cast<int>(fidx.size()   );
        SpMat A(row_A,col_A);
        Eigen::VectorXd x;
        Eigen::VectorXd b;

        b.resize(row_A);
        b.setZero();

        for(int i = triple_size_1; i < triple_size_2; ++i){
            b(i) = valid_Z[i-triple_size_1];
        }

        A.setFromTriplets(triple.begin(), triple.end() );

        Solve *p_A = new Solve(A);
        x = p_A->solve(b);

        cv::Mat Z_smooth = Z.clone();
        for(int i = 0; i < fidx.size(); ++i){
            Z_smooth.at<double>(fidx[i].y, fidx[i].x) = x(i);
        }

        return Z_smooth;
    }

    void generate_triple(vector<T> &triple, vector<Pidx> &fidx, int m_width, int m_height, double lambda_grad, double lambda_curve){
        triple.clear();

        int n = static_cast<int>(fidx.size() );
        int rownumber = 0;
        for(int i = 0; i < n; ++i){
            int idx0 = fidx[i].x;
            int idy0 = fidx[i].y;

            if(fidx[i].valid == 0){


                int idx1, idy1;
                idx1 = idx0 - 1;
                idy1 = idy0;
                if(idx1 >= 0){
                    int idn1 = idy1*m_width + idx1;
                    add2_triple(triple, fidx, rownumber, i, idn1, 0, lambda_grad);
                }

                idx1 = idx0;
                idy1 = idy0 - 1;
                if(idy1 >= 0){
                    int idn1 = idy1*m_width + idx1;
                    add2_triple(triple, fidx, rownumber, i, idn1, 0, lambda_grad);
                }

                idx1 = idx0 + 1;
                idy1 = idy0;
                if(idx1 < m_width){
                    int idn1 = idy1*m_width + idx1;
                    add2_triple(triple, fidx, rownumber, i, idn1, 1, lambda_grad);
                }

                idx1 = idx0;
                idy1 = idy0 + 1;
                if(idy1 < m_height){
                    int idn1 = idy1*m_width + idx1;
                    add2_triple(triple, fidx, rownumber, i, idn1, 1, lambda_grad);
                }

            }

            if((idx0-1 >= 0) && (idx0+1 < m_width)){
                int idn0 = idy0*m_width + idx0 - 1;
                int k1  = findk(fidx, i, idn0, 0);

                int idn1 = idy0*m_width + idx0 + 1;
                int k2 = findk(fidx, i, idn1, 1);

                if(k1 > -1 && k2 > -1){
                    if(fidx[k1].valid == 0 || fidx[k2].valid == 0 || fidx[i].valid){
                        if((fidx[i].label == fidx[k1].label) && (fidx[i].label == fidx[k2].label)){
                            triple.push_back(T(rownumber, k1, -lambda_curve));
                            triple.push_back(T(rownumber, k2, -lambda_curve));
                            triple.push_back(T(rownumber, i, 2.*lambda_curve));
                            rownumber++;
                        }
                    }
                }
            }

            if((idy0-1 >= 0) && (idy0+1 < m_height)){
                int idn0 = (idy0-1)*m_width + idx0;
                int k1  = findk(fidx, i, idn0, 0);

                int idn1 = (idy0+1)*m_width + idx0;
                int k2 = findk(fidx, i, idn1, 1);

                if(k1 > -1 && k2 > -1){
                    if(fidx[k1].valid == 0 || fidx[k2].valid == 0 || fidx[i].valid){
                        if((fidx[i].label == fidx[k1].label) && (fidx[i].label == fidx[k2].label)){
                            triple.push_back(T(rownumber, k1, -lambda_curve));
                            triple.push_back(T(rownumber, k2, -lambda_curve));
                            triple.push_back(T(rownumber, i, 2.*lambda_curve));
                            rownumber++;
                        }
                    }
                }
            }

        }

    }

    int findk(vector<Pidx> &fidx, int i, int idn0, int mode){
        if(mode == 0){
            for(int k = i; k >= 0; k--){
                if(fidx[k].idn == idn0){
                    return k;
                }
            }
        }else{
            for(int k = i; k < fidx.size(); k++){
                if(fidx[k].idn == idn0){
                    return k;
                }
            }
        }

        return -1;
    }

    void add2_triple(vector<T> &triple, vector<Pidx> &fidx, int &rownumber, int i, int idn0, int mode,double lambda_grad){
        int k_found = findk(fidx, i, idn0, mode);

        if(fidx[k_found].label == fidx[i].label){
            if(mode == 0){
                if(fidx[k_found].valid != 0 ){
                    triple.push_back(T(rownumber, k_found, lambda_grad));
                    triple.push_back(T(rownumber, i, -lambda_grad));
                    rownumber++;
                }
            }else{
                triple.push_back(T(rownumber, k_found, -lambda_grad));
                triple.push_back(T(rownumber, i, lambda_grad));
                rownumber++;
            }
        }

    }
     // Z填充 ======================================================   ===

    // 点云排序 =========================================================
    struct pointC{
        double x;
        double y;
        int index;
    };

    static bool compareC(const pointC &point1, const pointC &point2){
        if(point1.y < point2.y){
            return true;
        }else if(point1.y > point2.y){
            return false;
        }else if(point1.y == point2.y){
            return point1.x < point2.x;
        }
    }


    vector<int> cloud_sort(vector<double> &y_, vector<double> &x_){
        vector<pointC> points;
        points.resize(y_.size() );

        for(size_t i = 0; i < y_.size() ; ++i){
            points[i].y = y_[i];
            points[i].x = x_[i];
            points[i].index = i;
        }

        sort(points.begin(), points.end(), compareC);

        vector <int> return_index(y_.size() );
        for(size_t i = 0; i < y_.size() ; ++i){
            return_index[i] = points[i].index;
        }
        return return_index;
    }
    // 点云排序 =========================================================

};


}
}


#endif // IMG_UTILITY_H
