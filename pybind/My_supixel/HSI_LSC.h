#ifndef HSI_LSC_H
#define HSI_LSC_H

#include <iostream>
#include <memory>
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
#include <ctime>
#include <map>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

namespace PPP {
namespace HSI_supixel {

class HSI_LSC
{
public:
    HSI_LSC(){};

    ~HSI_LSC();

    void setHSI(Mat _image);
    void setParameter(float _rho , int redion_size, float ratio);

    void initImg();

    double m_rho;//系数

    int m_width;//图像宽度

    int m_height;//图像长度

    int m_nr_channels;//图像的波段数

    int m_imgSize;//图像的像素数目

    Mat m_Im;//图像每个像素各波段的原始数据

//    Mat m_Im2;

//    Mat m_Imc;

    Mat m_R;//图像每个像素的本征值

//    Mat m_S;//图像每个像素的shading值

//    Mat m_ntscIm;//图像每个像素的Y值

    double m_epsilon ;
   // ==========================

    //superpixel


    void iterate_Sup( int num_iterations = 10 );

    int getNumberofSuppixels_Sup() const;

    void getLabels_Sup( OutputArray labels_out ) const;

    // get mask image with contour
     cv::Mat getLabelContourMask_Sup( bool thick_line = true ) const;

    // enforce connectivity over labels
     void enforceLabelConnectivity_Sup( int min_element_size );

     Mat m_klabels;

     int computeLabel( int x, int y) {
        return std::min(y / (m_height / m_RowNum), m_RowNum- 1) * m_ColNum
                + std::min((x / (m_width / m_ColNum)), m_ColNum - 1);
    }
     void redo(   int region_size,float  ratio)
     {
         m_region_size = region_size;
         m_ratio = ratio;
     }

     ///////////////////////////////////////////////////////////
     cv::Mat Generate_supmap(cv::Mat RGBimg){
         int Dx[2] = {-1, 1};
         int Dy[2] = {-1, 1};
         for (int y = 0; y < m_height; y++) {
             for (int x = 0; x < m_width; x++) {
                 int count = 0;
                 int L = m_klabels.at<int>(y, x);
                 for(int dx : Dx){
                     for(int dy : Dy){
                         int nei_x = std::min(x+dx, m_width-1);
                         int nei_y = std::min(y+dy, m_height-1);
                         nei_x = std::max(nei_x, 0);
                         nei_y = std::max(nei_y, 0);

                         if(m_klabels.at<int>(nei_y, nei_x) != L){
                             count++;
                         }
                     }
                 }

                 if(count > 0){
                     auto *p = RGBimg.ptr<uchar>(y,x);
                     p[0] = 255;
                     p[1] = 0;
                     p[2] = 0;
                 }
             }
         }

         return  RGBimg;
     }

     ////////////////////////////////////
     cv::Mat GetSuppixelLabel(){
         return m_klabels;
     }
protected:


    // seeds stepx
    int m_stepx;

    // seeds stepy
    int m_stepy;


    // region size
    int m_region_size;

    // ratio
    float m_ratio;

    int m_ColNum;
    int m_RowNum;
private:

    // labels no
    int m_numlabels;

    // color coefficient
    float m_color_coeff;

    // dist coefficient
    float m_dist_coeff;

    // threshold coeff
    int m_threshold_coeff;

    // max value from
    // image channels
    double m_chvec_max;

    // stacked channels
    // of original image
    vector<Mat_<double> > m_chvec;
    vector<Mat> m_spectrum;
    vector<Mat> m_Kspectrum;

    Mat Nsum;

    // seeds on x
    vector<float> m_kseedsx;

    // seeds on y
    vector<float> m_kseedsy;

    // W
    Mat m_W;

    // labels storage


    // initialization
    inline void initialize();

    // fetch seeds
    inline void GetChSeeds();

    // precompute vector space
    inline void GetFeatureSpace();

    // LSC
    inline void PerformLSC( const int& num_iterations );

    // pre-enforce connectivity over labels
    inline void PreEnforceLabelConnectivity( int min_element_size );

    // enforce connectivity over labels
    inline void PostEnforceLabelConnectivity( int threshold );

    // re-count superpixles
    inline void countSuperpixels();

    inline float square(float x){return x*x;}


};


void HSI_LSC::setParameter(float _rho , int region_size, float ratio){
    m_region_size = region_size;
    m_ratio = ratio;
    m_rho = _rho;
}

void HSI_LSC::setHSI(Mat _image )
{
//        Mat image = _image;
//        CV_Assert( !image.empty() );

        cout<<"184s"<<endl;
        m_epsilon = 0.0001;

        m_width  = _image.cols;
        m_height = _image.rows;

        m_imgSize = m_width*m_height ;
        m_nr_channels = _image.channels();
//        cout<<"185s"<<endl;

        _image.convertTo( m_Im, CV_64FC(m_nr_channels));
//        cout<<"1856"<<endl;
        cv::minMaxIdx( m_Im, NULL, &m_chvec_max, NULL, NULL );
//        cout<<"187s"<<" "<<m_chvec_max<<" "<<m_Im.cols<<" "<<m_Im.rows<<" "<<m_Im.channels()<<endl;
        if(m_chvec_max!=1)
            cv::add( m_Im/m_chvec_max, m_epsilon, m_Im );
        else
            cv::add( m_Im, m_epsilon, m_Im );
//        cout<<"200s"<<endl;

        m_R = m_Im;
        cv::split(m_R,m_chvec);
//        Mat m_Imc;
//        cv::log(m_Im, m_Imc);
//        cv::Mat m_Imean;
//        cv::reduce(m_Imc.reshape(1,m_imgSize), m_Imean, 1, cv::REDUCE_AVG);
//        m_Imc = m_Imc - cv::repeat( m_Imean, 1, m_nr_channels ).reshape(m_nr_channels,m_height);
//        cv::exp(m_Imc, m_R);
//        cv::split(m_R,m_chvec);


}

HSI_LSC::~HSI_LSC()
{

}


class Superpixel
{
public:

    int Label, Size;
    vector<int> Neighbor, xLoc, yLoc;

    Superpixel( int L = 0, int S = 0 ) : Label(L), Size(S) { }

    friend bool operator == ( Superpixel& S, int L )
    {
        return S.Label == L;
    }
    friend bool operator == ( int L, Superpixel& S )
    {
        return S.Label == L;
    }
};



int HSI_LSC::getNumberofSuppixels_Sup() const
{
    return m_numlabels;
}

void HSI_LSC::initialize()
{
    // basic coeffs
    m_color_coeff = 20.0f;
    m_threshold_coeff = 4;
    m_dist_coeff = m_color_coeff * m_ratio;

    // total amount of superpixels given region size
    m_numlabels = int(float(m_width * m_height)
                /  float(m_region_size * m_region_size));



    // intitialize label storage
    m_klabels = Mat( m_height, m_width, CV_32S, Scalar::all(0) );

    // init seeds
    GetChSeeds();
}

void HSI_LSC::iterate_Sup( int num_iterations )
{
    // init
    initialize();

    // feature space
    GetFeatureSpace();

    Mat T;
    cv::minMaxIdx( m_R, NULL, &m_chvec_max, NULL, NULL );

    cv::split(m_R/m_chvec_max,m_chvec);

    PerformLSC( num_iterations );

}

void HSI_LSC::getLabels_Sup(OutputArray labels_out) const
{
    labels_out.create(m_height,m_width,CV_32S);
    labels_out.getMat()=m_klabels;
}

cv::Mat HSI_LSC::getLabelContourMask_Sup( bool _thick_line) const
{
    // default width
    int line_width = 2;

    if ( !_thick_line ) line_width = 1;

    cv::Mat mask;
    mask.create( m_height, m_width, CV_8UC1 );

    mask.setTo(0);

    const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
    const int dy8[8] = {  0, -1, -1, -1, 0, 1, 1,  1 };

    int sz = m_width*m_height;

    vector<bool> istaken(sz, false);

    int mainindex = 0;
    for( int j = 0; j < m_height; j++ )
    {
      for( int k = 0; k < m_width; k++ )
      {
        int np = 0;
        for( int i = 0; i < 8; i++ )
        {
          int x = k + dx8[i];
          int y = j + dy8[i];

          if( (x >= 0 && x < m_width) && (y >= 0 && y < m_height) )
          {
            int index = y*m_width + x;

            if( false == istaken[index] )
            {
              if( m_klabels.at<int>(j,k) != m_klabels.at<int>(y,x) ) np++;
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

/*
 * enforceLabelConnectivity
 *
 *   1. finding an adjacent label for each new component at the start
 *   2. if a certain component is too small, assigning the previously found
 *      adjacent label to this component, and not incrementing the label.
 *
 */
void HSI_LSC::enforceLabelConnectivity_Sup( int min_element_size )
{
    int threshold = (m_width * m_height)
                  / (m_numlabels * m_threshold_coeff);

    PreEnforceLabelConnectivity( min_element_size );
    PostEnforceLabelConnectivity( threshold );
    countSuperpixels();
}

/*
 * countSuperpixels()
 *
 *   1. count unique superpixels
 *   2. relabel all superpixels
 *
 */
inline void HSI_LSC::countSuperpixels()
{
    std::map<int,int> labels;

    int labelNum = 0;
    int prev_label = -1;
    int mark_label = 0;
    for( int x = 0; x < m_width; x++ )
    {
      for( int y = 0; y < m_height; y++ )
      {
        int curr_label = m_klabels.at<int>(y,x);

        // relax, just do relabel
        if ( curr_label == prev_label )
        {
          m_klabels.at<int>(y,x) = mark_label;
          continue;
        }

        // on label change do map lookup
        map<int,int>::iterator it = labels.find( curr_label );

        // if new label seen
        if ( it == labels.end() )
        {
          mark_label = labelNum; labelNum++;
          labels.insert( pair<int,int>( curr_label, mark_label ) );
          m_klabels.at<int>(y,x) = mark_label;
        } else
        {
          mark_label = it->second;
          m_klabels.at<int>(y,x) = mark_label;
        }
        prev_label = curr_label;
      }
    }
    m_numlabels = (int) labels.size();
}

/*
 * PreEnforceLabelConnectivity
 *
 *   1. finding an adjacent label for each new component at the start
 *   2. if a certain component is too small, assigning the previously found
 *      adjacent label to this component, and not incrementing the label.
 *
 */
inline void HSI_LSC::PreEnforceLabelConnectivity( int min_element_size )
{
    const int dx8[8] = { -1, -1,  0,  1,  1,  1,  0, -1 };
    const int dy8[8] = {  0, -1, -1, -1,  0,  1,  1,  1 };

    int adj = 0;
    vector<int> xLoc, yLoc;
    cv::Mat mask( m_height, m_width, CV_8U , Scalar::all(0) );

    for( int i = 0; i < m_width; i++ )
    {
      for( int j = 0; j < m_height; j++)
      {
        if( mask.at<uchar>(j,i) == 0 )
        {
          int L = m_klabels.at<int>(j,i);

          for( int k = 0; k < 8; k++ )
          {
            int x = i + dx8[k];
            int y = j + dy8[k];
            if ( x >= 0 && x <= m_width -1
              && y >= 0 && y <= m_height-1)
            {
              if ( mask.at<uchar>(y,x) == 1
                 && m_klabels.at<int>(y,x) != L )
                adj = m_klabels.at<int>(y,x);
              break;
            }
          }

          mask.at<uchar>(j,i) = 1;
          xLoc.insert( xLoc.end(), i );
          yLoc.insert( yLoc.end(), j );

          size_t indexMarker = 0;
          while( indexMarker < xLoc.size() )
          {
            int x = xLoc[indexMarker];
            int y = yLoc[indexMarker];

            indexMarker++;

            int minX = ( x-1 <= 0 ) ? 0 : x-1;
            int minY = ( y-1 <= 0 ) ? 0 : y-1;
            int maxX = ( x+1 >= m_width -1 ) ? m_width -1 : x+1;
            int maxY = ( y+1 >= m_height-1 ) ? m_height-1 : y+1;
            for( int m = minX; m <= maxX; m++ )
            {
              for( int n = minY; n <= maxY; n++ )
              {
                if (   mask.at<uchar>(n,m) == 0
                 && m_klabels.at<int>(n,m) == L )
                {
                  mask.at<uchar>(n,m) = 1;
                  xLoc.insert( xLoc.end(), m );
                  yLoc.insert( yLoc.end(), n );
                }
              }
            }
          }
          if ( indexMarker < (size_t) min_element_size )
          {
            for( size_t k = 0; k < xLoc.size(); k++ )
            {
              int x = xLoc[k];
              int y = yLoc[k];
              m_klabels.at<int>(y,x) = adj;
            }
          }
          xLoc.clear();
          yLoc.clear();
        }
      }
    }
}

/*
 * PostEnforceLabelConnectivity
 *
 */
inline void HSI_LSC::PostEnforceLabelConnectivity( int threshold )
{
    float PI2 = float(CV_PI / 2.0f);

    vector<float> centerW;
    queue <int> xLoc, yLoc;
    vector<float> centerX1, centerX2;
    vector<float> centerY1, centerY2;
    vector<int> strayX, strayY, Size;
    vector< vector<float> > centerC1( m_nr_channels );
    vector< vector<float> > centerC2( m_nr_channels );

    cv::Mat mask( m_height, m_width, CV_8U, Scalar::all(0) );

    int L;
    int sLabel = -1;
    for( int i = 0; i < m_width; i++ )
    {
      for( int j = 0; j < m_height; j++ )
      {
        if( mask.at<uchar>(j,i) == 0 )
        {
          sLabel++;
          int count = 1;

          centerW.insert( centerW.end(), 0 );
          for ( int b = 0; b < m_nr_channels; b++ )
          {
            centerC1[b].insert( centerC1[b].end(), 0 );
            centerC2[b].insert( centerC2[b].end(), 0 );
          }
          centerX1.insert( centerX1.end(), 0 );
          centerX2.insert( centerX2.end(), 0 );
          centerY1.insert( centerY1.end(), 0 );
          centerY2.insert( centerY2.end(), 0 );

          strayX.insert( strayX.end(), i );
          strayY.insert( strayY.end(), j );

          float Weight = m_W.at<float>(j,i);

          // accumulate dists
          centerW[sLabel] += Weight;
          for ( int b = 0; b < m_nr_channels; b++ )
          {

            float thetaC = 0.0f;
            switch ( m_chvec[b].depth() )
            {
              case CV_8U:
                thetaC = ( (float) m_chvec[b].at<uchar>(j,i)  / m_chvec_max ) * PI2;
                break;
              case CV_8S:
                thetaC = ( (float) m_chvec[b].at<char>(j,i)   / m_chvec_max ) * PI2;
                break;
              case CV_16U:
                thetaC = ( (float) m_chvec[b].at<ushort>(j,i) / m_chvec_max ) * PI2;
                break;
              case CV_16S:
                thetaC = ( (float) m_chvec[b].at<short>(j,i)  / m_chvec_max ) * PI2;
                break;
              case CV_32S:
                thetaC = ( (float) m_chvec[b].at<int>(j,i)    / m_chvec_max ) * PI2;
                break;
              case CV_32F:
                thetaC = ( (float) m_chvec[b].at<float>(j,i)  / m_chvec_max ) * PI2;
                break;
              case CV_64F:
                thetaC = ( (float) m_chvec[b].at<double>(j,i) / m_chvec_max ) * PI2;
                break;
            }

            // we do not store pre-computed C1[b], C2[b]
            float C1 = m_color_coeff * cos(thetaC) / m_nr_channels;
            float C2 = m_color_coeff * sin(thetaC) / m_nr_channels;

            centerC1[b][sLabel] += C1; centerC2[b][sLabel] += C2;
          }

          float thetaX = ( (float) i / (float) m_stepx ) * PI2;
          // we do not store pre-computed x1, x2
          float X1 = m_dist_coeff * cos(thetaX);
          float X2 = m_dist_coeff * sin(thetaX);

          float thetaY = ( (float) j / (float) m_stepy ) * PI2;
          // we do not store pre-computed y1, y2
          float Y1 = m_dist_coeff * cos(thetaY);
          float Y2 = m_dist_coeff * sin(thetaY);

          centerX1[sLabel] += X1; centerX2[sLabel] += X2;
          centerY1[sLabel] += Y1; centerY2[sLabel] += Y2;

          L = m_klabels.at<int>(j,i);
          m_klabels.at<int>(j,i) = sLabel;

          mask.at<uchar>(j,i) = 1;

          xLoc.push( i ); yLoc.push( j );
          while( !xLoc.empty() )
          {
            int x = xLoc.front(); xLoc.pop();
            int y = yLoc.front(); yLoc.pop();
            int minX = ( x-1 <=0 ) ? 0 : x-1;
            int minY = ( y-1 <=0 ) ? 0 : y-1;
            int maxX = ( x+1 >= m_width -1 ) ? m_width -1 : x+1;
            int maxY = ( y+1 >= m_height-1 ) ? m_height-1 : y+1;
            for( int m = minX; m <= maxX; m++ )
            {
              for( int n = minY; n <= maxY; n++ )
              {
                if(   mask.at<uchar>(n,m) == 0
                && m_klabels.at<int>(n,m) == L )
                {
                  count++;
                  xLoc.push(m); yLoc.push(n);

                  mask.at<uchar>(n,m) = 1;
                  m_klabels.at<int>(n,m) = sLabel;

                  Weight = m_W.at<float>(n,m);
                  centerW[sLabel] += Weight;
                  for ( int b = 0; b < m_nr_channels; b++ )
                  {
                    float thetaC = 0.0f;
                    switch ( m_chvec[b].depth() )
                    {
                      case CV_8U:
                        thetaC = ( (float) m_chvec[b].at<uchar>(j,i)  / m_chvec_max ) * PI2;
                        break;
                      case CV_8S:
                        thetaC = ( (float) m_chvec[b].at<char>(j,i)   / m_chvec_max ) * PI2;
                        break;
                      case CV_16U:
                        thetaC = ( (float) m_chvec[b].at<ushort>(j,i) / m_chvec_max ) * PI2;
                        break;
                      case CV_16S:
                        thetaC = ( (float) m_chvec[b].at<short>(j,i)  / m_chvec_max ) * PI2;
                        break;
                      case CV_32S:
                        thetaC = ( (float) m_chvec[b].at<int>(j,i)    / m_chvec_max ) * PI2;
                        break;
                      case CV_32F:
                        thetaC = ( (float) m_chvec[b].at<float>(j,i)  / m_chvec_max ) * PI2;
                        break;
                      case CV_64F:
                        thetaC = ( (float) m_chvec[b].at<double>(j,i) / m_chvec_max ) * PI2;
                        break;

                    }
                    // we do not store pre-computed C1[b], C2[b]
                    float C1 = m_color_coeff * cos(thetaC) / m_nr_channels;
                    float C2 = m_color_coeff * sin(thetaC) / m_nr_channels;

                    centerC1[b][sLabel] += C1; centerC2[b][sLabel] += C2;
                  }

                  thetaX = ( (float) m / (float) m_stepx ) * PI2;
                  // we do not store pre-computed x1, x2
                  X1 = m_dist_coeff * cos(thetaX);
                  X2 = m_dist_coeff * sin(thetaX);

                  thetaY = ( (float) n / (float) m_stepy ) * PI2;
                  // we do not store pre-computed y1, y2
                  Y1 = m_dist_coeff * cos(thetaY);
                  Y2 = m_dist_coeff * sin(thetaY);

                  centerX1[sLabel] += X1; centerX2[sLabel] += X2;
                  centerY1[sLabel] += Y1; centerY2[sLabel] += Y2;

                }
              }
            }
          }
          Size.insert( Size.end(), count );
          for ( int b = 0; b < m_nr_channels; b++ )
          {
            centerC1[b][sLabel] /= centerW[sLabel];
            centerC2[b][sLabel] /= centerW[sLabel];
          }
          centerX1[sLabel] /= centerW[sLabel];
          centerX2[sLabel] /= centerW[sLabel];
          centerY1[sLabel] /= centerW[sLabel];
          centerY2[sLabel] /= centerW[sLabel];
        }
      }
    }
    sLabel++;

    vector<Superpixel> Sarray;
    vector<int>::iterator Pointer;
    for( int i = 0; i < sLabel; i++)
    {
      if( Size[i] < threshold )
      {
        int x = strayX[i];
        int y = strayY[i];

        L = m_klabels.at<int>(y,x);
        mask.at<uchar>(y,x) = 0;

        size_t indexMark = 0;
        Superpixel S( L, Size[i] );

        S.xLoc.insert( S.xLoc.end(),x );
        S.yLoc.insert( S.yLoc.end(),y );
        while( indexMark < S.xLoc.size() )
        {
          x = S.xLoc[indexMark];
          y = S.yLoc[indexMark];

          indexMark++;

          int minX = ( x-1 <= 0 ) ? 0 : x-1;
          int minY = ( y-1 <= 0 ) ? 0 : y-1;
          int maxX = ( x+1 >= m_width -1 ) ? m_width -1 : x+1;
          int maxY = ( y+1 >= m_height-1 ) ? m_height-1 : y+1;
          for( int m = minX; m <= maxX; m++ )
          {
            for( int n = minY; n <= maxY; n++ )
            {
              if(   mask.at<uchar>(n,m) == 1
              && m_klabels.at<int>(n,m) == L )
              {
                mask.at<uchar>(n,m) = 0;

                S.xLoc.insert( S.xLoc.end(), m );
                S.yLoc.insert( S.yLoc.end(), n );
              }
              else if( m_klabels.at<int>(n,m) != L )
              {
                int NewLabel = m_klabels.at<int>(n,m);
                Pointer = find( S.Neighbor.begin(), S.Neighbor.end(), NewLabel );
                if ( Pointer == S.Neighbor.end() )
                {
                  S.Neighbor.insert( S.Neighbor.begin(), NewLabel );
                }
              }
            }
          }
        }
        Sarray.insert(Sarray.end(),S);
      }
    }

    vector<int>::iterator I, I2;
    vector<Superpixel>::iterator S;

    S = Sarray.begin();
    while( S != Sarray.end() )
    {
      int Label1 = (*S).Label;
      int Label2 = -1;

      double MinDist = DBL_MAX;
      for ( I = (*S).Neighbor.begin(); I != (*S).Neighbor.end(); I++ )
      {
        double D = 0.0f;

        for ( int b = 0; b < m_nr_channels; b++ )
        {
          float diffcenterC1 = centerC1[b][Label1]
                             - centerC1[b][*I];
          float diffcenterC2 = centerC2[b][Label1]
                             - centerC2[b][*I];

          D += (diffcenterC1 * diffcenterC1)
             + (diffcenterC2 * diffcenterC2);
        }

        float diffcenterX1 = centerX1[Label1] - centerX1[*I];
        float diffcenterX2 = centerX2[Label1] - centerX2[*I];
        float diffcenterY1 = centerY1[Label1] - centerY1[*I];
        float diffcenterY2 = centerY2[Label1] - centerY2[*I];

        D += (diffcenterX1 * diffcenterX1)
           + (diffcenterX2 * diffcenterX2)
           + (diffcenterY1 * diffcenterY1)
           + (diffcenterY2 * diffcenterY2);

        // if within dist
        if ( D < MinDist )
        {
          MinDist = D;
          Label2 = (*I);
        }
      }

      double W1 = centerW[Label1];
      double W2 = centerW[Label2];

      double W = W1 + W2;

      for ( int b = 0; b < m_nr_channels; b++ )
      {
        centerC1[b][Label2] = float((W2*centerC1[b][Label2] + W1*centerC1[b][Label1]) / W);
        centerC2[b][Label2] = float((W2*centerC2[b][Label2] + W1*centerC2[b][Label1]) / W);
      }
      centerX1[Label2] = float((W2*centerX1[Label2] + W1*centerX1[Label1]) / W);
      centerX2[Label2] = float((W2*centerX2[Label2] + W1*centerX2[Label1]) / W);
      centerY1[Label2] = float((W2*centerY1[Label2] + W1*centerY1[Label1]) / W);
      centerY2[Label2] = float((W2*centerY2[Label2] + W1*centerY2[Label1]) / W);

      centerW[Label2] = (float)W;
      for( size_t i = 0; i < (*S).xLoc.size(); i++ )
      {
        int x = (*S).xLoc[i];
        int y = (*S).yLoc[i];
        m_klabels.at<int>(y,x) = Label2;
      }

      vector<Superpixel>::iterator Stmp;
      Stmp = find( Sarray.begin(), Sarray.end(), Label2 );
      if( Stmp != Sarray.end() )
      {
        Size[Label2] = Size[Label1] + Size[Label2];
        if( Size[Label2] >= threshold )
        {
          Sarray.erase( S );
          Stmp = find( Sarray.begin(), Sarray.end(), Label2 );
          Sarray.erase( Stmp );
        }
        else
        {
          (*Stmp).xLoc.insert( (*Stmp).xLoc.end(), (*S).xLoc.begin(), (*S).xLoc.end() );
          (*Stmp).yLoc.insert( (*Stmp).yLoc.end(), (*S).yLoc.begin(), (*S).yLoc.end() );

          (*Stmp).Neighbor.insert( (*Stmp).Neighbor.end(), (*S).Neighbor.begin(), (*S).Neighbor.end() );

          sort( (*Stmp).Neighbor.begin(), (*Stmp).Neighbor.end() );

          I = unique( (*Stmp).Neighbor.begin(), (*Stmp).Neighbor.end() );
          (*Stmp).Neighbor.erase( I, (*Stmp).Neighbor.end() );

          I = find  ( (*Stmp).Neighbor.begin(), (*Stmp).Neighbor.end(), Label1 );
          (*Stmp).Neighbor.erase( I );

          I = find  ( (*Stmp).Neighbor.begin(), (*Stmp).Neighbor.end(), Label2 );
          (*Stmp).Neighbor.erase( I );

          Sarray.erase( S );
        }
      } else Sarray.erase( S );

      for( size_t i = 0; i < Sarray.size(); i++ )
      {
        I  = find( Sarray[i].Neighbor.begin(), Sarray[i].Neighbor.end(), Label1 );
        I2 = find( Sarray[i].Neighbor.begin(), Sarray[i].Neighbor.end(), Label2 );

        if ( I  != Sarray[i].Neighbor.end()
        &&   I2 != Sarray[i].Neighbor.end() )
        {
          Sarray[i].Neighbor.erase( I );
        }
        else
        if ( I  != Sarray[i].Neighbor.end()
         &&  I2 == Sarray[i].Neighbor.end() )
        {
          (*I) = Label2;
        }
      }

      S = Sarray.begin();

    }
}

/*
 * GetChannelsSeeds_ForGivenStepSize
 *
 * The k seed values are
 * taken as uniform spatial
 * pixel samples.
 *
 */
inline void HSI_LSC::GetChSeeds()
{
    int ColNum = (int) sqrt( (double) m_numlabels
               * ((double)m_width / (double)m_height) );
    int RowNum = m_numlabels / ColNum;

    m_stepx = m_width / ColNum;
    m_stepy = m_height / RowNum;

    int Col_remain = m_width  - (m_stepx*ColNum);
    int Row_remain = m_height - (m_stepy*RowNum);

    int count = 0;
    int t1 = 1, t2 = 1;
    int centerx, centery;

    for( int x = 0; x < ColNum; x++ )
    {
      t2 = 1;
      centerx = int((x*m_stepx) + (0.5f*m_stepx) + t1);
      if ( centerx >= m_width -1 ) centerx = m_width -1;

      for( int y = 0; y < RowNum; y++ )
      {
        centery = int((y*m_stepy) + (0.5f*m_stepy) + t2);

        if ( t2 < Row_remain ) t2++;
        if ( centery >= m_height-1 ) centery = m_height-1;

        m_kseedsx.push_back( (float)centerx );
        m_kseedsy.push_back( (float)centery );

        count++;
      }
      if ( t1 < Col_remain ) t1++;
    }
    // update amount
    m_numlabels = count;
}

struct FeatureSpaceSigmas
{
    FeatureSpaceSigmas( const vector< Mat_<double> >& _chvec, const int _nr_channels,
                        const float _chvec_max, const float _dist_coeff,
                        const float _color_coeff, const int _stepx, const int _stepy )
    {
      chvec = _chvec;
      stepx = _stepx;
      stepy = _stepy;
      chvec_max = _chvec_max;
      dist_coeff = _dist_coeff;
      nr_channels = _nr_channels;
      color_coeff = _color_coeff;

      PI2 = float(CV_PI / 2.0f);
      sigmaX1 = 0; sigmaX2 = 0;
      sigmaY1 = 0; sigmaY2 = 0;
      sigmaC1.resize( nr_channels );
      sigmaC2.resize( nr_channels );
      fill( sigmaC1.begin(), sigmaC1.end(), 0 );
      fill( sigmaC2.begin(), sigmaC2.end(), 0 );
      // previous block state
      double tmp_sigmaX1 = sigmaX1;
      double tmp_sigmaX2 = sigmaX2;
      double tmp_sigmaY1 = sigmaY1;
      double tmp_sigmaY2 = sigmaY2;
      vector<double> tmp_sigmaC1( nr_channels );
      vector<double> tmp_sigmaC2( nr_channels );
      for( int b = 0; b < nr_channels; b++ )
      {
        tmp_sigmaC1[b] = sigmaC1[b];
        tmp_sigmaC2[b] = sigmaC2[b];
      }

      for ( int x = 0; x <chvec[0].cols; x++ )
      {
        float thetaX = ( (float) x / (float) stepx ) * PI2;
        // we do not store pre-computed x1, x2
        float x1 = dist_coeff * cos(thetaX);
        float x2 = dist_coeff * sin(thetaX);

        for( int y = 0; y < chvec[0].rows; y++ )
        {
          float thetaY = ( (float) y / (float) stepy ) * PI2;
          // we do not store pre-computed y1, y2
          float y1 = dist_coeff * cos(thetaY);
          float y2 = dist_coeff * sin(thetaY);

          // accumulate distance sigmas
          tmp_sigmaX1 += x1; tmp_sigmaX2 += x2;
          tmp_sigmaY1 += y1; tmp_sigmaY2 += y2;

          for( int b = 0; b < nr_channels; b++ )
          {


            double thetaC =  chvec[b](y,x) * PI2;

            // we do not store pre-computed C1[b], C2[b]
            float C1 = color_coeff * cos(thetaC) / nr_channels;
            float C2 = color_coeff * sin(thetaC) / nr_channels;

            // accumulate sigmas per channels
            tmp_sigmaC1[b] += C1; tmp_sigmaC2[b] += C2;
          }
        }
      }
      sigmaX1 = tmp_sigmaX1; sigmaX2 = tmp_sigmaX2;
      sigmaY1 = tmp_sigmaY1; sigmaY2 = tmp_sigmaY2;
      for( int b = 0; b < nr_channels; b++ )
      {
        sigmaC1[b] = tmp_sigmaC1[b];
        sigmaC2[b] = tmp_sigmaC2[b];
      }
    }



    float PI2;
    int nr_channels;
    int stepx, stepy;

    double sigmaX1, sigmaX2;
    double sigmaY1, sigmaY2;

    float chvec_max;
    float dist_coeff;
    float color_coeff;

    vector<Mat_<double> > chvec;
    vector<double> sigmaC1;
    vector<double> sigmaC2;
};

struct FeatureSpaceWeights : ParallelLoopBody
{
    FeatureSpaceWeights( const vector< Mat_<double> >& _chvec, Mat* _W,
                         const double _sigmaX1, const double _sigmaX2,
                         const double _sigmaY1, const double _sigmaY2,
                         vector<double>& _sigmaC1, vector<double>& _sigmaC2,
                         const int _nr_channels, const float _chvec_max,
                         const float _dist_coeff, const float _color_coeff,
                         const int _stepx, const int _stepy )
    {
      W = _W;
      chvec = _chvec;
      stepx = _stepx;
      stepy = _stepy;
      chvec_max = _chvec_max;
      dist_coeff = _dist_coeff;
      nr_channels = _nr_channels;
      color_coeff = _color_coeff;

      PI2 = float(CV_PI / 2.0f);

      sigmaX1 = _sigmaX1; sigmaX2 = _sigmaX2;
      sigmaY1 = _sigmaY1; sigmaY2 = _sigmaY2;
      sigmaC1 = _sigmaC1; sigmaC2 = _sigmaC2;
    }

    void operator()( const Range& range ) const
    {
      for( int x = range.start; x < range.end; x++ )
      {
        float thetaX = ( (float) x / (float) stepx ) * PI2;

        for( int y = 0; y < chvec[0].rows; y++ )
        {
          float thetaY = ( (float) y / (float) stepy ) * PI2;

          // accumulate distance channels weighted by sigmas
          W->at<float>(y,x) += float((dist_coeff * cos(thetaX)) * sigmaX1);
          W->at<float>(y,x) += float((dist_coeff * sin(thetaX)) * sigmaX2);
          W->at<float>(y,x) += float((dist_coeff * cos(thetaY)) * sigmaY1);
          W->at<float>(y,x) += float((dist_coeff * sin(thetaY)) * sigmaY2);

          for( int b = 0; b < nr_channels; b++ )
          {


            double thetaC = chvec[b](y,x)* PI2;

            // accumulate color channels weighted by sigmas
            W->at<float>(y,x) += float((color_coeff * cos(thetaC) / nr_channels) * sigmaC1[b]);
            W->at<float>(y,x) += float((color_coeff * sin(thetaC) / nr_channels) * sigmaC2[b]);
          }
          W->at<float>(y,x)=1/W->at<float>(y,x);
        }
      }

    }

    Mat* W;
    float PI2;
    int nr_channels;
    int stepx, stepy;

    double sigmaX1, sigmaX2;
    double sigmaY1, sigmaY2;

    float chvec_max;
    float dist_coeff;
    float color_coeff;

    vector<Mat_<double> > chvec;
    vector<double> sigmaC1;
    vector<double> sigmaC2;
};

/*
 * Compute Feature Space
 *
 *
 */
inline void HSI_LSC::GetFeatureSpace()
{
    double sigmaX1 = 0.0f, sigmaX2 = 0.0f;
    double sigmaY1 = 0.0f, sigmaY2 = 0.0f;
    vector<double> sigmaC1( m_nr_channels , 0.0f );
    vector<double> sigmaC2( m_nr_channels , 0.0f );

    // compute feature space accumulation sigmas
    FeatureSpaceSigmas fss( m_chvec, m_nr_channels, m_chvec_max,
                            m_dist_coeff, m_color_coeff, m_stepx, m_stepy );


    sigmaX1 = fss.sigmaX1; sigmaX2 = fss.sigmaX2;
    sigmaY1 = fss.sigmaY1; sigmaY2 = fss.sigmaY2;
    for( int b = 0; b < m_nr_channels; b++ )
    {
      sigmaC1[b] = fss.sigmaC1[b];
      sigmaC2[b] = fss.sigmaC2[b];
    }

    // normalize sigmas
    sigmaY1 /= m_width*m_height;
    sigmaY2 /= m_width*m_height;
    sigmaX1 /= m_width*m_height;
    sigmaX2 /= m_width*m_height;
    for( int b = 0; b < m_nr_channels; b++ )
    {
      sigmaC1[b] /= m_width*m_height;
      sigmaC2[b] /= m_width*m_height;
    }

    // compute m_W normalization array
    m_W = Mat( m_height, m_width, CV_32F,Scalar(0) );
    parallel_for_( Range(0, m_width), FeatureSpaceWeights( m_chvec, &m_W,
                   sigmaX1, sigmaX2, sigmaY1, sigmaY2, sigmaC1, sigmaC2,
                   m_nr_channels, m_chvec_max, m_dist_coeff, m_color_coeff,
                   m_stepx, m_stepy ) );

}

struct FeatureSpaceCenters : ParallelLoopBody
{
    FeatureSpaceCenters( const vector< Mat_<double> >& _chvec, const Mat& _W,
                         const vector<float>& _kseedsx, const vector<float>& _kseedsy,
                         vector<float>* _centerX1, vector<float>* _centerX2,
                         vector<float>* _centerY1, vector<float>* _centerY2,
                         vector< vector<float> >* _centerC1, vector< vector<float> >* _centerC2,
                         const int _nr_channels, const float _chvec_max,
                         const float _dist_coeff, const float _color_coeff,
                         const int _stepx, const int _stepy )
    {
      W = _W;
      chvec = _chvec;
      stepx = _stepx;
      stepy = _stepy;
      chvec_max = _chvec_max;
      dist_coeff = _dist_coeff;
      nr_channels = _nr_channels;
      color_coeff = _color_coeff;

      PI2 = float(CV_PI / 2.0f);

      kseedsx = _kseedsx;
      kseedsy = _kseedsy;
      width  = chvec[0].cols;
      height = chvec[0].rows;

      centerX1 = _centerX1; centerX2 = _centerX2;
      centerY1 = _centerY1; centerY2 = _centerY2;
      centerC1 = _centerC1; centerC2 = _centerC2;
    }

    void operator()( const Range& range ) const
    {
      for( int i = range.start; i < range.end; i++ )
      {
        centerX1->at(i) = 0.0f; centerX2->at(i) = 0.0f;
        centerY1->at(i) = 0.0f; centerY2->at(i) = 0.0f;
        for( int b = 0; b < nr_channels; b++ )
        {
          centerC1->at(b)[i] = 0.0f; centerC2->at(b)[i] = 0.0f;
        }

        int X = (int)kseedsx[i]; int Y = (int)kseedsy[i];
        int minX = (X-stepx/4 <= 0) ? 0 : X-stepx/4;
        int minY = (Y-stepy/4 <= 0) ? 0 : Y-stepy/4;
        int maxX = (X+stepx/4 >= width -1) ? width -1 : X+stepx/4;
        int maxY = (Y+stepy/4 >= height-1) ? height-1 : Y+stepy/4;

        int count = 0;
        for( int x = minX; x <= maxX; x++ )
        {
          float thetaX = ( (float) x / (float) stepx ) * PI2;

          float tx1 = dist_coeff * cos(thetaX);
          float tx2 = dist_coeff * sin(thetaX);

          for( int y = minY; y <= maxY; y++ )
          {
            count++;
            float thetaY = ( (float) y / (float) stepy ) * PI2;

            // we do not store pre-computed x1, x2
            float x1 = tx1 / W.at<float>(y,x);
            float x2 = tx2 / W.at<float>(y,x);

            // we do not store pre-computed y1, y2
            float y1 = (dist_coeff * cos(thetaY)) / W.at<float>(y,x);
            float y2 = (dist_coeff * sin(thetaY)) / W.at<float>(y,x);

            centerX1->at(i) += x1; centerX2->at(i) += x2;
            centerY1->at(i) += y1; centerY2->at(i) += y2;

            for( int b = 0; b < nr_channels; b++ )
            {


              double thetaC =chvec[b](y,x)  * PI2;

              // we do not store pre-computed C1[b], C2[b]
              float C1 = (color_coeff * cos(thetaC) / nr_channels) / W.at<float>(y,x);
              float C2 = (color_coeff * sin(thetaC) / nr_channels) / W.at<float>(y,x);

              centerC1->at(b)[i] += C1; centerC2->at(b)[i] += C2;
            }
          }
        }
        // normalize
        centerX1->at(i) /= count; centerX2->at(i) /= count;
        centerY1->at(i) /= count; centerY2->at(i) /= count;
        for( int b = 0; b < nr_channels; b++ )
        {
          centerC1->at(b)[i] /= count; centerC2->at(b)[i] /= count;
        }
      }
    }

    Mat W;
    float PI2;
    int nr_channels;
    int stepx, stepy;
    int width, height;

    float chvec_max;
    float dist_coeff;
    float color_coeff;

    vector<Mat_<double> > chvec;
    vector<float> kseedsx, kseedsy;
    vector< vector<float> >* centerC1;
    vector< vector<float> >* centerC2;
    vector<float> *centerX1, *centerX2;
    vector<float> *centerY1, *centerY2;
};

struct FeatureSpaceKmeans : ParallelLoopBody
{
    FeatureSpaceKmeans( Mat* _klabels, Mat* _dist,
                        const vector< Mat_<double> >& _chvec, const Mat& _W,
                        const vector<float>& _kseedsx, const vector<float>& _kseedsy,
                        vector<float>& _centerX1, vector<float>& _centerX2,
                        vector<float>& _centerY1, vector<float>& _centerY2,
                        vector< vector<float> >& _centerC1, vector< vector<float> >& _centerC2,
                        const int _nr_channels, const float _chvec_max,
                        const float _dist_coeff, const float _color_coeff,
                        const int _stepx, const int _stepy )
    {
      W = _W;
      dist = _dist;
      chvec = _chvec;
      stepx = _stepx;
      stepy = _stepy;
      klabels = _klabels;
      chvec_max = _chvec_max;
      dist_coeff = _dist_coeff;
      nr_channels = _nr_channels;
      color_coeff = _color_coeff;

      PI2 = float(CV_PI / 2.0f);

      kseedsx = _kseedsx;
      kseedsy = _kseedsy;
      width  = chvec[0].cols;
      height = chvec[0].rows;

      centerX1 = _centerX1; centerX2 = _centerX2;
      centerY1 = _centerY1; centerY2 = _centerY2;
      centerC1 = _centerC1; centerC2 = _centerC2;
    }

    void operator()( const Range& range ) const
    {
      for( int i = range.start; i < range.end; i++ )
      {
        int X = (int)kseedsx[i]; int Y = (int)kseedsy[i];
        int minX = (X-(stepx) <= 0) ? 0 : X-stepx;
        int minY = (Y-(stepy) <= 0) ? 0 : Y-stepy;
        int maxX = (X+(stepx) >= width -1) ? width -1 : X+stepx;
        int maxY = (Y+(stepy) >= height-1) ? height-1 : Y+stepy;

        for( int x = minX; x <= maxX; x++ )
        {
          float thetaX = ( (float) x / (float) stepx ) * PI2;

          float tx1 = dist_coeff * cos(thetaX);
          float tx2 = dist_coeff * sin(thetaX);

          for( int y = minY; y <= maxY; y++ )
          {
            float thetaY = ( (float) y / (float) stepy ) * PI2;

            // we do not store pre-computed x1, x2
            float x1 = tx1 / W.at<float>(y,x);
            float x2 = tx2 / W.at<float>(y,x);
            // we do not store pre-computed y1, y2
            float y1 = (dist_coeff * cos(thetaY)) / W.at<float>(y,x);
            float y2 = (dist_coeff * sin(thetaY)) / W.at<float>(y,x);

            float diffx1 = x1 - centerX1[i]; float diffx2 = x2 - centerX2[i];
            float diffy1 = y1 - centerY1[i]; float diffy2 = y2 - centerY2[i];

            // compute distance given distance terms
            double D = (diffx1 * diffx1) + (diffx2 * diffx2)
                     + (diffy1 * diffy1) + (diffy2 * diffy2);

            // compute distance given channels terms
            for( int b = 0; b < nr_channels; b++ )
            {


               double  thetaC =  chvec[b](y,x)  * PI2;


              // we do not store pre-computed C1[b], C2[b]
              float C1 = (color_coeff * cos(thetaC) / nr_channels) / W.at<float>(y,x);
              float C2 = (color_coeff * sin(thetaC) / nr_channels) / W.at<float>(y,x);

              float diffC1 = C1 - centerC1[b][i]; float diffC2 = C2 - centerC2[b][i];

              D += (diffC1 * diffC1) + (diffC2 * diffC2);
            }

            // assign label if within D
            if ( D < dist->at<float>(y,x) )
            {
              dist->at<float>(y,x) = (float)D;
              klabels->at<int>(y,x) = i;
            }
          }
        }

      }
    }

    Mat W;
    float PI2;
    int nr_channels;
    int stepx, stepy;
    int width, height;

    float chvec_max;
    float dist_coeff;
    float color_coeff;

    Mat* dist;
    Mat* klabels;
    vector<Mat_<double> > chvec;
    vector<float> kseedsx, kseedsy;
    vector<float> centerX1, centerX2;
    vector<float> centerY1, centerY2;
    vector< vector<float> > centerC1;
    vector< vector<float> > centerC2;
};

struct FeatureCenterDists
{
    FeatureCenterDists( const vector< Mat_<double> >& _chvec, const Mat& _W, const Mat& _klabels,
                        const int _nr_channels, const float _chvec_max, const float _dist_coeff,
                        const float _color_coeff, const int _stepx, const int _stepy, const int _numlabels )
    {
      W = _W;
      chvec = _chvec;
      stepx = _stepx;
      stepy = _stepy;
      klabels = _klabels;
      numlabels = _numlabels;
      chvec_max = _chvec_max;
      dist_coeff = _dist_coeff;
      nr_channels = _nr_channels;
      color_coeff = _color_coeff;

      PI2 = float(CV_PI / 2.0f);

      Wsum.resize(numlabels);
      kseedsx.resize(numlabels);
      kseedsy.resize(numlabels);
      centerX1.resize(numlabels);
      centerX2.resize(numlabels);
      centerY1.resize(numlabels);
      centerY2.resize(numlabels);
      centerC1.resize(nr_channels);
      centerC2.resize(nr_channels);
      clusterSize.resize(numlabels);
      for( int b = 0; b < nr_channels; b++ )
      {
        centerC1[b].resize(numlabels);
        centerC2[b].resize(numlabels);
      }
      // refill with zero all arrays
      fill(centerX1.begin(), centerX1.end(), 0.0f);
      fill(centerX2.begin(), centerX2.end(), 0.0f);
      fill(centerY1.begin(), centerY1.end(), 0.0f);
      fill(centerY2.begin(), centerY2.end(), 0.0f);
      for( int b = 0; b < nr_channels; b++ )
      {
        fill(centerC1[b].begin(), centerC1[b].end(), 0.0f);
        fill(centerC2[b].begin(), centerC2[b].end(), 0.0f);
      }
      fill(Wsum.begin(), Wsum.end(), 0.0f);
      fill(kseedsx.begin(), kseedsx.end(), 0.0f);
      fill(kseedsy.begin(), kseedsy.end(), 0.0f);
      fill(clusterSize.begin(), clusterSize.end(), 0);
    }


    void renew()
    {
      // previous block state
      vector<float> tmp_Wsum = Wsum;
      vector<float> tmp_kseedsx = kseedsx;
      vector<float> tmp_kseedsy = kseedsy;
      vector<float> tmp_centerX1 = centerX1;
      vector<float> tmp_centerX2 = centerX2;
      vector<float> tmp_centerY1 = centerY1;
      vector<float> tmp_centerY2 = centerY2;
      vector< vector<float> > tmp_centerC1 = centerC1;
      vector< vector<float> > tmp_centerC2 = centerC2;
      vector<int> tmp_clusterSize = clusterSize;

      for ( int x = 0; x < chvec[0].cols; x++ )
      {

        float thetaX = ( (float) x / (float) stepx ) * PI2;

        // we do not store pre-computed x1, x2
        float x1 = (dist_coeff * cos(thetaX));
        float x2 = (dist_coeff * sin(thetaX));

        for( int y = 0; y < chvec[0].rows; y++ )
        {
          float thetaY = ( (float) y / (float) stepy ) * PI2;

          // we do not store pre-computed y1, y2
          float y1 = (dist_coeff * cos(thetaY));
          float y2 = (dist_coeff * sin(thetaY));

          int L = klabels.at<int>(y,x);

          tmp_centerX1[L] += x1; tmp_centerX2[L] += x2;
          tmp_centerY1[L] += y1; tmp_centerY2[L] += y2;

          // compute distance given channels terms
          for( int b = 0; b < nr_channels; b++ )
          {

             double thetaC =  chvec[b](y,x)* PI2;


            // we do not store pre-computed C1[b], C2[b]
            float C1 = (color_coeff * cos(thetaC) / nr_channels);
            float C2 = (color_coeff * sin(thetaC) / nr_channels);

            tmp_centerC1[b][L] += C1; tmp_centerC2[b][L] += C2;

          }
          tmp_clusterSize[L]++;
          tmp_Wsum[L] += W.at<float>(y,x);
          tmp_kseedsx[L] += x; tmp_kseedsy[L] += y;
        }
      }

      Wsum = tmp_Wsum;
      kseedsx = tmp_kseedsx;
      kseedsy = tmp_kseedsy;
      clusterSize = tmp_clusterSize;
      centerX1 = tmp_centerX1; centerX2 = tmp_centerX2;
      centerY1 = tmp_centerY1; centerY2 = tmp_centerY2;
      centerC1 = tmp_centerC1; centerC2 = tmp_centerC2;
    }

    void join( FeatureCenterDists& fcd )
    {
      for (int l = 0; l < numlabels; l++)
      {
        Wsum[l] += fcd.Wsum[l];
        kseedsx[l] += fcd.kseedsx[l];
        kseedsy[l] += fcd.kseedsy[l];
        centerX1[l] += fcd.centerX1[l];
        centerX2[l] += fcd.centerX2[l];
        centerY1[l] += fcd.centerY1[l];
        centerY2[l] += fcd.centerY2[l];
        clusterSize[l] += fcd.clusterSize[l];
        for( int b = 0; b < nr_channels; b++ )
        {
            centerC1[b][l] += fcd.centerC1[b][l];
            centerC2[b][l] += fcd.centerC2[b][l];
        }
      }
    }

    Mat W;
    float PI2;
    int numlabels;
    int nr_channels;
    int stepx, stepy;

    float chvec_max;
    float dist_coeff;
    float color_coeff;

    Mat klabels;
    vector<Mat_<double> > chvec;

    vector<float> Wsum;
    vector<int> clusterSize;
    vector<float> kseedsx, kseedsy;
    vector<float> centerX1, centerX2;
    vector<float> centerY1, centerY2;
    vector< vector<float> > centerC1, centerC2;

};

struct FeatureNormals : ParallelLoopBody
{
    FeatureNormals( const vector<float>& _Wsum, const vector<int>& _clusterSize,
                    vector<float>* _kseedsx, vector<float>* _kseedsy,
                    vector<float>* _centerX1, vector<float>* _centerX2,
                    vector<float>* _centerY1, vector<float>* _centerY2,
                    vector< vector<float> >* _centerC1, vector< vector<float> >* _centerC2,
                    const int _numlabels, const int _nr_channels )
    {
      Wsum = _Wsum;
      numlabels = _numlabels;
      clusterSize = _clusterSize;
      nr_channels = _nr_channels;

      kseedsx = _kseedsx; kseedsy = _kseedsy;
      centerX1 = _centerX1; centerX2 = _centerX2;
      centerY1 = _centerY1; centerY2 = _centerY2;
      centerC1 = _centerC1; centerC2 = _centerC2;
    }

    void operator()( const Range& range ) const
    {
      for( int i = range.start; i < range.end; i++ )
      {
        if ( Wsum[i] != 0 )
        {
          centerX1->at(i) /= Wsum[i]; centerX2->at(i) /= Wsum[i];
          centerY1->at(i) /= Wsum[i]; centerY2->at(i) /= Wsum[i];
          for( int b = 0; b < nr_channels; b++ )
          {
            centerC1->at(b)[i] /= Wsum[i]; centerC2->at(b)[i] /= Wsum[i];
          }
        }
        if ( clusterSize[i] != 0 )
        {
          kseedsx->at(i) /= clusterSize[i];
          kseedsy->at(i) /= clusterSize[i];
        }
      }
    }

    int numlabels;
    vector<float> Wsum;
    vector<int> clusterSize;
    int nr_channels;

    vector<float> *kseedsx, *kseedsy;
    vector<float> *centerX1, *centerX2;
    vector<float> *centerY1, *centerY2;
    vector< vector<float> > *centerC1;
    vector< vector<float> > *centerC2;
};


/*
 *    PerformSuperpixelLSC
 *
 *    Performs weighted kmeans segmentation
 *    in (4 + 2*m_nr_channels) dimensional space
 *
 */
inline void HSI_LSC::PerformLSC( const int&  itrnum )
{
    // allocate initial workspaces
    cv::Mat dist( m_height, m_width, CV_32F );
    //cout<<"1595s "<<endl;
    vector<float> centerX1( m_numlabels );
    vector<float> centerX2( m_numlabels );
    vector<float> centerY1( m_numlabels );
    vector<float> centerY2( m_numlabels );
    vector< vector<float> > centerC1( m_nr_channels );
    vector< vector<float> > centerC2( m_nr_channels );
    for( int b = 0; b < m_nr_channels; b++ )
    {
      centerC1[b].resize( m_numlabels );
      centerC2[b].resize( m_numlabels );
    }
    vector<float> Wsum( m_numlabels );
    vector<int> clusterSize( m_numlabels );

    // compute weighted distance centers
    parallel_for_( Range(0, m_numlabels), FeatureSpaceCenters(
                   m_chvec, m_W, m_kseedsx, m_kseedsy,
                   &centerX1, &centerX2, &centerY1, &centerY2,
                   &centerC1, &centerC2, m_nr_channels, m_chvec_max,
                   m_dist_coeff, m_color_coeff, m_stepx, m_stepy ) );

    // parallel reduce structure
    FeatureCenterDists fcd( m_chvec, m_W, m_klabels, m_nr_channels, m_chvec_max,
                            m_dist_coeff, m_color_coeff, m_stepx, m_stepy, m_numlabels );

    // K-Means
    for( int itr = 0; itr < itrnum; itr++ )
    {

      dist.setTo( FLT_MAX );

      // k-mean
      parallel_for_( Range(0, m_numlabels), FeatureSpaceKmeans(
                     &m_klabels, &dist, m_chvec, m_W, m_kseedsx, m_kseedsy,
                     centerX1, centerX2, centerY1, centerY2, centerC1, centerC2,
                     m_nr_channels, m_chvec_max, m_dist_coeff, m_color_coeff,
                     m_stepx, m_stepy ) );

      // accumulate center distances
      fcd.renew();

      // featch out the results
      Wsum = fcd.Wsum; clusterSize = fcd.clusterSize;
      m_kseedsx = fcd.kseedsx; m_kseedsy = fcd.kseedsy;
      centerX1 = fcd.centerX1; centerX2 = fcd.centerX2;
      centerY1 = fcd.centerY1; centerY2 = fcd.centerY2;
      centerC1 = fcd.centerC1; centerC2 = fcd.centerC2;


      // normalize accumulated distances
      parallel_for_( Range(0, m_numlabels), FeatureNormals(
                     Wsum, clusterSize, &m_kseedsx, &m_kseedsy,
                     &centerX1, &centerX2, &centerY1, &centerY2,
                     &centerC1, &centerC2, m_numlabels, m_nr_channels ) );
    }


    float PI2=float(CV_PI/2.0f);
    float W=0;
    for(int x=0;x<m_width;x++)
    {
        float thetaX=((float)x/(float)m_stepx)*PI2;
        float thx1=cos(thetaX)*m_dist_coeff;
        float thx2=sin(thetaX)*m_dist_coeff;

        for(int y=0;y<m_height;y++)
        {
            W+=m_W.at<float>(y,x);
            int i=m_klabels.at<int>(y,x);

            float x1=thx1/m_W.at<float>(y,x);
            float x2=thx2/m_W.at<float>(y,x);

            float thetaY=((float)y/(float)m_stepy)*PI2;
            float thy1=cos(thetaY)*m_dist_coeff;
            float thy2=sin(thetaY)*m_dist_coeff;
            float y1=thy1/m_W.at<float>(y,x);
            float y2=thy2/m_W.at<float>(y,x);

            float diffx1=x1-centerX1[i];
            float diffx2=x2-centerX2[i];
            float diffy1=y1-centerY1[i];
            float diffy2=y2-centerY2[i];

        }
    }

}

}//namespace segmentation
}//namespace PPP


#endif // HSI_FH_H
