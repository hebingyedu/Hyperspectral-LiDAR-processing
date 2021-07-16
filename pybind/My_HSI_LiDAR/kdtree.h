#ifndef KDTREE_H
#define KDTREE_H
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
#include <flann/flann.hpp>

using namespace std;


namespace PPP{
namespace geometry {

class KDTreeSearchParam {
public:
    enum class SearchType {
        Knn = 0,
        Radius = 1,
        Hybrid = 2,
    };

public:
    virtual ~KDTreeSearchParam() {}

protected:
    KDTreeSearchParam(SearchType type) : search_type_(type) {}

public:
    SearchType GetSearchType() const { return search_type_; }

private:
    SearchType search_type_;
};

class KDTreeSearchParamKNN : public KDTreeSearchParam {
public:
    KDTreeSearchParamKNN(int knn = 30)
        : KDTreeSearchParam(SearchType::Knn), knn_(knn) {}

public:
    int knn_;
};

class KDTreeSearchParamRadius : public KDTreeSearchParam {
public:
    KDTreeSearchParamRadius(double radius)
        : KDTreeSearchParam(SearchType::Radius), radius_(radius) {}

public:
    double radius_;
};

class KDTreeSearchParamHybrid : public KDTreeSearchParam {
public:
    KDTreeSearchParamHybrid(double radius, int max_nn)
        : KDTreeSearchParam(SearchType::Hybrid),
          radius_(radius),
          max_nn_(max_nn) {}

public:
    double radius_;
    int max_nn_;
};

}  // namespace geometry
} //namespace PPP

namespace flann {
template <typename T>
class Matrix;
template <typename T>
struct L2;
template <typename T>
class Index;
}  // namespace flann

namespace PPP{
namespace geometry {

class KDTreeFlann {
public:
    KDTreeFlann(){}
    KDTreeFlann( const std::vector<Eigen::Vector3d> &points){setCloud (points);}
    ~KDTreeFlann(){}
    KDTreeFlann( KDTreeFlann &) = delete;
    KDTreeFlann &operator=(const KDTreeFlann &) = delete;

public:
    void setCloud(const std::vector<Eigen::Vector3d> &points) {
        SetRawData(Eigen::Map<const Eigen::MatrixXd>(
                (const double *)points.data(),
                3, points.size()));
    }


    template <typename T>
    int Search(const T &query,
               const KDTreeSearchParam &param,
               std::vector<int> &indices,
               std::vector<double> &distance2) const{
        switch (param.GetSearchType()) {
            case KDTreeSearchParam::SearchType::Knn:
                return SearchKNN(query, ((const KDTreeSearchParamKNN &)param).knn_,
                                 indices, distance2);
            case KDTreeSearchParam::SearchType::Radius:
                return SearchRadius(
                        query, ((const KDTreeSearchParamRadius &)param).radius_,
                        indices, distance2);
            case KDTreeSearchParam::SearchType::Hybrid:
                return SearchHybrid(
                        query, ((const KDTreeSearchParamHybrid &)param).radius_,
                        ((const KDTreeSearchParamHybrid &)param).max_nn_, indices,
                        distance2);
            default:
                return -1;
        }
        return -1;
    }

    template <typename T>
    int SearchKNN(const T &query,
                  int knn,
                  std::vector<int> &indices,
                  std::vector<double> &distance2) const{
        // This is optimized code for heavily repeated search.
        // Other flann::Index::knnSearch() implementations lose performance due to
        // memory allocation/deallocation.
        if (data_.empty() || dataset_size_ <= 0 ||
            size_t(query.rows()) != dimension_ || knn < 0) {
            return -1;
        }
        flann::Matrix<double> query_flann((double *)query.data(), 1, dimension_);
        indices.resize(knn);
        distance2.resize(knn);
        flann::Matrix<int> indices_flann(indices.data(), query_flann.rows, knn);
        flann::Matrix<double> dists_flann(distance2.data(), query_flann.rows, knn);
        int k = flann_index_->knnSearch(query_flann, indices_flann, dists_flann,
                                        knn, flann::SearchParams(-1, 0.0));
        indices.resize(k);
        distance2.resize(k);
        return k;
    }

    template <typename T>
    int SearchRadius(const T &query,
                     double radius,
                     std::vector<int> &indices,
                     std::vector<double> &distance2) const{
        // This is optimized code for heavily repeated search.
        // Since max_nn is not given, we let flann to do its own memory management.
        // Other flann::Index::radiusSearch() implementations lose performance due
        // to memory management and CPU caching.
        if (data_.empty() || dataset_size_ <= 0 ||
            size_t(query.rows()) != dimension_) {
            return -1;
        }
        flann::Matrix<double> query_flann((double *)query.data(), 1, dimension_);
        flann::SearchParams param(-1, 0.0);
        param.max_neighbors = -1;
        std::vector<std::vector<int>> indices_vec(1);
        std::vector<std::vector<double>> dists_vec(1);
        int k = flann_index_->radiusSearch(query_flann, indices_vec, dists_vec,
                                           float(radius * radius), param);
        indices = indices_vec[0];
        distance2 = dists_vec[0];
        return k;
    }

    template <typename T>
    int SearchHybrid(const T &query,
                     double radius,
                     int max_nn,
                     std::vector<int> &indices,
                     std::vector<double> &distance2) const{
        // This is optimized code for heavily repeated search.
        // It is also the recommended setting for search.
        // Other flann::Index::radiusSearch() implementations lose performance due
        // to memory allocation/deallocation.
        if (data_.empty() || dataset_size_ <= 0 ||
            size_t(query.rows()) != dimension_ || max_nn < 0) {
            return -1;
        }
        flann::Matrix<double> query_flann((double *)query.data(), 1, dimension_);
        flann::SearchParams param(-1, 0.0);
        param.max_neighbors = max_nn;
        indices.resize(max_nn);
        distance2.resize(max_nn);
        flann::Matrix<int> indices_flann(indices.data(), query_flann.rows, max_nn);
        flann::Matrix<double> dists_flann(distance2.data(), query_flann.rows,
                                          max_nn);
        int k = flann_index_->radiusSearch(query_flann, indices_flann, dists_flann,
                                           float(radius * radius), param);
        indices.resize(k);
        distance2.resize(k);
        return k;
    }


    void SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data){
        dimension_ = data.rows();
        dataset_size_ = data.cols();
        if (dimension_ == 0 || dataset_size_ == 0) {
            assert("[KDTreeFlann::SetRawData] Failed due to no data.");
        }
        data_.resize(dataset_size_ * dimension_);
        memcpy(data_.data(), data.data(),
               dataset_size_ * dimension_ * sizeof(double));
        flann_dataset_.reset(new flann::Matrix<double>((double *)data_.data(),
                                                       dataset_size_, dimension_));
        flann_index_.reset(new flann::Index<flann::L2<double>>(
                *flann_dataset_, flann::KDTreeSingleIndexParams(15)));
        flann_index_->buildIndex();
    }

public:
    std::vector<double> data_;
    std::unique_ptr<flann::Matrix<double>> flann_dataset_;
    std::unique_ptr<flann::Index<flann::L2<double>>> flann_index_;
    size_t dimension_ = 0;
    size_t dataset_size_ = 0;
};

}  // namespace geometry
} // namespace PPP

#endif //KDTrEEE
