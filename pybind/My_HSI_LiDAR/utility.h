#ifndef BASE_H
#define BASE_H

#include <pybind11/detail/internals.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>  // Include first to suppress compiler warnings
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <iostream>
#include <memory>
#include <thread>
#include <cstdio>
#include <vector>
#include <queue>
#include <Eigen/Dense>

namespace py = pybind11;

namespace pybind11 {

template <typename Vector,
          typename holder_type = std::unique_ptr<Vector>,
          typename... Args>
py::class_<Vector, holder_type> my_bind_vector_without_repr(
        py::module &m, std::string const &name, Args &&... args) {
    // hack function to disable __repr__ for the convenient function
    // bind_vector()
    using Class_ = py::class_<Vector, holder_type>;
    Class_ cl(m, name.c_str(), std::forward<Args>(args)...);
    cl.def(py::init<>());
    cl.def("__bool__", [](const Vector &v) -> bool { return !v.empty(); },
           "Check whether the list is nonempty");
    cl.def("__len__", &Vector::size);
    return cl;
}

// - This function is used by Pybind for std::vector<SomeEigenType> constructor.
//   This optional constructor is added to avoid too many Python <-> C++ API
//   calls when the vector size is large using the default biding method.
//   Pybind matches np.float64 array to py::array_t<double> buffer.
// - Directly using templates for the py::array_t<double> and py::array_t<int>
//   and etc. doesn't work. The current solution is to explicitly implement
//   bindings for each py array types.
template <typename EigenVector>
std::vector<EigenVector> my_py_array_to_vectors_double(
        py::array_t<double, py::array::c_style | py::array::forcecast> array) {
    size_t eigen_vector_size = EigenVector::SizeAtCompileTime;
    if (array.ndim() != 2 || array.shape(1) != eigen_vector_size) {
        throw py::cast_error();
    }
    std::vector<EigenVector> eigen_vectors(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        // The EigenVector here must be a double-typed eigen vector, since only
        // open3d::Vector3dVector binds to py_array_to_vectors_double.
        // Therefore, we can use the memory map directly.
        eigen_vectors[i] = Eigen::Map<EigenVector>(&array_unchecked(i, 0));
    }
    return eigen_vectors;
}

template <typename EigenVector>
std::vector<EigenVector> my_py_array_to_vectors_int(
        py::array_t<int, py::array::c_style | py::array::forcecast> array) {
    size_t eigen_vector_size = EigenVector::SizeAtCompileTime;
    if (array.ndim() != 2 || array.shape(1) != eigen_vector_size) {
        throw py::cast_error();
    }
    std::vector<EigenVector> eigen_vectors(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        eigen_vectors[i] = Eigen::Map<EigenVector>(&array_unchecked(i, 0));
    }
    return eigen_vectors;
}

template <typename EigenVector,
          typename EigenAllocator = Eigen::aligned_allocator<EigenVector>>
std::vector<EigenVector, EigenAllocator>
my_py_array_to_vectors_int_eigen_allocator(
        py::array_t<int, py::array::c_style | py::array::forcecast> array) {
    size_t eigen_vector_size = EigenVector::SizeAtCompileTime;
    if (array.ndim() != 2 || array.shape(1) != eigen_vector_size) {
        throw py::cast_error();
    }
    std::vector<EigenVector, EigenAllocator> eigen_vectors(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        eigen_vectors[i] = Eigen::Map<EigenVector>(&array_unchecked(i, 0));
    }
    return eigen_vectors;
}

template <typename EigenVector,
          typename EigenAllocator = Eigen::aligned_allocator<EigenVector>>
std::vector<EigenVector, EigenAllocator>
my_py_array_to_vectors_int64_eigen_allocator(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> array) {
    size_t eigen_vector_size = EigenVector::SizeAtCompileTime;
    if (array.ndim() != 2 || array.shape(1) != eigen_vector_size) {
        throw py::cast_error();
    }
    std::vector<EigenVector, EigenAllocator> eigen_vectors(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        eigen_vectors[i] = Eigen::Map<EigenVector>(&array_unchecked(i, 0));
    }
    return eigen_vectors;
}

}  // namespace pybind11

namespace  {
template <typename EigenVector,
          typename Vector = std::vector<EigenVector>,
          typename holder_type = std::unique_ptr<Vector>,
          typename InitFunc>
py::class_<Vector, holder_type> my_pybind_eigen_vector_of_vector(
        py::module &m,
        const std::string &bind_name,
        const std::string &repr_name,
        InitFunc init_func) {
    typedef typename EigenVector::Scalar Scalar;
    auto vec1 = py::my_bind_vector_without_repr<std::vector<EigenVector>>(
            m, bind_name, py::buffer_protocol());
    vec1.def(py::init(init_func));
    vec1.def_buffer([](std::vector<EigenVector> &v) -> py::buffer_info {
        size_t rows = EigenVector::RowsAtCompileTime;
        return py::buffer_info(v.data(), sizeof(Scalar),
                               py::format_descriptor<Scalar>::format(), 2,
                               {v.size(), rows},
                               {sizeof(EigenVector), sizeof(Scalar)});
    });
    vec1.def("__repr__", [repr_name](const std::vector<EigenVector> &v) {
        return repr_name + std::string(" with ") + std::to_string(v.size()) +
               std::string(" elements.\n") +
               std::string("Use numpy.asarray() to access data.");
    });
    vec1.def("__copy__", [](std::vector<EigenVector> &v) {
        return std::vector<EigenVector>(v);
    });
    vec1.def("__deepcopy__", [](std::vector<EigenVector> &v, py::dict &memo) {
        return std::vector<EigenVector>(v);
    });

    // py::detail must be after custom constructor
    using Class_ = py::class_<Vector, std::unique_ptr<Vector>>;
    py::detail::vector_if_copy_constructible<Vector, Class_>(vec1);
    py::detail::vector_if_equal_operator<Vector, Class_>(vec1);
    py::detail::vector_modifiers<Vector, Class_>(vec1);
    py::detail::vector_accessor<Vector, Class_>(vec1);

    return vec1;
}

}

namespace PPP{
namespace geometry{
//模仿python numpy unique函数=================================================
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


// 计算点云边框===================================================================
struct BoundingBox {
    BoundingBox() {}

    BoundingBox(std::vector<Eigen::Vector3d> points_) {
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
// 计算点云边框===================================================================

// 计算点云中心===================================================================
template <typename Iterator>
Eigen::Vector3d Centroid3D(Iterator first, Iterator last,
                    const std::vector<double>& weights) {

    double sum = 0.0;
    Eigen::Vector3d centroid(0.0, 0.0, 0.0);
    int i = 0;
    for (Iterator p = first; p != last; ++p, ++i) {
        double w = weights[i];
        assert(w >= 0.0);

        centroid(0) += w * (*p)(0);
        centroid(1) += w * (*p)(1);
        centroid(2) += w * (*p)(2);
        sum += w;
    }
    assert(sum != 0.0);

    sum = 1.0 / sum;
    centroid(0) *= sum;
    centroid(1) *= sum;
    centroid(2) *= sum;

    return centroid;
}
// 计算点云中心===================================================================

// 计算点云邻域法线===================================================================
template <typename Iterator>
void PCAEstimateNormal(Iterator first, Iterator last,
                       const std::vector<double>& weights, Eigen::Vector4d& normal) {
    assert(first != last);

    Eigen::Vector3d centroid = Centroid3D(first, last, weights);
    double a00 = 0.0, a01 = 0.0, a02 = 0.0, a11 = 0.0, a12 = 0.0, a22 = 0.0;
    int i = 0;
    double sum = 0.0;
    for (Iterator p = first; p != last; ++p, ++i) {
        double x = (*p)(0) - centroid(0);
        double y = (*p)(1) - centroid(1);
        double z = (*p)(2) - centroid(2);
        double w = weights[i];

        a00 += w * x * x;
        a01 += w * x * y;
        a02 += w * x * z;
        a11 += w * y * y;
        a12 += w * y * z;
        a22 += w * z * z;

        sum += w;
    }

    double t = 1.0 / sum;
    a00 = a00 * t;
    a01 = a01 * t;
    a02 = a02 * t;
    a11 = a11 * t;
    a12 = a12 * t;
    a22 = a22 * t;
//    std::cout<<"A: "<<a00<<" "<<a01<<" "<<a02<<" "<<a11<<" "<<a12<<" "<<a22<<std::endl;

    // Computing the least eigenvalue of the covariance matrix.
    double m = (a00 + a11 + a22) / 3.0;
    double p = (a00 - m) * (a11 - m) + (a00 - m) * (a22 - m) + (a11 - m) * (a22 - m) -
                    (a12 * a12 + a01 * a01 + a02 * a02);
    double q = a12 * a12 * (a00 - m) + a01 * a01 * (a22 - m) + a02 * a02 * (a11 - m) -
                    (a00 - m) * (a11 - m) * (a22 - m) - 2 * a01 * a02 * a12;

    double r = sqrt(- p / 3.);
    double mr = pow(r, 3);
    double phi = - q / mr / 2.;
    phi = acos(phi) / 3.;
    // 有 r>0,  0<phi<pi/3, eig1 > eig3> eig2, 而法线对应最小特征值点特征向量
    double eig1 = 2 * r * cos(phi  ) + m;
    double eig2 = 2 * r * cos(phi + M_PI * (2.0 / 3.0) ) + m;
    double eig3 = 2 * r * cos(phi - M_PI * (2.0 / 3.0) ) + m;

    // Computing the corresponding eigenvector.
    double x = (a11 - eig2) * (a22 - eig2) - a12 * a12;
    double y = a02 * a12 - a01 * (a22 - eig2);
    double z = a01 * a12 - a02 * (a11 - eig2);

    double norm = x * x + y * y + z * z;
    norm = sqrt(norm);
    x = x / norm;
    y = y / norm;
    z = z / norm;


    normal(0) = x;
    normal(1) = y;
    normal(2) = z;
    normal(3) = eig2 / (eig1 + eig2 + eig3);
}
// 计算点云邻域法线===================================================================


//open3d 计算法线========================================================================
Eigen::Vector3d ComputeEigenvector0(const Eigen::Matrix3d &A, double eval0) {
    Eigen::Vector3d row0(A(0, 0) - eval0, A(0, 1), A(0, 2));
    Eigen::Vector3d row1(A(0, 1), A(1, 1) - eval0, A(1, 2));
    Eigen::Vector3d row2(A(0, 2), A(1, 2), A(2, 2) - eval0);
    Eigen::Vector3d r0xr1 = row0.cross(row1);
    Eigen::Vector3d r0xr2 = row0.cross(row2);
    Eigen::Vector3d r1xr2 = row1.cross(row2);
    double d0 = r0xr1.dot(r0xr1);
    double d1 = r0xr2.dot(r0xr2);
    double d2 = r1xr2.dot(r1xr2);

    double dmax = d0;
    int imax = 0;
    if (d1 > dmax) {
        dmax = d1;
        imax = 1;
    }
    if (d2 > dmax) {
        imax = 2;
    }

    if (imax == 0) {
        return r0xr1 / std::sqrt(d0);
    } else if (imax == 1) {
        return r0xr2 / std::sqrt(d1);
    } else {
        return r1xr2 / std::sqrt(d2);
    }
}

Eigen::Vector3d ComputeEigenvector1(const Eigen::Matrix3d &A,
                                    const Eigen::Vector3d &evec0,
                                    double eval1) {
    Eigen::Vector3d U, V;
    if (std::abs(evec0(0)) > std::abs(evec0(1))) {
        double inv_length =
                1 / std::sqrt(evec0(0) * evec0(0) + evec0(2) * evec0(2));
        U << -evec0(2) * inv_length, 0, evec0(0) * inv_length;
    } else {
        double inv_length =
                1 / std::sqrt(evec0(1) * evec0(1) + evec0(2) * evec0(2));
        U << 0, evec0(2) * inv_length, -evec0(1) * inv_length;
    }
    V = evec0.cross(U);

    Eigen::Vector3d AU(A(0, 0) * U(0) + A(0, 1) * U(1) + A(0, 2) * U(2),
                       A(0, 1) * U(0) + A(1, 1) * U(1) + A(1, 2) * U(2),
                       A(0, 2) * U(0) + A(1, 2) * U(1) + A(2, 2) * U(2));

    Eigen::Vector3d AV = {A(0, 0) * V(0) + A(0, 1) * V(1) + A(0, 2) * V(2),
                          A(0, 1) * V(0) + A(1, 1) * V(1) + A(1, 2) * V(2),
                          A(0, 2) * V(0) + A(1, 2) * V(1) + A(2, 2) * V(2)};

    double m00 = U(0) * AU(0) + U(1) * AU(1) + U(2) * AU(2) - eval1;
    double m01 = U(0) * AV(0) + U(1) * AV(1) + U(2) * AV(2);
    double m11 = V(0) * AV(0) + V(1) * AV(1) + V(2) * AV(2) - eval1;

    double absM00 = std::abs(m00);
    double absM01 = std::abs(m01);
    double absM11 = std::abs(m11);
    double max_abs_comp;
    if (absM00 >= absM11) {
        max_abs_comp = std::max(absM00, absM01);
        if (max_abs_comp > 0) {
            if (absM00 >= absM01) {
                m01 /= m00;
                m00 = 1 / std::sqrt(1 + m01 * m01);
                m01 *= m00;
            } else {
                m00 /= m01;
                m01 = 1 / std::sqrt(1 + m00 * m00);
                m00 *= m01;
            }
            return m01 * U - m00 * V;
        } else {
            return U;
        }
    } else {
        max_abs_comp = std::max(absM11, absM01);
        if (max_abs_comp > 0) {
            if (absM11 >= absM01) {
                m01 /= m11;
                m11 = 1 / std::sqrt(1 + m01 * m01);
                m01 *= m11;
            } else {
                m11 /= m01;
                m01 = 1 / std::sqrt(1 + m11 * m11);
                m11 *= m01;
            }
            return m11 * U - m01 * V;
        } else {
            return U;
        }
    }
}

Eigen::Vector4d FastEigen3x3(Eigen::Matrix3d &A) {
    // Previous version based on:
    // https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
    // Current version based on
    // https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    // which handles edge cases like points on a plane

    double max_coeff = A.maxCoeff();
    if (max_coeff == 0) {
        return Eigen::Vector4d::Zero();
    }
    A /= max_coeff;

    double norm = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
    if (norm > 0) {
        Eigen::Vector3d eval;
        Eigen::Vector3d evec0;
        Eigen::Vector3d evec1;
        Eigen::Vector3d evec2;

        double q = (A(0, 0) + A(1, 1) + A(2, 2)) / 3;

        double b00 = A(0, 0) - q;
        double b11 = A(1, 1) - q;
        double b22 = A(2, 2) - q;

        double p =
                std::sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);

        double c00 = b11 * b22 - A(1, 2) * A(1, 2);
        double c01 = A(0, 1) * b22 - A(1, 2) * A(0, 2);
        double c02 = A(0, 1) * A(1, 2) - b11 * A(0, 2);
        double det = (b00 * c00 - A(0, 1) * c01 + A(0, 2) * c02) / (p * p * p);

        double half_det = det * 0.5;
        half_det = std::min(std::max(half_det, -1.0), 1.0);

        double angle = std::acos(half_det) / (double)3;
        double const two_thirds_pi = 2.09439510239319549;
        double beta2 = std::cos(angle) * 2;
        double beta0 = std::cos(angle + two_thirds_pi) * 2;
        double beta1 = -(beta0 + beta2);

        eval(0) = q + p * beta0;
        eval(1) = q + p * beta1;
        eval(2) = q + p * beta2;

        Eigen::Vector3d normal;

        if (half_det >= 0) {
            evec2 = ComputeEigenvector0(A, eval(2));
            if (eval(2) < eval(0) && eval(2) < eval(1)) {
                A *= max_coeff;
                normal = evec2;
            }
            evec1 = ComputeEigenvector1(A, evec2, eval(1));
            A *= max_coeff;
            if (eval(1) < eval(0) && eval(1) < eval(2)) {
                normal =  evec1;
            }
            evec0 = evec1.cross(evec2);
            normal =  evec0;
        } else {
            evec0 = ComputeEigenvector0(A, eval(0));
            if (eval(0) < eval(1) && eval(0) < eval(2)) {
                A *= max_coeff;
                normal = evec0;
            }
            evec1 = ComputeEigenvector1(A, evec0, eval(1));
            A *= max_coeff;
            if (eval(1) < eval(0) && eval(1) < eval(2)) {
                normal = evec1;
            }
            evec2 = evec0.cross(evec1);
            normal = evec2;
        }

        double min_eval = eval.minCoeff();
        double curvature = min_eval/(eval.sum() );
        return Eigen::Vector4d(normal(0), normal(1), normal(2), curvature);

    } else {
        A *= max_coeff;
        if (A(0, 0) < A(1, 1) && A(0, 0) < A(2, 2)) {
            return Eigen::Vector4d(1, 0, 0, 0);
        } else if (A(1, 1) < A(0, 0) && A(1, 1) < A(2, 2)) {
            return Eigen::Vector4d(0, 1, 0, 0);
        } else {
            return Eigen::Vector4d(0, 0, 1, 0);
        }
    }
}
//open3d 计算法线========================================================================




}
}

#endif // BASE_H
