#include <iostream>

#include <pybind11/detail/internals.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>  // Include first to suppress compiler warnings
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/detail/internals.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>  // Include first to suppress compiler warnings
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "HSI_FH.h"
#include "HSI_LSC.h"
#include "ndarray_converter.h"

using namespace std;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3i>);

// some helper functions
namespace pybind11 {
namespace detail {

template <typename T, typename Class_>
void bind_default_constructor(Class_ &cl) {
    cl.def(py::init([]() { return new T(); }), "Default constructor");
}

template <typename T, typename Class_>
void bind_copy_functions(Class_ &cl) {
    cl.def(py::init([](const T &cp) { return new T(cp); }), "Copy constructor");
    cl.def("__copy__", [](T &v) { return T(v); });
    cl.def("__deepcopy__", [](T &v, py::dict &memo) { return T(v); });
}

}  // namespace detail
}  // namespace pybind11


//TBBSupervoxel=======================================================
namespace PPP{
namespace HSI_supixel  {

void pybind_FH_supixel(py::module &m) {
    py::class_<HSI_FH>
            HSI_FH(m, "HSI_FH",
                       "HSI_FH class. 超像素分割 "
                       );
    py::detail::bind_default_constructor<HSI_supixel::HSI_FH>(HSI_FH);
    py::detail::bind_copy_functions<HSI_supixel::HSI_FH>(HSI_FH);
    HSI_FH
            .def(py::init<> ())
            .def("__repr__",
                 [](const HSI_supixel::HSI_FH &FHS) {
                     return std::string("一些函数 ") +
                             "example1:\n" ;
                 })
            .def("setHSI",
                 &HSI_FH::setHSI,
                 "input the point cloud and init",
                 "HSI_"_a)
            .def("graph_seg",
                 &HSI_FH::graph_seg,
                 "graph_seg",
                 "ratio"_a, "min_size"_a)
            .def("compute_edge",
                 &HSI_FH::compute_edge,
                 "compute_edge")
            .def("Generate_supmask",
                 &HSI_FH::Generate_supmask,
                 "Generate_supmask",
                 "is_thick_line"_a)
            .def("GetSuppixelLabel",
                 &HSI_FH::GetSuppixelLabel,
                 "GetSuppixelLabel");
}

}
}
//TBBSupervoxel=======================================================

//LSCSupervoxel=======================================================
namespace PPP{
namespace HSI_supixel  {

void pybind_LSC_supixel(py::module &m) {
    py::class_<HSI_LSC>
            HSI_LSC(m, "HSI_LSC",
                       "HSI_LSC class. 超像素分割 "
                       );
    py::detail::bind_default_constructor<HSI_supixel::HSI_LSC>(HSI_LSC);
    py::detail::bind_copy_functions<HSI_supixel::HSI_LSC>(HSI_LSC);
    HSI_LSC
            .def(py::init<> ())
            .def("__repr__",
                 [](const HSI_supixel::HSI_LSC &LSC) {
                     return std::string("一些函数 ") +
                             "example1:\n" ;
                 })
            .def("setHSI",
                 &HSI_LSC::setHSI,
                 "input the point cloud and init",
                 "HSI_"_a)
            .def("setParameter",
                 &HSI_LSC::setParameter,
                 "setParameter",
                 "rho"_a, "region_size"_a, "ratio"_a)
            .def("iterate_Sup",
                 &HSI_LSC::iterate_Sup,
                 "iterate_Sup",
                 "num_iter"_a)
            .def("enforceLabelConnectivity_Sup",
                 &HSI_LSC::enforceLabelConnectivity_Sup,
                 "enforceLabelConnectivity_Sup",
                 "min_size"_a)
            .def("getLabelContourMask_Sup",
                 &HSI_LSC::getLabelContourMask_Sup,
                 "getLabelContourMask_Sup",
                 "is_thick_line"_a)
            .def("GetSuppixelLabel",
                 &HSI_LSC::GetSuppixelLabel,
                 "GetSuppixelLabel");
}

}
}
//LSCSupervoxel=======================================================

//segmentation=======================================================
void pybind_segmentation(py::module &m) {
    py::module m_submodule = m.def_submodule("segmentation");
    PPP::HSI_supixel::pybind_FH_supixel(m_submodule);
    PPP::HSI_supixel::pybind_LSC_supixel(m_submodule);

}
//segmantation=======================================================

PYBIND11_MODULE(My_suppixel, m) {
    NDArrayConverter::init_numpy();
    m.doc() = "Python binding of Open3D";

    // Register this first, other submodule (e.g. odometry) might depend on this
    pybind_segmentation(m);

}
