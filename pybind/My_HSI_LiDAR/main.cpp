#include <iostream>

#include <pybind11/detail/internals.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>  // Include first to suppress compiler warnings
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <Python.h>

#include "utility.h"
#include "PointCloud.h"
#include "CloudToImg.h"
#include "ndarray_converter.h"
#include "img_utility.h"
#include "TBBSupervoxel.h"
#include "regionGrow.h"
#include "SupervoxMerging.h"
#include "supervoxel_structure.h"
#include "FH_supervoxel.h"

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

//utility==========================================================

void pybind_eigen(py::module &m) {

    auto MyVector3dVector1 = my_pybind_eigen_vector_of_vector<Eigen::Vector3d>(
            m, "Vector3dVector", "std::vector<Eigen::Vector3d>",
            py::my_py_array_to_vectors_double<Eigen::Vector3d>);

    auto MyVector3iVector = my_pybind_eigen_vector_of_vector<Eigen::Vector3i>(
            m, "Vector3iVector", "std::vector<Eigen::Vector3i>",
            py::my_py_array_to_vectors_int<Eigen::Vector3i>);
}

void pybind_utility(py::module &m) {
    py::module m_submodule = m.def_submodule("utility");
    pybind_eigen(m_submodule);
}
//utility==========================================================

//PointCloud=======================================================
namespace PPP{
namespace geometry {

void pybind_pointcloud(py::module &m) {
    py::class_<PointCloud>
            pointcloud(m, "PointCloud",
                       "PointCloud class. A point cloud consists of point "
                       "coordinates, and optionally point colors and point "
                       "normals.");
    py::detail::bind_default_constructor<geometry::PointCloud>(pointcloud);
    py::detail::bind_copy_functions<geometry::PointCloud>(pointcloud);
    pointcloud
            .def(py::init<const std::vector<Eigen::Vector3d> &>(),
                 "Create a PointCloud from points", "points"_a)
            .def("__repr__",
                 [](const geometry::PointCloud &pcd) {
                     return std::string("geometry::PointCloud with ") +
                            std::to_string(pcd.points_.size()) + " points.\n" +
                             "example:\n" +
                             "import My_HSI_LiDAR as HSIL\n"+
                             "import numpy as np\n"+
                             "import open3d as o3d\n"+
                             "pcd = HSIL.geometry.PointCloud()\n"+
                             "np_points = np.random.rand(100, 3)\n"+
                             "pcd.points = o3d.utility.Vector3dVector(np_points)\n"+
                             "pcd.ComputeNeighbor(5)\n"
                             "pcd.NormalEstimate()\n"+
                             "#pcd.ClearNeighbor()\n"
                             "np.asarray(pcd.normals)\n"+
                             "\n"+
                             "\n"+
                             "example2:\n"+
                             "pcd += pcd\n"+
                             "pcd+pcd";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def_readwrite("points", &geometry::PointCloud::points_,
                                       "``float64`` array of shape ``(num_points, 3)``, "
                                       "use ``numpy.asarray()`` to access data: Points "
                                       "coordinates.")
            .def_readwrite("normals", &geometry::PointCloud::normals_,
                           "``float64`` array of shape ``(num_points, 3)``, "
                           "use ``numpy.asarray()`` to access data: Points "
                           "normals.")
            .def_readwrite("colors", &geometry::PointCloud::colors_,
                    "``float64`` array of shape ``(num_points, 3)``, "
                    "range ``[0, 1]`` , use ``numpy.asarray()`` to access "
                    "data: RGB colors of points.")
            .def_readwrite("curvatures", &geometry::PointCloud::curvatures_,
                    "``float64`` array of shape ``(num_points, )``, "
                    "use ``numpy.asarray()`` to access "
                    "data: curvatures of points.")
            .def("ComputeNeighbor",
                 &PointCloud::ComputeNeighbor,
                 "generate n nrighbors",
                 "nn"_a)
            .def("NormalEstimate",
                 &PointCloud::NormalEstimate,
                 "compute normals whthin n nrighbors")
            .def("NormalEstimatek",
                 &PointCloud::NormalEstimatek,
                 "compute normals whthin n nrighbors")
            .def("ClearNeighbor",
                 &PointCloud::ClearNeighbor,
                 "ClearNeighbor")
            .def("HasNeighbors",
                 &PointCloud::HasNeighbors,
                 "HasNeighbors")
            .def("NeighborsSize",
                 &PointCloud::NeighborsSize,
                 "NeighborsSize")
            .def("HasColors",
                 &PointCloud::HasColors,
                 "HasColors")
            .def("HasNormals",
                 &PointCloud::HasNormals,
                 "HasNormals")
            .def("Clear",
                 &PointCloud::Clear,
                 "Clear")
            .def("cube_cloud", &PointCloud::cube_cloud,
                 "generate example point clojud",
                 "sixe"_a, "longx"_a, "longy"_a, "longz"_a,
                 "off_x"_a = 0., "off_y"_a = 0., "off_z"_a = 0)
            .def("SelectDownSample",
                 &PointCloud::SelectDownSample,
                 "SelectDownSample",
                 "cloud"_a, "indices"_a, "invert"_a=false)
            .def("project",
                 &PointCloud::project,
                 "project",
                 "cloud_output"_a, "Ln"_a)
            .def("GetNeighbor",
                 &PointCloud::GetNeighbor,
                 "GetNeighbor",
                 "point_xuhao"_a);
}

}
}
//PointCloud=======================================================

//geometry=======================================================
void pybind_geometry(py::module &m) {
    py::module m_submodule = m.def_submodule("geometry");
    PPP::geometry::pybind_pointcloud(m_submodule);

}
//geometry=======================================================


//CloudToImg=======================================================
namespace PPP{
namespace resample {

void pybind_CloudToImg(py::module &m) {
    py::class_<CloudToImg>
            CloudToImg(m, "CloudToImg",
                       "CloudToImg class. resample point cloud to image "
                       );
    py::detail::bind_default_constructor<resample::CloudToImg>(CloudToImg);
    py::detail::bind_copy_functions<resample::CloudToImg>(CloudToImg);
    CloudToImg
            .def(py::init<> ())
            .def(py::init<const geometry::PointCloud &> (),
                 "PointCloud"_a)
            .def("__repr__",
                 [](const resample::CloudToImg &CTI) {
                     return std::string("点云转图像 ") +
                             "example:\n" +
                             "import My_HSI_LiDAR as HSIL\n"+
                             "import numpy as np\n"+
                             "import open3d as o3d\n"+
                             "pcd = HSIL.geometry.PointCloud()\n"+
                             "np_points = np.random.rand(100, 3)\n"+
                             "pcd.points = o3d.utility.Vector3dVector(np_points)\n"+
                             "\n\n"+
                             "CTI = HSIL.resample.CloudToImg(pcd)\n"+
                             "#CTI = HSIL.resample.CloudToImg()\n"+
                             "#CTI.setCloud(pcd)\n"+
                             "CTI.initialize()\n"+
                             "CTI.setGeodis(0.01)\n"+
                             "CTI.compute_idn()\n"+
                             "CTI.compute_point_in_grid()\n"+
                             "dsm = CTI.compute_dsm()";
                 })
            .def("setCloud",
                 &CloudToImg::setCloud,
                 "input the point cloud and init",
                 "pcd"_a)
            .def("initialize", &CloudToImg::initialize, "initialize")
            .def("setGeodis",
                 &CloudToImg::setGeodis,
                 "set the resample geo distance",
                 "geo_dis"_a)
            .def("setGeoPrj",
                 &CloudToImg::setGeoPrj,
                 "set the resample geotransform",
                 "geo_prj"_a, "m_height"_a, "m_width"_a)
            .def("compute_idn",
                 &CloudToImg::compute_idn,
                 "compute the cloud_idn")
            .def("compute_point_in_grid",
                 &CloudToImg::compute_point_in_grid,
                 "compute point in grid")
            .def("compute_M",
                 &CloudToImg::compute_M,
                 py::return_value_policy::reference_internal ,
                 "compute_M",
                 "shadow_idn"_a)
            .def("compute_Sm",
                 &CloudToImg::compute_Sm,
                 py::return_value_policy::reference_internal ,
                 "compute_Sm",
                 "vox_label"_a, "Ln"_a, "shadow_idn"_a)
            .def("compute_idn_noshadow",
                 &CloudToImg::compute_idn_noshadow,
                 "compute_idn_noshadow",
                 "shadow_idn"_a)
            .def("compute_dsm",
                 &CloudToImg::compute_dsm,
                 "compute and return dsm")
            .def_readwrite("resample_width",
                           &CloudToImg::resample_width_,
                           "resample_width ")
            .def_readwrite("resample_height_", &CloudToImg::resample_height_,
                    "resample_height_ ")
            .def_readwrite("cloud_idn_", &CloudToImg::cloud_idn_,
                    "点云每个坐标映射到DSM上对应到位置 ")
            .def_readwrite("cloud_idn_u", &CloudToImg::cloud_idn_u_,
                    "cloud_idn_u_ ")
            .def_readwrite("u_reverse_indices", &CloudToImg::u_reverse_indices_,
                    "u_reverse_indices_ ")
            .def_readwrite("geo_dis", &CloudToImg::geo_dis_,
                    "geo_dis_ ")
            .def_readwrite("geo_prj", &CloudToImg::geo_prj_,
                    "geo_prj_ ")
            .def("get_points_in_grid",
                 &CloudToImg::get_points_in_grid,
                 "return get_points_in_grid",
                 "pixel_number"_a)
            .def("SelectByMinMax",
                 &CloudToImg::SelectByMinMax,
                 "SelectByMinMax",
                 "max_th"_a, "min_th"_a)
            .def("SelectByPixel",
                 &CloudToImg::SelectByPixel,
                 "SelectByPixel",
                 "pixels"_a);
}

}
}
//CloudToImg=======================================================

//resample=======================================================
void pybind_resample(py::module &m) {
    py::module m_submodule = m.def_submodule("resample");
    PPP::resample::pybind_CloudToImg(m_submodule);

}
//resample=======================================================

//CloudToImg=======================================================
namespace PPP{
namespace Iutility {

void pybind_Iutility(py::module &m) {
    py::class_<Img_utility>
            Img_utility(m, "Img_utility",
                       "Iutility class. resample point cloud to image "
                       );
    py::detail::bind_default_constructor<Iutility::Img_utility>(Img_utility);
    py::detail::bind_copy_functions<Iutility::Img_utility>(Img_utility);
    Img_utility
            .def(py::init<> ())
            .def("__repr__",
                 [](const Iutility::Img_utility &IUI) {
                     return std::string("一些函数 ") +
                             "example1:\n" +
                             "import My_HSI_LiDAR as HSIL\n"+
                             "import numpy as np\n"+
                             "import open3d as o3d\n"+
                             "import matplotlib.pyplot as plt\n"+
                             "Iutility = HSIL.Iutility.Img_utility()\n"+
                             "dsm = np.random.random((50,50))\n"+
                             "dsm[20:28,30:39] = -1\n"+
                             "dsm_vis = Iutility.visualizeDEM(dsm, 0, 0.75)\n"+
                             "plt.figure()\n"+
                             "plt.imshow(dsm_vis)\n"+
                             "plt.show()\n"+
                             "dsm_smooth = Iutility.inpaintZ(dsm, label, 1, 1,  0.1)\n"+
                             "dsm_smooth_vis = Iutility.visualizeDEM(dsm_smooth, 0, 0.75)\n"+
                             "plt.figure()\n"+
                             "plt.imshow(dsm_smooth_vis)\n"+
                             "plt.show()\n"+
                              "\n"+
                             "\n"+
                             "example2:\n"+
                             "np_points = np.random.rand(100, 3)\n"+
                             "index = Iutility.cloud_sort(np_points[:,0], np_points[:,1])\n"+
                             "np_points_sort = np_points[index]";
                 })
            .def("visualizeDEM",
                 &Img_utility::visualizeDEM,
                 "visualizeDEM",
                 "Z"_a, "valid_min"_a, "contrast"_a = 0.75)
            .def("inpaintZ",
                 &Img_utility::inpaintZ,
                 "inpaintZ",
                 "dsm"_a, "label"_a, "lambda_grad"_a, "lambda_curve"_a, "valid_min"_a)
            .def("cloud_sort",
                 &Img_utility::cloud_sort,
                 "cloud_sort",
                 "y_"_a, "x_"_a);
}

}
}
//CloudToImg=======================================================

//Iutility=======================================================
void pybind_Iutility(py::module &m) {
    py::module m_submodule = m.def_submodule("Iutility");
    PPP::Iutility::pybind_Iutility(m_submodule);

}
//Iutility=======================================================


//=============================================================

//TBBSupervoxel=======================================================
namespace PPP{
namespace segmentation  {

void pybind_TBBsupervoxel(py::module &m) {
    py::class_<TBBSupervoxel>
            TBBSupervoxel(m, "TBBSupervoxel",
                       "TBBSupervoxel class. 超体素分割 "
                       );
    py::detail::bind_default_constructor<segmentation::TBBSupervoxel>(TBBSupervoxel);
    py::detail::bind_copy_functions<segmentation::TBBSupervoxel>(TBBSupervoxel);
    TBBSupervoxel
            .def(py::init<> ())
            .def("__repr__",
                 [](const segmentation::TBBSupervoxel &TBB) {
                     return std::string("一些函数 ") +
                             "example1:\n" +
                             "import My_HSI_LiDAR as HSIL\n"+
                             "import numpy as np\n"+
                             "import open3d as o3d\n"+
                             "import matplotlib.pyplot as plt\n"+
                             "pcd = HSIL.geometry.PointCloud()\n"+
                             "pcd.cube_cloud(1000, 1., 1., 1.)\n"+
                             "TBB = HSIL.segmentation.TBBSupervoxel()\n"+
                             "TBB.setCloud(pcd)\n"+
                             "TBB.FindNeighbor(50)\n"+
                             "TBB.set_n_sup1(100)\n"+
                             "TBB.set_z_scale(10.)\n"+
                             "TBB.StartSegmentation()\n"+
                             "label_supvox = np.asarray(TBB.labels)\n"+
                             "\n"+
                             "sup_map = HSIL.geometry.PointCloud()\n"+
                             "TBB.Generate_supmap(sup_map)\n"+
                             "\n"+
                             "cloud_vis = o3d.geometry.PointCloud()\n"+
                             "cloud_vis.points = sup_map.points\n"+
                             "cloud_vis.colors = sup_map.colors\n"+
                             "o3d.visualization.draw_geometries([cloud_vis])\n";
                 })
            .def("setCloud",
                 &TBBSupervoxel::setCloud,
                 "input the point cloud and init",
                 "cloud"_a)
            .def("FindNeighbor",
                 &TBBSupervoxel::FindNeighbor,
                 "FindNeighbor",
                 "number_of_neighbor"_a)
            .def("set_n_sup1",
                 &TBBSupervoxel::set_n_sup1,
                 "set the expected number of supervxels",
                 "number_of_supervoxel"_a)
            .def("set_n_sup2",
                 &TBBSupervoxel::set_n_sup2,
                 "set the expected number of supervxels",
                 "size_of_supervoxel"_a)
            .def("set_z_scale",
                 &TBBSupervoxel::set_z_scale,
                 "set the weight of height z",
                 "z_scale"_a)
            .def("StartSegmentation",
                 &TBBSupervoxel::StartSegmentation,
                 "start supvoxel segemtation")
            .def_readwrite("labels",
                 &TBBSupervoxel::labels_,
                 "labels_ ")
            .def_readwrite("n_supervoxels",
                 &TBBSupervoxel::n_supervoxels_,
                 "n_supervoxels_ ")
            .def("getLabels",
                 &TBBSupervoxel::getLabels,
                 "return supvoxel labels")
            .def("Generate_supmap",
                 &TBBSupervoxel::Generate_supmap,
                 "Generate_supmap");
}

}
}
//TBBSupervoxel=======================================================

//RegionGrow=======================================================
namespace PPP{
namespace segmentation  {

void pybind_RegionGrow(py::module &m) {
    py::class_<RegionGrowing>
            RegionGrowing(m, "RegionGrowing",
                       "RegionGrowing class.  "
                       );
    py::detail::bind_default_constructor<segmentation::RegionGrowing>(RegionGrowing);
    py::detail::bind_copy_functions<segmentation::RegionGrowing>(RegionGrowing);
    RegionGrowing
            .def(py::init<> ())
            .def("__repr__",
                 [](const segmentation::RegionGrowing &RG) {
                     return std::string("一些函数 ") +
                             "example1:\n" +
                             "import My_HSI_LiDAR as HSIL\n"+
                             "import numpy as np\n"+
                             "import open3d as o3d\n"+
                             "import matplotlib.pyplot as plt\n"+
                             "pcd = HSIL.geometry.PointCloud()\n"+
                             "pcd.cube_cloud(1000, 1., 1., 1.)\n"+
                             "pcd1 = HSIL.geometry.PointCloud()\n"+
                             "pcd1.cube_cloud(1000, 1., 1., 1., 1.5)\n"+
                             "pcd = pcd + pcd1\n"+
                             "\n"+
                             "reg = HSIL.segmentation.RegionGrowing()\n"+
                             "reg.setCloud(pcd)\n"+
                             "reg.setMinclusterSize(300 )\n"+
                             "reg.setMaxClusterSize(100000000 )\n"+
                             "reg.setNumberOfNeighbours(60 )\n"+
                             "reg.setSmoothThreshold(10 / 180.0 * np.pi)\n"+
                             "reg.setCurvatureThreshold(2)\n"+
                             "reg.extract()\n"+
                             "\n"+
                             "sup_map = HSIL.geometry.PointCloud()\n"+
                             "reg.Generate_supmap(sup_map)\n"+
                             "\n"+
                             "cloud_vis = o3d.geometry.PointCloud()\n"+
                             "cloud_vis.points = sup_map.points\n"+
                             "cloud_vis.colors = sup_map.colors\n"+
                             "o3d.visualization.draw_geometries([cloud_vis])\n";
                 })
            .def("setCloud",
                 &RegionGrowing::setCloud,
                 "input the point cloud and init",
                 "cloud"_a)
            .def("setMinclusterSize",
                 &RegionGrowing::setMinclusterSize,
                 "set the minimum size of a seg",
                 "minsize"_a)
            .def("setMaxClusterSize",
                 &RegionGrowing::setMaxClusterSize,
                 "set the maximum size of a seg",
                 "maxsize"_a)
            .def("setSmoothThreshold",
                 &RegionGrowing::setSmoothThreshold,
                 "set the Smooth Threshold",
                 "threshold"_a)
            .def("setCurvatureThreshold",
                 &RegionGrowing::setCurvatureThreshold,
                 "set the Curvature Threshold",
                 "threshold"_a)
            .def("setNumberOfNeighbours",
                 &RegionGrowing::setNumberOfNeighbours,
                 "setNumberOfNeighbours",
                 "number"_a)
            .def("setIndices",
                 &RegionGrowing::setIndices,
                 "setIndices",
                 "indices"_a)
            .def("extract", &RegionGrowing::extract, "extract")
            .def_readwrite("n_supervoxels",
                 &RegionGrowing::number_of_segments_,
                 "n_supervoxels_ ")
            .def_readwrite("labels",
                 &RegionGrowing::point_labels_,
                 "point_labels_ ")
            .def("Generate_supmap",
                 &RegionGrowing::Generate_supmap,
                 "Generate_supmap");
}

}
}
//RegionGrow=======================================================


//SupervoxMerging=======================================================
namespace PPP{
namespace segmentation  {

void pybind_SupervoxMerging(py::module &m) {
    py::class_<SupervoxMerging>
            SupervoxMerging(m, "SupervoxMerging",
                       "SupervoxMerging class.  "
                       );
    py::detail::bind_default_constructor<segmentation::SupervoxMerging>(SupervoxMerging);
    py::detail::bind_copy_functions<segmentation::SupervoxMerging>(SupervoxMerging);
    SupervoxMerging
            .def(py::init<> ())
            .def("__repr__",
                 [](const segmentation::SupervoxMerging &SM) {
                     return std::string("一些函数 ") +
                             "example1:\n" +
                             "import My_HSI_LiDAR as HSIL\n"+
                             "import numpy as np\n"+
                             "import open3d as o3d\n"+
                             "import matplotlib.pyplot as plt\n"+
                             "pcd = HSIL.geometry.PointCloud()\n"+
                             "pcd.cube_cloud(1000, 1., 1., 1.)\n"+
                             "pcd1 = HSIL.geometry.PointCloud()\n"+
                             "pcd1.cube_cloud(1000, 1., 1., 1., 1.5)\n"+
                             "pcd = pcd + pcd1\n"+
                             "\n"+
                             "TBB = HSIL.segmentation.TBBSupervoxel()\n"+
                             "TBB.setCloud(pcd)\n"+
                             "TBB.FindNeighbor(50)\n"+
                             "TBB.set_n_sup1(100)\n"+
                             "TBB.set_z_scale(10.)\n"+
                             "TBB.StartSegmentation()\n"+
                             "label_TBB = np.asarray(TBB.labels)\n"+
                             "\n"+
                             "reg = HSIL.segmentation.RegionGrowing()\n"+
                             "reg.setCloud(pcd)\n"+
                             "reg.setMinclusterSize(300 )\n"+
                             "reg.setMaxClusterSize(100000000 )\n"+
                             "reg.setNumberOfNeighbours(60 )\n"+
                             "reg.setSmoothThreshold(10 / 180.0 * np.pi)\n"+
                             "reg.setCurvatureThreshold(2)\n"+
                             "reg.extract()\n"+
                             "label_regien = np.asarray(reg.labels)\n"
                             "\n"+
                             "voxMergin = HSIL.segmentation.SupervoxMerging()\n"+
                             "voxMergin.setCloud(pcd, label_voxel, label_regien)\n"+
                             "voxMergin.setTreeSegPrameter(1.2, 30)\n"+
                             "voxMergin.point_label2voxel()\n"
                             "label_merge = voxMergin.getPoint_Labels()\n"
                             "\n"+
                             "sup_map = HSIL.geometry.PointCloud()\n"+
                             "voxMergin.Generate_supmap(sup_map)\n"+
                             "cloud_vis = o3d.geometry.PointCloud()\n"+
                             "cloud_vis.points = sup_map.points\n"+
                             "cloud_vis.colors = sup_map.colors\n"+
                             "o3d.visualization.draw_geometries([cloud_vis])\n";
                 })
            .def("setCloud",
                 &SupervoxMerging::setCloud,
                 "input the point cloud and init",
                 "cloud"_a, "point_labels"_a, "voxel_labels"_a)
            .def("setTreeSegPrameter",
                 &SupervoxMerging::setTreeSegPrameter,
                 "setTreeSegPrameter",
                 "distance_thre"_a, "nn_neiber"_a)
            .def("point_label2voxel",
                 &SupervoxMerging::point_label2voxel,
                 "point_label2voxel")
            .def("getPoint_Labels",
                 &SupervoxMerging::getPoint_Labels,
                 "getPoint_Labels")
            .def("getPoint_Classes",
                &SupervoxMerging::getPoint_Classes,
                "getPoint_Classes")
            .def("Generate_supmap",
                 &SupervoxMerging::Generate_supmap,
                 "Generate_supmap");
}

}
}
//SupervoxMerging=======================================================

//supervoxel_structure=======================================================
namespace PPP{
namespace segmentation  {

void pybind_supervoxel_structure(py::module &m) {
    py::class_<supervoxel_structure>
            supervoxel_structure(m, "supervoxel_structure",
                       "supervoxel_structure class.  "
                       );
    py::detail::bind_default_constructor<segmentation::supervoxel_structure>(supervoxel_structure);
    py::detail::bind_copy_functions<segmentation::supervoxel_structure>(supervoxel_structure);
    supervoxel_structure
            .def(py::init<> ())
            .def("__repr__",
                 [](const segmentation::supervoxel_structure &SS) {
                     return std::string("一些函数 ") +
                             "example1:\n" +
                             "import My_HSI_LiDAR as HSIL\n"+
                             "import numpy as np\n"+
                             "import open3d as o3d\n"+
                             "import matplotlib.pyplot as plt\n"+
                             "pcd = HSIL.geometry.PointCloud()\n"+
                             "pcd.cube_cloud(1000, 1., 1., 1.)\n"+
                             "pcd1 = HSIL.geometry.PointCloud()\n"+
                             "pcd1.cube_cloud(1000, 1., 1., 1., 1.5)\n"+
                             "pcd = pcd + pcd1\n"+
                             "\n"+
                             "TBB = HSIL.segmentation.TBBSupervoxel()\n"+
                             "TBB.setCloud(pcd)\n"+
                             "TBB.FindNeighbor(50)\n"+
                             "TBB.set_n_sup1(100)\n"+
                             "TBB.set_z_scale(10.)\n"+
                             "TBB.StartSegmentation()\n"+
                             "label_TBB = np.asarray(TBB.labels)\n"+
                             "\n"+
                             "reg = HSIL.segmentation.RegionGrowing()\n"+
                             "reg.setCloud(pcd)\n"+
                             "reg.setMinclusterSize(300 )\n"+
                             "reg.setMaxClusterSize(100000000 )\n"+
                             "reg.setNumberOfNeighbours(60 )\n"+
                             "reg.setSmoothThreshold(10 / 180.0 * np.pi)\n"+
                             "reg.setCurvatureThreshold(2)\n"+
                             "reg.extract()\n"+
                             "label_regien = np.asarray(reg.labels)\n"
                             "\n"+
                             "voxMergin = HSIL.segmentation.SupervoxMerging()\n"+
                             "voxMergin.setCloud(pcd, label_voxel, label_regien)\n"+
                             "voxMergin.setTreeSegPrameter(1.2, 30)\n"+
                             "voxMergin.point_label2voxel()\n"
                             "label_merge = voxMergin.getPoint_Labels()\n"
                             "\n"+
                             "sup_map = HSIL.geometry.PointCloud()\n"+
                             "voxMergin.Generate_supmap(sup_map)\n"+
                             "cloud_vis = o3d.geometry.PointCloud()\n"+
                             "cloud_vis.points = sup_map.points\n"+
                             "cloud_vis.colors = sup_map.colors\n"+
                             "o3d.visualization.draw_geometries([cloud_vis])\n";
                 })
            .def("setCloud",
                 &supervoxel_structure::setCloud,
                 "input the point cloud and init",
                 "cloud"_a)
            .def("setVoxLabel",
                 &supervoxel_structure::setVoxLabel,
                 "setVoxLabel",
                 "voxlabel"_a)
            .def("compute_voxBoundary",
                 &supervoxel_structure::compute_voxBoundary,
                 "compute_voxBoundary",
                 "num_neigh"_a)
            .def("GetIndice",
                 &supervoxel_structure::GetIndice,
                 "GetIndice",
                 "indice"_a)
            .def("GetNeighbors",
                 &supervoxel_structure::GetNeighbors,
                 "GetNeighbors",
                 "indice"_a)
            .def("GetCentroid",
                 &supervoxel_structure::GetCentroid,
                 "GetCentroid",
                 "outcloud"_a)
            .def("GetMeanColor",
                &supervoxel_structure::GetMeanColor,
                "GetMeanColor",
                 "outcloud"_a)
            .def_readwrite("n_supervoxels_", &supervoxel_structure::n_supervoxels_,
                            "n_supervoxels_ ")
            .def("project",
                 &supervoxel_structure::project,
                 "project",
                 "Ln"_a)
            .def("light_project",
                 &supervoxel_structure::light_project,
                 "light_project",
                 "margin"_a, "z_th"_a);

}

}
}
//supervoxel_structure=======================================================


//TBBSupervoxel=======================================================
namespace PPP{
namespace segmentation  {

void pybind_FH_supvoxel(py::module &m) {
    py::class_<FH_supvoxel>
            FH_supvoxel(m, "FH_supvoxel",
                       "FH_supvoxel class. 超体素分割 "
                       );
    py::detail::bind_default_constructor<segmentation::FH_supvoxel>(FH_supvoxel);
    py::detail::bind_copy_functions<segmentation::FH_supvoxel>(FH_supvoxel);
    FH_supvoxel
            .def(py::init<> ())
            .def("__repr__",
                 [](const segmentation::FH_supvoxel &FHS) {
                     return std::string("一些函数 ") +
                             "example1:\n" +
                             "import My_HSI_LiDAR as HSIL\n"+
                             "import numpy as np\n"+
                             "import open3d as o3d\n"+
                             "import matplotlib.pyplot as plt\n"+
                             "pcd = HSIL.geometry.PointCloud()\n"+
                             "pcd.cube_cloud(1000, 1., 1., 1.)\n"+
                             "TBB = HSIL.segmentation.TBBSupervoxel()\n"+
                             "TBB.setCloud(pcd)\n"+
                             "TBB.FindNeighbor(50)\n"+
                             "TBB.set_n_sup1(100)\n"+
                             "TBB.set_z_scale(10.)\n"+
                             "TBB.StartSegmentation()\n"+
                             "label_supvox = np.asarray(TBB.labels)\n"+
                             "\n"+
                             "sup_map = HSIL.geometry.PointCloud()\n"+
                             "TBB.Generate_supmap(sup_map)\n"+
                             "\n"+
                             "cloud_vis = o3d.geometry.PointCloud()\n"+
                             "cloud_vis.points = sup_map.points\n"+
                             "cloud_vis.colors = sup_map.colors\n"+
                             "o3d.visualization.draw_geometries([cloud_vis])\n";
                 })
            .def("setCloud",
                 &FH_supvoxel::setCloud,
                 "input the point cloud and init",
                 "cloud"_a)
            .def("FindNeighbor",
                 &FH_supvoxel::FindNeighbor,
                 "FindNeighbor",
                 "number_of_neighbor"_a)
            .def("setHSIFeature",
                 &FH_supvoxel::setHSIFeature,
                 "set the setHSIFeature",
                 "HSIFeature"_a)
            .def("compute_edge",
                 &FH_supvoxel::compute_edge,
                 "compute_edge")
            .def("graph_seg",
                 &FH_supvoxel::graph_seg,
                 "start segmentation",
                 "ratio"_a, "minsize"_a)
            .def_readwrite("labels",
                 &FH_supvoxel::labels_,
                 "labels_ ")
            .def_readwrite("n_supervoxels",
                 &FH_supvoxel::n_supervoxels_,
                 "n_supervoxels_ ")
            .def("Generate_supmap",
                 &FH_supvoxel::Generate_supmap,
                 "Generate_supmap");
}

}
}
//TBBSupervoxel=======================================================

//segmentation=======================================================
void pybind_segmentation(py::module &m) {
    py::module m_submodule = m.def_submodule("segmentation");
    PPP::segmentation::pybind_TBBsupervoxel(m_submodule);
    PPP::segmentation::pybind_RegionGrow(m_submodule);
    PPP::segmentation::pybind_SupervoxMerging(m_submodule);
    PPP::segmentation::pybind_supervoxel_structure(m_submodule);
    PPP::segmentation::pybind_FH_supvoxel(m_submodule);

}
//segmantation=======================================================

PYBIND11_MODULE(My_HSI_LiDAR, m) {
    NDArrayConverter::init_numpy();
    m.doc() = "Python binding of Open3D";

    // Register this first, other submodule (e.g. odometry) might depend on this
    pybind_utility(m);
    pybind_geometry(m);
    pybind_resample(m);
    pybind_Iutility(m);
    pybind_segmentation(m);

}
