#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cost_functions.h"

#include <unordered_map>
#include <vector>

#include <ceres/ceres.h>

namespace py = pybind11;
using namespace pybind11::literals;

// Forward declarations from track_selection.cc
py::tuple ComputeTracksToDeleteWrapper(
    py::array_t<int64_t, py::array::c_style> point3d_ids,
    py::array_t<int64_t, py::array::c_style> track_image_ids,
    py::array_t<int64_t, py::array::c_style> track_pt2d_idxs,
    py::array_t<int32_t, py::array::c_style> track_lengths,
    const std::unordered_map<int64_t, int> &sift_count,
    int min_num_support_abs);

py::tuple ComputeVirtualTracksToDeleteWrapper(
    py::array_t<int64_t, py::array::c_style> point3d_ids,
    py::array_t<int64_t, py::array::c_style> track_image_ids,
    py::array_t<int64_t, py::array::c_style> track_pt2d_idxs,
    py::array_t<int32_t, py::array::c_style> track_lengths,
    const std::unordered_map<uint64_t, int> &pair_count_in,
    int min_num_support_abs);

// Helper function to create a ProductManifold for 7D pose (quat + trans)
ceres::Manifold *CreatePoseManifold() {
  return new ceres::ProductManifold<ceres::EigenQuaternionManifold,
                                    ceres::EuclideanManifold<3>>();
}

// Helper function to create a ProductManifold for 7D pose with fixed
// translation component
ceres::Manifold *
CreatePoseManifoldWithFixedTransComponent(int fixed_component) {
  std::vector<int> constant_indices = {fixed_component};
  return new ceres::ProductManifold<ceres::EigenQuaternionManifold,
                                    ceres::SubsetManifold>(
      ceres::EigenQuaternionManifold(),
      ceres::SubsetManifold(3, constant_indices));
}

// Helper function to create a manifold that fixes rotation but allows
// translation
ceres::Manifold *CreateTranslationOnlyManifold() {
  // Fix all 4 quaternion components, allow 3 translation components
  std::vector<int> constant_quat = {0, 1, 2, 3};
  return new ceres::ProductManifold<ceres::SubsetManifold,
                                    ceres::EuclideanManifold<3>>(
      ceres::SubsetManifold(4, constant_quat), ceres::EuclideanManifold<3>());
}

// Solve a Ceres problem with CUDA GPU acceleration
void SolveCUDA(const ceres::Solver::Options &input_options,
               ceres::Problem *problem, ceres::Solver::Summary *summary) {
  ceres::Solver::Options options = input_options;

#ifdef CERES_HAS_CUDA
  switch (options.linear_solver_type) {
  case ceres::SPARSE_NORMAL_CHOLESKY:
  case ceres::SPARSE_SCHUR:
    options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    break;
  case ceres::DENSE_NORMAL_CHOLESKY:
  case ceres::DENSE_SCHUR:
  case ceres::DENSE_QR:
    options.dense_linear_algebra_library_type = ceres::CUDA;
    break;
  default:
    break;
  }
#endif

  py::gil_scoped_release release;
  ceres::Solve(options, problem, summary);
}

bool IsCUDAAvailable() {
#ifdef CERES_HAS_CUDA
  return true;
#else
  return false;
#endif
}

PYBIND11_MODULE(pygluemap, m) {
  py::module_::import("pyceres");

  m.def("RotationGeodesicError",
        &RotationGeodesicError::Create<const Eigen::Vector4d &>,
        py::arg("i_q_j"));

  m.def("BATAPairwiseDirectionError",
        &BATAPairwiseDirectionError::Create<const Eigen::Vector3d &>,
        py::arg("translation_obs"));

  m.def("ScaledObsPairwiseDirectionError",
        &ScaledObsPairwiseDirectionError::Create<const Eigen::Vector3d &>,
        py::arg("translation_obs"));

  m.def("ReprojErrorCost",
        &colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor,
                                          const Eigen::Vector2d &>,
        "camera_model_id"_a, "point2D"_a, "Reprojection error.");

  m.def(
      "ReprojErrorCostWithNegativeDepth",
      &colmap::CreateCameraCostFunction<ReprojErrorCostWithNegativeDepthFunctor,
                                        const Eigen::Vector2d &>,
      "camera_model_id"_a, "point2D"_a,
      "Reprojection error with negative depth.");

  // Manifold creation helpers for 7D pose (quaternion + translation)
  m.def("CreatePoseManifold", &CreatePoseManifold,
        py::return_value_policy::take_ownership,
        "Create a ProductManifold for 7D pose (quaternion + translation).");

  m.def("CreatePoseManifoldWithFixedTransComponent",
        &CreatePoseManifoldWithFixedTransComponent, py::arg("fixed_component"),
        py::return_value_policy::take_ownership,
        "Create a ProductManifold for 7D pose with one translation component "
        "fixed.");

  m.def(
      "CreateTranslationOnlyManifold", &CreateTranslationOnlyManifold,
      py::return_value_policy::take_ownership,
      "Create a manifold that fixes rotation but allows translation to vary.");

  // CUDA GPU solver
  m.def("solve_cuda", &SolveCUDA, py::arg("options"), py::arg("problem"),
        py::arg("summary"),
        "Solve a Ceres problem with CUDA GPU acceleration.");

  m.def("is_cuda_available", &IsCUDAAvailable,
        "Returns True if the module was compiled with CUDA support.");

  // Numpy-based track selection: returns point3D IDs to delete.
  // Python then calls reconstruction.delete_point3d(id) for each.
  m.def("compute_tracks_to_delete", &ComputeTracksToDeleteWrapper,
        py::arg("point3d_ids"), py::arg("track_image_ids"),
        py::arg("track_pt2d_idxs"), py::arg("track_lengths"),
        py::arg("sift_count"), py::arg("min_num_support_abs") = 512,
        "Classify and select tracks. Returns (ids_to_delete, pair_count) where "
        "ids_to_delete is an int64 array of point3D IDs to delete and "
        "pair_count is a dict mapping (img_low, img_high) tuples to coverage "
        "counts after selection.");

  m.def("compute_virtual_tracks_to_delete",
        &ComputeVirtualTracksToDeleteWrapper, py::arg("point3d_ids"),
        py::arg("track_image_ids"), py::arg("track_pt2d_idxs"),
        py::arg("track_lengths"), py::arg("pair_count"),
        py::arg("min_num_support_abs") = 512,
        "Select virtual tracks given existing pair coverage. Returns "
        "(ids_to_delete, updated_pair_count). Tracks whose image pairs are "
        "all above min_num_support_abs are removed.");
}
