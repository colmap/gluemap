# setup PYTHON_EXECUTABLE
cmake_minimum_required(VERSION 3.18)

project(pygluemap)
cmake_policy(SET CMP0057 NEW)  # Fix IN_LIST handling

set(CMAKE_BUILD_TYPE Release)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Python_EXECUTABLE: " ${Python_EXECUTABLE})

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(pygluemap MODULE "bindings.cc" "track_selection.cc" "merge_reconstruction.cc")

target_include_directories(pygluemap PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
)
target_link_libraries(pygluemap PRIVATE pybind11::module Eigen3::Eigen ${CERES_LIBRARIES} Boost::headers)
target_compile_definitions(pygluemap PRIVATE cxx_std_17)
target_compile_definitions(pygluemap PRIVATE CERES_HAS_CUDA)

# Bake the directories of linked libraries (Ceres, etc.) into the
# extension's RPATH so loading pygluemap also picks up the matching
# libstdc++ / libceres from $CONDA_PREFIX/lib at runtime, without
# requiring the user to set LD_LIBRARY_PATH.
set_target_properties(pygluemap PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# scikit-build-core sets CMAKE_INSTALL_PREFIX to the wheel staging dir;
# combined with `wheel.install-dir = "/"` in pyproject.toml, this lands
# pygluemap at the top of site-packages (matching the legacy install).
install(TARGETS pygluemap DESTINATION .)
