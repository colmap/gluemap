# setup PYTHON_EXECUTABLE
cmake_minimum_required(VERSION 3.18)

project(pygluemap)
cmake_policy(SET CMP0057 NEW)  # Fix IN_LIST handling

set(CMAKE_BUILD_TYPE Release)

find_package (Python COMPONENTS Interpreter Development)
message(STATUS "Python_EXECUTABLE: " ${Python_EXECUTABLE})

execute_process(
    COMMAND python3 -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(pygluemap MODULE "bindings.cc" "track_selection.cc")

target_include_directories(pygluemap PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
)
target_link_libraries(pygluemap PRIVATE pybind11::module Eigen3::Eigen ${CERES_LIBRARIES} Boost::headers)
target_compile_definitions(pygluemap PRIVATE cxx_std_17)
target_compile_definitions(pygluemap PRIVATE CERES_HAS_CUDA)

# install the module to the site-packages directory
install(TARGETS pygluemap DESTINATION ${PYTHON_SITE_PACKAGES})
