cmake_minimum_required(VERSION 3.22)
project(Realsense_RGBD_extract_ROI)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
find_package(realsense2 REQUIRED)
find_package(Open3D REQUIRED)
find_package(spdlog REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} main.cpp)

# Link librealsense2 to the target
target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY} spdlog::spdlog Open3D::Open3D ${CUDA_LIBRARIES} ${OpenCV_LIBS})
