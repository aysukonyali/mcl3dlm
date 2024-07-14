cmake_minimum_required(VERSION 3.9)
# export CMAKE_PREFIX_PATH=/home/aysu/open3d_install/lib/cmake/Open3D:$CMAKE_PREFIX_PATH buildde bunu runla
set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(mcl3dlm)

# Options
set(CMAKE_CXX_STANDARD 17)
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs CACHE PATH "Path to libs folder")
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src CACHE PATH "Path to src folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(Open3D_DIR "/home/aysu/open3d_install/lib/cmake/Open3D" CACHE PATH "Path to Open3DConfig.cmake directory")

# Set CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${Open3D_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)

find_package(Open3D REQUIRED)
include_directories(${Open3D_INCLUDE_DIRS})
# include_directories(
#     /home/aysu/open3d_install/include
#     /home/aysu/open3d_install/include/open3d/3rdparty)

set(HEADER_FILES 
    ${SOURCE_DIR}/local_reconstruction.h
    ${SOURCE_DIR}/data_association.h
    ${SOURCE_DIR}/alignment.h

)

set(SOURCE_FILES 
    ${SOURCE_DIR}/local_reconstruction.cpp
    ${SOURCE_DIR}/data_association.cpp
    ${SOURCE_DIR}/alignment.cpp

)
add_executable(${PROJECT_NAME} ${SOURCE_DIR}/main.cpp ${HEADER_FILES} ${SOURCE_FILES})
#target_link_libraries(${PROJECT_NAME} Eigen3::Eigen ${OpenCV_LIBS})
#target_link_libraries(${PROJECT_NAME} Eigen3::Eigen ${OpenCV_LIBS} ${Open3D_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} Open3D::Open3D Eigen3::Eigen)
