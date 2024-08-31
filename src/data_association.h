#ifndef DATA_ASSOCIATION_H
#define DATA_ASSOCIATION_H
#include <unordered_set>
#include <filesystem>
#include <open3d/Open3D.h>
#include <opencv2/core.hpp>
#include "open3d/geometry/PointCloud.h"

// Struct to store voxel information
struct VoxelData
{
    Eigen::Matrix4d transformation_matrix; // Transformation matrix T
    Eigen::Vector3d diagonal_vector;       // Diagonal vector of T*covariance*T_inverse
    int num_points;                        // Number of points in the voxel
};
void drawCorrespondences(
    const open3d::geometry::PointCloud &source,
    const open3d::geometry::PointCloud &target,
    const open3d::pipelines::registration::CorrespondenceSet &correspondences);
void drawRefinedCorrespondences(
    const open3d::geometry::PointCloud &source,
    const open3d::geometry::PointCloud &target,
    const open3d::pipelines::registration::CorrespondenceSet &correspondences1,
    const open3d::pipelines::registration::CorrespondenceSet &correspondences2);
        
void computeVoxelMaps(open3d::geometry::VoxelGrid &voxel_grid,
                      open3d::geometry::PointCloud &lidar_t0_t1_t2,
                      std::unordered_map<Eigen::Vector3d, Eigen::Vector3i, open3d::utility::hash_eigen<Eigen::Vector3d>> &point_to_voxel_map,
                      std::unordered_map<Eigen::Vector3i, VoxelData, open3d::utility::hash_eigen<Eigen::Vector3i>> &voxel_to_data_map);

open3d::pipelines::registration::CorrespondenceSet refineCorrespondences(
    open3d::pipelines::registration::CorrespondenceSet &correspondences,
    open3d::geometry::PointCloud &source,
    open3d::geometry::PointCloud &target,
    std::unordered_map<Eigen::Vector3i, VoxelData, open3d::utility::hash_eigen<Eigen::Vector3i>> &voxel_to_data_map,
    std::unordered_map<Eigen::Vector3d, Eigen::Vector3i, open3d::utility::hash_eigen<Eigen::Vector3d>> &point_to_voxel_map,
    int N_min,
    int N_standard_deviation);

#endif