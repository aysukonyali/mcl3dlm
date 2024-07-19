#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp> // For WLS filter
#include <open3d/Open3D.h>
#include "open3d/geometry/PointCloud.h"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <iostream>
#include "data_association.h"
void visualize_correspondences(
     open3d::geometry::PointCloud &source,
     open3d::geometry::PointCloud &target,
     open3d::pipelines::registration::CorrespondenceSet &correspondences) 
{
    // Create a vector to hold the point clouds and linesets
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geometries;

    // Create copies of the source and target point clouds to avoid modifying the originals
    auto source_copy = std::make_shared<open3d::geometry::PointCloud>(source);
    auto target_copy = std::make_shared<open3d::geometry::PointCloud>(target);

    // Set colors for visualization
    source_copy->PaintUniformColor(Eigen::Vector3d(1, 0, 0));  // Red for source
    target_copy->PaintUniformColor(Eigen::Vector3d(0, 1, 0));  // Green for target

    // Add the point clouds to the geometries vector
    geometries.push_back(source_copy);
    geometries.push_back(target_copy);

    // Create a LineSet to show correspondences
    auto line_set = std::make_shared<open3d::geometry::LineSet>();

    // Collect all points from source and target point clouds
    std::vector<Eigen::Vector3d> points;
    points.insert(points.end(), source.points_.begin(), source.points_.end());
    points.insert(points.end(), target.points_.begin(), target.points_.end());

    // Add points to the LineSet
    line_set->points_ = points;

    // Add lines connecting corresponding points
    for (const auto &correspondence : correspondences) {
        int src_idx = correspondence(0);
        int tgt_idx = correspondence(1);

        // Ensure the indices are within bounds
        if (src_idx < source.points_.size() && tgt_idx < target.points_.size()) {
            // The target index is offset by the number of points in the source point cloud
            line_set->lines_.emplace_back(src_idx, source.points_.size() + tgt_idx);
            line_set->colors_.emplace_back(0.0, 0.0, 1.0);  // Blue color for correspondences
        } else {
            std::cerr << "Correspondence index out of bounds: src_idx = " << src_idx << ", tgt_idx = " << tgt_idx << std::endl;
        }
    }

    // Add the LineSet to the geometries vector
    geometries.push_back(line_set);

    // Visualize
    open3d::visualization::DrawGeometries(geometries, "Correspondences", 800, 600);
}

// Function to compute the covariance matrix for a set of points
Eigen::Matrix3d computeCovarianceMatrix(std::vector<Eigen::Vector3d> &points, Eigen::Vector3d &mean)
{
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    mean = Eigen::Vector3d::Zero();

    for (const auto &point : points)
    {
        mean += point;
    }
    mean /= static_cast<double>(points.size());

    for (const auto &point : points)
    {
        Eigen::Vector3d centered = point - mean;
        covariance += centered * centered.transpose();
    }
    covariance /= static_cast<double>(points.size());

    return covariance;
}
// Custom comparison function to sort pairs by their first element in descending order
bool compareEigenPairs(std::pair<double, Eigen::Vector3d> &a, std::pair<double, Eigen::Vector3d> &b)
{
    return a.first > b.first; // Descending order
}

void findLargestEigenvectors(Eigen::Matrix3d &covariance_matrix, Eigen::Vector3d &v1, Eigen::Vector3d &v2)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver( (covariance_matrix + covariance_matrix.transpose()) * 0.5);
    Eigen::Vector3d eigenvalues = solver.eigenvalues();
    Eigen::Matrix3d eigenvectors = solver.eigenvectors();

    // Store eigenvalues and corresponding eigenvectors as pairs
    std::vector<std::pair<double, Eigen::Vector3d>> eigen_pairs;
    for (int i = 0; i < 3; ++i)
    {
        eigen_pairs.push_back(std::make_pair(eigenvalues(i), eigenvectors.col(i)));
    }

    // Sort the eigen_pairs based on the eigenvalues in descending order
    std::sort(eigen_pairs.begin(), eigen_pairs.end(), compareEigenPairs);

    // The largest eigenvectors
    v1 = eigen_pairs[0].second;
    v2 = eigen_pairs[1].second;
}

void computeVoxelMaps(open3d::geometry::VoxelGrid &voxel_grid,
                      open3d::geometry::PointCloud &lidar_t0_t1_t2,
                      std::unordered_map<Eigen::Vector3d, Eigen::Vector3i, open3d::utility::hash_eigen<Eigen::Vector3d>> &point_to_voxel_map,
                      std::unordered_map<Eigen::Vector3i, VoxelData, open3d::utility::hash_eigen<Eigen::Vector3i>> &voxel_to_data_map)
{

    // Map to store points for each voxel
    std::unordered_map<Eigen::Vector3i, std::vector<Eigen::Vector3d>, open3d::utility::hash_eigen<Eigen::Vector3i>> voxel_to_point_map;

    for (auto &point : lidar_t0_t1_t2.points_)
    {
        Eigen::Vector3i voxel_index = voxel_grid.GetVoxel(point);
        voxel_to_point_map[voxel_index].push_back(point);
        point_to_voxel_map[point] = voxel_index;
    }
    // Process each voxel to compute covariance matrix and eigenvectors
    for (auto &voxel_points_pair : voxel_to_point_map)
    {
        auto voxel_index = voxel_points_pair.first;
        auto points = voxel_points_pair.second;

        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        Eigen::Matrix3d covariance = computeCovarianceMatrix(points, mean);

        Eigen::Vector3d v1 = Eigen::Vector3d::Zero();
        Eigen::Vector3d v2 = Eigen::Vector3d::Zero();
        findLargestEigenvectors(covariance, v1, v2);

        Eigen::Vector3d v3 = v1.cross(v2);

        // Create the rotation matrix R
        Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
        R.col(0) = v1;
        R.col(1) = v2;
        R.col(2) = v3;

        // Create the transformation matrix T
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d covariance_4_4 = Eigen::Matrix4d::Identity();
        covariance_4_4.block<3, 3>(0, 0)= covariance;
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = mean;

        T = T.inverse().eval();

        // Compute the diagonal vector of T*covariance*T_inverse
        // Eigen::Matrix3d T_inv = T.inverse();
        // Eigen::Matrix3d diag_matrix = T.block<3, 3>(0, 0) * covariance * R;   //  ???????
        Eigen::Matrix4d diag_matrix = T * covariance_4_4 * T.inverse();
        Eigen::Vector3d diagonal_vector = diag_matrix.diagonal().segment<3>(0).cwiseSqrt();

        // Store the transformation matrix, diagonal vector, and number of points in the voxel
        voxel_to_data_map[voxel_index] = {
            T,
            diagonal_vector,
            static_cast<int>(points.size())};
    }
}

bool isComponentWiseLessThan(Eigen::Vector3d &a, Eigen::Vector3d &b)
{
    return (a.x() < b.x()) && (a.y() < b.y()) && (a.z() < b.z());
}
std::vector<Eigen::Vector3i> getNeighboringVoxels(Eigen::Vector3i &voxel_index)
{
    std::vector<Eigen::Vector3i> neighbors;

    // Generate the 27 possible neighbors (3x3x3 cube including the center voxel)
    for (int dx = -1; dx <= 1; ++dx)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dz = -1; dz <= 1; ++dz)
            {
                Eigen::Vector3i neighbor_index = voxel_index + Eigen::Vector3i(dx, dy, dz);
                neighbors.push_back(neighbor_index);
            }
        }
    }

    return neighbors;
}

open3d::pipelines::registration::CorrespondenceSet refineCorrespondences(
    open3d::pipelines::registration::CorrespondenceSet correspondences,
    open3d::geometry::PointCloud &source,
    open3d::geometry::PointCloud &target,
    std::unordered_map<Eigen::Vector3i, VoxelData, open3d::utility::hash_eigen<Eigen::Vector3i>> &voxel_to_data_map,
    std::unordered_map<Eigen::Vector3d, Eigen::Vector3i, open3d::utility::hash_eigen<Eigen::Vector3d>> &point_to_voxel_map,
    int N_min,
    int N_standard_deviation)
{
    open3d::pipelines::registration::CorrespondenceSet refined_corr;
    for (const auto &correspondence : correspondences)
    {
        int source_index = correspondence[0];
        int target_index = correspondence[1];

        Eigen::Vector3d source_point = source.points_[source_index];
        Eigen::Vector3d target_point = target.points_[target_index];
        Eigen::Vector3i voxel_index = point_to_voxel_map[target_point];

        std::vector<Eigen::Vector3i> voxels_to_check = getNeighboringVoxels(voxel_index);
        for (Eigen::Vector3i v : voxels_to_check)
        {
            // if the neighboring voxel really exists
            if (voxel_to_data_map.find(v) != voxel_to_data_map.end())
            {
                VoxelData voxel_data = voxel_to_data_map[v];
                Eigen::Matrix4d transformation_matrix = voxel_data.transformation_matrix;
                Eigen::Vector3d diagonal_vector = voxel_data.diagonal_vector;
                int num_points = voxel_data.num_points;

                Eigen::Vector4d source_point_homogeneous;
                source_point_homogeneous << source_point, 1.0;

                // Apply the transformation
                Eigen::Vector4d transformed_point_homogeneous = transformation_matrix * source_point_homogeneous;

                // Convert back to 3D coordinates
                Eigen::Vector3d transformed_point = transformed_point_homogeneous.head<3>();
                Eigen::Vector3d multiple_standard_deviation = N_standard_deviation * diagonal_vector;

                if (num_points >= N_min && isComponentWiseLessThan(transformed_point, multiple_standard_deviation))
                {
                    refined_corr.push_back(correspondence);
                    // no need to check the other neigboring voxels now
                    break;
                }
            }
        }
    }
    return refined_corr;
}
