#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <open3d/Open3D.h>
#include "open3d/geometry/PointCloud.h"
#include <opencv2/opencv.hpp>
#include <opencv2/surface_matching/icp.hpp>
#include <vector>

// Function to rotate points
std::vector<cv::Vec3f> rotatePoints(std::vector<cv::Vec3f> &points, double angle_deg)
{
    double angle_rad = angle_deg * M_PI / 180.0;
    Eigen::Matrix3d rotation;
    rotation = Eigen::AngleAxisd(angle_rad, Eigen::Vector3d::UnitZ());

    std::vector<cv::Vec3f> rotated_points;
    for (auto &point : points)
    {
        // Convert cv::Vec3f to Eigen::Vector3d
        Eigen::Vector3d point_eigen(point[0], point[1], point[2]);

        // Rotate the point using Eigen
        Eigen::Vector3d rotated_point_eigen = rotation * point_eigen;

        // Convert Eigen::Vector3d back to cv::Vec3f
        cv::Vec3f rotated_point(rotated_point_eigen.x(), rotated_point_eigen.y(), rotated_point_eigen.z());
        rotated_points.push_back(rotated_point);
    }
    return rotated_points;
}

// Convert points and colors to Open3D PointCloud
void convertSourceToOpen3dPointCloud(std::vector<cv::Vec3f> &points, std::vector<cv::Vec3b> &colors,
                                     open3d::geometry::PointCloud &point_cloud)
{
    for (const auto &point : points)
    {
        // Convert cv::Vec3f to Eigen::Vector3d
        Eigen::Vector3d point_eigen(static_cast<double>(point[0]),
                                    static_cast<double>(point[1]),
                                    static_cast<double>(point[2]));

        // Add point to Open3D point cloud
        point_cloud.points_.push_back(point_eigen);
    }
    for (size_t i = 0; i < colors.size(); ++i)
    {

        // Convert color from cv::Vec3b (BGR) to Open3D RGB format
        point_cloud.colors_.emplace_back(
            static_cast<double>(colors[i][2]) / 255.0, // B channel to R
            static_cast<double>(colors[i][1]) / 255.0, // G channel remains G
            static_cast<double>(colors[i][0]) / 255.0  // R channel to B
        );
    }
}

void convertTargetToOpen3dPointCloud(std::vector<Eigen::Vector4f> &target_points,
                                     open3d::geometry::PointCloud &point_cloud)
{
    for (const auto &target_point : target_points)
    {
        // Extract XYZ coordinates from Eigen::Vector4f (homogeneous coordinates)
        Eigen::Vector3d point_xyz(static_cast<double>(target_point[0]),
                                  static_cast<double>(target_point[1]),
                                  static_cast<double>(target_point[2]));

        // Add point to Open3D point cloud
        point_cloud.points_.push_back(point_xyz);
        // No color information provided in Eigen::Vector4f, so add default color (white)
        point_cloud.colors_.push_back(Eigen::Vector3d(0.0, 0.0, 0.0)); // White color
    }
}
// Function to visualize point clouds
void visualizePointClouds(open3d::geometry::PointCloud &source,
                          open3d::geometry::PointCloud &target,
                          std::string window_name)
{
    open3d::visualization::Visualizer visualizer;
    visualizer.CreateVisualizerWindow(window_name, 800, 600);
    visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(source));
    visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(target));
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
}

// Run ICP
void runIcp(open3d::geometry::PointCloud &source,
            open3d::geometry::PointCloud &target,
            double max_correspondence_distance, int max_iterations, double tolerance,
            Eigen::Matrix4d &initial_transformation)
{

    // Visualize source and target before ICP
    // std::cout << "Visualizing point clouds before ICP..." << std::endl;
    // visualizePointClouds(source, target, "Before ICP");

    // Run ICP
    auto registration_result = open3d::pipelines::registration::RegistrationICP(
        source, target, max_correspondence_distance, initial_transformation,
        open3d::pipelines::registration::TransformationEstimationPointToPoint(),
        open3d::pipelines::registration::ICPConvergenceCriteria(max_iterations, tolerance));

    std::cout << "Transformation matrix after icp:" << std::endl;
    std::cout << registration_result.transformation_ << std::endl;

    // Transform the source point cloud using the final transformation
    source.Transform(registration_result.transformation_);

    // Visualize source and target after ICP
    // std::cout << "Visualizing point clouds after ICP..." << std::endl;
    // visualizePointClouds(source, target, "After ICP");

    // Save the transformed source point cloud to a file
    // open3d::io::WritePointCloud("transformed_source.ply", source);
    // std::cout << "Transformed source point cloud saved to transformed_source.ply" << std::endl;
}

void cvMatToEigenMat(cv::Mat &cv_mat, Eigen::Matrix4d &eigen_mat)
{
    eigen_mat = Eigen::Matrix4d(cv_mat.rows, cv_mat.cols);
    for (int i = 0; i < cv_mat.rows; ++i)
    {
        for (int j = 0; j < cv_mat.cols; ++j)
        {
            eigen_mat(i, j) = cv_mat.at<double>(i, j);
        }
    }
}
// Function to merge two point clouds
open3d::geometry::PointCloud mergePointClouds(open3d::geometry::PointCloud &pc1,open3d::geometry::PointCloud &pc2)
{
    open3d::geometry::PointCloud merged_pcd;

    merged_pcd.points_.insert(merged_pcd.points_.end(), pc1.points_.begin(), pc1.points_.end());
    merged_pcd.points_.insert(merged_pcd.points_.end(), pc2.points_.begin(), pc2.points_.end());

    return merged_pcd;
}
std::vector<Eigen::Vector4f> mergeColors(std::vector<Eigen::Vector4f> color1, std::vector<Eigen::Vector4f> color2)
{
    std::vector<Eigen::Vector4f> merged_colors;

    merged_colors.insert(merged_colors.end(), color1.begin(), color1.end());
    merged_colors.insert(merged_colors.end(), color2.begin(), color2.end());

    return merged_colors;
}