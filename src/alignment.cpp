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
#include "local_reconstruction.h"
#include "data_association.h"

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
// Function to rotate points
open3d::geometry::PointCloud rotateLidarMap(std::vector<Eigen::Vector4f> &lidar, double angle_deg, std::string filename)
{  
    open3d::geometry::PointCloud point_cloud;
        for (const auto &lidar_point : lidar)
    {
        // Extract XYZ coordinates from Eigen::Vector4f (homogeneous coordinates)
        Eigen::Vector3d point_xyz(static_cast<double>(lidar_point[0]),
                                  static_cast<double>(lidar_point[1]),
                                  static_cast<double>(lidar_point[2]));

        // Add point to Open3D point cloud
        point_cloud.points_.push_back(point_xyz);
    }
    double angle_rad = angle_deg * M_PI / 180.0;
    Eigen::Matrix3d rotation;
    rotation = Eigen::AngleAxisd(angle_rad, Eigen::Vector3d::UnitZ());

    std::vector<Eigen::Vector3d> rotated_points;
    for (auto &point : point_cloud.points_)
    {
        // Rotate the point using Eigen
        Eigen::Vector3d rotated_point = rotation * point;

        // Add the rotated point to the vector
        rotated_points.push_back(rotated_point);
    }
    point_cloud.points_=rotated_points;
    write_point_cloud(point_cloud.points_,lidar, filename);
    return point_cloud;
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
open3d::geometry::PointCloud mergePointClouds(open3d::geometry::PointCloud &pc1, open3d::geometry::PointCloud &pc2)
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
open3d::geometry::PointCloud get_denser_lidar_map(std::vector<Eigen::Vector4f> lidar_map_t0,
                                                  std::vector<Eigen::Vector4f> lidar_map_t1,
                                                  std::vector<Eigen::Vector4f> lidar_map_t2,
                                                  open3d::geometry::PointCloud open3d_stereo_cam2velo)
{
    double max_correspondence_distance = 0.02;
    int max_iterations = 100;
    double tolerance = 1e-6;
    open3d::geometry::PointCloud target_lidar_map_t0;
    open3d::geometry::PointCloud source_lidar_map_t1;
    open3d::geometry::PointCloud source_lidar_map_t2;
    convertTargetToOpen3dPointCloud(lidar_map_t0, target_lidar_map_t0);
    convertTargetToOpen3dPointCloud(lidar_map_t1, source_lidar_map_t1);
    convertTargetToOpen3dPointCloud(lidar_map_t2, source_lidar_map_t2);

    std::cout << "t1_0 - t0_0 " << lidar_map_t0[0].norm() - lidar_map_t1[0].norm() << std::endl;
    std::cout << "t2_0 - t0_0 " << lidar_map_t0[0].norm() - lidar_map_t2[0].norm() << std::endl;
    Eigen::Matrix4d identity_t1_t0 = Eigen::Matrix4d::Identity();
    identity_t1_t0(0, 3) = std::abs(lidar_map_t0[0].norm() - lidar_map_t1[0].norm());
    runIcp(source_lidar_map_t1, target_lidar_map_t0, 0.1, 100, tolerance, identity_t1_t0);
    write_point_cloud(source_lidar_map_t1.points_, lidar_map_t1, "icp_lidar_map_t0_t1.ply");
    open3d::geometry::PointCloud merged_pcds_t0_t1 = mergePointClouds(target_lidar_map_t0, source_lidar_map_t1);
    std::vector<Eigen::Vector4f> merged_colors_t0_t1 = mergeColors(lidar_map_t0, lidar_map_t1);
    write_point_cloud(merged_pcds_t0_t1.points_, merged_colors_t0_t1, "merged_pcds_t0_t1.ply");

    // aligning with stereo reconstruction  ---commented out-----
    // Eigen::Matrix4d identity =Eigen::Matrix4d::Identity();
    // runIcp(merged_pcds_t0_t1, open3d_stereo_cam2velo, 0.01, 100, tolerance, identity);
    // write_point_cloud(merged_pcds_t0_t1.points_, merged_colors_t0_t1, "icp_lidar_t0_t1_stereo.ply");

    Eigen::Matrix4d identity_t2_t0 = Eigen::Matrix4d::Identity();
    identity_t2_t0(0, 3) = std::abs(lidar_map_t0[0].norm() - lidar_map_t2[0].norm());
    // identity_t2_t0(0,3) = 1.3163+1.4493;
    // identity_t2_t0(0,3) = 2.6;
    runIcp(source_lidar_map_t2, merged_pcds_t0_t1, 0.2, 100, tolerance, identity_t2_t0);
    write_point_cloud(source_lidar_map_t2.points_, lidar_map_t2, "icp_lidar_map_t0_t1_t2.ply");
    open3d::geometry::PointCloud merged_pcds_t0_t1_t2 = mergePointClouds(merged_pcds_t0_t1, source_lidar_map_t2);
    std::vector<Eigen::Vector4f> merged_colors_t0_t1_t2 = mergeColors(merged_colors_t0_t1, lidar_map_t2);

    // aligning with stereo reconstruction ---commented out-----
    // runIcp(merged_pcds_t0_t1_t2, open3d_stereo_cam2velo, 0.01,100, tolerance, identity);
    write_point_cloud(merged_pcds_t0_t1_t2.points_, merged_colors_t0_t1_t2, "icp_lidar_t0_t1_t2_final.ply");

    return merged_pcds_t0_t1_t2;
}
// Custom function to refine correspondences
std::vector<Eigen::Vector2i> refine_correspondences(
    const open3d::geometry::PointCloud& source,
    const open3d::geometry::PointCloud& target,
    const std::vector<Eigen::Vector2i>& initial_correspondences) {

    std::vector<Eigen::Vector2i> refined_correspondences;

    // Example custom refinement strategy: Filter out correspondences based on some criterion
    // This is a placeholder; you should replace it with your actual refinement logic
    for (const auto& correspondence : initial_correspondences) {
        const Eigen::Vector3d& source_point = source.points_[correspondence(0)];
        const Eigen::Vector3d& target_point = target.points_[correspondence(1)];

        // Example criterion: Only keep correspondences where the source and target points are within a certain distance
        double distance = (source_point - target_point).norm();
        if (distance < 0.05) { // Replace 0.05 with your threshold
            refined_correspondences.push_back(correspondence);
        }
    }

    return refined_correspondences;
}

void run_improved_icp(double voxel_size,
                      int N_min,
                      int N_standard_devitation,
                      std::vector<Eigen::Vector4f> &lidar_map_t0,
                      open3d::geometry::PointCloud &lidar_t0_t1_t2,
                      double max_correspondence_distance, 
                      int max_iterations, 
                      double tolerance)
{

    // Voxelize the source point cloud
    auto voxel_grid = open3d::geometry::VoxelGrid::CreateFromPointCloud(lidar_t0_t1_t2, voxel_size);
    open3d::io::WriteVoxelGrid("lidar_voxel_grid.ply", *voxel_grid);
    std::cout << "voxel_grid size: "<< voxel_grid->GetVoxels().size() << std::endl;
    
    // each point in the point cloud is in a voxel
    std::unordered_map<Eigen::Vector3d, Eigen::Vector3i, open3d::utility::hash_eigen<Eigen::Vector3d>> point_to_voxel_map;
    // each voxel has its Voxel data (T, standart deviation vector, number of points)
    std::unordered_map<Eigen::Vector3i, VoxelData, open3d::utility::hash_eigen<Eigen::Vector3i>> voxel_to_data_map;
    computeVoxelMaps(*voxel_grid, lidar_t0_t1_t2, point_to_voxel_map, voxel_to_data_map);

    std::cout << "point_to_voxel_map size: "<< point_to_voxel_map.size() << std::endl;
    std::cout << "voxel_to_data_map size: "<< voxel_to_data_map.size() << std::endl;

    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    open3d::geometry::PointCloud rotated_lidar_map_t0 = rotateLidarMap(lidar_map_t0,2,"rotated_lidar_t0.ply");
    open3d::geometry::PointCloud rotated_lidar_map_t0_2 = rotated_lidar_map_t0;

    auto registration_result = open3d::pipelines::registration::RegistrationICP(
        rotated_lidar_map_t0, lidar_t0_t1_t2, 0.1, t,
        open3d::pipelines::registration::TransformationEstimationPointToPoint(),
        open3d::pipelines::registration::ICPConvergenceCriteria(100, tolerance));

    write_point_cloud(rotated_lidar_map_t0.Transform(registration_result.transformation_).points_,lidar_map_t0,"rotated_lidar_with_normal_correspondences.ply");
    std::cout << "Transformation matrix for rotated lidar after icp:" << std::endl;
    std::cout << registration_result.transformation_ << std::endl;    

    open3d::pipelines::registration::CorrespondenceSet correspondences = registration_result.correspondence_set_;
    std::cout << "correspondences size: "<< correspondences.size() << std::endl;
    //visualize_correspondences(rotated_lidar_map_t0,lidar_t0_t1_t2,correspondences);
    open3d::pipelines::registration::CorrespondenceSet refined_correspondences = refineCorrespondences(
        correspondences,
        rotated_lidar_map_t0_2,
        lidar_t0_t1_t2,
        voxel_to_data_map,
        point_to_voxel_map,
        N_min, // N_min
        N_standard_devitation); // N_standard_deviation
    std::cout << "refined_correspondences size: "<< refined_correspondences.size() << std::endl;
    
    // Estimate the transformation matrix using point-to-point transformation estimation
    open3d::pipelines::registration::TransformationEstimationPointToPoint estimation;
    Eigen::Matrix4d transformation = estimation.ComputeTransformation(
        rotated_lidar_map_t0_2, lidar_t0_t1_t2, refined_correspondences);
    //visualize_correspondences(rotated_lidar_map_t0_2,lidar_t0_t1_t2,refined_correspondences);    
    std::cout << "Transformation matrix using refined correspondences:" << std::endl;
    std::cout << transformation << std::endl;
    write_point_cloud(rotated_lidar_map_t0_2.Transform(transformation).points_,lidar_map_t0,"rotated_lidar_with_refined_correspondences.ply");
}