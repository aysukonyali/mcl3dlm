#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <open3d/Open3D.h>
#include "open3d/geometry/PointCloud.h"
#include <opencv2/opencv.hpp>
#include <opencv2/surface_matching/icp.hpp>
#include <vector>
#include <random>
#include "local_reconstruction.h"
#include "data_association.h"
#include "simple_odometry.h"
#include "alignment.h"

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
    point_cloud.points_ = rotated_points;
    write_point_cloud(point_cloud.points_, lidar, filename);
    return point_cloud;
}
Eigen::Vector3d generateRandomTranslation(double min_val, double max_val) {
    // Random number generation setup
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 eng(rd()); // Seed the generator
    std::uniform_real_distribution<> distr(min_val, max_val); // Define the range

    // Generate random translation vector
    Eigen::Vector3d translation;
    translation << distr(eng), distr(eng), distr(eng);
    return translation;
}
void rotation_plot(int angles, int range, open3d::geometry::PointCloud &stereo, open3d::geometry::PointCloud &lidar,std::vector<double> &errors,std::vector<double>&errors_icp){
    double tolerance = 1e-6;
    
    for (size_t i = 0; i < angles; i++)
    {   
        double error = 0.0;
        double error_icp = 0.0;
        for (size_t j = 0; j < range; j++)
        {
            Sophus::SO3d so3d_r = random_rotation(i);
            Eigen::Matrix4d r = Eigen::Matrix4d::Identity();
            r.block<3, 3>(0, 0) = so3d_r.matrix();
            Eigen::Matrix4d r_t = Eigen::Matrix4d::Identity();
            Eigen::Matrix4d r_icp = Eigen::Matrix4d::Identity();
            double d = (double)(i+2);
            correspondences_filtering_plot(0.5, 15, 3, stereo, lidar,r , r_t, r_icp, d, 30, tolerance);
            error += computeError(r, r_t);
            error_icp += computeError(r, r_icp);
        }
        error /= range;
        error_icp /= range;
        errors.push_back(error);
        errors_icp.push_back(error_icp);
    }
 
}
Sophus::SO3d random_rotation(double angle_deg) {
    // Convert angle from degrees to radians
    double angle_rad = angle_deg * M_PI / 180.0;

    // Generate a random axis
    Eigen::Vector3d axis = Eigen::Vector3d::Random();
    axis.normalize(); // Normalize the axis vector

    // Create an axis-angle representation
    Eigen::Vector3d axis_angle = axis * angle_rad;

    // Convert axis-angle to rotation matrix using SO3d::exp
    Sophus::SO3d rotation = Sophus::SO3d::exp(axis_angle);

    return rotation;
}
double computeError(Eigen::Matrix4d& A, Eigen::Matrix4d& B) {
    // Multiply the two matrices
    Eigen::Matrix4d product = A * B;

    // Compute the matrix logarithm
    // Eigen::Matrix4d logProduct;
    // try {
    //     logProduct = product.log();
    // } catch (const std::exception& e) {
    //     std::cerr << "Matrix logarithm computation failed: " << e.what() << std::endl;
    //     return std::numeric_limits<double>::infinity(); // Return a large number to indicate error
    // }


    // Compute the norm of the matrix logarithm
    //double norm = logProduct.norm();
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    double norm = (product-t).norm();

    return norm;
}
// Function to rotate points
open3d::geometry::PointCloud rotateOpen3dPointCloud(open3d::geometry::PointCloud point_cloud, double angle_deg, Eigen::Vector3d &translation, Eigen::Matrix4d &transformation)
{
    double angle_rad = angle_deg * M_PI / 180.0;
    Eigen::Matrix3d rotation;
    rotation = Eigen::AngleAxisd(angle_rad, Eigen::Vector3d::UnitZ());
    //Eigen::Vector3d translation = generateRandomTranslation(0.0, 0.75);
    transformation.block<3, 3>(0, 0) = rotation;
    transformation.block<3, 1>(0, 3) = translation;
    std::vector<Eigen::Vector3d> rotated_points;
    for (auto &point : point_cloud.points_)
    {
        // Rotate the point using Eigen
        Eigen::Vector3d rotated_point = rotation * point + translation;

        // Add the rotated point to the vector
        rotated_points.push_back(rotated_point);
    }

    point_cloud.points_ = rotated_points;
    return point_cloud;
}
open3d::geometry::PointCloud rotatePointCloud(open3d::geometry::PointCloud point_cloud, Eigen::Matrix4d &transformation)
{
    //Eigen::Vector3d translation = generateRandomTranslation(0.0, 0.75);
    Eigen::Matrix3d r = transformation.block<3, 3>(0, 0);
    std::vector<Eigen::Vector3d> rotated_points;
    for (auto &point : point_cloud.points_)
    {
        // Rotate the point using Eigen
        Eigen::Vector3d rotated_point = r * point;

        // Add the rotated point to the vector
        rotated_points.push_back(rotated_point);
    }

    point_cloud.points_ = rotated_points;
    return point_cloud;
}
std::pair<double, double> computeAlignmentQuality(
    open3d::geometry::PointCloud &source,
    open3d::geometry::PointCloud &target,
    Eigen::Matrix4d transformation)
{
    auto source_transformed = source.Transform(transformation);
    auto evaluation = open3d::pipelines::registration::EvaluateRegistration(
        source_transformed, target, 0.1); // distance threshold
    return {evaluation.fitness_, evaluation.inlier_rmse_};
}
// Convert points and colors to Open3D PointCloud
void convertSourceToOpen3dPointCloud(std::vector<cv::Vec3f> &points, std::vector<cv::Vec3b> &colors,
                                     open3d::geometry::PointCloud &point_cloud)
{

    for (size_t i = 0; i < points.size(); ++i)
    {
        if (points[i][0] < 65.0 && points[i][1] > -20.0)
        {
            Eigen::Vector3d point_eigen(static_cast<double>(points[i][0]),
                                        static_cast<double>(points[i][1]),
                                        static_cast<double>(points[i][2]));

            // Add point to Open3D point cloud
            point_cloud.points_.push_back(point_eigen);

            // Convert color from cv::Vec3b (BGR) to Open3D RGB format
            point_cloud.colors_.emplace_back(
                static_cast<double>(colors[i][2]) / 255.0, // B channel to R
                static_cast<double>(colors[i][1]) / 255.0, // G channel remains G
                static_cast<double>(colors[i][0]) / 255.0  // R channel to B
            );
        }
    }
}

void convertTargetToOpen3dPointCloud(std::vector<Eigen::Vector4f> &target_points,
                                     open3d::geometry::PointCloud &point_cloud)
{

    // Iterate over the last half of the points
    for (std::size_t i = 0; i < target_points.size(); ++i)
    {
        // Extract XYZ coordinates from Eigen::Vector4f (homogeneous coordinates)
        if (target_points[i][0] > 0.0)
        {
            Eigen::Vector3d point_xyz(static_cast<double>(target_points[i][0]),
                                      static_cast<double>(target_points[i][1]),
                                      static_cast<double>(target_points[i][2]));

            // Add point to Open3D point cloud

            point_cloud.points_.push_back(point_xyz);
            // No color information provided in Eigen::Vector4f, so add default color (white)
            double intensity = static_cast<double>(target_points[i][3]);
            point_cloud.colors_.push_back(Eigen::Vector3d(intensity, intensity, intensity));
        }
    }
}
// Function to visualize point clouds
void visualizePointClouds(open3d::geometry::PointCloud &source,
                          open3d::geometry::PointCloud &target,
                          std::string window_name)
{
    open3d::visualization::Visualizer visualizer;
    visualizer.CreateVisualizerWindow(window_name, 1200, 1800);
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

    merged_pcd.colors_.insert(merged_pcd.colors_.end(), pc1.colors_.begin(), pc1.colors_.end());
    merged_pcd.colors_.insert(merged_pcd.colors_.end(), pc2.colors_.begin(), pc2.colors_.end());

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
open3d::geometry::PointCloud get_stereo_open3d(cv::Mat left_img,
                                               cv::Mat right_img,
                                               cv::Mat P_rect_02,
                                               cv::Mat T_03,
                                               cv::Mat R,
                                               cv::Mat T)
{

    cv::Mat disparity;
    computeDisparity(left_img, right_img, disparity);
    cv::Mat disp_normalized;
    cv::normalize(disparity, disp_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Disparity Map", disp_normalized);
    cv::waitKey(1);

    std::vector<cv::KeyPoint> keypoints_left;
    extractKeypoints(left_img, keypoints_left);
    cv::Mat img_with_keypoints;
    cv::drawKeypoints(left_img, keypoints_left, img_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("Keypoints", img_with_keypoints);
    cv::waitKey(1);

    cv::Mat Q = cv::Mat::zeros(4, 4, CV_64F);
    compute_Q_matrix(P_rect_02, T_03, Q);

    cv::Mat points_3d;
    reconstruct3DPoints(disparity, Q, points_3d);

    std::vector<cv::Vec3f> keypoints_3d;
    std::vector<cv::Vec3b> keypoints_colors;
    cv::Mat colors;
    cv::cvtColor(left_img, colors, cv::COLOR_BGR2RGB);
    getKeypoints_3d(colors, points_3d, keypoints_3d, keypoints_colors, keypoints_left);

    cv::Mat t_cam2velo = cv::Mat::eye(4, 4, CV_64F);

    transformCam2Velodyn(R, T, t_cam2velo, keypoints_3d);

    open3d::geometry::PointCloud open3d_stereo_cam2velo;
    convertSourceToOpen3dPointCloud(keypoints_3d, keypoints_colors, open3d_stereo_cam2velo);
    return open3d_stereo_cam2velo;
}
open3d::geometry::PointCloud get_denser_stereo(cv::Mat left_img_0,
                                               cv::Mat right_img_0,
                                               cv::Mat left_img_1,
                                               cv::Mat right_img_1,
                                               cv::Mat left_img_2,
                                               cv::Mat right_img_2,
                                               cv::Mat P_rect_02,
                                               cv::Mat T_03,
                                               cv::Mat R,
                                               cv::Mat T)
{

    double tolerance = 1e-6;
    open3d::geometry::PointCloud open3d_stereo_0 = get_stereo_open3d(left_img_0, right_img_0, P_rect_02, T_03, R, T);
    open3d::io::WritePointCloud("open3d_stereo_0.ply", open3d_stereo_0);
    open3d::geometry::PointCloud open3d_stereo_1 = get_stereo_open3d(left_img_1, right_img_1, P_rect_02, T_03, R, T);
    open3d::io::WritePointCloud("open3d_stereo_1.ply", open3d_stereo_1);
    open3d::geometry::PointCloud open3d_stereo_2 = get_stereo_open3d(left_img_2, right_img_2, P_rect_02, T_03, R, T);
    open3d::io::WritePointCloud("open3d_stereo_2.ply", open3d_stereo_2);

    double t1_t0 = std::abs(open3d_stereo_1.points_[0].norm() - open3d_stereo_0.points_[0].norm());
    double t2_t0 = std::abs(open3d_stereo_2.points_[0].norm() - open3d_stereo_0.points_[0].norm());
    std::cout << "t1_0 - t0_0 stereo " << t1_t0 << std::endl;
    std::cout << "t2_0 - t0_0 stereo " << t2_t0 << std::endl;

    Eigen::Matrix4d identity_t1_t0 = Eigen::Matrix4d::Identity();
    identity_t1_t0(0, 3) = 1.34;
    runIcp(open3d_stereo_1, open3d_stereo_0, 0.2, 100, tolerance, identity_t1_t0);
    open3d::geometry::PointCloud merged_pcds_t0_t1 = mergePointClouds(open3d_stereo_0, open3d_stereo_1);

    Eigen::Matrix4d identity_t2_t0 = Eigen::Matrix4d::Identity();
    identity_t2_t0(0, 3) = 2.68;
    runIcp(open3d_stereo_2, merged_pcds_t0_t1, 0.1, 100, tolerance, identity_t2_t0);
    open3d::geometry::PointCloud merged_pcds_t0_t1_t2 = mergePointClouds(merged_pcds_t0_t1, open3d_stereo_2);
    open3d::io::WritePointCloud("stereo_t0_t1_t2.ply", merged_pcds_t0_t1_t2);
    return merged_pcds_t0_t1_t2;
}
open3d::geometry::PointCloud get_denser_stereo_from_images(std::vector<cv::Mat> leftImages,
                                                           std::vector<cv::Mat> rightImages,
                                                           size_t image_size,
                                                           cv::Mat P_rect_02,
                                                           cv::Mat T_03,
                                                           cv::Mat R,
                                                           cv::Mat T)
{

    double tolerance = 1e-6;
    double max_corr_distance = 0.1;
    double distance = 1.34;
    // std::vector<open3d::geometry::PointCloud> stereos_open3d;
    open3d::geometry::PointCloud current_stereo = get_stereo_open3d(leftImages[0], rightImages[0], P_rect_02, T_03, R, T);
    Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
    for (size_t i = 1; i < image_size; i++)
    {
        open3d::geometry::PointCloud stereo_open3d = get_stereo_open3d(leftImages[i], rightImages[i], P_rect_02, T_03, R, T);
        identity(0, 3) = distance * i;
        runIcp(stereo_open3d, current_stereo, max_corr_distance, 100, tolerance, identity);
        open3d::geometry::PointCloud merged_pcds = mergePointClouds(current_stereo, stereo_open3d);
        current_stereo = merged_pcds;
        open3d::io::WritePointCloud("stereo_until" + std::to_string(i) + ".ply", current_stereo);
        max_corr_distance += 0.1;
        std::cout << "i: " << i << std::endl;
    }

    return current_stereo;
}
open3d::geometry::PointCloud get_denser_lidar_map_from_velodyns(std::vector<std::vector<Eigen::Vector4f>> lidar_maps, size_t map_size)
{
    double tolerance = 1e-6;
    double max_corr_distance = 0.1;
    double distance = 1.34;
    open3d::geometry::PointCloud current_lidar;
    convertTargetToOpen3dPointCloud(lidar_maps[0], current_lidar);
    open3d::io::WritePointCloud("lidar_until0.ply", current_lidar);
    Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
    for (size_t i = 1; i < map_size; i++)
    {
        open3d::geometry::PointCloud lidar_open3d;
        convertTargetToOpen3dPointCloud(lidar_maps[i],lidar_open3d);
        identity(0, 3) = distance * i;
        runIcp(lidar_open3d, current_lidar, max_corr_distance, 100, tolerance, identity);
        open3d::geometry::PointCloud merged_pcds = mergePointClouds(current_lidar, lidar_open3d);
        current_lidar = merged_pcds;
        
        max_corr_distance += 0.1;
        std::cout << "i: " << i << std::endl;open3d::io::WritePointCloud("lidar_until" + std::to_string(i) + ".ply", current_lidar);
    }
    return current_lidar;
}

// Custom function to refine correspondences
std::vector<Eigen::Vector2i> refine_correspondences(
    const open3d::geometry::PointCloud &source,
    const open3d::geometry::PointCloud &target,
    const std::vector<Eigen::Vector2i> &initial_correspondences)
{

    std::vector<Eigen::Vector2i> refined_correspondences;

    // Example custom refinement strategy: Filter out correspondences based on some criterion
    // This is a placeholder; you should replace it with your actual refinement logic
    for (const auto &correspondence : initial_correspondences)
    {
        const Eigen::Vector3d &source_point = source.points_[correspondence(0)];
        const Eigen::Vector3d &target_point = target.points_[correspondence(1)];

        // Example criterion: Only keep correspondences where the source and target points are within a certain distance
        double distance = (source_point - target_point).norm();
        if (distance < 0.05)
        { // Replace 0.05 with your threshold
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
    std::cout << "voxel_grid size: " << voxel_grid->GetVoxels().size() << std::endl;

    // each point in the point cloud is in a voxel
    std::unordered_map<Eigen::Vector3d, Eigen::Vector3i, open3d::utility::hash_eigen<Eigen::Vector3d>> point_to_voxel_map;
    // each voxel has its Voxel data (T, standart deviation vector, number of points)
    std::unordered_map<Eigen::Vector3i, VoxelData, open3d::utility::hash_eigen<Eigen::Vector3i>> voxel_to_data_map;
    computeVoxelMaps(*voxel_grid, lidar_t0_t1_t2, point_to_voxel_map, voxel_to_data_map);

    std::cout << "point_to_voxel_map size: " << point_to_voxel_map.size() << std::endl;
    std::cout << "voxel_to_data_map size: " << voxel_to_data_map.size() << std::endl;

    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    open3d::geometry::PointCloud rotated_lidar_map_t0 = rotateLidarMap(lidar_map_t0, 2, "rotated_lidar_t0.ply");
    open3d::geometry::PointCloud rotated_lidar_map_t0_2 = rotated_lidar_map_t0;

    auto registration_result = open3d::pipelines::registration::RegistrationICP(
        rotated_lidar_map_t0, lidar_t0_t1_t2, 0.1, t,
        open3d::pipelines::registration::TransformationEstimationPointToPoint(),
        open3d::pipelines::registration::ICPConvergenceCriteria(100, tolerance));

    write_point_cloud(rotated_lidar_map_t0.Transform(registration_result.transformation_).points_, lidar_map_t0, "rotated_lidar_with_normal_correspondences.ply");
    std::cout << "Transformation matrix for rotated lidar after icp:" << std::endl;
    std::cout << registration_result.transformation_ << std::endl;

    open3d::pipelines::registration::CorrespondenceSet correspondences = registration_result.correspondence_set_;
    std::cout << "correspondences size: " << correspondences.size() << std::endl;
    // visualize_correspondences(rotated_lidar_map_t0,lidar_t0_t1_t2,correspondences);
    open3d::pipelines::registration::CorrespondenceSet refined_correspondences = refineCorrespondences(
        correspondences,
        rotated_lidar_map_t0_2,
        lidar_t0_t1_t2,
        voxel_to_data_map,
        point_to_voxel_map,
        N_min,                  // N_min
        N_standard_devitation); // N_standard_deviation
    std::cout << "refined_correspondences size: " << refined_correspondences.size() << std::endl;

    // Estimate the transformation matrix using point-to-point transformation estimation
    open3d::pipelines::registration::TransformationEstimationPointToPoint estimation;
    Eigen::Matrix4d transformation = estimation.ComputeTransformation(
        rotated_lidar_map_t0_2, lidar_t0_t1_t2, refined_correspondences);
    // visualize_correspondences(rotated_lidar_map_t0_2,lidar_t0_t1_t2,refined_correspondences);
    std::cout << "Transformation matrix using refined correspondences:" << std::endl;
    std::cout << transformation << std::endl;
    write_point_cloud(rotated_lidar_map_t0_2.Transform(transformation).points_, lidar_map_t0, "rotated_lidar_with_refined_correspondences.ply");
}
void savePointCloudAsImage(const open3d::geometry::PointCloud& point_cloud, std::string filename) {
  auto vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("Point Cloud Viewer", 640, 480);
    vis->AddGeometry(std::make_shared<open3d::geometry::PointCloud>(point_cloud));
    vis->UpdateGeometry();
    vis->PollEvents();
    vis->UpdateRender();
    
    // Wait a bit to ensure rendering is complete
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Capture the screen image
    vis->CaptureScreenImage(filename);
    
    vis->DestroyVisualizerWindow();
}
void correspondences_filtering(double voxel_size,
                               int N_min,
                               int N_standard_devitation,
                               open3d::geometry::PointCloud &stereo3d,
                               open3d::geometry::PointCloud &lidar_t0_t1_t2,
                               int rotation_degree,
                               Eigen::Vector3d translation,
                               double &error_corr,
                               double &error_icp,
                               double max_correspondence_distance,
                               int max_iterations,
                               double tolerance)
{

    // Voxelize the source point cloud
    open3d::io::WritePointCloud("frame_lidar" + std::to_string(rotation_degree) + ".ply", stereo3d);
    open3d::io::WritePointCloud("frame_stereo" + std::to_string(rotation_degree) + ".ply", lidar_t0_t1_t2);
    auto voxel_grid = open3d::geometry::VoxelGrid::CreateFromPointCloud(lidar_t0_t1_t2, voxel_size);
    open3d::io::WriteVoxelGrid("lidar_voxel_grid.ply", *voxel_grid);
    std::cout << "voxel_grid size: " << voxel_grid->GetVoxels().size() << std::endl;

    // each point in the point cloud is in a voxel
    std::unordered_map<Eigen::Vector3d, Eigen::Vector3i, open3d::utility::hash_eigen<Eigen::Vector3d>> point_to_voxel_map;
    // each voxel has its Voxel data (T, standart deviation vector, number of points)
    std::unordered_map<Eigen::Vector3i, VoxelData, open3d::utility::hash_eigen<Eigen::Vector3i>> voxel_to_data_map;
    computeVoxelMaps(*voxel_grid, lidar_t0_t1_t2, point_to_voxel_map, voxel_to_data_map);

    std::cout << "point_to_voxel_map size: " << point_to_voxel_map.size() << std::endl;
    std::cout << "voxel_to_data_map size: " << voxel_to_data_map.size() << std::endl;

    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d noise_t = Eigen::Matrix4d::Identity();
    //Eigen::Vector3d translation(0.1, 0.1, 0.1);
    open3d::geometry::PointCloud noisy_stereo3d = rotateOpen3dPointCloud(stereo3d, rotation_degree, translation, noise_t);
    open3d::io::WritePointCloud("frame_noisy_stereo" + std::to_string(rotation_degree) + ".ply", noisy_stereo3d);

    auto registration_result = open3d::pipelines::registration::RegistrationICP(
        noisy_stereo3d, lidar_t0_t1_t2, max_correspondence_distance, t,
        open3d::pipelines::registration::TransformationEstimationPointToPoint(),
        open3d::pipelines::registration::ICPConvergenceCriteria(max_iterations, tolerance));

    open3d::pipelines::registration::CorrespondenceSet correspondences = registration_result.correspondence_set_;
    std::cout << "correspondences size: " << correspondences.size() << std::endl;
    drawCorrespondences(noisy_stereo3d, lidar_t0_t1_t2, correspondences);

    open3d::pipelines::registration::CorrespondenceSet refined_correspondences = refineCorrespondences(
        correspondences,
        noisy_stereo3d,
        lidar_t0_t1_t2,
        voxel_to_data_map,
        point_to_voxel_map,
        N_min,                  // N_min
        N_standard_devitation); // N_standard_deviation
    std::cout << "refined_correspondences size: " << refined_correspondences.size() << std::endl;
    drawRefinedCorrespondences(noisy_stereo3d, lidar_t0_t1_t2, correspondences, refined_correspondences);

    // Estimate the transformation matrix using point-to-point transformation estimation
    open3d::pipelines::registration::TransformationEstimationPointToPoint estimation;
    Eigen::Matrix4d correspondence_filtering_transformation = estimation.ComputeTransformation(
        noisy_stereo3d, lidar_t0_t1_t2, refined_correspondences);
    open3d::geometry::PointCloud pcd_gif = noisy_stereo3d;
    open3d::io::WritePointCloud("frame_transformed" + std::to_string(rotation_degree) + ".ply", pcd_gif.Transform(correspondence_filtering_transformation));
    open3d::geometry::PointCloud transformed_noisy_stereo_icp = noisy_stereo3d;
    open3d::geometry::PointCloud transformed_noisy_stereo_corr_filtering = noisy_stereo3d;
    // transformed_noisy_stereo_icp.Transform(registration_result.transformation_);
    // transformed_noisy_stereo_corr_filtering.Transform(correspondence_filtering_transformation);
    error_corr = ((correspondence_filtering_transformation*noise_t)-t).norm();
    error_icp = ((registration_result.transformation_*noise_t)-t).norm();
    auto [fitness1, rmse1] = computeAlignmentQuality(transformed_noisy_stereo_corr_filtering, lidar_t0_t1_t2, correspondence_filtering_transformation);
    auto [fitness2, rmse2] = computeAlignmentQuality(transformed_noisy_stereo_icp, lidar_t0_t1_t2, registration_result.transformation_);

    std::cout << "Fitness corr_refinement: " << fitness1 << ", RMSE corr_refinement: " << rmse1 << std::endl;
    std::cout << "Fitness icp : " << fitness2 << ", RMSE icp: " << rmse2 << std::endl;
    std::cout << "error corr_refinement : " << error_corr << std::endl;
    std::cout << "error icp : " << error_icp << std::endl;
}
void correspondences_filtering_plot(double voxel_size,
                               int N_min,
                               int N_standard_devitation,
                               open3d::geometry::PointCloud &stereo3d,
                               open3d::geometry::PointCloud &lidar_t0_t1_t2,
                               Eigen::Matrix4d &r,
                               Eigen::Matrix4d &r_t,
                               Eigen::Matrix4d &r_icp,
                               double max_correspondence_distance,
                               int max_iterations,
                               double tolerance)
{


    auto voxel_grid = open3d::geometry::VoxelGrid::CreateFromPointCloud(lidar_t0_t1_t2, voxel_size);
    open3d::io::WriteVoxelGrid("lidar_voxel_grid.ply", *voxel_grid);

    // each point in the point cloud is in a voxel
    std::unordered_map<Eigen::Vector3d, Eigen::Vector3i, open3d::utility::hash_eigen<Eigen::Vector3d>> point_to_voxel_map;
    // each voxel has its Voxel data (T, standart deviation vector, number of points)
    std::unordered_map<Eigen::Vector3i, VoxelData, open3d::utility::hash_eigen<Eigen::Vector3i>> voxel_to_data_map;
    computeVoxelMaps(*voxel_grid, lidar_t0_t1_t2, point_to_voxel_map, voxel_to_data_map);

    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();

    open3d::geometry::PointCloud noisy_stereo3d = rotatePointCloud(stereo3d, r);

    auto registration_result = open3d::pipelines::registration::RegistrationICP(
        noisy_stereo3d, lidar_t0_t1_t2, max_correspondence_distance, t,
        open3d::pipelines::registration::TransformationEstimationPointToPoint(),
        open3d::pipelines::registration::ICPConvergenceCriteria(max_iterations, tolerance));
    r_icp = registration_result.transformation_;
    open3d::pipelines::registration::CorrespondenceSet correspondences = registration_result.correspondence_set_;
    std::cout << "correspondences size: " << correspondences.size() << std::endl;
    //drawCorrespondences(noisy_stereo3d, lidar_t0_t1_t2, correspondences);

    open3d::pipelines::registration::CorrespondenceSet refined_correspondences = refineCorrespondences(
        correspondences,
        noisy_stereo3d,
        lidar_t0_t1_t2,
        voxel_to_data_map,
        point_to_voxel_map,
        N_min,                  // N_min
        N_standard_devitation); // N_standard_deviation
    std::cout << "refined_correspondences size: " << refined_correspondences.size() << std::endl;
    //drawRefinedCorrespondences(noisy_stereo3d, lidar_t0_t1_t2, correspondences, refined_correspondences);

    // Estimate the transformation matrix using point-to-point transformation estimation
    open3d::pipelines::registration::TransformationEstimationPointToPoint estimation;
    Eigen::Matrix4d correspondence_filtering_transformation = estimation.ComputeTransformation(
        noisy_stereo3d, lidar_t0_t1_t2, refined_correspondences);

    r_t = correspondence_filtering_transformation;

}