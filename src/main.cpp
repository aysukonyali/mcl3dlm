#include <opencv2/opencv.hpp>
#include <open3d/Open3D.h>
#include "open3d/geometry/PointCloud.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <opencv2/core/base.hpp>
#include "local_reconstruction.h"
#include "alignment.h"


// Function to compute the covariance matrix for a set of points
Eigen::Matrix3d computeCovarianceMatrix(const std::vector<Eigen::Vector3d> &points) {
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();

    for (const auto &point : points) {
        mean += point;
    }
    mean /= static_cast<double>(points.size());

    for (const auto &point : points) {
        Eigen::Vector3d centered = point - mean;
        covariance += centered * centered.transpose();
    }
    covariance /= static_cast<double>(points.size());

    return covariance;
}
// // Function to compute the eigenvectors with the two largest eigenvalues inside each voxel
// void computeEigenVectorsInVoxels(const open3d::geometry::VoxelGrid &voxel_grid,
//                                  std::unordered_map<Eigen::Vector3i, std::pair<Eigen::Vector3d, Eigen::Vector3d>, open3d::utility::hash_eigen<Eigen::Vector3i>> &voxel_eigenvectors) {
//     for (const auto &voxel : voxel_grid.voxels_) {
//         std::vector<Eigen::Vector3d> points;
//         for (const auto &point : voxel_grid.GetPointIndicesInVoxel(voxel.first)) {
//             points.push_back(voxel_grid.points_[point]);
//         }

//         if (points.size() < 3) {
//             continue; // Skip if not enough points to form a covariance matrix
//         }

//         Eigen::Matrix3d covariance = computeCovarianceMatrix(points);

//         // Perform eigenvalue decomposition
//         Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
//         if (solver.info() != Eigen::Success) {
//             continue;
//         }

//         // Sort eigenvalues and select the two largest eigenvectors
//         Eigen::Matrix3d eigenvectors = solver.eigenvectors();
//         Eigen::Vector3d eigenvalues = solver.eigenvalues();

//         std::vector<std::pair<double, Eigen::Vector3d>> eigen_pairs;
//         for (int i = 0; i < 3; ++i) {
//             eigen_pairs.emplace_back(eigenvalues[i], eigenvectors.col(i));
//         }
//         std::sort(eigen_pairs.rbegin(), eigen_pairs.rend(), [](const auto &a, const auto &b) {
//             return a.first < b.first;
//         });

//         voxel_eigenvectors[voxel.first] = {eigen_pairs[0].second, eigen_pairs[1].second};
//     }
// }


int main(int argc, char **argv)
{
    std::string binFile0 = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000.bin";
    std::string binFile1 = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000001.bin";
    std::string binFile2 = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000002.bin";
    std::vector<Eigen::Vector4f> lidar_map_t0 = readLidarMap(binFile0,"3d_lidar_map_t0.ply");
    std::vector<Eigen::Vector4f> lidar_map_t1 = readLidarMap(binFile1,"3d_lidar_map_t1.ply");
    std::vector<Eigen::Vector4f> lidar_map_t2 = readLidarMap(binFile2,"3d_lidar_map_t2.ply");

    cv::Mat left_img = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000_l.png", cv::IMREAD_COLOR);
    cv::Mat right_img = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000_r.png", cv::IMREAD_COLOR);
    std::cout << "left img size " << left_img.size() << std::endl;

    std::vector<cv::KeyPoint> keypoints_left;
    cv::Mat descriptors_left;
    extractKeypointsAndDescriptors(left_img, keypoints_left, descriptors_left);

    std::string calibFile_cam_cam = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/calib_cam_to_cam.txt";
    std::string calibFile_velo_cam = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/calib_velo_to_cam.txt";
    cv::Mat P_rect_02, T_03, R, T;

    readCalibrationFile(calibFile_cam_cam, P_rect_02, T_03, R, T);
    readCalibrationFile(calibFile_velo_cam, P_rect_02, T_03, R, T);

    cv::Mat disparity;
    computeDisparity(left_img, right_img, disparity);
    std::cout << "disparity size " << disparity.size() << std::endl;
    cv::Mat disp_normalized;
    cv::normalize(disparity, disp_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Disparity Map", disp_normalized);
    cv::waitKey(0); // Wait indefinitely for a key press


    cv::Mat filtered_disparity;
    applyWLSFilter(left_img, right_img, filtered_disparity);
    std::cout << "disparity size " << filtered_disparity.size() << std::endl;
    cv::normalize(filtered_disparity, filtered_disparity, 0, 255, cv::NORM_MINMAX, CV_8U);
    //filtered_disparity *= 16.0;
    cv::imshow("Disparity Map", filtered_disparity);
    cv::waitKey(0); // Wait indefinitely for a key press


    cv::Mat Q = cv::Mat::zeros(4, 4, CV_64F);
    compute_Q_matrix(P_rect_02, T_03, Q);
    std::cout << "Q matrix " << std::endl;
    printMat(Q);

    cv::Mat points_3d;
    reconstruct3DPoints(filtered_disparity, Q, points_3d);
    points_3d *= 16.0;

    std::vector<cv::Vec3f> keypoints_3d;
    std::vector<cv::Vec3b> keypoints_colors;
    cv::Mat colors;
    cv::cvtColor(left_img, colors, cv::COLOR_BGR2RGB);
    getKeypoints_3d(colors, points_3d, keypoints_3d, keypoints_colors, keypoints_left);
    //std::vector<cv::Vec3f> icp_source_keypoints = keypoints_3d;

    cv::Mat img_with_keypoints;
    cv::drawKeypoints(left_img, keypoints_left, img_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("Keypoints", img_with_keypoints);
    cv::waitKey(0);


    std::vector<cv::Vec3f> all_3d_points;
    std::vector<cv::Vec3b> all_colors;
    open3d::geometry::PointCloud open3d_stereo;
    getAllPoints_3d(colors, points_3d, all_3d_points, all_colors);
    std::vector<cv::Vec3f> icp_source_keypoints = all_3d_points;
    

    cv::Mat t_cam2velo = cv::Mat::eye(4, 4, CV_64F);
    transformCam2Velodyn(R, T, t_cam2velo, keypoints_3d);
    std::string point_cloud_cam_to_velodyn = "point_cloud_keypoints_cam_to_velodyn.ply";
    write2PLY(keypoints_3d, keypoints_colors, point_cloud_cam_to_velodyn);

    transformCam2Velodyn(R, T, t_cam2velo, all_3d_points);
    std::string point_cloud_all_points_cam_to_velodyn = "point_cloud_all_points_cam_to_velodyn.ply";
    write2PLY(all_3d_points, all_colors, point_cloud_all_points_cam_to_velodyn);
    convertSourceToOpen3dPointCloud(all_3d_points, all_colors, open3d_stereo);
    write2PLY2(open3d_stereo.points_, all_colors, "open3d_stereo.ply");

    // Rotate points to add noise
    std::vector<cv::Vec3f> noisy_keypoints_3d = rotatePoints(icp_source_keypoints, 3.0);
    std::string noisy_keypoints = "noisy_keypoints.ply";
    write2PLY(noisy_keypoints_3d, keypoints_colors, noisy_keypoints);

    // Convert to Open3D PointCloud
    open3d::geometry::PointCloud source, target;
    convertSourceToOpen3dPointCloud(noisy_keypoints_3d, keypoints_colors, source);
    convertTargetToOpen3dPointCloud(lidar_map_t0, target);


    // Set parameters
    double max_correspondence_distance = 0.02; 
    int max_iterations = 100;                 
    double tolerance = 1e-6;              

    // Run ICP
    Eigen::Matrix4d initial_transform;
    cvMatToEigenMat(t_cam2velo, initial_transform);
    runIcp(source, target, max_correspondence_distance, max_iterations, tolerance, initial_transform);
    write2PLY2(source.points_,keypoints_colors,"icp_transformed_noisy_keypoints.ply");
    

    // Convert to Open3D PointCloud
    open3d::geometry::PointCloud open3d_lidar_map_t0;
    convertTargetToOpen3dPointCloud(lidar_map_t0, open3d_lidar_map_t0);
    open3d::geometry::PointCloud source_lidar_map_t1;
    Eigen::Matrix4d identity_t1_t0 =Eigen::Matrix4d::Identity();
    identity_t1_t0(0,3) = 1.4493;
    std::cout << "t1_0 - t0_0 "<< lidar_map_t0[0].norm()-lidar_map_t1[0].norm() << std::endl;
    std::cout << "t2_0 - t0_0 "<< lidar_map_t0[0].norm()-lidar_map_t2[0].norm() << std::endl;
    //identity_t1_t0(0,3) = 1.3;
    convertTargetToOpen3dPointCloud(lidar_map_t1, source_lidar_map_t1);
    runIcp(source_lidar_map_t1, target, 0.1, 100, tolerance, identity_t1_t0);
    write_point_cloud(source_lidar_map_t1.points_, lidar_map_t1, "icp_lidar_map_t0_t1.ply");
    open3d::geometry::PointCloud merged_pcds_t0_t1 = mergePointClouds(open3d_lidar_map_t0,source_lidar_map_t1);
    std::vector<Eigen::Vector4f> merged_colors_t0_t1 = mergeColors(lidar_map_t0,lidar_map_t1);
    write_point_cloud(merged_pcds_t0_t1.points_, merged_colors_t0_t1, "merged_pcds_t0_t1.ply");

    Eigen::Matrix4d identity =Eigen::Matrix4d::Identity();
    runIcp(merged_pcds_t0_t1, open3d_stereo, 0.01, 100, tolerance, identity);
    write_point_cloud(merged_pcds_t0_t1.points_, merged_colors_t0_t1, "icp_lidar_t0_t1_stereo.ply");

    open3d::geometry::PointCloud source_lidar_map_t2;
    Eigen::Matrix4d identity_t2_t0 =Eigen::Matrix4d::Identity();
    identity_t2_t0(0,3) = 1.3163+1.4493;
    //identity_t2_t0(0,3) = 2.6;
    convertTargetToOpen3dPointCloud(lidar_map_t2, source_lidar_map_t2);
    runIcp(source_lidar_map_t2, merged_pcds_t0_t1, 0.2, 100, tolerance, identity_t2_t0);
    write_point_cloud(source_lidar_map_t2.points_, lidar_map_t2, "icp_lidar_map_t0_t1_t2.ply");
    open3d::geometry::PointCloud merged_pcds_t0_t1_t2 = mergePointClouds(merged_pcds_t0_t1,source_lidar_map_t2);
    std::vector<Eigen::Vector4f> merged_colors_t0_t1_t2 = mergeColors(merged_colors_t0_t1,lidar_map_t2);
    runIcp(merged_pcds_t0_t1_t2, open3d_stereo, 0.01,100, tolerance, identity);
    write_point_cloud(merged_pcds_t0_t1_t2.points_, merged_colors_t0_t1_t2, "icp_lidar_t0_t1_t2_stereo.ply");


    // open3d::geometry::PointCloud merged_pcds = mergePointClouds(source_lidar_map_t1,source_lidar_map_t2);
    // std::vector<Eigen::Vector4f> merged_colors = mergeColors(lidar_map_t1,lidar_map_t2);
    // write_point_cloud(merged_pcds.points_, merged_colors, "merged_t1_t2.ply");

    // Voxelization resolution
    //double voxel_size = 1.5;

    // Voxelize the source point cloud
    // auto voxel_grid = open3d::geometry::VoxelGrid::CreateFromPointCloud(target, voxel_size);
    // open3d::io::WriteVoxelGrid("lidar_voxel_grid.ply", *voxel_grid);
    // open3d::geometry::Voxel v = voxel_grid->getV
    return 0;
}


// create a bigger lidar map from 2-3 frames
// gıven tımespamps t1 t2 t3 we know the vehicle poses from calıbratıon 
// fınd realtıve poses to t1 
// transform lidars to camera and then to timestamp t1 
// This gives denser 3D reconstructıon and then implement fılterıng from the paper and run ICP 
// May be ıt wouldbe better to fılter stereo and not lıder (Up to the test)
