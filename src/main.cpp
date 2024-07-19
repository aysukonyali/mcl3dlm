#include <opencv2/opencv.hpp>
#include <open3d/Open3D.h>
#include "open3d/geometry/PointCloud.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
// #include <opencv2/viz.hpp>
// #include <opencv2/viz/vizcore.hpp>
// #include <opencv2/viz/types.hpp>
// #include <opencv2/viz/widgets.hpp>
// #include <opencv2/viz/viz3d.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <opencv2/core/base.hpp>
#include "local_reconstruction.h"
#include "alignment.h"
#include "data_association.h"


void visualize_correspondences(
    const open3d::geometry::PointCloud &source,
    const open3d::geometry::PointCloud &target,
    const open3d::pipelines::registration::CorrespondenceSet &correspondences) 
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
        line_set->lines_.emplace_back(correspondence(0), source.points_.size() + correspondence(1));
        line_set->colors_.emplace_back(0.0, 0.0, 1.0);  // Blue color for correspondences
    }

    // Add the LineSet to the geometries vector
    geometries.push_back(line_set);

    // Visualize
    open3d::visualization::DrawGeometries(geometries, "Correspondences", 800, 600);
}
int main(int argc, char **argv)
{
    std::string binFile0 = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000.bin";
    std::string binFile1 = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000001.bin";
    std::string binFile2 = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000002.bin";
    std::vector<Eigen::Vector4f> lidar_map_t0 = readLidarMap(binFile0, "3d_lidar_map_t0.ply");
    std::vector<Eigen::Vector4f> lidar_map_t1 = readLidarMap(binFile1, "3d_lidar_map_t1.ply");
    std::vector<Eigen::Vector4f> lidar_map_t2 = readLidarMap(binFile2, "3d_lidar_map_t2.ply");

    cv::Mat left_img = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000_l.png", cv::IMREAD_COLOR);
    cv::Mat right_img = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000_r.png", cv::IMREAD_COLOR);
    std::cout << "left img size " << left_img.size() << std::endl;

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

    std::vector<cv::KeyPoint> keypoints_left;
    extractKeypoints(left_img, keypoints_left);
    cv::Mat img_with_keypoints;
    cv::drawKeypoints(left_img, keypoints_left, img_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("Keypoints", img_with_keypoints);
    cv::waitKey(0);


    // cv::Mat filtered_disparity;
    // applyWLSFilter(left_img, right_img, filtered_disparity);
    // std::cout << "disparity size " << filtered_disparity.size() << std::endl;
    // cv::normalize(filtered_disparity, filtered_disparity, 0, 255, cv::NORM_MINMAX, CV_8U);
    // //cv::normalize(filtered_disparity, filtered_disparity, 0, 50, cv::NORM_MINMAX, CV_32F);
    // cv::imshow("Disparity Map", filtered_disparity);
    // cv::waitKey(0); // Wait indefinitely for a key press

    cv::Mat Q = cv::Mat::zeros(4, 4, CV_64F);
    //cv::Mat Q = cv::Mat::zeros(4, 4, CV_32F);
    compute_Q_matrix(P_rect_02, T_03, Q);
    std::cout << "Q matrix " << std::endl;
    printMat(Q);

    cv::Mat points_3d;
    //reconstruct3DPoints(filtered_disparity, Q, points_3d);
    reconstruct3DPoints(disparity, Q, points_3d);
    //points_3d *= 16.0;
    // cv::viz::Viz3d window("3D Point Cloud");

    // // Create a point cloud widget
    // cv::viz::WCloud cloud_widget(points_3d);

    // // Add the point cloud widget to the window
    // window.showWidget("Point Cloud", cloud_widget);

    // // Start event loop
    // window.spin();

    std::vector<cv::Vec3f> keypoints_3d;
    std::vector<cv::Vec3b> keypoints_colors;
    cv::Mat colors;
    cv::cvtColor(left_img, colors, cv::COLOR_BGR2RGB);
    getKeypoints_3d(colors, points_3d, keypoints_3d, keypoints_colors, keypoints_left);

    //cv::Mat colors;
    //cv::cvtColor(left_img, colors, cv::COLOR_BGR2RGB);
    std::vector<cv::Vec3f> all_3d_points;
    std::vector<cv::Vec3b> all_colors;
    getAllPoints_3d(colors, points_3d, all_3d_points, all_colors);

    // Rotate points to add noise
    std::vector<cv::Vec3f> icp_source_keypoints = all_3d_points;
    std::vector<cv::Vec3f> noisy_stereo = rotatePoints(icp_source_keypoints, 3.0);
    std::string noisy_points = "noisy_stereo.ply";
    write2PLY(noisy_stereo, all_colors, noisy_points);

    cv::Mat t_cam2velo = cv::Mat::eye(4, 4, CV_64F);
    transformCam2Velodyn(R, T, t_cam2velo, all_3d_points);
    std::string point_cloud_all_points_cam_to_velodyn = "point_cloud_all_points_cam_to_velodyn.ply";
    write2PLY(all_3d_points, all_colors, point_cloud_all_points_cam_to_velodyn);

    transformCam2Velodyn(R, T, t_cam2velo, keypoints_3d);
    std::string point_cloud_key_points_cam_to_velodyn = "point_cloud_key_points_cam_to_velodyn.ply";
    write2PLY(keypoints_3d, keypoints_colors, point_cloud_key_points_cam_to_velodyn);

    open3d::geometry::PointCloud open3d_stereo_cam2velo;
    convertSourceToOpen3dPointCloud(keypoints_3d, keypoints_colors, open3d_stereo_cam2velo);
    write2PLY2(open3d_stereo_cam2velo.points_, keypoints_colors, "open3d_stereo_cam2velo.ply");
    // convertSourceToOpen3dPointCloud(all_3d_points, all_colors, open3d_stereo_cam2velo);
    // write2PLY2(open3d_stereo_cam2velo.points_, all_colors, "open3d_stereo_cam2velo.ply");

    // Set parameters
    // Run an example icp with noisy stereo and lidar

    open3d::geometry::PointCloud open3d_noisy_stereo, open3d_lidar_map_t0;
    convertSourceToOpen3dPointCloud(noisy_stereo, all_colors, open3d_noisy_stereo);
    convertTargetToOpen3dPointCloud(lidar_map_t0, open3d_lidar_map_t0);

    double max_correspondence_distance = 0.02;
    int max_iterations = 100;
    double tolerance = 1e-6;

    Eigen::Matrix4d initial_transform;
    cvMatToEigenMat(t_cam2velo, initial_transform);
    runIcp(open3d_noisy_stereo, open3d_lidar_map_t0, max_correspondence_distance, max_iterations, tolerance, initial_transform);
    write2PLY2(open3d_noisy_stereo.points_, all_colors, "icp_transformed_noisy_stereo.ply");

    // get denser lidar map from t0 t1 and t2
    open3d::geometry::PointCloud lidar_t0_t1_t2 = get_denser_lidar_map(lidar_map_t0, lidar_map_t1, lidar_map_t2, open3d_stereo_cam2velo);


    // run icp and then refine correspondences
    //run_improved_icp(0.5,15,5,lidar_map_t0,lidar_t0_t1_t2,0.02,2000,tolerance);



    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    auto registration_result = open3d::pipelines::registration::RegistrationICP(
        open3d_stereo_cam2velo, lidar_t0_t1_t2, 0.1, t,
        open3d::pipelines::registration::TransformationEstimationPointToPoint(),
        open3d::pipelines::registration::ICPConvergenceCriteria(100, tolerance));

    std::cout << "registration_result corr: "<< registration_result.correspondence_set_.size() << std::endl; 
    std::cout << "T: "<< registration_result.transformation_ << std::endl;   

    // auto voxel_grid = open3d::geometry::VoxelGrid::CreateFromPointCloud(lidar_t0_t1_t2, 1.5);
    // open3d::io::WriteVoxelGrid("lidar_voxel_grid.ply", *voxel_grid);
    // std::cout << "voxel_grid size: "<< voxel_grid->GetVoxels().size() << std::endl;
    
    // std::unordered_map<Eigen::Vector3d, Eigen::Vector3i, open3d::utility::hash_eigen<Eigen::Vector3d>> point_to_voxel_map;
    // std::unordered_map<Eigen::Vector3i, VoxelData, open3d::utility::hash_eigen<Eigen::Vector3i>> voxel_to_data_map;
    // computeVoxelMaps(*voxel_grid, lidar_t0_t1_t2, point_to_voxel_map, voxel_to_data_map);

    // open3d::pipelines::registration::CorrespondenceSet correspondences = registration_result.correspondence_set_;
    // std::cout << "correspondences size: "<< correspondences.size() << std::endl;
    // open3d::pipelines::registration::CorrespondenceSet refined_correspondences = refineCorrespondences(
    //     correspondences,
    //     open3d_stereo_cam2velo,
    //     lidar_t0_t1_t2,
    //     voxel_to_data_map,
    //     point_to_voxel_map,
    //     15, // N_min
    //     5); // N_standard_deviation
    // std::cout << "refined_correspondences size: "<< refined_correspondences.size() << std::endl;
    

    return 0;
}

// create a bigger lidar map from 2-3 frames
// gıven tımespamps t1 t2 t3 we know the vehicle poses from calıbratıon
// fınd realtıve poses to t1
// transform lidars to camera and then to timestamp t1
// This gives denser 3D reconstructıon and then implement fılterıng from the paper and run ICP
// May be ıt wouldbe better to fılter stereo and not lıder (Up to the test)
