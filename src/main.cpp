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
#include <filesystem>
#include <opencv2/core/base.hpp>
#include "local_reconstruction.h"
#include "alignment.h"
#include "data_association.h"

// struct TriangulationResult {
//     std::vector<cv::Vec3f> points3D;
//     std::vector<cv::Vec3b> colors; // Store color information
//     std::vector<cv::Point2f> pointsLeft;
//     std::vector<cv::Point2f> pointsRight;
// };
// void computeDescriptors(cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
//     cv::Ptr<cv::Feature2D> extractor = cv::SIFT::create();
//     extractor->compute(img, keypoints, descriptors);
// }
// void filterMatchesWithRatioTest(const std::vector<std::vector<cv::DMatch>>& knn_matches, std::vector<cv::DMatch>& good_matches, float ratio_thresh = 0.75) {
//     for (size_t i = 0; i < knn_matches.size(); ++i) {
//         if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
//             good_matches.push_back(knn_matches[i][0]);
//         }
//     }
// }
// void detectAndMatchKeypoints(cv::Mat& imgLeft, cv::Mat& imgRight,
//                              std::vector<cv::KeyPoint>& keypointsLeft, std::vector<cv::KeyPoint>& keypointsRight,
//                              std::vector<cv::DMatch>& matches) {

//     cv::Mat descriptorsLeft, descriptorsRight;

//     extractKeypoints(imgLeft,keypointsLeft);
//     extractKeypoints(imgRight, keypointsRight);
//     computeDescriptors(imgLeft,keypointsLeft,descriptorsLeft);
//     computeDescriptors(imgRight, keypointsRight,descriptorsRight);

//     cv::Mat descLeft32F, descRight32F;
//     descriptorsLeft.convertTo(descLeft32F, CV_32F);
//     descriptorsRight.convertTo(descRight32F, CV_32F);

//     cv::BFMatcher matcher(cv::NORM_L2);
//     matcher.match(descLeft32F, descRight32F, matches);

//         // KNN match to get two nearest neighbors
//     std::vector<std::vector<cv::DMatch>> knn_matches;
//     matcher.knnMatch(descLeft32F, descRight32F, knn_matches, 2);

//     // Apply ratio test to filter matches
//     filterMatchesWithRatioTest(knn_matches, matches);

//     cv::Mat imgMatches;
//     cv::drawMatches(imgLeft, keypointsLeft, imgRight, keypointsRight, matches, imgMatches);
//     cv::imshow("Matches", imgMatches);
//     cv::waitKey(0);
// }
// TriangulationResult triangulatePoints(std::vector<cv::KeyPoint>& keypointsLeft,
//                                       std::vector<cv::KeyPoint>& keypointsRight,
//                                       std::vector<cv::DMatch>& matches,
//                                       cv::Mat& P1, cv::Mat& P2,
//                                       cv::Mat& imgLeft) {
//     TriangulationResult result;

//     for (auto& match : matches) {
//         result.pointsLeft.push_back(keypointsLeft[match.queryIdx].pt);
//         result.pointsRight.push_back(keypointsRight[match.trainIdx].pt);
//     }

//     // cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F); // Projection matrix for the first camera
//     // cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F); // Projection matrix for the second camera
//     // R.copyTo(P2(cv::Rect(0, 0, 3, 3)));
//     // T.copyTo(P2(cv::Rect(3, 0, 1, 3)));

//     cv::Mat points4D;
//     cv::triangulatePoints(P1, P2, result.pointsLeft, result.pointsRight, points4D);

//     for (int i = 0; i < points4D.cols; i++) {
//         cv::Mat x = points4D.col(i);
//         x /= x.at<float>(3);
//         if (cv::checkRange(x, true, nullptr, -1e4, 1e4))
//         {
//         result.points3D.push_back(cv::Vec3f(x.at<float>(0), x.at<float>(1), x.at<float>(2)));

//         // // Extract color information from the left image
//         int x2d = static_cast<int>(result.pointsLeft[i].x);
//         int y2d = static_cast<int>(result.pointsLeft[i].y);
//         result.colors.push_back(imgLeft.at<cv::Vec3b>(y2d, x2d));
//         }
//     }
//     std::string offff = "puff.ply";
//     write2PLY(result.points3D,result.colors,offff);
//     return result;
// }

// std::shared_ptr<open3d::geometry::PointCloud> createPointCloudOpen3D(std::vector<cv::Vec3f>& points3D, std::vector<cv::Vec3b>& colors) {
//     auto cloud = std::make_shared<open3d::geometry::PointCloud>();
//     for (size_t i = 0; i < points3D.size(); ++i) {
//         if(points3D[i][2]<50.0 &&points3D[i][2]>15.0 && points3D[i][0]<30.0 && points3D[i][0]>-25 &&points3D[i][1]>-10.0 ){
//         cloud->points_.emplace_back(static_cast<double>(points3D[i][0]),
//                                     static_cast<double>(points3D[i][1]),
//                                     static_cast<double>(points3D[i][2]));
//         cloud->colors_.emplace_back(static_cast<double>(colors[i][2]) / 255.0,
//                                     static_cast<double>(colors[i][1]) / 255.0,
//                                     static_cast<double>(colors[i][0]) / 255.0);
//                                     }
//     }
//     return cloud;
// }

// void savePointCloudOpen3D(std::shared_ptr<open3d::geometry::PointCloud> cloud, std::string filename) {
//     open3d::io::WritePointCloud(filename, *cloud);
// }

// void transformCam2Velodynn(cv::Mat& R, cv::Mat& T, cv::Mat& t_cam2velo, std::shared_ptr<open3d::geometry::PointCloud> cloud) {
//     // Initialize transformation matrix
//     cv::Mat transformation = cv::Mat::eye(4, 4, CV_64F);
//     R.copyTo(transformation(cv::Rect(0, 0, 3, 3)));
//     T.copyTo(transformation(cv::Rect(3, 0, 1, 3)));

//     // Compute inverse of transformation matrix
//     t_cam2velo = transformation.inv();

//     // Transform points
//     for (auto& point : cloud->points_) {
//         cv::Mat homogenous_point = (cv::Mat_<double>(4, 1) << point.x(), point.y(), point.z(), 1.0);
//         cv::Mat transformed_point = t_cam2velo * homogenous_point;
//         point.x() = transformed_point.at<double>(0, 0);
//         point.y() = transformed_point.at<double>(1, 0);
//         point.z() = transformed_point.at<double>(2, 0);
//     }
// }
int main(int argc, char **argv)
{
    // Example x and y values
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 3, 5, 7, 11};

    // Write data to file
    std::ofstream data_file("data.dat");
    for (size_t i = 0; i < x.size(); ++i)
    {
        data_file << x[i] << " " << y[i] << "\n";
    }
    data_file.close();

    // Create a Gnuplot script
    std::ofstream script_file("plot_script.gp");
    script_file << "set terminal pngcairo size 800,600 enhanced font 'Verdana,10'\n";
    script_file << "set output 'plot.png'\n";
    script_file << "set xlabel 'X axis'\n";
    script_file << "set ylabel 'Y axis'\n";
    script_file << "set title 'Scatter Plot of X vs Y'\n";
    script_file << "plot 'data.dat' with points pt 7 ps 1.5 title 'Data points'\n";
    script_file.close();

    // Call Gnuplot to execute the script
    std::system("gnuplot plot_script.gp");

    std::cout << "Plot created as plot.png" << std::endl;

    std::string binFile0 = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000.bin";
    std::string binFile1 = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000001.bin";
    std::string binFile2 = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000002.bin";
    std::vector<Eigen::Vector4f> lidar_map_t0 = readLidarMap(binFile0, "3d_lidar_map_t0.ply");
    std::vector<Eigen::Vector4f> lidar_map_t1 = readLidarMap(binFile1, "3d_lidar_map_t1.ply");
    std::vector<Eigen::Vector4f> lidar_map_t2 = readLidarMap(binFile2, "3d_lidar_map_t2.ply");
    // open3d::geometry::PointCloud test;
    // convertTargetToOpen3dPointCloud(lidar_map_t0, test);
    // open3d::visualization::Visualizer visualizer1;
    // visualizer1.CreateVisualizerWindow("test", 1600, 1200);
    // visualizer1.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(test));
    // visualizer1.Run();
    // visualizer1.DestroyVisualizerWindow();

    cv::Mat left_img = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000_l.png", cv::IMREAD_COLOR);
    cv::Mat right_img = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000_r.png", cv::IMREAD_COLOR);

    cv::Mat left_img_1 = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000001_l.png", cv::IMREAD_COLOR);
    cv::Mat right_img_1 = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000001_r.png", cv::IMREAD_COLOR);

    cv::Mat left_img_2 = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000002_l.png", cv::IMREAD_COLOR);
    cv::Mat right_img_2 = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000002_r.png", cv::IMREAD_COLOR);

    // cv::Mat left_img = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000_bwl.png", cv::IMREAD_GRAYSCALE);
    // cv::Mat right_img = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000_bwr.png", cv::IMREAD_GRAYSCALE);

    std::string calibFile_cam_cam = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/calib_cam_to_cam.txt";
    std::string calibFile_velo_cam = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/calib_velo_to_cam.txt";
    cv::Mat P_rect_02, T_03, R_rect_00, T_00, R, T, K, D;

    readCalibrationFile(calibFile_cam_cam, P_rect_02, R_rect_00, T_00, T_03, R, T, K, D);
    readCalibrationFile(calibFile_velo_cam, P_rect_02, R_rect_00, T_00, T_03, R, T, K, D);

    // // Detect and match keypoints
    // std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    // std::vector<cv::DMatch> matches;
    // detectAndMatchKeypoints(left_img, right_img, keypointsLeft, keypointsRight, matches);

    // // Triangulate points
    // TriangulationResult result = triangulatePoints(keypointsLeft, keypointsRight, matches, P_rect_02, R_rect_00, left_img);

    // // Create the point cloud
    // std::shared_ptr<open3d::geometry::PointCloud> cloud = createPointCloudOpen3D(result.points3D, result.colors);

    // // Transform points from camera to Velodyne coordinates
    // cv::Mat t_cam2velo1;
    // transformCam2Velodynn(R, T, t_cam2velo1, cloud);
    // std::cout << cloud->points_.size() << std::endl;
    // open3d::geometry::PointCloud cloudd = *cloud;
    // open3d::io::WritePointCloud("off.ply", *cloud);

    cv::Mat disparity;
    computeDisparity(left_img, right_img, disparity);
    // std::cout << "disparity size " << disparity.size() << std::endl;
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
    compute_Q_matrix(P_rect_02, T_03, Q);
    std::cout << "Q matrix " << std::endl;
    printMat(Q);

    cv::Mat points_3d;
    // reconstruct3DPoints(filtered_disparity, Q, points_3d);
    reconstruct3DPoints(disparity, Q, points_3d);
    // points_3d *= 16.0;
    //  cv::viz::Viz3d window("3D Point Cloud");

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

    // cv::Mat colors;
    // cv::cvtColor(left_img, colors, cv::COLOR_BGR2RGB);
    std::vector<cv::Vec3f> all_3d_points;
    std::vector<cv::Vec3b> all_colors;
    getAllPoints_3d(colors, points_3d, all_3d_points, all_colors);

    // Rotate points to add noise
    std::vector<cv::Vec3f> icp_source_keypoints = keypoints_3d;
    std::vector<cv::Vec3f> noisy_stereo = rotatePoints(icp_source_keypoints, 3.0);
    std::string noisy_points = "noisy_stereo.ply";
    write2PLY(noisy_stereo, keypoints_colors, noisy_points);
    // comment out if you dont want noise
    // keypoints_3d = noisy_stereo;

    cv::Mat t_cam2velo = cv::Mat::eye(4, 4, CV_64F);
    // transformCam2Velodyn(R, T, t_cam2velo, all_3d_points);
    // std::string point_cloud_all_points_cam_to_velodyn = "point_cloud_all_points_cam_to_velodyn.ply";
    // write2PLY(all_3d_points, all_colors, point_cloud_all_points_cam_to_velodyn);

    transformCam2Velodyn(R, T, t_cam2velo, keypoints_3d);
    std::string point_cloud_key_points_cam_to_velodyn = "point_cloud_key_points_cam_to_velodyn.ply";
    write2PLY(keypoints_3d, keypoints_colors, point_cloud_key_points_cam_to_velodyn);

    open3d::geometry::PointCloud open3d_stereo_cam2velo;
    convertSourceToOpen3dPointCloud(keypoints_3d, keypoints_colors, open3d_stereo_cam2velo);

    // write2PLY2(open3d_stereo_cam2velo.points_, keypoints_colors, "open3d_stereo_cam2velo.ply");
    // convertSourceToOpen3dPointCloud(all_3d_points, all_colors, open3d_stereo_cam2velo);
    //  write2PLY2(open3d_stereo_cam2velo.points_, all_colors, "open3d_stereo_cam2velo.ply");

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

    // get denser stereo reconstruction from t0 t1 and t2
    open3d::geometry::PointCloud stereo_t0_t1_t2 = get_denser_stereo(left_img, right_img, left_img_1, right_img_1, left_img_2, right_img_2, P_rect_02, T_03, R, T);
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("PointCloud Viewer dkfjlsjfl", 1600, 1200);
    vis.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(lidar_t0_t1_t2));
    // vis.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(open3d_stereo_cam2velo));
    vis.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(stereo_t0_t1_t2));
    vis.GetRenderOption().point_size_ = 11.0;
    vis.Run();
    vis.DestroyVisualizerWindow();

    // run icp and then refine correspondences
    // run_improved_icp(0.5,15,5,lidar_map_t0,lidar_t0_t1_t2,0.02,2000,tolerance);

    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    auto registration_result = open3d::pipelines::registration::RegistrationICP(
        stereo_t0_t1_t2, lidar_t0_t1_t2, 0.1, t,
        open3d::pipelines::registration::TransformationEstimationPointToPoint(),
        open3d::pipelines::registration::ICPConvergenceCriteria(1000, tolerance));

    std::cout << "registration_result corr: " << registration_result.correspondence_set_.size() << std::endl;
    std::cout << "T: " << registration_result.transformation_ << std::endl;
    drawCorrespondences(stereo_t0_t1_t2, lidar_t0_t1_t2, registration_result.correspondence_set_);

    // correspondences_filtering(0.5, 15, 3, open3d_stereo_cam2velo, lidar_t0_t1_t2, 2, 100, tolerance);
    int rotation_degree = 3;
    Eigen::Vector3d translation(0.1, 0.1, 0.1);
    double error_corr = 0.0;
    double error_icp = 0.0;
    correspondences_filtering(0.5, 15, 3, stereo_t0_t1_t2, lidar_t0_t1_t2, rotation_degree, translation, error_corr, error_icp, 3, 100, tolerance);

    // auto voxel_grid = open3d::geometry::VoxelGrid::CreateFromPointCloud(open3d_lidar_map_t0, 0.5);
    // Eigen::Vector3d voxel_color(0.9, 0.9, 0.9); // Bright white color
    // for (auto &voxel : voxel_grid->voxels_)
    // {
    //     voxel.second.color_ = voxel_color;
    // }

    // open3d::visualization::Visualizer visualizer;
    // visualizer.CreateVisualizerWindow("Voxel Grid Visualization", 1920, 1080);

    // // Add the voxel grid to the visualizer
    // visualizer.AddGeometry(voxel_grid);

    // // Add the original point cloud to the visualizer (optional)
    // visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(open3d_lidar_map_t0));
    // visualizer.GetRenderOption().point_size_ = 5.0;

    // // Run the visualizer
    // visualizer.Run();
    // visualizer.DestroyVisualizerWindow();

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
