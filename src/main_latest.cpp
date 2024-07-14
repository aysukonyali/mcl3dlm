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
// 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   github PAT
void printMat(const cv::Mat &mat)
{
    std::cout << "Matrix rows: " << mat.rows << ", cols: " << mat.cols << std::endl;

    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            std::cout << mat.at<double>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// Function to read the .bin file
std::vector<Eigen::Vector4f> readVeloToCam(std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    std::vector<Eigen::Vector4f> points;

    if (!file)
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return points;
    }

    Eigen::Vector4f point;
    while (file.read(reinterpret_cast<char *>(&point), sizeof(point)))
    {
        // Eigen::Vector4f new_pt;
        // new_pt << -point(1), -point(2), point(0), point(3);
        points.push_back(point);
    }

    return points;
}

int main(int argc, char **argv)
{
    std::string binFile = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000.bin";
    std::string plyFileMap = "3d_lidar_map.ply";
    std::vector<Eigen::Vector4f> pointss = readVeloToCam(binFile);
    writePLY2(pointss, plyFileMap);

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

    //std::vector<Eigen::Vector3d> keypoints_3d;
    std::vector<cv::Vec3f> keypoints_3d;
    std::vector<cv::Vec3b> keypoints_colors;
    cv::Mat colors;
    cv::cvtColor(left_img, colors, cv::COLOR_BGR2RGB);
    getKeypoints_3d(colors, points_3d, keypoints_3d, keypoints_colors, keypoints_left);
    std::vector<cv::Vec3f> icp_source_keypoints = keypoints_3d;

    cv::Mat img_with_keypoints;
    cv::drawKeypoints(left_img, keypoints_left, img_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("Keypoints", img_with_keypoints);
    cv::waitKey(0);

    std::cout << "keypoints_left size " << keypoints_left.size() << std::endl;
    // cvMatToEigen(colors, points_3d, keypoints_3d, keypoints_colors, keypoints_left);
    // std::string point_cloud_keypoints = "point_cloud_keypoints.ply";
    // writePLY(keypoints_3d, keypoints_colors, point_cloud_keypoints);

    //std::vector<Eigen::Vector3d> all_3d_points;
    std::vector<cv::Vec3f> all_3d_points;
    std::vector<cv::Vec3b> all_colors;
    getAllPoints_3d(colors, points_3d, all_3d_points, all_colors);
    // cvMatToEigenAllPoints(colors, points_3d, all_3d_points, all_colors);
    // std::string point_cloud_all_points = "point_cloud_all_points.ply";
    // writePLY(all_3d_points, all_colors, point_cloud_all_points);
    
    //Eigen::Matrix4d t_cam2velo = Eigen::Matrix4d::Identity();
    cv::Mat t_cam2velo = cv::Mat::eye(4, 4, CV_64F);
    transformCam2Velodyn(R, T, t_cam2velo, keypoints_3d);
    std::string point_cloud_cam_to_velodyn = "point_cloud_keypoints_cam_to_velodyn.ply";
    write2PLY(keypoints_3d, keypoints_colors, point_cloud_cam_to_velodyn);

    transformCam2Velodyn(R, T, t_cam2velo, all_3d_points);
    std::string point_cloud_all_points_cam_to_velodyn = "point_cloud_all_points_cam_to_velodyn.ply";
    write2PLY(all_3d_points, all_colors, point_cloud_all_points_cam_to_velodyn);

    // Rotate points to add noise
    std::vector<cv::Vec3f> noisy_keypoints_3d = rotatePoints(icp_source_keypoints, 5.0);
    std::string noisy_keypoints = "noisy_keypoints.ply";
    write2PLY(noisy_keypoints_3d, keypoints_colors, noisy_keypoints);

    // Convert to Open3D PointCloud
    open3d::geometry::PointCloud source, target;
    convertSourceToOpen3dPointCloud(noisy_keypoints_3d, keypoints_colors, source);
    convertTargetToOpen3dPointCloud(pointss, target);


    // Set parameters
    double max_correspondence_distance = 0.02; 
    int max_iterations = 100;                 
    double tolerance = 1e-6;              

    // Run ICP
    Eigen::Matrix4d initial_transform;
    cvMatToEigenMat(t_cam2velo, initial_transform);
    runIcp(source, target, max_correspondence_distance, max_iterations, tolerance, initial_transform);
    write2PLY2(source.points_,keypoints_colors,"icp_transformed_noisy_keypoints.ply");

    return 0;
}