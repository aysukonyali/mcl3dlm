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
std::vector<Eigen::Vector3d> readBinFile(std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    std::vector<Eigen::Vector3d> points;

    if (!file)
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return points;
    }

    float point[4];
    while (file.read(reinterpret_cast<char *>(point), sizeof(point)))
    {
        points.emplace_back(point[0], point[1], point[2]);
    }

    return points;
}

// Function to read the .bin file
std::vector<Eigen::Vector4f> readBinFile2(std::string &filename)
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
        Eigen::Vector4f new_pt;
        new_pt << -point(1), -point(2), point(0), point(3);
        points.push_back(point);
    }

    return points;
}
void readBinFileToPointCloud(std::string &filename, open3d::geometry::PointCloud &pointCloud)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file)
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    Eigen::Vector4f point;
    while (file.read(reinterpret_cast<char *>(&point), sizeof(point)))
    {
        // Extract the position
        Eigen::Vector3d position(point.x(), point.y(), point.z());

        // Extract the color directly from the point
        // Assuming point.w() contains a valid color representation (e.g., normalized to [0, 1] for RGB)
        // Adjust based on your data format
        Eigen::Vector3d color(point.w(), point.w(), point.w());
        // Extract the intensity and map to a color
        // float intensity = point.w();
        // cv::Mat intensityMat(1, 1, CV_32F, &intensity);
        // intensityMat.convertTo(intensityMat, CV_8U, 255.0);  // Normalize intensity to [0, 255]

        // // Use a colormap (e.g., COLORMAP_JET) to map intensity to color
        // cv::Mat colorMat;
        // cv::applyColorMap(intensityMat, colorMat, cv::COLORMAP_JET);

        // // Extract the color
        // cv::Vec3b colorVec = colorMat.at<cv::Vec3b>(0, 0);
        // Eigen::Vector3d color(colorVec[2] / 255.0, colorVec[1] / 255.0, colorVec[0] / 255.0);  // BGR to RGB

        // // Add the position and color to the point cloud
        pointCloud.points_.push_back(position);
        pointCloud.colors_.push_back(color);
    }
}

int main(int argc, char **argv)
{
    std::string binFile = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000.bin";
    std::string plyFileMap = "3d_lidar_map.ply";
    std::vector<Eigen::Vector4f> pointss = readBinFile2(binFile);
    writePLY2(pointss, plyFileMap);

    // open3d::geometry::PointCloud velodynPointCloud;
    // readBinFileToPointCloud(binFile, velodynPointCloud);
    open3d::visualization::Visualizer visualizer;
    // visualizer.CreateVisualizerWindow("Velodyn Point Cloud Viewer", 800, 600);
    // visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(velodynPointCloud));
    // visualizer.Run();
    // visualizer.DestroyVisualizerWindow();
    // open3d::io::WritePointCloud("velodyn_point_cloud_open3d.ply", velodynPointCloud);

    cv::Mat left_img = cv::imread("/home/aysu/Downloads/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png", cv::IMREAD_COLOR);
    cv::Mat right_img = cv::imread("/home/aysu/Downloads/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/0000000000.png", cv::IMREAD_COLOR);
    // addGaussianNoise(left_img, 0.0, 10.0);
    // addGaussianNoise(right_img, 10.0, 100.0);
    
    // cv::Mat left_img = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000_l.png", cv::IMREAD_COLOR);
    // cv::Mat right_img = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000_r.png", cv::IMREAD_COLOR);
    // std::cout << "left img size " << left_img.size() << std::endl;
    // std::vector<cv::KeyPoint> keypoints_left, keypoints_right;
    // cv::Mat descriptors_left, descriptors_right;
    // extractKeypointsAndDescriptors(left_img, keypoints_left, descriptors_left);
    // extractKeypointsAndDescriptors(right_img, keypoints_right, descriptors_right);
    //     // Match descriptors
    // std::vector<cv::DMatch> matches;
    // matchDescriptors(descriptors_left, descriptors_right, matches);

    std::string calibFile_cam_cam = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/calib_cam_to_cam.txt";
    std::string calibFile_velo_cam = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/calib_velo_to_cam.txt";
    cv::Mat P_rect_02, P_rect_03, K1, D1, K2, D2, T_02, R_02, R, T;

    readCalibrationFile(calibFile_cam_cam, P_rect_02, P_rect_03, K1, D1, K2, D2, R_02, T_02, R, T);
    readCalibrationFile(calibFile_velo_cam, P_rect_02, P_rect_03, K1, D1, K2, D2, R_02, T_02, R, T);
    printMat(R);
    printMat(T);

    cv::Mat disparity;
    computeDisparity(left_img, right_img, disparity);
    std::cout << "left_img rows: " << left_img.rows << ", left_img: " << left_img.cols << std::endl;
    std::cout << "disparity rows: " << disparity.rows << ", disparity: " << disparity.cols << std::endl;
    cv::Mat disp_normalized;
    cv::normalize(disparity, disp_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Display the normalized disparity map
    cv::imshow("Disparity Map", disp_normalized);
    cv::waitKey(0); // Wait indefinitely for a key press

    cv::Mat Q = cv::Mat::zeros(4, 4, CV_64F);
    compute_Q_matrix(P_rect_02, T_02, Q);
    std::cout << "Q matrix " << std::endl;
    printMat(Q);

    // Rectify images and compute Q matrix
    cv::Size imageSize = left_img.size();
    cv::Mat R1, R2, P1, P2, Q1;
    cv::stereoRectify(K1, D1, K2, D2, imageSize, R_02, T_02, R1, R2, P1, P2, Q1);
    printMat(Q1);

    // Extract intrinsic matrices
    // cv::Mat K11 = P_rect_02(cv::Range(0, 3), cv::Range(0, 3)).clone();
    // cv::Mat K22 = P_rect_03(cv::Range(0, 3), cv::Range(0, 3)).clone();

    // // Initialize matrices for stereoRectify
    // cv::Mat R11, R22, P11, P22, Q2;

    // Perform stereo rectification
    // cv::stereoRectify(K11, cv::Mat::zeros(1, 5, CV_64F), K22, cv::Mat::zeros(1, 5, CV_64F),
    //                   left_img.size(), cv::Mat::eye(3, 3, CV_64F), cv::Mat(cv::Vec3d(T.at<double>(0, 0), 0.0, 0.0)),
    //                   R11, R22, P11, P22, Q2, cv::CALIB_ZERO_DISPARITY, -1, left_img.size());
    // printMat(Q2);

    cv::Mat points_3d;
    reconstruct3DPoints(disparity, Q, points_3d);
    // Reflect on x-axis
    cv::Mat reflect_matrix = cv::Mat::eye(3, 3, CV_64F);
    reflect_matrix.at<double>(0, 0) = -1;
    //cv::transform(points_3d, points_3d, reflect_matrix);
    //cv::imshow("Display Window", points_3d);

    // Wait for a key press
    cv::waitKey(0);

    std::vector<Eigen::Vector3d> eigen_points;
    std::vector<cv::Vec3b> out_colors;
    cv::Mat colors;
    cv::cvtColor(left_img, colors, cv::COLOR_BGR2RGB);
    std::cout << "colors rows: " << colors.rows << ", colors: " << colors.cols << std::endl;
    cvMatToEigen(colors, points_3d, eigen_points, out_colors, disparity);
    std::cout << "aa " << std::endl;
    std::cout << "eigen_points size: " << eigen_points.size() << std::endl;
    std::cout << "out_colors size: " << out_colors.size() << std::endl;
    std::string plyFile = "point_cloud_opencv.ply";
    writePLY(eigen_points, out_colors, plyFile);

    open3d::geometry::PointCloud pointCloud;
    generatePointCloud(disparity, left_img, Q, pointCloud);

    Eigen::Matrix4d reflect_matrix_ = Eigen::Matrix4d::Identity();
    reflect_matrix_(0, 0) = -1; // Reflect on the x-axis

    // Apply the transformation to the point cloud
    //pointCloud.Transform(reflect_matrix_);
    filterPointCloud(pointCloud, disparity);
    visualizer.CreateVisualizerWindow("Point Cloud Viewer", 800, 600);
    visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(pointCloud));
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    open3d::io::WritePointCloud("point_cloud_open3d.ply", pointCloud);

    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d T_eigen;
    cvMatToEigenMat(R, R_eigen);
    for (int i = 0; i < 3; ++i)
        T_eigen(i, 0) = T.at<double>(i, 0);
    transformation.block<3, 3>(0, 0) = R_eigen;
    transformation.block<3, 1>(0, 3) = T_eigen;
    Eigen::Matrix4d transformation_inverse = transformation.inverse();

    // for icp
    open3d::geometry::PointCloud noisyPointCloud = pointCloud;
    open3d::geometry::PointCloud manuallytransformedPointCloud = pointCloud;
    
    pointCloud.Transform(transformation_inverse);
    for (size_t i = 0; i < manuallytransformedPointCloud.points_.size(); ++i) {
        Eigen::Vector3d &point = manuallytransformedPointCloud.points_[i];
        Eigen::Vector4d homogenous_point(point.x(), point.y(), point.z(), 1.0); // Convert to homogenous coordinates
        Eigen::Vector4d transformed_point = transformation_inverse * homogenous_point;  // Apply transformation
        point = transformed_point.head<3>();                                    // Convert back to 3D coordinates
    }
    open3d::io::WritePointCloud("manuallytransformedPointCloud.ply", manuallytransformedPointCloud);

    
    visualizer.CreateVisualizerWindow("Cam to Velodyn Point Cloud Viewer", 800, 600);
    visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(pointCloud));
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    open3d::io::WritePointCloud("point_cloud_open3d_cam_to_velodyn.ply", pointCloud);

    Eigen::Affine3d transform(transformation_inverse);
    for (auto &point : eigen_points)
    {
        point = transform * point;
    }
    std::string plyFiletransformed = "point_cloud_opencv_cam_to_velodyn.ply";
    writePLY(eigen_points, out_colors, plyFiletransformed);



    for (auto &point : pointss)
    {
        Eigen::Vector4d point_d(point.x(), point.y(), point.z(), point.w());

        // Apply the transformation
        Eigen::Vector4d transformed_point_d = transformation * point_d;

        // Convert back to Eigen::Vector4f
        point = transformed_point_d.cast<float>();
    }
    std::string velodyn_to_cam_lidar_map = "rotated_lidar_map.ply";
    writePLY2(pointss, velodyn_to_cam_lidar_map);

    // add noise to the local reconstructed point cloud
    
    // addNoise(noisyPointCloud, 10);
    // visualizer.CreateVisualizerWindow("Noisy Point Cloud Viewer", 800, 600);
    // visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(noisyPointCloud));
    // visualizer.Run();
    // visualizer.DestroyVisualizerWindow();

    Eigen::Matrix4d noise_transformation = Eigen::Matrix4d::Identity();
    double theta = M_PI / 6; // 30 degrees
    noise_transformation(0, 0) = cos(theta);
    noise_transformation(0, 1) = -sin(theta);
    noise_transformation(1, 0) = sin(theta);
    noise_transformation(1, 1) = cos(theta);
    noise_transformation(0, 3) = 0.5; // 0.5 units along x-axis
    noise_transformation(1, 3) = 0.5; // 0.5 units along y-axis

    //Apply the transformation to the source point cloud
    noisyPointCloud.Transform(noise_transformation);
    open3d::io::WritePointCloud("noisy_point_cloud.ply", noisyPointCloud);
    visualizer.CreateVisualizerWindow("noisy Point Cloud", 800, 600);
    //visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(velodynPointCloud));
    visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(noisyPointCloud));
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();

    // ICP parameters
    double max_correspondence_distance = 0.02; // Maximum distance for matching points
    int max_iterations = 50;                   // Maximum number of ICP iterations
    double tolerance = 1e-6;                   // Convergence criteria

    // Run ICP
    run_icp(noisyPointCloud, velodynPointCloud, max_correspondence_distance, max_iterations, tolerance, transformation_inverse);

    // Visualize the aligned point clouds
    //open3d::visualization::Visualizer visualizer;
    visualizer.CreateVisualizerWindow("ICP Alignment", 800, 600);
    visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(velodynPointCloud));
    visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(noisyPointCloud));
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    open3d::io::WritePointCloud("noisy_point_cloud_icp.ply", noisyPointCloud);

    //std::cout << "points_3d size " << points_3d.size() << std::endl;
    // Reflect on x-axis
    // cv::Mat reflect_matrix = cv::Mat::eye(3, 3, CV_64F);
    // reflect_matrix.at<double>(0, 0) = -1;
    //cv::transform(points_3d, points_3d, reflect_matrix);

    //cv::imshow("Display Window", points_3d);
    //cv::waitKey(0);


    // for (auto &point : pointss)
    // {
    //     Eigen::Vector4d point_d(point.x(), point.y(), point.z(), point.w());

    //     // Apply the transformation
    //     Eigen::Vector4d transformed_point_d = transformation * point_d;

    //     // Convert back to Eigen::Vector4f
    //     point = transformed_point_d.cast<float>();
    // }
    // std::string velodyn_to_cam_lidar_map = "rotated_lidar_map.ply";
    // writePLY2(pointss, velodyn_to_cam_lidar_map);

    return 0;
}