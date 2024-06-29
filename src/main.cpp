#include <opencv2/opencv.hpp>
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

bool readCalibrationFile(const std::string &filename, cv::Mat &K1, cv::Mat &D1, cv::Mat &K2, cv::Mat &D2)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        std::string key;
        if (std::getline(ss, key, ':'))
        {
            if (key == "K_02")
            {
                K1 = cv::Mat(3, 3, CV_64F);
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        ss >> K1.at<double>(i, j);
            }
            else if (key == "D_02")
            {
                D1 = cv::Mat(1, 5, CV_64F); // Assuming 5 distortion coefficients
                for (int i = 0; i < 5; ++i)
                    ss >> D1.at<double>(i);
            }
            else if (key == "K_03")
            {
                K2 = cv::Mat(3, 3, CV_64F);
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        ss >> K2.at<double>(i, j);
            }
            else if (key == "D_03")
            {
                D2 = cv::Mat(1, 5, CV_64F); // Assuming 5 distortion coefficients
                for (int i = 0; i < 5; ++i)
                    ss >> D2.at<double>(i);
            }
        }
    }

    file.close();
    return true;
}

// Function to read the .bin file
std::vector<Eigen::Vector3d> readBinFile(std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<Eigen::Vector3d> points;

    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return points;
    }

    float point[4];
    while (file.read(reinterpret_cast<char*>(point), sizeof(point))) {
        points.emplace_back(point[0], point[1], point[2]);
    }

    return points;
}



// Function to read the .bin file
std::vector<Eigen::Vector4f> readBinFile2(std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<Eigen::Vector4f> points;

    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return points;
    }

    Eigen::Vector4f point;
    while (file.read(reinterpret_cast<char*>(&point), sizeof(point))) {
        points.push_back(point);
    }

    return points;
}


int main(int argc, char **argv)
{

    const cv::Mat left_img = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000_l.png", cv::IMREAD_GRAYSCALE);
    const cv::Mat right_img = cv::imread("/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000_r.png", cv::IMREAD_GRAYSCALE);
    std::cout << "hello " << std::endl;
    if (left_img.empty() || right_img.empty())
    {
        std::cerr << "Could not open the images!" << std::endl;
        return -1;
    }
    std::cout << "hello " << std::endl;
    std::string calibFile = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/calib_cam_to_cam.txt";
    cv::Mat K1, D1, K2, D2;
    if (!readCalibrationFile(calibFile, K1, D1, K2, D2))
    {
        return -1;
    }

    std::vector<cv::Point2f> points1, points2;
    matchSIFTFeatures(left_img, right_img, points1, points2);

    cv::Mat disparity;
    computeDisparity(left_img, right_img, disparity);

    cv::Mat Q;
    compute_Q_matrix(points1, points2, K1, K2, D1, D2, left_img.size(), Q);

    cv::Mat points_3d;
    reconstruct3DPoints(disparity, Q, points_3d);

    std::vector<Eigen::Vector3d> eigen_points;
    cvMatToEigen(points_3d, eigen_points);


    // Write to PLY file
    std::string plyFile = "local_reconstructed_3d_points.ply";
    writePLY(eigen_points, plyFile);


    std::string binFile = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/0000000000.bin"; 
    std::string plyFileMap = "3d_lidar_map.ply";
    std::string plyFileMap2 = "3d_lidar_map2.ply";

    // Read the .bin file
    std::vector<Eigen::Vector3d> points = readBinFile(binFile);
    // Write to a .ply file
    writePLY(points, plyFileMap);

    std::vector<Eigen::Vector4f> pointss = readBinFile2(binFile);
    writePLY2(pointss, plyFileMap2);
    return 0;
}