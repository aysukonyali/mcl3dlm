#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "local_reconstruction.h"
// Function prototypes
void computeSparseDisparity(std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2,
                            std::vector<cv::DMatch> &matches, std::vector<float> &disparities,
                            std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);
void reconstruct3DFromSparseDisparity(std::vector<cv::Point2f> &points1, std::vector<float> &disparities,
                                      cv::Mat &Q, std::vector<cv::Vec3f> &points3D);

void readCalibrationFile_F(std::string &filename, cv::Mat &P_rect_02, cv::Mat &T_03);
void compute_Q_matrix_float(cv::Mat &P_rect_02, cv::Mat &T_02, cv::Mat &Q);
void cvMatToEigen_f(cv::Mat &colors, cv::Mat &points_3d, std::vector<Eigen::Vector3f> &eigen_points, std::vector<cv::Vec3b> &out_colors);

void writePLY_f(std::vector<Eigen::Vector3f> &points, std::vector<cv::Vec3b> &colors, std::string &filename);


int main()
{
    // Load images

    cv::Mat img_left = cv::imread("/home/aysu/Downloads/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png", cv::IMREAD_COLOR);
    cv::Mat img_right = cv::imread("/home/aysu/Downloads/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/0000000000.png", cv::IMREAD_COLOR);

    // Extract keypoints and descriptors
    std::vector<cv::KeyPoint> keypoints_left, keypoints_right;
    cv::Mat descriptors_left, descriptors_right;
    extractKeypointsAndDescriptors(img_left, keypoints_left, descriptors_left);
    extractKeypointsAndDescriptors(img_right, keypoints_right, descriptors_right);

    // Match descriptors
    std::vector<cv::DMatch> matches;
    matchDescriptors(descriptors_left, descriptors_right, matches);

    // Compute sparse disparity from matched keypoints
    std::vector<float> disparities;
    std::vector<cv::Point2f> points_left, points_right;
    computeSparseDisparity(keypoints_left, keypoints_right, matches, disparities, points_left, points_right);

    // Compute Q matrix
    std::string calibFile_cam_cam = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/calib_cam_to_cam.txt";

    cv::Mat P_rect_02, T_02;

    readCalibrationFile_F(calibFile_cam_cam, P_rect_02, T_02);

    cv::Mat Q = cv::Mat::zeros(4, 4, CV_64F);
    compute_Q_matrix_float(P_rect_02, T_02, Q);

    // Reconstruct 3D points from sparse disparity
    std::vector<cv::Vec3f> points3D;
    reconstruct3DFromSparseDisparity(points_left, disparities, Q, points3D);


    std::vector<Eigen::Vector3f> eigen_points;
    std::vector<cv::Vec3b> out_colors;
    cv::Mat colors;
    cv::cvtColor(img_left, colors, cv::COLOR_BGR2RGB);
    cvMatToEigen_f(colors, points3D, eigen_points, out_colors);


    std::cout << "3D points saved to output.ply" << std::endl;
    return 0;
}

void computeSparseDisparity(std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2,
                            std::vector<cv::DMatch> &matches, std::vector<float> &disparities,
                            std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2)
{
    for (const auto &match : matches)
    {
        float disparity = keypoints1[match.queryIdx].pt.x - keypoints2[match.trainIdx].pt.x;
        disparities.push_back(disparity);
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }
}

void reconstruct3DFromSparseDisparity(std::vector<cv::Point2f> &points1, std::vector<float> &disparities,
                                      cv::Mat &Q, std::vector<cv::Vec3f> &points3D)
{
    for (size_t i = 0; i < points1.size(); ++i)
    {
        float x = points1[i].x;
        float y = points1[i].y;
        float disparity = disparities[i];

        cv::Mat_<float> vec(4, 1);
        vec(0) = x;
        vec(1) = y;
        vec(2) = disparity;
        vec(3) = 1.0;

        // Ensure Q and vec are of the same data type (e.g., CV_32F or CV_64F)
        if (Q.type() != vec.type())
        {
            Q.convertTo(Q, vec.type());
        }

        // Perform the matrix multiplication
        cv::Mat_<float> xyz = Q * vec;

        // Extract 3D point from the homogeneous coordinates
        cv::Vec3f point(xyz(0) / xyz(3), xyz(1) / xyz(3), xyz(2) / xyz(3));
        points3D.push_back(point);
    }
}

void compute_Q_matrix_float(cv::Mat &P_rect_02, cv::Mat &T_02, cv::Mat &Q)
{
    // Compute the Q matrix
    float fx = P_rect_02.at<float>(0, 0);
    float cx = P_rect_02.at<float>(0, 2);
    float cy = P_rect_02.at<float>(1, 2);

    Q.at<float>(0, 0) = 1.0;
    Q.at<float>(0, 3) = -cx;
    Q.at<float>(1, 1) = 1.0;
    Q.at<float>(1, 3) = -cy;
    Q.at<float>(2, 3) = fx;
    Q.at<float>(3, 2) = -1.0 / T_02.at<float>(0, 0);
}

void readCalibrationFile_F(std::string &filename, cv::Mat &P_rect_02, cv::Mat &T_03)
{
    std::ifstream file(filename);

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        std::string key;
        if (std::getline(ss, key, ':'))
        {
            if (key == "P_rect_02")
            {
                P_rect_02 = cv::Mat(3, 4, CV_64F);
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 4; ++j)
                        ss >> P_rect_02.at<float>(i, j);
            }

            else if (key == "T_03")
            {
                T_03 = cv::Mat(3, 1, CV_64F);
                for (int i = 0; i < 3; ++i)
                    ss >> T_03.at<float>(i, 0);
            }
        }
    }

    file.close();
}
void cvMatToEigen_f(cv::Mat &colors, std::vector<cv::Vec3f> &points_3d, std::vector<Eigen::Vector3f> &eigen_points, std::vector<cv::Vec3b> &out_colors)
{



    
        for (int j = 0; j < points_3d.size(); ++j)
        {
            cv::Vec3f point = points_3d[j];
            if (cv::checkRange(point, true, nullptr, -1e4, 1e4))
            {
                Eigen::Vector3f eigen_point(point[0], point[1], point[2]);
                eigen_points.push_back(eigen_point);

            }
        }
    
}

void writePLY_f(std::vector<Eigen::Vector3f> &points, std::vector<cv::Vec3b> &colors, std::string &filename)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << points.size() << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";
    ofs << "end_header\n";

    for (size_t i = 0; i < points.size(); ++i)
    {
        ofs << points[i][0] << " " << points[i][1] << " " << points[i][2] << " "
            << (int)colors[i][0] << " " << (int)colors[i][1] << " " << (int)colors[i][2] << "\n";
    }

    ofs.close();
}