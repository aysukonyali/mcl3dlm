#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <iostream>
#include "local_reconstruction.h"

void matchSIFTFeatures(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2)
{
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    sift->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    for (const auto &match : matches)
    {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }
}

void computeDisparity(const cv::Mat &left_img, const cv::Mat &right_img, cv::Mat &disparity)
{
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
    stereo->compute(left_img, right_img, disparity);
}

void reconstruct3DPoints(cv::Mat &disparity, cv::Mat &Q, cv::Mat &points_3d)
{
    cv::reprojectImageTo3D(disparity, points_3d, Q);
}

void compute_Q_matrix(std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2,  cv::Mat &K1,  cv::Mat &K2, cv::Mat &D1,  cv::Mat &D2, cv::Size imageSize, cv::Mat &Q)
{
    cv::Mat E = cv::findEssentialMat(points1, points2, K1);

    // Decompose Essential Matrix to get rotation and translation
    cv::Mat R, T;

    cv::recoverPose(E, points1, points2, K1, R, T);

    // Rectification parameters
    cv::Mat R1, R2, P1, P2;

    // Compute rectification transforms and Q matrix
    cv::stereoRectify(K1, D1, K2, D2, imageSize, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 1, imageSize);
}
void cvMatToEigen(cv::Mat &points_3d, std::vector<Eigen::Vector3d> &eigen_points)
{
    for (int i = 0; i < points_3d.rows; ++i)
    {
        for (int j = 0; j < points_3d.cols; ++j)
        {
            cv::Vec3f point = points_3d.at<cv::Vec3f>(i, j);
            if (!cv::checkRange(point))
                continue; // Skip invalid points
            Eigen::Vector3d eigen_point(point[0], point[1], point[2]);
            eigen_points.push_back(eigen_point);
        }
    }
}

void writePLY(std::vector<Eigen::Vector3d>& points, std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << points.size() << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "end_header\n";
     
    for (const auto& point : points) {
        ofs << point.x() << " " << point.y() << " " << point.z() << "\n";
    }
    std::cout << "hello "<< std::endl;
    ofs.close();
}



void writePLY2(std::vector<Eigen::Vector4f> &points, std::string &filename) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << points.size() << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property float intensity\n";
    ofs << "end_header\n";

    for (const auto &point : points) {
        ofs << point.x() << " " << point.y() << " " << point.z() << " " << point.w() << "\n";
    }

    ofs.close();
}
