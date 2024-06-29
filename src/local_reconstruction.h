#ifndef LOCAL_RECONSTRUCTION_H
#define LOCAL_RECONSTRUCTION_H
#include <unordered_set>
#include <filesystem>

#include <opencv2/core.hpp>

void matchSIFTFeatures(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);

void computeDisparity(const cv::Mat &left_img, const cv::Mat &right_img, cv::Mat &disparity);

void reconstruct3DPoints(cv::Mat &disparity, cv::Mat &Q, cv::Mat &points_3d);

void compute_Q_matrix(std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, cv::Mat &K1, cv::Mat &K2, cv::Mat &D1, cv::Mat &D2, cv::Size imageSize, cv::Mat &Q);

void cvMatToEigen(cv::Mat &points_3d, std::vector<Eigen::Vector3d> &eigen_points);
void writePLY(std::vector<Eigen::Vector3d>& points, std::string& filename);
void writePLY2(std::vector<Eigen::Vector4f> &points, std::string &filename);

#endif