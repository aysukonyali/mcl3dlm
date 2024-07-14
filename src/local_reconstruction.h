#ifndef LOCAL_RECONSTRUCTION_H
#define LOCAL_RECONSTRUCTION_H
#include <unordered_set>
#include <filesystem>
#include <open3d/Open3D.h>
#include <opencv2/core.hpp>
#include "open3d/geometry/PointCloud.h"

void printMat(cv::Mat &mat);
std::vector<Eigen::Vector4f> readLidarMap(std::string &filename, std::string filename2save);
void computeDisparity(cv::Mat &left_img, cv::Mat &right_img, cv::Mat &disparity);

void reconstruct3DPoints(cv::Mat &disparity, cv::Mat &Q, cv::Mat &points_3d);

void compute_Q_matrix(cv::Mat &P_rect_02, cv::Mat &P_rect_03, cv::Mat &Q);

void cvMatToEigen(cv::Mat &colors, cv::Mat &points_3d, std::vector<Eigen::Vector3d> &eigen_points, std::vector<cv::Vec3b> &out_colors, std::vector<cv::KeyPoint> &keypoint_coord);
void write2PLY(std::vector<cv::Vec3f> &points, std::vector<cv::Vec3b> &colors, std::string &filename);
void write2PLY2(std::vector<Eigen::Vector3d> &points, std::vector<cv::Vec3b> &colors, std::string filename);
void writePLY2(std::vector<Eigen::Vector4f> &points, std::string &filename);

void readCalibrationFile(std::string &filename, cv::Mat &P_rect_02, cv::Mat &T_03, cv::Mat &R, cv::Mat &T);
void generatePointCloud(cv::Mat &disparity, cv::Mat &left_img, cv::Mat &Q, open3d::geometry::PointCloud &pointCloud);
void filterPointCloud(open3d::geometry::PointCloud &pointCloud, cv::Mat &disparity);
void cvMatToEigenMat(cv::Mat &cvMat, Eigen::Matrix3d &eigenMat);
void extractKeypointsAndDescriptors(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

void matchDescriptors(cv::Mat &descriptors1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches);

void filter3DPoints(std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2,
                    std::vector<cv::DMatch> &matches, std::vector<Eigen::Vector3d> &points3D,
                    std::vector<Eigen::Vector3d> &filteredPoints3D);

void getKeyPointCoordinatesFromMatches(std::vector<cv::KeyPoint> &keypoints_left,
                                     std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &keypoint_coord);

void cvMatToEigenAllPoints(cv::Mat &colors, cv::Mat &points_3d, std::vector<Eigen::Vector3d> &eigen_points, std::vector<cv::Vec3b> &out_colors);
void transform_cam2velodyn(cv::Mat &R,  cv::Mat &T, Eigen::Matrix4d &t_cam2velo, std::vector<Eigen::Vector3d> &eigen_points);

void applyWLSFilter(cv::Mat& img_left, cv::Mat& img_right, cv::Mat& filtered_disparity);
void getKeypoints_3d(cv::Mat &colors, cv::Mat &points_3d, std::vector<cv::Vec3f> &keypoints_3d, std::vector<cv::Vec3b> &keypoint_colors, std::vector<cv::KeyPoint> &keypoint_coord);
void getAllPoints_3d(cv::Mat &colors, cv::Mat &points_3d, std::vector<cv::Vec3f> &all_3d_points, std::vector<cv::Vec3b> &all_colors);
void transformCam2Velodyn(cv::Mat &R, cv::Mat &T, cv::Mat &t_cam2velo, std::vector<cv::Vec3f> &points);
void write_point_cloud(std::vector<Eigen::Vector3d> &points,std::vector<Eigen::Vector4f> &intensity,std::string filename);
#endif
