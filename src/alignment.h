#ifndef ALIGNMENT_H
#define ALIGNMENT_H
#include <unordered_set>
#include <filesystem>
#include <open3d/Open3D.h>
#include <opencv2/core.hpp>
#include "open3d/geometry/PointCloud.h"

std::vector<cv::Vec3f> rotatePoints(std::vector<cv::Vec3f> &points, double angle_deg);

void convertSourceToOpen3dPointCloud(std::vector<cv::Vec3f> &points, std::vector<cv::Vec3b> &colors,
                               open3d::geometry::PointCloud &point_cloud);

void convertTargetToOpen3dPointCloud(std::vector<Eigen::Vector4f>& target_points,
                                     open3d::geometry::PointCloud& point_cloud);

void visualizePointClouds(open3d::geometry::PointCloud &source,
                          open3d::geometry::PointCloud &target,
                          std::string window_name);

void runIcp(open3d::geometry::PointCloud &source,
             open3d::geometry::PointCloud &target,
             double max_correspondence_distance, int max_iterations, double tolerance,
             Eigen::Matrix4d &initial_transformation);

void cvMatToEigenMat(cv::Mat& cv_mat, Eigen::Matrix4d& eigen_mat);        
open3d::geometry::PointCloud mergePointClouds(open3d::geometry::PointCloud &pc1,open3d::geometry::PointCloud &pc2);
std::vector<Eigen::Vector4f> mergeColors(std::vector<Eigen::Vector4f> color1, std::vector<Eigen::Vector4f> color2);
#endif