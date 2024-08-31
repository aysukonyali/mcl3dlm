#ifndef ALIGNMENT_H
#define ALIGNMENT_H
#include <unordered_set>
#include <filesystem>
#include <sophus/so3.hpp>
#include <open3d/Open3D.h>
#include <opencv2/core.hpp>
#include "open3d/geometry/PointCloud.h"
#include "simple_odometry.h"

std::vector<cv::Vec3f> rotatePoints(std::vector<cv::Vec3f> &points, double angle_deg);

void convertSourceToOpen3dPointCloud(std::vector<cv::Vec3f> &points, std::vector<cv::Vec3b> &colors,
                                     open3d::geometry::PointCloud &point_cloud);

void convertTargetToOpen3dPointCloud(std::vector<Eigen::Vector4f> &target_points,
                                     open3d::geometry::PointCloud &point_cloud);

void visualizePointClouds(open3d::geometry::PointCloud &source,
                          open3d::geometry::PointCloud &target,
                          std::string window_name);

void runIcp(open3d::geometry::PointCloud &source,
            open3d::geometry::PointCloud &target,
            double max_correspondence_distance, int max_iterations, double tolerance,
            Eigen::Matrix4d &initial_transformation);

void cvMatToEigenMat(cv::Mat &cv_mat, Eigen::Matrix4d &eigen_mat);

open3d::geometry::PointCloud mergePointClouds(open3d::geometry::PointCloud &pc1, open3d::geometry::PointCloud &pc2);

std::vector<Eigen::Vector4f> mergeColors(std::vector<Eigen::Vector4f> color1, std::vector<Eigen::Vector4f> color2);

open3d::geometry::PointCloud get_denser_lidar_map(std::vector<Eigen::Vector4f> lidar_map_t0,
                                                  std::vector<Eigen::Vector4f> lidar_map_t1,
                                                  std::vector<Eigen::Vector4f> lidar_map_t2,
                                                  open3d::geometry::PointCloud open3d_stereo_cam2velo);
open3d::geometry::PointCloud rotateLidarMap(std::vector<Eigen::Vector4f> &lidar, double angle_deg, std::string filename);
open3d::geometry::PointCloud rotateOpen3dPointCloud(open3d::geometry::PointCloud point_cloud, double angle_deg, Eigen::Vector3d &translation, Eigen::Matrix4d &transformation);
void run_improved_icp(double voxel_size,
                      int N_min,
                      int N_standard_devitation,
                      std::vector<Eigen::Vector4f> &lidar_map_t0,
                      open3d::geometry::PointCloud &lidar_t0_t1_t2,
                      double max_correspondence_distance,
                      int max_iterations,
                      double tolerance);

void correspondences_filtering(double voxel_size,
                               int N_min,
                               int N_standard_devitation,
                               open3d::geometry::PointCloud &stereo3d,
                               open3d::geometry::PointCloud &lidar_t0_t1_t2,
                               int rotation_degree,
                               Eigen::Vector3d translation,
                               double &error_corr,
                               double &error_icp,
                               double max_correspondence_distance,
                               int max_iterations,
                               double tolerance);

std::pair<double, double> computeAlignmentQuality(
    open3d::geometry::PointCloud &source,
    open3d::geometry::PointCloud &target,
    Eigen::Matrix4d transformation);

open3d::geometry::PointCloud get_denser_stereo(cv::Mat left_img_0,
                                               cv::Mat right_img_0,
                                               cv::Mat left_img_1,
                                               cv::Mat right_img_1,
                                               cv::Mat left_img_2,
                                               cv::Mat right_img_2,
                                               cv::Mat P_rect_02,
                                               cv::Mat T_03,
                                               cv::Mat R,
                                               cv::Mat T);

open3d::geometry::PointCloud get_denser_stereo_from_images(std::vector<cv::Mat> leftImages,
                                                           std::vector<cv::Mat> rightImages,
                                                           size_t image_size,
                                                           cv::Mat P_rect_02,
                                                           cv::Mat T_03,
                                                           cv::Mat R,
                                                           cv::Mat T);

open3d::geometry::PointCloud get_denser_lidar_map_from_velodyns(std::vector<std::vector<Eigen::Vector4f>> lidar_maps, size_t map_size);
void rotation_plot(int angles, int range, open3d::geometry::PointCloud &stereo, open3d::geometry::PointCloud &lidar,std::vector<double> &errors,std::vector<double>&errors_icp);
void correspondences_filtering_plot(double voxel_size,
                                    int N_min,
                                    int N_standard_devitation,
                                    open3d::geometry::PointCloud &stereo3d,
                                    open3d::geometry::PointCloud &lidar_t0_t1_t2,
                                    Eigen::Matrix4d &r,
                                    Eigen::Matrix4d &r_t,
                                    Eigen::Matrix4d &r_icp,
                                    double max_correspondence_distance,
                                    int max_iterations,
                                    double tolerance);

Sophus::SO3d random_rotation(double angle_deg);
double computeError(Eigen::Matrix4d& A, Eigen::Matrix4d& B);
#endif