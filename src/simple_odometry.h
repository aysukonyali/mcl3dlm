#ifndef SIMPLE_ODOMETRY_H
#define SIMPLE_ODOMETRY_H
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <filesystem>
#include <open3d/Open3D.h>
#include <opencv2/core.hpp>
#include "open3d/geometry/PointCloud.h"

// Custom comparator for cv::KeyPoint
struct KeyPointComparato {
    bool operator()(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2) const {
        // Compare based on the x coordinate
        if (kp1.pt.x != kp2.pt.x)
            return kp1.pt.x < kp2.pt.x;
        // Compare based on the y coordinate if x coordinates are equal
        if (kp1.pt.y != kp2.pt.y)
            return kp1.pt.y < kp2.pt.y;
        // Compare based on the size if both coordinates are equal
        if (kp1.size != kp2.size)
            return kp1.size < kp2.size;
        // Compare based on the angle if size is equal
        if (kp1.angle != kp2.angle)
            return kp1.angle < kp2.angle;
        // Compare based on the response if angle is equal
        return kp1.response < kp2.response;
    }
};
struct Image
{
    // ID for the image
    int id;

    cv::Mat left_img;
    
    cv::Mat right_img;

    // // Map from keypoints to descriptors
    // std::unordered_map<cv::KeyPoint, cv::Mat, KeyPointHash> keypointsToDescriptors;

    // // Map from keypoints to 3D points (cv::Vec3f)
    // std::unordered_map<cv::KeyPoint, cv::Vec3f, KeyPointHash> keypointsTo3DPoints;

    // // Map from keypoints to color (cv::Vec3b)
    // std::unordered_map<cv::KeyPoint, cv::Vec3b, KeyPointHash> keypointsToColor;

    std::map<cv::KeyPoint, cv::Mat, KeyPointComparato> keypointsToDescriptors;

    // Map from keypoints to 3D points using comparator
    std::map<cv::KeyPoint, cv::Vec3f, KeyPointComparato> keypointsTo3DPoints;

    // Map from keypoints to color using comparator
    std::map<cv::KeyPoint, cv::Vec3b, KeyPointComparato> keypointsToColor;

    // 4x4 transformation matrix
    Eigen::Matrix4d T_i1_i2;

    // 4x4 refined transformation matrix
    Eigen::Matrix4d T_refined;

    open3d::geometry::PointCloud pc;

    cv::Mat descriptors;

    std::vector<cv::KeyPoint> keypoints;

    std::vector<cv::Vec3b> colors;



    // Constructor to initialize the matrices as identity matrices
    Image() : id(0), T_i1_i2(Eigen::Matrix4d::Identity()), T_refined(Eigen::Matrix4d::Identity()) {}
};
void get3dPointsInVelodyn(Image &i,
                          cv::Mat &colors,
                          cv::Mat &points_3d,
                          std::vector<cv::Vec3f> &keypoints_3d,
                          std::vector<cv::Vec3b> &keypoint_colors,
                          std::vector<cv::KeyPoint> &keypoint_coord,
                          cv::Mat &R,
                          cv::Mat &T);

void get_stereo_first(Image &i,
                      cv::Mat P_rect_02,
                      cv::Mat T_03,
                      cv::Mat R,
                      cv::Mat T);

void computeDescriptors(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);
void getLeftKeypoints_odometry(Image &i);
void extractKeypoints_odometry(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, std::vector<cv::Vec3b> &colors_odometry);
void keypointsMatchingAndPnP(Image &current, Image &next, cv::Mat K, cv::Mat D, cv::Mat P_rect_02, cv::Mat T_03);
void drawPnPMatches(
    const open3d::geometry::PointCloud &source,
    const open3d::geometry::PointCloud &target,
    const std::vector<cv::DMatch> &matches);

Eigen::Matrix4d cvMatToEigen4(cv::Mat& mat);    
void correspondences_filtering_odometry(Image &i,
                               double voxel_size,
                               int N_min,
                               int N_standard_devitation,
                               open3d::pipelines::registration::CorrespondenceSet &correspondences,
                               open3d::geometry::PointCloud &stereo3d,
                               open3d::geometry::PointCloud &lidar_t0_t1_t2);

void runSimpleOdometry(std::vector<cv::Mat> leftImages, std::vector<cv::Mat> rightImages,std::vector<std::vector<Eigen::Vector4f>> lidarMaps,cv::Mat P_rect_02, cv::Mat T_03, cv::Mat  R, cv::Mat T, cv::Mat K, cv::Mat D);
open3d::geometry::PointCloud get_stereo_second(Image &i,
                                               cv::Mat P_rect_02,
                                               cv::Mat T_03,
                                               std::vector<cv::KeyPoint> keypoints_left);
#endif
