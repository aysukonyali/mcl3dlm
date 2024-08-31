#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp> // For WLS filter
#include <open3d/Open3D.h>
#include "open3d/geometry/PointCloud.h"
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
#include "simple_odometry.h"

void printMat(cv::Mat &mat)
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
std::vector<Eigen::Vector4f> readLidarMap(std::string &filename, std::string filename2save)
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
    writePLY2(points, filename2save);
    return points;
}
void computeDisparity(cv::Mat left_img, cv::Mat right_img, cv::Mat &disparity)
{
    cv::Mat left_img_gaussian;
    cv::Mat right_img_gaussian;
    cv::GaussianBlur(left_img, left_img_gaussian, cv::Size(5, 5), 0);
    cv::GaussianBlur(right_img, right_img_gaussian, cv::Size(5, 5), 0);
    left_img_gaussian.convertTo(left_img_gaussian, -1, 0.5, 0);
    right_img_gaussian.convertTo(right_img_gaussian, -1, 0.5, 0);
    int minDisparity = 0;
    int numDisparities = 112; // Ensure this is divisible by 16
    int blockSize = 5;
    int P1 = 8 * 3 * blockSize * blockSize;
    int P2 = 32 * 3 * blockSize * blockSize;
    int disp12MaxDiff = 1;
    int preFilterCap = 31;
    int uniquenessRatio = 12;
    int speckleWindowSize = 150;
    int speckleRange = 2;
    int mode = cv::StereoSGBM::MODE_HH4;
    cv::Ptr<cv::StereoSGBM> stereo1 = cv::StereoSGBM::create(
        minDisparity,
        numDisparities,
        blockSize,
        P1,
        P2,
        disp12MaxDiff,
        preFilterCap,
        uniquenessRatio,
        speckleWindowSize,
        speckleRange,
        mode);
    // cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
    // stereo->compute(left_img, right_img, disparity);
    stereo1->compute(left_img_gaussian, right_img_gaussian, disparity);
    disparity.convertTo(disparity, CV_32F, 1.0 / 16.0);

    // disparity.convertTo(disparity, CV_8U, 255 / (numDisparities * 16.));
    //  cv::Mat bilateral_filtered_disp;
    //  cv::bilateralFilter(disparity, bilateral_filtered_disp, 5, 25, 25);
    //  disparity = bilateral_filtered_disp;
    //  cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8U);
}
void applyCLAHE(cv::Mat &image)
{
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    cv::Mat result;
    clahe->apply(image, result);
    image = result;
}

// Custom comparator for cv::KeyPoint with integer casting
struct KeyPointComparator
{
    bool operator()(const cv::KeyPoint &lhs, const cv::KeyPoint &rhs) const
    {
        // int lhs_x = static_cast<int>(lhs.pt.x);
        // int lhs_y = static_cast<int>(lhs.pt.y);
        // int rhs_x = static_cast<int>(rhs.pt.x);
        // int rhs_y = static_cast<int>(rhs.pt.y);
        // if (lhs_x != rhs_x) return lhs_x < rhs_x;
        // return lhs_y < rhs_y;
        if (lhs.pt.x != rhs.pt.x)
            return lhs.pt.x < rhs.pt.x;
        return lhs.pt.y < rhs.pt.y;
    }
};

// Function to filter unique keypoints with integer casting
std::vector<cv::KeyPoint> filterUniqueKeypoints(const std::vector<cv::KeyPoint> &keypoints)
{
    std::set<cv::KeyPoint, KeyPointComparator> unique_keypoints(keypoints.begin(), keypoints.end());
    return std::vector<cv::KeyPoint>(unique_keypoints.begin(), unique_keypoints.end());
}

void extractKeypoints(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints)
{
    // cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
    // detector->detect(img, keypoints); cv::SiftFeatureDetector::create();
    std::vector<cv::KeyPoint> keypoints1;
    cv::Ptr<cv::Feature2D> detector1 = cv::SiftFeatureDetector::create();
    detector1->detect(img, keypoints1);
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat contrast_brightness_img;
    img.convertTo(contrast_brightness_img, -1, 3.0, 45);
    cv::Ptr<cv::Feature2D> detector2 = cv::SiftFeatureDetector::create();
    detector2->detect(contrast_brightness_img, keypoints2);
    std::vector<cv::KeyPoint> all_keypoints = keypoints1;                            // Initialize with keypoints1
    all_keypoints.insert(all_keypoints.end(), keypoints2.begin(), keypoints2.end()); // Insert keypoints2
    // allkeypoints.push_back(cv::KeyPoint(cv::Point2f(3000.1f, 4000.3f), 1.0f));
    // allkeypoints.push_back(cv::KeyPoint(cv::Point2f(3000.2f, 4000.4f), 1.0f));
    // std::cout << "allkeypoints size " << allkeypoints.size() << std::endl;
    std::vector<cv::KeyPoint> unique_keypoints = filterUniqueKeypoints(all_keypoints);
    keypoints = unique_keypoints;

    // std::cout << "keypoints size " << keypoints.size() << std::endl;

    // cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
    // detector->detect(contrast_brightness_img, keypoints);
}

void getKeypoints_3d(cv::Mat &colors, cv::Mat &points_3d, std::vector<cv::Vec3f> &keypoints_3d, std::vector<cv::Vec3b> &keypoint_colors, std::vector<cv::KeyPoint> &keypoint_coord)
{
    int counter = 0;
    for (int j = 0; j < keypoint_coord.size(); ++j)
    {
        // int x = static_cast<int>(keypoint_coord[j].pt.x);
        // int y = static_cast<int>(keypoint_coord[j].pt.y);
        int x = static_cast<int>(std::round(keypoint_coord[j].pt.x));
        int y = static_cast<int>(std::round(keypoint_coord[j].pt.y));

        if (x >= 0 && x < points_3d.cols && y >= 0 && y < points_3d.rows)
        {
            cv::Vec3f point = points_3d.at<cv::Vec3f>(y, x);
            if (cv::checkRange(point, true, nullptr, -1e4, 1e4))
            {
                counter++;
                keypoints_3d.push_back(point);
                keypoint_colors.push_back(colors.at<cv::Vec3b>(y, x));
            }
        }
    }
    std::cout << "counter size " << counter << std::endl;
    std::string point_cloud_keypoints = "point_cloud_keypoints.ply";
    write2PLY(keypoints_3d, keypoint_colors, point_cloud_keypoints);
}

void applyWLSFilter(cv::Mat &img_left, cv::Mat &img_right, cv::Mat &filtered_disparity)
{

    // cv::Mat left_for_matcher, right_for_matcher;
    // cv::resize(img_left, left_for_matcher, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR_EXACT);
    // cv::resize(img_right, right_for_matcher, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR_EXACT);
    // cv::cvtColor(left_for_matcher, left_for_matcher, cv::COLOR_BGR2GRAY);
    // cv::cvtColor(right_for_matcher, right_for_matcher, cv::COLOR_BGR2GRAY);
    // cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create();
    // cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
    // cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
    // cv::Mat left_disp, right_disp;
    // left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
    // right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
    // left_disp.convertTo(left_disp, CV_32F, 1.0 / 16.0);
    // right_disp.convertTo(right_disp, CV_32F, 1.0 / 16.0);
    // wls_filter->setLambda(8000.0);
    // wls_filter->setSigmaColor(1.5);
    // wls_filter->filter(left_disp, img_left, filtered_disparity, right_disp);
    // img_left.convertTo(img_left, -1, 1.2, 0);
    // img_right.convertTo(img_right, -1, 1.2, 0);
    // cv::Mat imgL_gray, imgR_gray;
    // cv::cvtColor(img_left, img_left, cv::COLOR_BGR2GRAY);
    // cv::cvtColor(img_right, img_right, cv::COLOR_BGR2GRAY);

    // // Apply CLAHE to both images
    // applyCLAHE(img_left);
    // applyCLAHE(img_right);

    cv::Mat left_img_gaussian;
    cv::Mat right_img_gaussian;
    cv::GaussianBlur(img_left, left_img_gaussian, cv::Size(5, 5), 0);
    cv::GaussianBlur(img_right, right_img_gaussian, cv::Size(5, 5), 0);
    left_img_gaussian.convertTo(left_img_gaussian, -1, 0.5, 0);
    right_img_gaussian.convertTo(right_img_gaussian, -1, 0.5, 0);
    int minDisparity = 0;
    int numDisparities = 16; // Ensure this is divisible by 16
    int blockSize = 3;
    int P1 = 8 * 3 * blockSize * blockSize;
    int P2 = 32 * 3 * blockSize * blockSize;
    int disp12MaxDiff = 1;
    int preFilterCap = 1;
    int uniquenessRatio = 11;
    int speckleWindowSize = 200;
    int speckleRange = 1;
    int mode = cv::StereoSGBM::MODE_HH4;
    cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(
        minDisparity,
        numDisparities,
        blockSize,
        P1,
        P2,
        disp12MaxDiff,
        preFilterCap,
        uniquenessRatio,
        speckleWindowSize,
        speckleRange,
        mode);

    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
    cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
    cv::Mat left_disp, right_disp;
    left_matcher->compute(img_left, img_right, left_disp);
    right_matcher->compute(img_right, img_left, right_disp);
    left_disp.convertTo(left_disp, CV_32F, 1.0 / 16.0);
    right_disp.convertTo(right_disp, CV_32F, 1.0 / 16.0);
    wls_filter->setLambda(8000.0);
    wls_filter->setSigmaColor(1.5);
    wls_filter->filter(left_disp, img_left, filtered_disparity, right_disp);
    // filtered_disparity *= 16.0*16.0;
    // filtered_disparity.convertTo(filtered_disparity, CV_32F, 1.0 / 16.0);
}

void reconstruct3DPoints(cv::Mat &disparity, cv::Mat &Q, cv::Mat &points_3d)
{
    cv::reprojectImageTo3D(disparity, points_3d, Q, true);
    // cv::perspectiveTransform(points_3d, points_3d, cv::Mat::eye(4, 4, CV_64F));
}

void compute_Q_matrix(cv::Mat &P_rect_02, cv::Mat &T_02, cv::Mat &Q)
{
    // Compute the Q matrix

    double fx = P_rect_02.at<double>(0, 0);
    double cx = P_rect_02.at<double>(0, 2);
    double cy = P_rect_02.at<double>(1, 2);

    Q.at<double>(0, 0) = 1.0;
    Q.at<double>(0, 3) = -cx;
    Q.at<double>(1, 1) = 1.0;
    Q.at<double>(1, 3) = -cy;
    Q.at<double>(2, 3) = fx;
    Q.at<double>(3, 2) = -1.0 / T_02.at<double>(0, 0);
    // Q.at<double>(3, 2) = -1.0/(-0.51);
}
void write2PLY(std::vector<cv::Vec3f> &points, std::vector<cv::Vec3b> &colors, std::string &filename)
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
        // ofs << points[i][0] << " " << points[i][1] << " " << points[i][2] << "\n";
    }

    ofs.close();
}

// Function to convert all points
void getAllPoints_3d(cv::Mat &colors, cv::Mat &points_3d, std::vector<cv::Vec3f> &all_3d_points, std::vector<cv::Vec3b> &all_colors)
{
    for (int i = 0; i < points_3d.rows * 0.75; ++i)
    {
        for (int j = 0; j < points_3d.cols; ++j)
        {
            cv::Vec3f point = points_3d.at<cv::Vec3f>(i, j);
            if (cv::checkRange(point, true, nullptr, -1e4, 1e4))
            {
                all_3d_points.push_back(point);
                all_colors.push_back(colors.at<cv::Vec3b>(i, j));
            }
        }
    }
    std::string point_cloud_all_points = "point_cloud_all_points.ply";
    write2PLY(all_3d_points, all_colors, point_cloud_all_points);
}

void cvMatToEigen(cv::Mat &colors, cv::Mat &points_3d, std::vector<Eigen::Vector3d> &eigen_points, std::vector<cv::Vec3b> &out_colors, std::vector<cv::KeyPoint> &keypoint_coord)
{
    int counter = 0;
    std::cout << "keypoint_coord size " << keypoint_coord.size() << std::endl;
    std::cout << "points_3d.cols size " << points_3d.cols << std::endl;
    std::cout << "points_3d.rows size " << points_3d.rows << std::endl;
    for (int j = 0; j < keypoint_coord.size(); ++j)
    {
        int x = static_cast<int>(keypoint_coord[j].pt.x);
        int y = static_cast<int>(keypoint_coord[j].pt.y);

        if (x >= 0 && x < points_3d.cols && y >= 0 && y < points_3d.rows)
        {
            cv::Vec3f point = points_3d.at<cv::Vec3f>(y, x);
            if (cv::checkRange(point, true, nullptr, -1e4, 1e4))
            {
                counter++;
                Eigen::Vector3d eigen_point(point[0], point[1], point[2]);
                eigen_points.push_back(eigen_point);
                out_colors.push_back(colors.at<cv::Vec3b>(y, x));
            }
        }
    }
    std::cout << "counter size " << counter << std::endl;
}
void cvMatToEigenAllPoints(cv::Mat &colors, cv::Mat &points_3d, std::vector<Eigen::Vector3d> &eigen_points, std::vector<cv::Vec3b> &out_colors)
{
    for (int i = 0; i < points_3d.rows; ++i)
    {
        for (int j = 0; j < points_3d.cols; ++j)
        {
            cv::Vec3f point = points_3d.at<cv::Vec3f>(i, j);
            // if (cv::checkRange(point, true, nullptr, -1e4, 1e4) && mask.at<uchar>(i, j))
            if (cv::checkRange(point, true, nullptr, -1e4, 1e4))
            {
                Eigen::Vector3d eigen_point(point[0], point[1], point[2]);
                eigen_points.push_back(eigen_point);
                out_colors.push_back(colors.at<cv::Vec3b>(i, j));
            }
        }
    }
}

void cvMatToEigenMat(cv::Mat &cvMat, Eigen::Matrix3d &eigenMat)
{

    for (int i = 0; i < cvMat.rows; ++i)
    {
        for (int j = 0; j < cvMat.cols; ++j)
        {

            eigenMat(i, j) = cvMat.at<double>(i, j);
        }
    }
}

void writePLY2(std::vector<Eigen::Vector4f> &points, std::string &filename)
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
    ofs << "property float intensity\n";
    ofs << "end_header\n";

    for (const auto &point : points)
    {
        ofs << point.x() << " " << point.y() << " " << point.z() << " " << point.w() << "\n";
    }

    ofs.close();
}
void write2PLY2(std::vector<Eigen::Vector3d> &points, std::vector<cv::Vec3b> &colors, std::string filename)
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
    ofs << "property double x\n";
    ofs << "property double y\n";
    ofs << "property double z\n";
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
void readCalibrationFile(std::string &filename, cv::Mat &P_rect_02, cv::Mat &R_rect_00, cv::Mat &T_00, cv::Mat &T_03, cv::Mat &R, cv::Mat &T, cv::Mat &K, cv::Mat &D)
{
    std::ifstream file(filename);

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        std::string key;
        if (std::getline(ss, key, ':'))
        {
            if (key == "P_rect_00")
            {
                P_rect_02 = cv::Mat(3, 4, CV_64F);
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 4; ++j)
                        ss >> P_rect_02.at<double>(i, j);
            }
            if (key == "P_rect_01")
            {
                R_rect_00 = cv::Mat(3, 4, CV_64F);
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 4; ++j)
                        ss >> R_rect_00.at<double>(i, j);
            }
            else if (key == "T_01")
            {
                T_03 = cv::Mat(3, 1, CV_64F);
                for (int i = 0; i < 3; ++i)
                    ss >> T_03.at<double>(i, 0);
            }
            else if (key == "K_00")
            {
                K = cv::Mat(3, 3, CV_64F);
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        ss >> K.at<double>(i, j);
            }
            else if (key == "D_00")
            {
                D = cv::Mat(5, 1, CV_64F);
                for (int i = 0; i < 5; ++i)
                    ss >> D.at<double>(i, 0);
            }
            else if (key == "T_00")
            {
                T_00 = cv::Mat(3, 1, CV_64F);
                for (int i = 0; i < 3; ++i)
                    ss >> T_00.at<double>(i, 0);
            }
            else if (key == "R")
            {
                R = cv::Mat(3, 3, CV_64F);
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        ss >> R.at<double>(i, j);
            }
            else if (key == "T")
            {
                T = cv::Mat(3, 1, CV_64F);
                for (int i = 0; i < 3; ++i)
                    ss >> T.at<double>(i, 0);
            }
        }
    }

    file.close();
}

void transform_cam2velodyn(cv::Mat &R, cv::Mat &T, Eigen::Matrix4d &t_cam2velo, std::vector<Eigen::Vector3d> &eigen_points)
{
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d T_eigen;
    cvMatToEigenMat(R, R_eigen);
    for (int i = 0; i < 3; ++i)
        T_eigen(i, 0) = T.at<double>(i, 0);
    transformation.block<3, 3>(0, 0) = R_eigen;
    transformation.block<3, 1>(0, 3) = T_eigen;
    t_cam2velo = transformation.inverse();

    for (auto &point : eigen_points)
    {
        Eigen::Vector4d homogenous_point(point.x(), point.y(), point.z(), 1.0);
        Eigen::Vector4d transformed_point = t_cam2velo * homogenous_point;
        point = transformed_point.head<3>();
    }
}

void transformCam2Velodyn(cv::Mat &R, cv::Mat &T, cv::Mat &t_cam2velo, std::vector<cv::Vec3f> &points)
{
    // Initialize transformation matrix
    cv::Mat transformation = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(transformation(cv::Rect(0, 0, 3, 3)));
    T.copyTo(transformation(cv::Rect(3, 0, 1, 3)));

    // Compute inverse of transformation matrix
    t_cam2velo = transformation.inv();
    std::cout << "Original Transformation Matrix Cam to Velodyn" << std::endl;
    printMat(t_cam2velo);
    // Transform points
    for (auto &point : points)
    {
        cv::Mat homogenous_point = (cv::Mat_<double>(4, 1) << point[0], point[1], point[2], 1.0);
        cv::Mat transformed_point = t_cam2velo * homogenous_point;
        point[0] = transformed_point.at<double>(0, 0);
        point[1] = transformed_point.at<double>(1, 0);
        point[2] = transformed_point.at<double>(2, 0);
    }
}
void write_point_cloud(std::vector<Eigen::Vector3d> &points, std::vector<Eigen::Vector4f> &intensity, std::string filename)
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
    ofs << "property double x\n";
    ofs << "property double y\n";
    ofs << "property double z\n";
    ofs << "property float intensity\n";
    ofs << "end_header\n";

    for (size_t i = 0; i < points.size(); ++i)
    {
        ofs << points[i][0] << " " << points[i][1] << " " << points[i][2] << " " << intensity[i].w() << "\n";
    }

    ofs.close();
}