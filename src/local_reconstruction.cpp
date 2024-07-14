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
void computeDisparity(cv::Mat &left_img, cv::Mat &right_img, cv::Mat &disparity)
{
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
    stereo->compute(left_img, right_img, disparity);
    disparity.convertTo(disparity, CV_32F, 1.0 / 16.0);
}
void applyCLAHE(cv::Mat& image) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    cv::Mat result;
    clahe->apply(image, result);
    image = result;
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

    int minDisparity = 0;
    int numDisparities = 16; // Must be divisible by 16
    int blockSize = 7;       // Must be odd
    int P1 = 1 * 3 * blockSize * blockSize;
    int P2 = 4 * 3 * blockSize * blockSize;
    int disp12MaxDiff = 0;
    int preFilterCap = 0;
    int uniquenessRatio = 50; // Adjust this to influence confidence
    int speckleWindowSize = 0;
    int speckleRange = 2;
    int mode = cv::StereoSGBM::MODE_HH4;
    cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(0, 16, 7, 0, 0, 0, 0, 30, 100, 2, cv::StereoSGBM::MODE_HH);
    // cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(
    //     minDisparity,
    //     numDisparities,
    //     blockSize,
    //     P1,
    //     P2,
    //     disp12MaxDiff,
    //     preFilterCap,
    //     uniquenessRatio,
    //     speckleWindowSize,
    //     speckleRange,
    //     mode);

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
    cv::reprojectImageTo3D(disparity, points_3d, Q);
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
    }

    ofs.close();
}

// Function to convert keypoint coordinates
void getKeypoints_3d(cv::Mat &colors, cv::Mat &points_3d, std::vector<cv::Vec3f> &keypoints_3d, std::vector<cv::Vec3b> &keypoint_colors, std::vector<cv::KeyPoint> &keypoint_coord)
{
    int counter = 0;
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
                keypoints_3d.push_back(point);
                keypoint_colors.push_back(colors.at<cv::Vec3b>(y, x));
            }
        }
    }
    std::cout << "counter size " << counter << std::endl;
    std::string point_cloud_keypoints = "point_cloud_keypoints.ply";
    write2PLY(keypoints_3d, keypoint_colors, point_cloud_keypoints);
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
    // for (int i = 0; i < points_3d.rows; ++i)
    // {
    //     for (int j = 0; j < points_3d.cols; ++j)
    //     {
    //         cv::Vec3f point = points_3d.at<cv::Vec3f>(i, j);
    //         // if (cv::checkRange(point, true, nullptr, -1e4, 1e4) && mask.at<uchar>(i, j))
    //         if (cv::checkRange(point, true, nullptr, -1e4, 1e4))
    //         {
    //             Eigen::Vector3d eigen_point(point[0], point[1], point[2]);
    //             eigen_points.push_back(eigen_point);
    //             out_colors.push_back(colors.at<cv::Vec3b>(i, j));
    //         }
    //     }
    // }
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
void readCalibrationFile(std::string &filename, cv::Mat &P_rect_02, cv::Mat &T_03, cv::Mat &R, cv::Mat &T)
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
                        ss >> P_rect_02.at<double>(i, j);
            }
            else if (key == "T_03")
            {
                T_03 = cv::Mat(3, 1, CV_64F);
                for (int i = 0; i < 3; ++i)
                    ss >> T_03.at<double>(i, 0);
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

void generatePointCloud(cv::Mat &disparity, cv::Mat &left_img, cv::Mat &Q, open3d::geometry::PointCloud &pointCloud)
{

    for (int i = 0; i < disparity.rows; i++)
    {
        for (int j = 0; j < disparity.cols; j++)
        {
            float d = disparity.at<float>(i, j);
            if (d <= 0 || d >= 96 || std::isnan(d) || std::isinf(d)) // Filter out invalid disparities
                continue;

            cv::Mat vec = (cv::Mat_<double>(4, 1) << j, i, d, 1.0);
            cv::Mat xyz = Q * vec;
            xyz /= xyz.at<double>(3, 0);

            double x = xyz.at<double>(0, 0);
            double y = xyz.at<double>(1, 0);
            double z = xyz.at<double>(2, 0);

            if (std::isnan(x) || std::isinf(x) ||
                std::isnan(y) || std::isinf(y) ||
                std::isnan(z) || std::isinf(z))
            {
                continue;
            }
            // if(mask.at<uchar>(i, j)) {
            Eigen::Vector3d point(x, y, z);
            cv::Vec3b colorVec = left_img.at<cv::Vec3b>(i, j);
            Eigen::Vector3d color(colorVec[2] / 255.0, colorVec[1] / 255.0, colorVec[0] / 255.0);

            pointCloud.points_.push_back(point);
            pointCloud.colors_.push_back(color);
            //}
        }
    }
}

void filterPointCloud(open3d::geometry::PointCloud &pointCloud, cv::Mat &disparity)
{
    double min_disparity;
    cv::minMaxIdx(disparity, &min_disparity, nullptr);
    cv::Mat mask = disparity > min_disparity;

    // Create a new point cloud to store filtered points
    open3d::geometry::PointCloud filteredPointCloud;
    for (size_t i = 0; i < pointCloud.points_.size(); ++i)
    {
        const auto &point = pointCloud.points_[i];
        // if (mask.at<uint8_t>(i) && fabs(point.x()) < 10.5) {
        if (mask.at<uint8_t>(i))
        {
            filteredPointCloud.points_.push_back(point);
            filteredPointCloud.colors_.push_back(pointCloud.colors_[i]);
        }
    }

    // Replace the original point cloud with the filtered one
    pointCloud = filteredPointCloud;
}

// Function implementations
//  ORB detected too little keypoints
//  FAST detected much more keypoints
//  AKAZE was not bad but less keypoints compared to SIFT
void extractKeypointsAndDescriptors(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    // cv::Ptr<cv::Feature2D> detector = cv::SiftFeatureDetector::create();
    // detector->detect(img, keypoints);

    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
    detector->detect(img, keypoints);

    // cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create();
    // detector->detect(img, keypoints);

    // cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    // detector->detect(img, keypoints);

    // cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
    // detector->detect(img, keypoints);

    // cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    // extractor->compute(img, keypoints, descriptors);

    //     // Sub-pixel refinement
    // cv::Mat gray;
    // cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // Convert to grayscale
    // std::vector<cv::Point2f> points;
    // for (auto &kp : keypoints) {
    //     points.push_back(kp.pt);
    // }

    // cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);
    // cv::cornerSubPix(gray, points, cv::Size(5, 5), cv::Size(-1, -1), criteria);

    // for (size_t i = 0; i < points.size(); ++i) {
    //     keypoints[i].pt = points[i];
    // }
}

void matchDescriptors(cv::Mat &descriptors1, cv::Mat &descriptors2, std::vector<cv::DMatch> &good_matches)
{
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
}

void filter3DPoints(std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2,
                    std::vector<cv::DMatch> &matches, std::vector<Eigen::Vector3d> &points3D,
                    std::vector<Eigen::Vector3d> &filteredPoints3D)
{
    std::set<int> used_indices;
    for (const auto &match : matches)
    {
        used_indices.insert(match.queryIdx);
    }

    for (size_t i = 0; i < points3D.size(); ++i)
    {
        if (used_indices.count(i))
        {
            filteredPoints3D.push_back(points3D[i]);
        }
    }
}

void getKeyPointCoordinatesFromMatches(std::vector<cv::KeyPoint> &keypoints_left,
                                       std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &keypoint_coord)
{
    // Iterate through matches
    for (size_t i = 0; i < matches.size(); ++i)
    {
        // Get the indices of keypoints in the left and right images
        int queryIdx = matches[i].queryIdx;

        // Get the keypoints from keypoints_left and keypoints_right
        cv::KeyPoint kp_left = keypoints_left[queryIdx];

        // Extract image coordinates
        cv::Point2f coord_left = kp_left.pt;
        keypoint_coord.push_back(coord_left);
    }
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