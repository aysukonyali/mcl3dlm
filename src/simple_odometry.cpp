#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <open3d/Open3D.h>
#include "open3d/geometry/PointCloud.h"
#include <opencv2/opencv.hpp>
#include <opencv2/surface_matching/icp.hpp>
#include <vector>
#include <random>
#include <opencv2/core/eigen.hpp>
#include "local_reconstruction.h"
#include "data_association.h"
#include "alignment.h"
#include "simple_odometry.h"

// bool operator<(cv::KeyPoint& lhs, cv::KeyPoint& rhs) {
//     if (lhs.pt.x != rhs.pt.x) return lhs.pt.x < rhs.pt.x;
//     if (lhs.pt.y != rhs.pt.y) return lhs.pt.y < rhs.pt.y;
//     if (lhs.size != rhs.size) return lhs.size < rhs.size;
//     if (lhs.angle != rhs.angle) return lhs.angle < rhs.angle;
//     if (lhs.response != rhs.response) return lhs.response < rhs.response;
//     if (lhs.octave != rhs.octave) return lhs.octave < rhs.octave;
//     return lhs.class_id < rhs.class_id;
// }

void get3dPointsInVelodyn(Image &i,
                          cv::Mat &colors,
                          cv::Mat &points_3d,
                          std::vector<cv::Vec3f> &keypoints_3d,
                          std::vector<cv::Vec3b> &keypoint_colors,
                          std::vector<cv::KeyPoint> &keypoint_coord,
                          cv::Mat &R,
                          cv::Mat &T)
{
    int counter = 0;
    cv::Mat t_cam2velo;
    cv::Mat transformation = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(transformation(cv::Rect(0, 0, 3, 3)));
    T.copyTo(transformation(cv::Rect(3, 0, 1, 3)));

    // Compute inverse of transformation matrix
    t_cam2velo = transformation.inv();
    for (int j = 0; j < keypoint_coord.size(); ++j)
    {

        int x = static_cast<int>(std::round(keypoint_coord[j].pt.x));
        int y = static_cast<int>(std::round(keypoint_coord[j].pt.y));

        if (x >= 0 && x < points_3d.cols && y >= 0 && y < points_3d.rows)
        {
            cv::Vec3f point = points_3d.at<cv::Vec3f>(y, x);
            cv::Vec3b color = colors.at<cv::Vec3b>(y, x);
            if (cv::checkRange(point, true, nullptr, -1e4, 1e4))
            {
                counter++;
                keypoints_3d.push_back(point);
                keypoint_colors.push_back(color);
                cv::Mat homogenous_point = (cv::Mat_<double>(4, 1) << point[0], point[1], point[2], 1.0);
                cv::Mat transformed_point = t_cam2velo * homogenous_point;
                point[0] = transformed_point.at<double>(0, 0);
                point[1] = transformed_point.at<double>(1, 0);
                point[2] = transformed_point.at<double>(2, 0);

                if (point[0] < 65.0 && point[1] > -20.0)
                {
                    Eigen::Vector3d point_eigen(static_cast<double>(point[0]),
                                                static_cast<double>(point[1]),
                                                static_cast<double>(point[2]));

                    // Add point to Open3D point cloud
                    i.pc.points_.push_back(point_eigen);

                    // Convert color from cv::Vec3b (BGR) to Open3D RGB format
                    i.pc.colors_.emplace_back(
                        static_cast<double>(color[2]) / 255.0, // B channel to R
                        static_cast<double>(color[1]) / 255.0, // G channel remains G
                        static_cast<double>(color[0]) / 255.0  // R channel to B
                    );
                    i.keypointsTo3DPoints[keypoint_coord[j]] = point;
                    i.keypointsToColor[keypoint_coord[j]] = colors.at<cv::Vec3b>(y, x);
                    i.colors.push_back(colors.at<cv::Vec3b>(y, x));
                    i.keypoints.push_back(keypoint_coord[j]);
                }
            }
        }
    }
    cv::Mat descriptors;
    computeDescriptors(i.left_img, i.keypoints, descriptors);
    cv::Mat descLeft32F;
    descriptors.convertTo(descLeft32F, CV_32F);
    i.descriptors = descLeft32F;
}
void computeDescriptors(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    cv::Ptr<cv::Feature2D> extractor = cv::SIFT::create();
    extractor->compute(img, keypoints, descriptors);
}
void get_stereo_first(Image &i,
                      cv::Mat P_rect_02,
                      cv::Mat T_03,
                      cv::Mat R,
                      cv::Mat T)
{

    cv::Mat disparity;
    computeDisparity(i.left_img, i.right_img, disparity);
    cv::Mat disp_normalized;
    cv::normalize(disparity, disp_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Disparity Map", disp_normalized);
    cv::waitKey(1);

    std::vector<cv::KeyPoint> keypoints_left;
    extractKeypoints(i.left_img, keypoints_left);
    cv::Mat img_with_keypoints;
    cv::drawKeypoints(i.left_img, keypoints_left, img_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("Keypoints", img_with_keypoints);
    cv::waitKey(1);

    cv::Mat Q = cv::Mat::zeros(4, 4, CV_64F);
    compute_Q_matrix(P_rect_02, T_03, Q);

    cv::Mat points_3d;
    reconstruct3DPoints(disparity, Q, points_3d);

    std::vector<cv::Vec3f> keypoints_3d;
    std::vector<cv::Vec3b> keypoints_colors;
    cv::Mat colors;
    cv::cvtColor(i.left_img, colors, cv::COLOR_BGR2RGB);

    get3dPointsInVelodyn(i, colors, points_3d, keypoints_3d, keypoints_colors, keypoints_left, R, T);
}
open3d::geometry::PointCloud get_stereo_second(Image &i,
                                               cv::Mat P_rect_02,
                                               cv::Mat T_03)
{

    cv::Mat disparity;
    computeDisparity(i.left_img, i.right_img, disparity);
    cv::Mat disp_normalized;
    cv::normalize(disparity, disp_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Disparity Map", disp_normalized);
    cv::waitKey(1);

    cv::Mat Q = cv::Mat::zeros(4, 4, CV_64F);
    compute_Q_matrix(P_rect_02, T_03, Q);

    cv::Mat points_3d;
    reconstruct3DPoints(disparity, Q, points_3d);

    std::vector<cv::Vec3f> keypoints_3d;
    std::vector<cv::Vec3b> keypoints_colors;
    cv::Mat colors;
    cv::cvtColor(i.left_img, colors, cv::COLOR_BGR2RGB);
    std::vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::Feature2D> detector1 = cv::SiftFeatureDetector::create();
    detector1->detect(i.left_img, keypoints);
    getKeypoints_3d(colors, points_3d, keypoints_3d, keypoints_colors, keypoints);
    open3d::geometry::PointCloud point_cloud;
    for (size_t i = 0; i < keypoints_3d.size(); ++i)
    {
        Eigen::Vector3d point_eigen(static_cast<double>(keypoints_3d[i][0]),
                                    static_cast<double>(keypoints_3d[i][1]),
                                    static_cast<double>(keypoints_3d[i][2]));

        // Add point to Open3D point cloud
        point_cloud.points_.push_back(point_eigen);

        // Convert color from cv::Vec3b (BGR) to Open3D RGB format
        point_cloud.colors_.emplace_back(
            static_cast<double>(keypoints_colors[i][2]) / 255.0, // B channel to R
            static_cast<double>(keypoints_colors[i][1]) / 255.0, // G channel remains G
            static_cast<double>(keypoints_colors[i][0]) / 255.0  // R channel to B
        );
    }
    return point_cloud;
}
void filterMatchesWithRatioTest(const std::vector<std::vector<cv::DMatch>> &knn_matches, std::vector<cv::DMatch> &good_matches, float ratio_thresh = 0.75)
{
    for (size_t i = 0; i < knn_matches.size(); ++i)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
}
void getKeypoints(cv::Mat &points_3d, std::vector<Eigen::Vector3d> &keypoints_3d, std::vector<cv::KeyPoint> &keypoint_coord)
{
    int counter = 0;
    for (int j = 0; j < keypoint_coord.size(); ++j)
    {

        int x = static_cast<int>(std::round(keypoint_coord[j].pt.x));
        int y = static_cast<int>(std::round(keypoint_coord[j].pt.y));

        if (x >= 0 && x < points_3d.cols && y >= 0 && y < points_3d.rows)
        {
            cv::Vec3f point = points_3d.at<cv::Vec3f>(y, x);
            Eigen::Vector3d eigenVec;
            eigenVec << point[0], point[1], point[2];
            if (cv::checkRange(point, true, nullptr, -1e4, 1e4))
            {
                counter++;
                keypoints_3d.push_back(eigenVec);
            }
        }
    }
}
void keypointsMatchingAndPnP(Image &current, Image &next, cv::Mat K, cv::Mat D, cv::Mat P_rect_02, cv::Mat T_03)
{

    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_L2);
    matcher.match(current.descriptors, next.descriptors, matches);

    // KNN match to get two nearest neighbors
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(current.descriptors, next.descriptors, knn_matches, 2);

    // Apply ratio test to filter matches
    filterMatchesWithRatioTest(knn_matches, matches);

    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point2f> points2d;

    std::vector<cv::Vec3b> new_colors;
    std::vector<cv::KeyPoint> new_keypoints;

    for (auto &match : matches)
    {
        cv::KeyPoint keypoint_query = current.keypoints[match.queryIdx];
        cv::Vec3f vec = current.keypointsTo3DPoints[keypoint_query];
        cv::Point3f point3d(vec[0], vec[1], vec[2]);
        points3d.push_back(point3d);
        cv::KeyPoint keypoint_train = next.keypoints[match.trainIdx];
        cv::Point2f point2d = keypoint_train.pt;
        points2d.push_back(point2d);
        cv::Vec3b color = next.colors[match.trainIdx];
        new_colors.push_back(color);
        new_keypoints.push_back(keypoint_train);
        next.pc.colors_.emplace_back(
            static_cast<double>(color[2]) / 255.0, // B channel to R
            static_cast<double>(color[1]) / 255.0, // G channel remains G
            static_cast<double>(color[0]) / 255.0  // R channel to B
        );
    }
    next.keypoints = new_keypoints;
    next.colors = new_colors;
    cv::Mat Q = cv::Mat::zeros(4, 4, CV_64F);
    compute_Q_matrix(P_rect_02, T_03, Q);
    cv::Mat disparity;
    computeDisparity(next.left_img, next.right_img, disparity);
    cv::Mat points_3d;
    reconstruct3DPoints(disparity, Q, points_3d);
    std::vector<Eigen::Vector3d> keypoints_3d;
    getKeypoints(points_3d, keypoints_3d, new_keypoints);
    for (const auto &kp : new_keypoints)
    {
        // Convert OpenCV keypoint coordinates to Eigen::Vector3d
        Eigen::Vector3d pixel2cam;
        pixel2cam << kp.pt.x, kp.pt.y, 1.0;

        // Convert to normalized camera coordinates using Eigen and the intrinsic matrix K
        Eigen::Matrix3d K_eigen;
        cv::cv2eigen(K, K_eigen); // Convert cv::Mat to Eigen::Matrix3d

        int x = static_cast<int>(std::round(kp.pt.x));
        int y = static_cast<int>(std::round(kp.pt.y));
        float disparityValue = disparity.at<float>(y, x);
        if (cv::checkRange(disparityValue, true, nullptr, -1e4, 1e4))
        {
            float focal_length = 7.215377e+02;
            float baseline = -5.370000e-01;
            float depth = (focal_length * baseline) / disparityValue;
            if (cv::checkRange(depth, true, nullptr, -1e4, 1e4))
            {
                Eigen::Vector3d pixel2cam_normalized = K_eigen.inverse() * pixel2cam;
                //pixel2cam_normalized = pixel2cam;

                // pixel2cam_normalized[0] = (kp.pt.x - K.at<double>(0, 2)) / K.at<double>(0, 0);
                // pixel2cam_normalized[1] = (kp.pt.y - K.at<double>(1, 2)) / K.at<double>(1, 1);
                // pixel2cam_normalized[2] *= depth;
                // Eigen::Vector3d p = (pixel2cam_normalized * depth);
                // cv::Mat cvMat(3, 1, CV_64F);

                // Copy data from Eigen::Vector3d to cv::Mat
                // cvMat.at<double>(0, 0) = p.x();
                // cvMat.at<double>(1, 0) = p.y();
                // cvMat.at<double>(2, 0) = p.z();
                // if (cv::checkRange(cvMat, true, nullptr, -1e4, 1e4))
                // {
                if (!(pixel2cam_normalized * depth).hasNaN())
                {
                    
                    next.pc.points_.push_back(pixel2cam_normalized * depth);
                }
                std::cout << "size " << next.pc.points_.size() << std::endl;
                //}
                // next.pc.points_.push_back(pixel2cam);
            }
        }
    }
    std::vector<Eigen::Vector3d> filteredPoints;

    // next.pc.points_ = keypoints_3d;
    open3d::io::WritePointCloud("outputfsfsd.ply", next.pc);
    // Output rotation and translation vectors
    cv::Mat rvec, tvec;
    cv::Mat d;
    cv::Mat a;
    cv::invert(K,d);
    bool success = cv::solvePnP(points3d, points2d, K, D, rvec, tvec,false, cv::SOLVEPNP_EPNP);
    if (success)
    {
        std::cout << "Rotation Vector (rvec): \n"
                  << rvec << std::endl;
        std::cout << "Translation Vector (tvec): \n"
                  << tvec << std::endl;

        // Convert rotation vector to rotation matrix
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        cv::Mat T = cv::Mat::zeros(4, 4, CV_64F);

        // Fill in the rotation matrix
        cv::Mat R_inv;
        cv::invert(R, R_inv);
        cv::Mat r = R.t();

        R.copyTo(T(cv::Rect(0, 0, 3, 3)));

        // Fill in the translation vector
        cv::Mat t_inv = -r * tvec; // Compute -R_inv * t
        T.at<double>(0, 3) = tvec.at<double>(0);
        T.at<double>(1, 3) = tvec.at<double>(1);
        T.at<double>(2, 3) = tvec.at<double>(2);

        // Set the bottom-right corner to 1
        T.at<double>(3, 3) = 1.0;
   
        // cv::Mat T_inv;
        // cv::invert(T, T_inv);
        Eigen::Matrix4d T_inv = cvMatToEigen4(T).inverse();

        Eigen::Matrix4d eigen_T = T_inv;

        next.T_i1_i2 = eigen_T;
        std::cout << "next.T_i1_i2_: first  " << next.T_i1_i2 << std::endl;

        // cv::Mat P = cv::Mat::zeros(4, 4, CV_64F);

        // // Create the 3x4 RT matrix
        // cv::Mat RT(3, 4, CV_64F);
        // cv::hconcat(R, tvec, RT);

        // // Fill the 3x3 top-left of the 4x4 matrix with the intrinsic matrix K
        // cv::Mat K_inv;
        // cv::invert(K, K_inv);
        // K_inv.copyTo(P(cv::Rect(0, 0, 3, 3)));

        // // Fill the 3x4 top-right of the 4x4 matrix with the extrinsic matrix RT
        // RT.copyTo(P(cv::Rect(0, 0, 4, 3)));

        // // Set the bottom row of the 4x4 matrix
        // P.at<double>(3, 3) = 1.0;
        // cv::Mat InvP;
        // cv::invert(P, InvP);
        // Eigen::Matrix4d eigen_T = cvMatToEigen4(InvP);
        // next.T_i1_i2 = eigen_T;
    }
    else
    {
        std::cout << "PnP solution was not successful." << std::endl;
    }

    //  cv::Mat K_inv;
    //  cv::invert(K, K_inv);

    // // Convert 2D points to normalized coordinates
    // cv::Mat points2d_homogeneous;
    // std::cout << "fsdfsdfsdfsdffsfsdfsdfsdfsdfds" << std::endl;
    // cv::convertPointsToHomogeneous(points2d, points2d_homogeneous);
    // cv::Mat R;
    // std::cout << "fsdfsdfsdfsdffsdfsdfsdfsdf"<< points2d_homogeneous.size << std::endl;
    // cv::Mat points3d_normalized = K_inv * points2d_homogeneous; // Transpose for matrix multiplication
    // std::cout << "fsdfsdfsdfsdf" << std::endl;
    // cv::Rodrigues(rvec, R);                                         // Convert rotation vector to rotation matrix

    // // Transform the normalized 3D points using the rotation and translation
    // cv::Mat tvec_reshaped = tvec.reshape(1, 3);                             // Make sure tvec is in the correct shape
    // cv::Mat points3d_transformed = R * points3d_normalized + tvec_reshaped; // Points in world coordinates

    // // Transpose to match original point orientation
    // points3d_transformed = points3d_transformed.t();

    // // Convert Eigen::MatrixXd to open3d::geometry::PointCloud
    // auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();

    // // Ensure the matrix is 3xN for the points
    // if (points3d_transformed.rows != 3 && points3d_transformed.cols > 0)
    // {
    //     // Extract points
    //     for (int i = 0; i < points3d_transformed.cols; ++i)
    //     {
    //         Eigen::Vector3d point(points3d_transformed.at<double>(0, i), points3d_transformed.at<double>(1, i), points3d_transformed.at<double>(2, i));
    //         point_cloud->points_.push_back(point);
    //     }
    // }
    // open3d::geometry::PointCloud second = get_stereo_second(next, P_rect_02, T_03, new_keypoints);
    // open3d::visualization::Visualizer visualizer;
    // visualizer.CreateVisualizerWindow("Point Cloud 2", 1600, 1200);
    // visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(current.pc));
    // visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(second));
    // // visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(second.Transform(next.T_i1_i2)));
    // //  visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(current.pc.Transform(next.T_i1_i2.inverse())));
    // // visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(next.pc));
    // visualizer.Run();
    // visualizer.DestroyVisualizerWindow();

    drawPnPMatches(current.pc, next.pc, matches);
}
void getLeftKeypoints_odometry(Image &i)
{

    cv::Mat descriptorsLeft;
    std::vector<cv::Vec3b> colors;
    std::vector<cv::KeyPoint> keypointsLeft;
    extractKeypoints_odometry(i.left_img, keypointsLeft, colors);
    computeDescriptors(i.left_img, keypointsLeft, descriptorsLeft);
    cv::Mat descLeft32F;
    descriptorsLeft.convertTo(descLeft32F, CV_32F);
    i.keypoints = keypointsLeft;
    i.colors = colors;
    i.descriptors = descLeft32F;
}
struct KeyPointComparator
{
    bool operator()(const cv::KeyPoint &lhs, const cv::KeyPoint &rhs) const
    {
        if (lhs.pt.x != rhs.pt.x)
            return lhs.pt.x < rhs.pt.x;
        return lhs.pt.y < rhs.pt.y;
    }
};
// Function to filter unique keypoints with integer casting
std::vector<cv::KeyPoint> filterUniqueKeypointss(const std::vector<cv::KeyPoint> &keypoints)
{
    std::set<cv::KeyPoint, KeyPointComparator> unique_keypoints(keypoints.begin(), keypoints.end());
    return std::vector<cv::KeyPoint>(unique_keypoints.begin(), unique_keypoints.end());
}
void extractKeypoints_odometry(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, std::vector<cv::Vec3b> &colors_odometry)
{

    std::vector<cv::KeyPoint> keypoints1;
    cv::Ptr<cv::Feature2D> detector1 = cv::SiftFeatureDetector::create();
    detector1->detect(img, keypoints);
    // std::vector<cv::KeyPoint> keypoints2;
    // cv::Mat contrast_brightness_img;
    // img.convertTo(contrast_brightness_img, -1, 3.0, 10);
    // cv::Ptr<cv::Feature2D> detector2 = cv::SiftFeatureDetector::create();
    // detector2->detect(img, keypoints2);
    // std::vector<cv::KeyPoint> all_keypoints = keypoints1;                            // Initialize with keypoints1
    // all_keypoints.insert(all_keypoints.end(), keypoints2.begin(), keypoints2.end()); // Insert keypoints2

    // std::vector<cv::KeyPoint> unique_keypoints = filterUniqueKeypointss(all_keypoints);
    // keypoints = unique_keypoints;
    cv::Mat colors;
    cv::cvtColor(img, colors, cv::COLOR_BGR2RGB);
    for (int j = 0; j < keypoints.size(); ++j)
    {
        int x = static_cast<int>(std::round(keypoints[j].pt.x));
        int y = static_cast<int>(std::round(keypoints[j].pt.y));

        colors_odometry.push_back(colors.at<cv::Vec3b>(y, x));
    }
}
void drawPnPMatches(
    const open3d::geometry::PointCloud &source,
    const open3d::geometry::PointCloud &target,
    const std::vector<cv::DMatch> &matches)
{
    // Create a LineSet to visualize the correspondences
    auto line_set = std::make_shared<open3d::geometry::LineSet>();

    // Offset to separate the two point clouds visually
    Eigen::Vector3d offset(0.0, 0.0, 12.0);

    // Add the points from both point clouds to the LineSet
    line_set->points_.resize(source.points_.size() + target.points_.size());
    for (size_t i = 0; i < source.points_.size(); ++i)
    {
        line_set->points_[i] = source.points_[i];
    }
    for (size_t i = 0; i < target.points_.size(); ++i)
    {
        line_set->points_[i + source.points_.size()] = target.points_[i] + offset;
    }
    int counter = 0;
    // Add lines for correspondences
    for (const auto &match : matches)
    {   
        if(counter%49==0){
        // The source index is match.queryIdx, target index is match.trainIdx
        int source_idx = match.queryIdx;
        int target_idx = match.trainIdx + source.points_.size();

        line_set->lines_.push_back(Eigen::Vector2i(source_idx, target_idx));
        line_set->colors_.push_back(Eigen::Vector3d(1, 0, 0)); // Red color for lines
        }
        counter++;
    }

    // Visualize the source and target point clouds along with the correspondences
    open3d::visualization::Visualizer visualizer;
    visualizer.CreateVisualizerWindow("Correspondences PnP", 1600, 1200);

    auto source_ptr = std::make_shared<open3d::geometry::PointCloud>(source);
    auto target_ptr = std::make_shared<open3d::geometry::PointCloud>(target);
    for (auto &point : target_ptr->points_)
    {
        point += offset;
    }

    visualizer.AddGeometry(source_ptr);
    visualizer.AddGeometry(target_ptr);
    visualizer.AddGeometry(line_set);
    visualizer.GetRenderOption().point_size_ = 11.0;
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
}
// Convert a cv::Mat (4x4, double type) to Eigen::Matrix4d
Eigen::Matrix4d cvMatToEigen4(cv::Mat &mat)
{

    Eigen::Matrix4d eigenMat = Eigen::Matrix4d::Identity();
    // Copy data from cv::Mat to Eigen::Matrix4d
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            eigenMat(i, j) = mat.at<double>(i, j);
        }
    }
    return eigenMat;
}
// Function to convert a series of transformation matrices to camera positions
std::vector<Eigen::Vector3d> ConvertMatricesToCameraPositions(const std::vector<Eigen::Matrix4d> &transforms)
{
    std::vector<Eigen::Vector3d> positions;
    for (const auto &transform : transforms)
    {
        positions.push_back(transform.block<3, 1>(0, 3)); // Extract the translation component
    }
    return positions;
}

// Function to convert a series of camera positions to a LineSet
std::shared_ptr<open3d::geometry::LineSet> CreateCameraTrajectoryLineSet(
    const std::vector<Eigen::Vector3d> &camera_positions, const Eigen::Vector3d &color)
{
    auto line_set = std::make_shared<open3d::geometry::LineSet>();

    std::vector<Eigen::Vector3d> points(camera_positions.begin(), camera_positions.end());
    line_set->points_.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i)
    {
        line_set->points_[i] = points[i];
    }

    std::vector<Eigen::Vector2i> lines;
    for (size_t i = 0; i < points.size() - 1; ++i)
    {
        lines.push_back(Eigen::Vector2i(i, i + 1));
    }
    line_set->lines_.resize(lines.size());
    for (size_t i = 0; i < lines.size(); ++i)
    {
        line_set->lines_[i] = lines[i];
    }

    std::vector<Eigen::Vector3d> colors(lines.size(), color);
    line_set->colors_.resize(colors.size());
    for (size_t i = 0; i < colors.size(); ++i)
    {
        line_set->colors_[i] = colors[i];
    }

    return line_set;
}

// Function to visualize the target point cloud and two camera trajectories
void VisualizeTrajectories(
    const std::vector<Eigen::Vector3d> &traj1,
    const std::vector<Eigen::Vector3d> &traj2,
    open3d::geometry::PointCloud lidar)
{

    // Create LineSets for the camera trajectories
    auto line_set1 = CreateCameraTrajectoryLineSet(traj1, Eigen::Vector3d(1, 0, 0)); // Red
    auto line_set2 = CreateCameraTrajectoryLineSet(traj2, Eigen::Vector3d(0, 0, 1)); // Blue

    // Create a visualizer and add geometries
    open3d::visualization::Visualizer visualizer;
    visualizer.CreateVisualizerWindow("Camera Trajectories Comparison", 800, 600);
    visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(lidar));
    visualizer.AddGeometry(line_set1);
    visualizer.AddGeometry(line_set2);

    // Run the visualizer
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
}
void correspondences_filtering_odometry(Image &i,
                                        double voxel_size,
                                        int N_min,
                                        int N_standard_devitation,
                                        open3d::pipelines::registration::CorrespondenceSet &correspondences,
                                        open3d::geometry::PointCloud &stereo3d,
                                        open3d::geometry::PointCloud &lidar_t0_t1_t2)
{

    // Voxelize the source point cloud

    auto voxel_grid = open3d::geometry::VoxelGrid::CreateFromPointCloud(lidar_t0_t1_t2, voxel_size);
    open3d::io::WriteVoxelGrid("lidar_voxel_grid.ply", *voxel_grid);

    // each point in the point cloud is in a voxel
    std::unordered_map<Eigen::Vector3d, Eigen::Vector3i, open3d::utility::hash_eigen<Eigen::Vector3d>> point_to_voxel_map;
    // each voxel has its Voxel data (T, standart deviation vector, number of points)
    std::unordered_map<Eigen::Vector3i, VoxelData, open3d::utility::hash_eigen<Eigen::Vector3i>> voxel_to_data_map;
    computeVoxelMaps(*voxel_grid, lidar_t0_t1_t2, point_to_voxel_map, voxel_to_data_map);
    drawCorrespondences(stereo3d, lidar_t0_t1_t2, correspondences);
    std::cout << "correspondences size: " << correspondences.size() << std::endl;
    open3d::pipelines::registration::CorrespondenceSet refined_correspondences = refineCorrespondences(
        correspondences,
        stereo3d,
        lidar_t0_t1_t2,
        voxel_to_data_map,
        point_to_voxel_map,
        N_min,                  // N_min
        N_standard_devitation); // N_standard_deviation
    std::cout << "refined_correspondences size: " << refined_correspondences.size() << std::endl;
    drawRefinedCorrespondences(stereo3d, lidar_t0_t1_t2, correspondences, refined_correspondences);

    // Estimate the transformation matrix using point-to-point transformation estimation
    open3d::pipelines::registration::TransformationEstimationPointToPoint estimation;
    Eigen::Matrix4d correspondence_filtering_transformation = estimation.ComputeTransformation(
        stereo3d, lidar_t0_t1_t2, refined_correspondences);

    i.T_refined = correspondence_filtering_transformation;

    stereo3d.Transform(correspondence_filtering_transformation);
     open3d::visualization::Visualizer visualizer;
    visualizer.CreateVisualizerWindow("Camera Trajectories Comparison", 800, 600);
    visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(stereo3d));
    visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(lidar_t0_t1_t2));
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
}

void runSimpleOdometry(std::vector<cv::Mat> leftImages, std::vector<cv::Mat> rightImages, std::vector<std::vector<Eigen::Vector4f>> lidarMaps, cv::Mat P_rect_02, cv::Mat T_03, cv::Mat R, cv::Mat T, cv::Mat K, cv::Mat D)
{
    std::vector<Eigen::Matrix4d> T_1_2;
    std::vector<Eigen::Matrix4d> T_1_2_refined;
    open3d::geometry::PointCloud dense_lidar = get_denser_lidar_map_from_velodyns(lidarMaps, 3);
    std::vector<Image> images;
    Image img;
    img.id = 0;
    img.left_img = leftImages[0];
    img.right_img = rightImages[0];
    get_stereo_first(img, P_rect_02, T_03, R, T);

    Image current = img;

    for (size_t i = 1; i < 10; i++)
    {
        Image next;
        next.id = i;
        next.left_img = leftImages[i];
        next.right_img = rightImages[i];
        getLeftKeypoints_odometry(next);
        keypointsMatchingAndPnP(current, next, K, D, P_rect_02, T_03);

        if (i > 2)
        {
            Eigen::Matrix4d t = next.T_i1_i2;
            next.T_i1_i2 = current.T_i1_i2 * t;
        }
        T_1_2.push_back(next.T_i1_i2);
        Eigen::Matrix4d t = next.T_i1_i2;
        std::cout << "next.T_i1_i2_: " << next.T_i1_i2 << std::endl;
        // get_stereo_first(next,P_rect_02,T_03,R,T);
        // open3d::visualization::Visualizer visualizer;
        // visualizer.CreateVisualizerWindow("Point Cloud sadsad", 1600, 1200);
        // visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(next.pc.Transform(t)));
        // visualizer.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(dense_lidar));
        // visualizer.Run();
        // visualizer.DestroyVisualizerWindow();
        //open3d::geometry::PointCloud second =get_stereo_second(next,P_rect_02,T_03);
        auto registration_result = open3d::pipelines::registration::RegistrationICP(
            next.pc, dense_lidar, 50, t, // Transformation from 2d keypoints to 3d stereo reconstruction as initialization T
            open3d::pipelines::registration::TransformationEstimationPointToPoint(),
            open3d::pipelines::registration::ICPConvergenceCriteria(100, 1e-5));
        std::cout << "registration_result.correspondence_set_: " << registration_result.correspondence_set_.size() << std::endl;
        correspondences_filtering_odometry(next, 0.5, 15, 3, registration_result.correspondence_set_, next.pc, dense_lidar);
        T_1_2_refined.push_back(next.T_refined);
        current = next;
    }

    // Convert matrices to camera positions
    std::vector<Eigen::Vector3d> camera_positions1 = ConvertMatricesToCameraPositions(T_1_2);
    std::vector<Eigen::Vector3d> camera_positions2 = ConvertMatricesToCameraPositions(T_1_2_refined);

    // Visualize the trajectories
    VisualizeTrajectories(camera_positions1, camera_positions2, dense_lidar);
}