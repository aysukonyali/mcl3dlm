#include <opencv2/opencv.hpp>
#include <open3d/Open3D.h>
#include "open3d/geometry/PointCloud.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <regex>
#include <cstdlib> 
#include <opencv2/core/base.hpp>
#include "local_reconstruction.h"
#include "alignment.h"
#include "data_association.h"
#include "simple_odometry.h"

int extractNumberFromFilename(const std::filesystem::path &filename)
{
    // Regular expression to match numbers in the filename
    std::regex number_regex("(\\d+)");
    std::smatch match;

    std::string filename_str = filename.filename().string();

    // Search for the first numeric sequence in the filename
    if (std::regex_search(filename_str, match, number_regex) && !match.empty())
    {
        return std::stoi(match.str()); // Convert matched string to integer
    }

    return -1; // Return -1 if no number is found
}

void loadImages(const std::string &directory, std::vector<cv::Mat> &images)
{
    // Clear the vector to avoid appending to an old list
    images.clear();

    // Vector to hold directory entries
    std::vector<std::filesystem::directory_entry> entries;

    // Collect all entries in the directory
    for (const auto &entry : std::filesystem::directory_iterator(directory))
    {
        if (entry.is_regular_file())
        {
            entries.push_back(entry);
        }
    }

    // Sort entries by the numeric part of the filename
    std::sort(entries.begin(), entries.end(), [](const std::filesystem::directory_entry &a, const std::filesystem::directory_entry &b)
              { return extractNumberFromFilename(a.path()) < extractNumberFromFilename(b.path()); });

    // Load images in sorted order
    for (const auto &entry : entries)
    {
        // Read image file
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cerr << "Error: Could not read image from path: " << entry.path() << std::endl;
            continue;
        }
        // Add image to vector
        images.push_back(img);
    }
}
std::vector<Eigen::Vector4f> readBinFile(const std::string &filename)
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
        points.push_back(point);
    }

    file.close();
    return points;
}
void loadLidarMaps(const std::string &directory, std::vector<std::vector<Eigen::Vector4f>> &lidarMaps)
{
    namespace fs = std::filesystem;

    // Vector to hold all .bin files
    std::vector<fs::directory_entry> binFiles;

    // Collect all .bin files from the directory
    for (const auto &entry : fs::directory_iterator(directory))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".bin")
        {
            binFiles.push_back(entry);
        }
    }

    // Sort files by the numeric part of their filenames
    std::sort(binFiles.begin(), binFiles.end(), [](const fs::directory_entry &a, const fs::directory_entry &b)
              { return extractNumberFromFilename(a.path()) < extractNumberFromFilename(b.path()); });

    // Read and store data from each file
    lidarMaps.clear(); // Clear previous data
    for (const auto &entry : binFiles)
    {
        std::vector<Eigen::Vector4f> lidar = readBinFile(entry.path().string());
        lidarMaps.push_back(lidar);
    }
}
void plotData(const std::vector<double>& y1, const std::vector<double>& y2, int num_points, const std::string plotTitle) {
    if (y1.size() != num_points) {
        std::cerr << "Error: y1 and y2 vectors must have the same size as num_points." << std::endl;
        return;
    }

    // Generate x values
    std::vector<double> x(num_points);
    for (int i = 0; i < num_points; ++i) {
        x[i] = i;
    }

    // Write data to file
    std::ofstream data_file("data.dat");
    for (int i = 0; i < num_points; ++i) {
        data_file << x[i] << " " << y1[i] << " " << y2[i]<< "\n";
    }
    data_file.close();

    // Create a Gnuplot script
    std::ofstream script_file("plot_script.gp");
    script_file << "set terminal pngcairo size 800,600 enhanced font 'Verdana,10'\n";
    script_file << "set output '" << plotTitle << "'\n";
    script_file << "set xlabel 'noise'\n";
    script_file << "set ylabel 'error'\n";
    script_file << "set title 'error in estimated T with increasing noise'\n";
    script_file << "plot 'data.dat' using 1:2 with points pt 7 ps 1.5 title 'corr', \\\n";
    script_file << "     'data.dat' using 1:3 with points pt 7 ps 1.5 title 'icp'\n";
    script_file.close();

    // Call Gnuplot to execute the script
    std::system("gnuplot plot_script.gp");

    std::cout << "Plot created as plot.png" << std::endl;
}
int main(int argc, char **argv)
{

    std::string calibFile_cam_cam = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/calib_cam_to_cam.txt";
    std::string calibFile_velo_cam = "/home/aysu/Desktop/projects/mcl3dlm_visnav/mcl3dlm/data/calib_velo_to_cam.txt";
    cv::Mat P_rect_02, T_03, R_rect_00, T_00, R, T, K, D;

    readCalibrationFile(calibFile_cam_cam, P_rect_02, R_rect_00, T_00, T_03, R, T, K, D);
    readCalibrationFile(calibFile_velo_cam, P_rect_02, R_rect_00, T_00, T_03, R, T, K, D);

    std::string leftImagesPath = "/home/aysu/Downloads/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_02/data";
    std::string rightImagesPath = "/home/aysu/Downloads/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_03/data";
    std::string lidarMapsPath = "/home/aysu/Downloads/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data";

    std::vector<cv::Mat> leftImages;
    std::vector<cv::Mat> rightImages;
    std::vector<std::vector<Eigen::Vector4f>> lidarMaps;

    loadImages(leftImagesPath, leftImages);
    loadImages(rightImagesPath, rightImages);
    loadLidarMaps(lidarMapsPath, lidarMaps);

    std::cout << "Number of left images: " << leftImages.size() << std::endl;
    std::cout << "Number of right images: " << rightImages.size() << std::endl;
    std::cout << "Number of right images: " << lidarMaps.size() << std::endl;

    //runSimpleOdometry(leftImages, rightImages, lidarMaps, P_rect_02, T_03, R, T, K, D);
    
    open3d::geometry::PointCloud dense_stereo = get_denser_stereo_from_images(leftImages, rightImages, 3, P_rect_02, T_03, R, T);
    open3d::geometry::PointCloud dense_lidar = get_denser_lidar_map_from_velodyns(lidarMaps, 3);
    //visualizePointClouds(dense_stereo, dense_lidar, "dense stereo & dense lidar");
    std::vector<double> error;
    std::vector<double> error_icp;
    rotation_plot(15,2,dense_stereo, dense_lidar,error, error_icp);
    plotData(error,error_icp,15,"fksjdfklsjkfs");
    // std::vector<double> errors_corr;
    // std::vector<double> errors_icp;
    // for (size_t i = 0; i < 15; i++)
    // {
    //     int rotation_degree = i;
    //     Eigen::Vector3d translation(0.0+0.5*i, 0.0+0.5*i, 0.0+0.5*i);
    //     double error_corr = 0.0;
    //     double error_icp = 0.0;
    //     correspondences_filtering(0.5, 15, 3, dense_stereo, dense_lidar, rotation_degree, translation, error_corr,error_icp, 0.1, 100, 1e-6);
    //     errors_corr.push_back(error_corr);
    //     errors_icp.push_back(error_icp);
    // }

    // plotData(errors_corr,errors_icp,15, "plot_4");
    // //
    // std::vector<cv::DMatch> matches;

    return 0;
}

// create a bigger lidar map from 2-3 frames
// gıven tımespamps t1 t2 t3 we know the vehicle poses from calıbratıon
// fınd realtıve poses to t1
// transform lidars to camera and then to timestamp t1
// This gives denser 3D reconstructıon and then implement fılterıng from the paper and run ICP
// May be ıt wouldbe better to fılter stereo and not lıder (Up to the test)
