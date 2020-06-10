#include <iostream>
#include <ros/ros.h>
#include <string>
#include <limits>
#include <cmath>
#include <Eigen/Dense>
// PCL specific libraries
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

const string TARGET_FILE_NAME ="/root/catkin_ws/src/car_project/target.pcd";
const string OUTPUT_FILE_NAME ="/root/catkin_ws/src/car_project/target_cropped.pcd";

// load PCD file for target
pcl::PointCloud<pcl::PointXYZ>::Ptr loadTarget(string targetFileName) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(targetFileName, *cloud) == -1) {
        PCL_ERROR("Couldn't read file \n");
    }
    cout << "Loaded " 
              << cloud->width * cloud->height
              << " data points from " << targetFileName << " with the following fields: "
              << endl;
    return cloud;
}

int main (int argc, char** argv) {
    // Container for source and target clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = loadTarget(TARGET_FILE_NAME);
    pcl::CropBox<pcl::PointXYZ> boxFilter;

    float minX = -0.75;
    float minY = -0.75;
    float minZ = -0.75;
    float maxX = 0.75;
    float maxY = 0.75;
    float maxZ = 0.75;

    boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
    boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
    boxFilter.setInputCloud(cloud);
    boxFilter.filter(*cloud);

    pcl::io::savePCDFileASCII(OUTPUT_FILE_NAME, *cloud);
    cout << "Saved PCD File" << endl;
    return(0);
}

