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

// load PCD file for target
pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadTarget(string targetFileName) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(targetFileName, *cloud) == -1) {
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
    char* input = argv[1];
    char* output = argv[2];

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = loadTarget(input);
    pcl::CropBox<pcl::PointXYZRGB> boxFilter;

    float minX = -0.3;
    float minY = -0.3;
    float minZ = -0.25;
    float maxX = 0.3;
    float maxY = 0.3;
    float maxZ = 0.27;

    boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
    boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
    boxFilter.setInputCloud(cloud);
    boxFilter.filter(*cloud);

    pcl::io::savePCDFileASCII(output, *cloud);
    cout << "Saved PCD File" << endl;
    return(0);
}

