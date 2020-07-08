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
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

using namespace std;

const string TARGET_FILE_NAME ="/root/catkin_ws/src/car_project/target.pcd";

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
    /*for (size_t i = 0; i < cloud->points.size(); ++i) {
        cout << "   " << cloud->points[i].x
                  << " "   << cloud->points[i].y
                  << " "   << cloud->points[i].z << endl;
    }*/

    return cloud;
}

void cloudTransform(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    // Container for source and target clouds
    pcl::PCLPointCloud2* cloud2 = new pcl::PCLPointCloud2;
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud2);
    pcl_conversions::toPCL(*cloud_msg, *cloud2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(*cloud2, *cloud);
    
    float minX = -0.5;
    float minY = -0.5;
    float minZ = -0.5;
    float maxX = 0.5;
    float maxY = 0.5;
    float maxZ = 0.5;

	pcl::CropBox<pcl::PointXYZRGB> boxFilter;
    boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
    boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
    boxFilter.setInputCloud(cloud);
    boxFilter.filter(*cloud);
    //cout << cloud_source->size() << endl;
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setLeafSize(0.005, 0.005, 0.005);
    
    // Create segmentation object for planar model and set params
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZRGB>());
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.02);

    int i = 0, nr_points = (int) cloud->points.size();
    while (cloud->points.size () > 0.3 * nr_points) {
        // Segment the largest planar compontent from the remaining cloud
        seg.setInputCloud (cloud);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0) {
            cout << "Could not estimate a planar model for the given dataset." << endl;
        seg.segment (*inliers, *coefficients);
            break;
        }

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud (cloud);
        extract.setIndices (inliers);
        extract.setNegative (false);

        // Get the points associated with the planar surface
        extract.filter(*cloud_plane);
        //cout << "PointCloud representing the planar component: " << cloud_plane->size () << " data points." << endl;

        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*cloud);

        ++i;
    }

    pcl::io::savePCDFile (TARGET_FILE_NAME, *cloud);
}

int main (int argc, char** argv) {
   // Initialize ROS
   ros::init (argc, argv, "transform");
   ros::NodeHandle nh;
  
   // Create a ROS subscriber for the input point cloud
   ros::Subscriber sub = nh.subscribe ("input", 1, cloudTransform);
  
   // Create a ROS publisher for the output point cloud
   //pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);
  
   // Spin
   ros::spin ();

   return(0);
}
