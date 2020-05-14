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

using namespace std;

const string TARGET_FILE_NAME ="/root/catkin_ws/src/car_project/target.pcd";

// load PCD file for target
pcl::PointCloud<pcl::PointXYZ>::Ptr
loadTarget(string targetFileName)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(targetFileName, *cloud) == -1) 
    {
        PCL_ERROR("Couldn't read file \n");
    }
    cout << "Loaded " 
              << cloud->width * cloud->height
              << " data points from " << targetFileName << " with the following fields: "
              << endl;
    /*for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        cout << "   " << cloud->points[i].x
                  << " "   << cloud->points[i].y
                  << " "   << cloud->points[i].z << endl;
    }*/

    return cloud;
}

void
cloudTransform(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    // Container for source and target clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target = loadTarget(TARGET_FILE_NAME);
    pcl::PCLPointCloud2* cloud_source2 = new pcl::PCLPointCloud2;
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud_source2);
    pcl_conversions::toPCL(*cloud_msg, *cloud_source2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(*cloud_source2, *cloud_source);

    pcl::PointCloud<pcl::PointXYZ> cloud_source_registered;
    
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    /*const float bad_point = std::numeric_limits<float>::quiet_NaN();
    for (size_t i = 0; i < cloud_source->points.size(); ++i)
    {
        pcl::PointCloud<pcl::PointXYZ> p = cloud_source->points[i];
        if(p.x == "nan" || p.y == "nan" || p.z == "nan")
            p.x = p.y = p.z = bad_point;
    }*/

    float minX = -3.0;
    float minY = -3.0;
    float minZ = -1.5;
    float maxX = 3.0;
    float maxY = 3.0;
    float maxZ = 1.5;
	pcl::CropBox<pcl::PointXYZ> boxFilter;
    boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
    boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
    boxFilter.setInputCloud(cloud_source);
    boxFilter.filter(*cloud_source);
    cout << cloud_source->size() << endl;
    boxFilter.setInputCloud(cloud_target);
    boxFilter.filter(*cloud_target);
    cout << cloud_target->size() << endl;

    //cout << cloud_source->is_dense << endl;
    icp.setInputSource(cloud_source);
    icp.setInputTarget(cloud_target);
     
    // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
    icp.setMaxCorrespondenceDistance(0.5);
    // Set the maximum number of iterations (criterion 1)
    icp.setMaximumIterations(50);
    // Set the transformation epsilon (criterion 2)
    icp.setTransformationEpsilon(1e-8);
    // Set the euclidean distance difference epsilon (criterion 3)
    icp.setEuclideanFitnessEpsilon(1);
     
    // Perform the alignment
    icp.align (cloud_source_registered);
    
    // Obtain the transformation that aligned cloud_source to cloud_source_registered
    Eigen::Matrix4f transformation = icp.getFinalTransformation();
    const Eigen::IOFormat fmt(4, 0, ", ", "\n", "[", "]");
    cout << "Transformation Matrix: \n" << transformation.format(fmt) << endl;
}

int
main (int argc, char** argv) 
{
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
