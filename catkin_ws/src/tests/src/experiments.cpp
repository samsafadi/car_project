#include <ros/ros.h>
//PCL specific libraries
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>

using namespace std;

ros::Publisher pub;

void 
cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    // Container for original and filtered data
    pcl::PCLPointCloud2* cloud2 = new pcl::PCLPointCloud2;
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud2);
    
    // Convert to PCL data type
    pcl_conversions::toPCL(*cloud_msg, *cloud2);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(*cloud2, *cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);


    float minX = -3.0;
    float minY = -3.0;
    float minZ = -1.5;
    float maxX = 3.0;
    float maxY = 3.0;
    float maxZ = 1.5;
    //cout << minX << minY << minZ << maxX << maxY << maxZ << endl;
	pcl::CropBox<pcl::PointXYZ> boxFilter;
    boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
    boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
    boxFilter.setInputCloud(cloud);
    //cout << cloud->size() << endl;
    boxFilter.filter(*cloud_filtered);
    //cout << cloud_filtered->size() << endl;
    // VoxelGrid Filtering 
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_filtered);
    sor.setLeafSize(0.005, 0.005, 0.005);
    sor.filter(*cloud_filtered);


    // Convert to ROS data type
    pcl::PCLPointCloud2* cloud_filtered2 = new pcl::PCLPointCloud2;
    pcl::toPCLPointCloud2(*cloud_filtered, *cloud_filtered2);
    sensor_msgs::PointCloud2 output;
    pcl_conversions::fromPCL(*cloud_filtered2, output);
    //cout << output.size() << endl;

    // Publish the data
    pub.publish (output);
}

int
main (int argc, char** argv)
{
    // Initialize ROS
    ros::init (argc, argv, "tests");
    ros::NodeHandle nh;
    
    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe ("input", 1, cloud_cb);

    // Create a ROS publisher for the output point cloud
    cout << "Advertising output \n";
    pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);

    // Spin
    ros::spin ();
    return(0);
}

