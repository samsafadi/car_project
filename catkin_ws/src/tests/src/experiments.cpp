#include <ros/ros.h>
//PCL specific libraries
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

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
    // VoxelGrid Filtering 
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_filtered);
    sor.setLeafSize(0.005, 0.005, 0.005);
    sor.filter(*cloud_filtered);
    //cout << "cloud_filtered size: " << cloud_filtered->size () << endl;

    // Create segmentation object for planar model and set params
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.05);
    
    int i = 0, nr_points = (int) cloud_filtered->points.size ();
    while (cloud_filtered->points.size () > 0.3 * nr_points) 
    {
        // Segment the largest planar compontent from the remaining cloud
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients);
        cout << inliers->indices.size () << endl;
        if (inliers->indices.size () == 0)
        {
            cout << "Could not estimate a planar model for the given dataset." << endl;
            break;
        }
        
        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud (cloud_filtered);
        extract.setIndices (inliers);
        extract.setNegative (false);

        // Get the points associated with the planar surface
        extract.filter(*cloud_plane);
        cout << "PointCloud representing the planar component: " << cloud_plane->size () << " data points." << endl;

        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*cloud_filtered);
        //cout << "cloud_filtered size: " << cloud_filtered->size () << endl;
        
        ++i;
    }

    // Create the KdTree for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered);

    vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.02);
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (10000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract(cluster_indices);

    int j = 0;
    for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            cloud_cluster->points.push_back (cloud_filtered->points[*pit]);
        
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
        j++;
    }


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

