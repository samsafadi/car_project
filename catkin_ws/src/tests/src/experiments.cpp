#include <ros/ros.h>
#include <math.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
// PCL specific libraries
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include "pcl_ros/point_cloud.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/filters/voxel_grid.h>
// OpenCV specific libraries
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;


ros::Publisher pub;
ros::Publisher marker_pub;

VideoCapture cap(0, CAP_V4L2); // change to whatever /dev/video[*] your device is on

// Params
float confThreshold = 0.5;
float nmsThreshold = 0.4;
int inpWidth = 416;
int inpHeight = 416;

vector<String> getOutputsNames(const dnn::Net& net) {
    static vector<String> names;
    if (names.empty()) {
        // Get indices of output layers
        vector<int> outLayers = net.getUnconnectedOutLayers();

        // get names of all layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i) {
            names[i] = layersNames[outLayers[i] - 1];
            cout << names[i] << endl;
        }
    }
    return names;
}

void drawPred(int classId, float conf, int left, int top, 
        int right, int bottom, Mat& frame, vector<string> classes) {
    // Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255));

    // get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    // Display the label at the top of the bounding box
    int baseLine;
    Size labelsize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelsize.height);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
}

tuple<vector<int>, vector<float>, vector<Rect>> getConfidentBoxes(Mat& frame, 
        const vector<Mat>& outs, vector<string> classes) {

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        /* Scan through the bounding boxes and keep the ones with
         * that have a high enough confidence score */
        float* data = (float*)outs[i].data;
        //cout << "OUTS ROWS: " << outs[i].rows << endl;

        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                cout << confidence << endl;
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y, 
                box.x + box.width, box.y + box.height, frame, classes);
    }

    return {classIds, confidences, boxes};
}

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    // Load weights and cfg
    string homedir;
    if ((homedir = getenv("HOME")) == "") {
        homedir = getpwuid(getuid())->pw_dir;
    }
    string weights = homedir + "/catkin_ws/src/tests/new_weights/yolov3-tiny_70000.weights";
    string cfg = homedir + "/catkin_ws/src/tests/yolov3-tiny.cfg";
    string classesFile = homedir + "/catkin_ws/src/tests/objects.names";

    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Container for original and filtered data
    pcl::PCLPointCloud2* cloud2 = new pcl::PCLPointCloud2;
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud2);

    const string FRAME_ID = cloud_msg->header.frame_id;
    
    // Convert to PCL data type
    pcl_conversions::toPCL(*cloud_msg, *cloud2);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(*cloud2, *cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    float minX = -100.0;
    float minY = -100.0;
    float minZ = -100.0;
    float maxX = 100.0;
    float maxY = 100.0;
    float maxZ = 100.0;
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
    sor.setLeafSize(0.005, 0.005, 0.005);
    sor.filter(*cloud_filtered);

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
    while (cloud_filtered->points.size () > 0.3 * nr_points) {
        // Segment the largest planar compontent from the remaining cloud
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0) {
            cout << "Could not estimate a planar model for the given dataset." << endl;
        seg.segment (*inliers, *coefficients);
            break;
        }
        
        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud (cloud_filtered);
        extract.setIndices (inliers);
        extract.setNegative (false);

        // Get the points associated with the planar surface
        extract.filter(*cloud_plane);
        //cout << "PointCloud representing the planar component: " << cloud_plane->size () << " data points." << endl;

        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*cloud_filtered);
        //cout << "cloud_filtered size: " << cloud_filtered->size () << endl;
        
        ++i;
    }

    /* 
    *** Find YOLOv3 bounding box ***
    */

    // Instantiate frame
    Mat frame, blob;
    cap >> frame;

    // get the frame from the cap
    if (!frame.data) cout << "THIS FRAME IS INVALID DAWG" << endl;
    // else
    //     cout << "noice" << endl;

    
    // Read in the net from the darknet cfg and weights files
    dnn::Net net = dnn::readNetFromDarknet(cfg, weights);
    dnn::blobFromImage(frame, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);

    // Set the input
    net.setInput(blob);
    
    // Get the output
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));
    //cout << outs.size() << endl;

    // Get confident boxes
    tuple<vector<int>, vector<float>, vector<Rect>> confident_boxes;
    confident_boxes = getConfidentBoxes(frame, outs, classes);
    
    vector<Rect> boxes = get<2>(confident_boxes);
    vector<float> confidences = get<1>(confident_boxes);

    cout << "BOXES: " << boxes.size() << endl;

    for (i = 0; i < boxes.size(); ++i)
        // Debugging the location of the centroids of any possible boxes
        // TODO: figure out how to convert opencv points to ROS
        cout << "BOX " << i << ":: x:" << boxes[i].x - boxes[i].width / 2
            << " y: " << boxes[i].y - boxes[i].height / 2;
    if (boxes.size() > 0) 
        cout << '\n';
    
    // show the image
    imshow("frame", frame);  // show the image inside of it
    waitKey(1);

    // Create the KdTree for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered);

    vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.03);
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract(cluster_indices);

    vector<visualization_msgs::Marker> markers; // Wanna store the markers
    int j = 0;
    for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); it++) {

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        
        Eigen::Vector4f centroid;        
        Eigen::Vector4f min;
        Eigen::Vector4f max;

        int k = 0;
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++) {
            cloud_cluster->points.push_back (cloud_filtered->points[*pit]);
            //pcl::PointXYZ p = cloud_filtered->points[*pit];
            k++;
        }

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        
        pcl::getMinMax3D (*cloud_cluster, min, max);
        pcl::compute3DCentroid (*cloud_cluster, centroid);

        visualization_msgs::Marker marker;
        marker.header.frame_id = FRAME_ID;
        marker.header.stamp = ros::Time::now();
        marker.lifetime = ros::Duration(0.25);

        marker.ns = "euclidean_clusters";
        marker.id = j;
        marker.type = visualization_msgs::Marker::CUBE;

        /*cout << "Centroid: (" <<
        centroid[0] << "," <<
        centroid[1] << "," <<
        centroid[2] << 
        ") Min: (" <<
        min[0] << "," <<
        min[1] << "," <<
        min[2] <<
        ") Max: (" <<
        max[0] << "," <<
        max[1] << "," <<
        max[2] << ")" << endl;*/

        marker.pose.position.x = centroid[0];
        marker.pose.position.y = centroid[1];
        marker.pose.position.z = centroid[2];
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = max[0] - min[0];
        marker.scale.y = max[1] - min[1];
        marker.scale.z = max[2] - min[2];

        if (marker.scale.x == 0)
            marker.scale.x = 0.1;

        if (marker.scale.y == 0)
            marker.scale.y = 0.1;

        if (marker.scale.z == 0)
            marker.scale.z = 0.1;

        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.3;

        markers.push_back (marker);
        marker_pub.publish (marker);

        //std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
        j++;

    }
    /*
    for (i = 0; i < markers.size(); ++i) {
        x = marker.pose.position.x;
        y = marker.pose.position.y;
        z = marker.pose.position.z;
        if (
        */

    // Convert to ROS data type
    pcl::PCLPointCloud2* cloud_filtered2 = new pcl::PCLPointCloud2;
    pcl::toPCLPointCloud2(*cloud_filtered, *cloud_filtered2);
    sensor_msgs::PointCloud2 output;
    pcl_conversions::fromPCL(*cloud_filtered2, output);
    output.header.frame_id = FRAME_ID;
    output.header.stamp = ros::Time::now();
    //cout << output.size() << endl;

    // Publish the data
    pub.publish (output);
}

int main (int argc, char** argv) {
    // Initialize ROS
    ros::init (argc, argv, "tests");
    ros::NodeHandle nh;

    if(!cap.open(0))
        cout << "Couldn't open RGB stream" << endl;
    else
        cout << "Opened RGB stream" << endl;
    
    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe ("input", 1, cloud_cb);

    // Create a ROS publisher for the output point cloud
    cout << "Advertising output \n";
    pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);
    //pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ>> ("output", 1);
    
    // Create marker publisher node
    marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);

    // Spin
    ros::spin ();

    // Close everything and exit
    cap.release();
    destroyAllWindows();
    return(0);
}

