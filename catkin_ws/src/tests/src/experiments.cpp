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
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <boost/make_shared.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
// OpenCV specific libraries
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <image_transport/image_transport.h>

using namespace std;
using namespace cv;

typedef pcl::PointNormal PointNT;
typedef pcl::PointXYZRGBNormal PointRGBNT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<pcl::PointXYZRGB,pcl::Normal,FeatureT> FeatureEstimationT;
typedef pcl::SHOT352 SHOTFT;
typedef pcl::ReferenceFrame RFType;
//typedef pcl::SHOTEstimationOMP<pcl::PointWithScale,pcl::Normal,SHOTFT> SHOTEstimationT;

ros::Publisher cloud_pub;
ros::Publisher target_pub;
ros::Publisher transform_pub;
ros::Publisher model_pub;
ros::Publisher keypoints_pub;
ros::Publisher marker_pub;

image_transport::Publisher image_pub;

// Shared information
vector<Rect> global_boxes;

// Params
float confThreshold = 0.5;
float nmsThreshold = 0.4;
int inpWidth = 416;
int inpHeight = 416;
float leaf = 0.003; // This is also the model resolution

vector<String> getOutputsNames(const dnn::Net& net) {
    static vector<String> names;
    if (names.empty()) {
        // Get indices of output layers
        vector<int> outLayers = net.getUnconnectedOutLayers();

        // get names of all layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        //cout << outLayers.size() << endl;
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
    // Draw small circle around the center
    //cout << "RIGHT: " << right << "LEFT: " << left << "TOP: " << top << "BOTTOM: " << bottom << endl;
    int cx = left + ((right - left) / 2);
    int cy = top + ((bottom - top) / 2);
    circle(frame, Point(cx, cy), 1, Scalar(0, 0, 255));

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

#pragma omp parallel for
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                //cout << confidence << endl;
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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadTarget(string targetFileName) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(targetFileName, *cloud) == -1) {
        PCL_ERROR("Couldn't read file \n");
    }
    return cloud;
}

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    
    // Container for original and filtered data
    pcl::PCLPointCloud2* cloud2 = new pcl::PCLPointCloud2;
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud2);

    const string FRAME_ID = cloud_msg->header.frame_id;
    
    // Convert to PCL data type
    pcl_conversions::toPCL(*cloud_msg, *cloud2);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(*cloud2, *cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

    float minX = -100.0;
    float minY = -100.0;
    float minZ = -100.0;
    float maxX = 100.0;
    float maxY = 100.0;
    float maxZ = 100.0;
    //cout << minX << minY << minZ << maxX << maxY << maxZ << endl;
	pcl::CropBox<pcl::PointXYZRGB> boxFilter;
    boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
    boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
    boxFilter.setInputCloud(cloud);
    //cout << cloud->size() << endl;
    boxFilter.filter(*cloud_filtered);
    // VoxelGrid Filtering 
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud_filtered);
    sor.setLeafSize(leaf, leaf, leaf);
    sor.filter(*cloud_filtered);

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

    int i = 0, nr_points = (int) cloud_filtered->points.size();
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
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud (cloud_filtered);
        extract.setIndices (inliers);
        extract.setNegative (false);

        // Get the points associated with the planar surface
        extract.filter(*cloud_plane);
        //cout << "PointCloud representing the planar component: " << cloud_plane->size () << " data points." << endl;

        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*cloud_filtered);

        ++i;
    }

    // Create the KdTree for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud (cloud_filtered);

    vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance (0.05);
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract(cluster_indices);

    vector<Rect> boxes = global_boxes;
    vector<visualization_msgs::Marker> markers; // Want to store the markers
    bool target_found = false; // To indicate to if we can do icp
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_pc(new pcl::PointCloud<pcl::PointXYZ>); // For icp if we find a target
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);

    int j = 0;
    for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); it++) {

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
        
        Eigen::Vector4f centroid;        
        Eigen::Vector4f min;
        Eigen::Vector4f max;

        int k = 0;
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++) {
            cloud_cluster->points.push_back (cloud_filtered->points[*pit]);
            //pcl::PointXYZRGB p = cloud_filtered->points[*pit];
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
        marker.lifetime = ros::Duration(2.0);

        marker.ns = "euclidean_clusters";
        marker.id = j;
        marker.type = visualization_msgs::Marker::CUBE;

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

        bool contains_centroid = false;
        for (int b = 0; b < boxes.size(); ++b) {
            Rect box = boxes[b];
            int cx = box.x + (box.width / 2);
            int cy = box.y + (box.height / 2);
            pcl::PointXYZRGB p = cloud->at(cx, cy);
            //cout << p.x << " " << p.y << " " << p.z << " " << endl;

            //cout << "CX: " << cx << " CY: " << cy << " MAX_X: " << max[0] << 
            //    " MIN_X: " << min[0] << " MAX_Y: " << max[1] << " MIN_Y: " << min[1]<< endl;

            if (p.x <= max[0] && p.x >= min[0] && p.y <= max[1] && p.y >= min[1])
                contains_centroid = true, target_found = true;
        }
        if (contains_centroid == true) {
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 0.3;
        } else {
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.color.a = 0.3;
        }

        if (contains_centroid) {
            pcl::CropBox<pcl::PointXYZRGB> targetFilter;

            targetFilter.setMin(Eigen::Vector4f(min[0], min[1], min[2], 1.0));
            targetFilter.setMax(Eigen::Vector4f(max[0], max[1], max[2], 1.0));
            targetFilter.setInputCloud(cloud_filtered);
            targetFilter.filter(*object_rgb);
            //target_pub.publish(object_rgb);

            pcl::copyPointCloud (*object_rgb, *target_pc);
        }
            
        marker_pub.publish(marker);
        markers.push_back (marker);
        j++;

    }
    // Load model (of mug)
    string homedir;
    if ((homedir = getenv("HOME")) == "") {
        homedir = getpwuid(getuid())->pw_dir;
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model = loadTarget(homedir + "/catkin_ws/src/tests/model.pcd");

    if (target_found) {
        // Downsample Model
        sor.setInputCloud(model);
        sor.setLeafSize(leaf, leaf, leaf);
        sor.filter(*model);

        // Compute Centroids
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr orig_object (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr orig_model (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud(*object_rgb, *orig_object);
        pcl::copyPointCloud(*model, *orig_model);

        Eigen::Vector4f object_centroid;
        Eigen::Vector4f model_centroid;
        pcl::compute3DCentroid (*object_rgb, object_centroid);
        pcl::compute3DCentroid (*model, model_centroid);
        pcl::demeanPointCloud (*orig_object, object_centroid, *object_rgb);
        pcl::demeanPointCloud (*orig_model, model_centroid, *model);
        model->header.frame_id = FRAME_ID;
        pcl_conversions::toPCL (ros::Time::now(), model->header.stamp);
        
        // Estimate Normals for extracted object (to use in feature extraction)
        pcl::PointCloud<pcl::Normal>::Ptr object_normals (new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr model_normals (new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr object_keypoint_normals (new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr model_keypoint_normals (new pcl::PointCloud<pcl::Normal>);
        //pcl::search::KdTree<pcl::PointWithScale>::Ptr pws_tree (new pcl::search::KdTree<pcl::PointWithScale>);

        pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> nest;
        nest.setSearchMethod (tree); 
        nest.setKSearch (20);

        nest.setInputCloud (object_rgb);
        nest.compute (*object_normals);
        nest.setInputCloud (model);
        nest.compute (*model_normals);

       
        // Estimate features (using Normals just computed)
        pcl::PointCloud<FeatureT>::Ptr object_features (new pcl::PointCloud<FeatureT>);
        pcl::PointCloud<FeatureT>::Ptr model_features (new pcl::PointCloud<FeatureT>);
        
        FeatureEstimationT fest;
        fest.setSearchMethod (tree);
        fest.setKSearch (30);

        fest.setInputCloud (object_rgb);
        fest.setInputNormals (object_normals);
        fest.compute (*object_features);
        fest.setInputCloud (model);
        fest.setInputNormals (model_normals);
        fest.compute (*model_features);

        // Join point cloud and normals
        pcl::PointCloud<PointRGBNT>::Ptr object_with_normals (new pcl::PointCloud<PointRGBNT>);
        pcl::PointCloud<PointRGBNT>::Ptr model_with_normals (new pcl::PointCloud<PointRGBNT>);
        //pcl::PointCloud<PointRGBNT>::Ptr object_kpts_with_normals (new pcl::PointCloud<PointRGBNT>);
        //pcl::PointCloud<PointRGBNT>::Ptr model_kpts_with_normals (new pcl::PointCloud<PointRGBNT>);

        pcl::concatenateFields (*object_rgb, *object_normals, *object_with_normals);
        pcl::concatenateFields (*model, *model_normals, *model_with_normals);
        //pcl::concatenateFields (*object_keypoints, *object_keypoint_normals, *object_kpts_with_normals);
        //pcl::concatenateFields (*model_keypoints, *model_keypoint_normals, *model_kpts_with_normals);

        // Peform initial pre-ICP alignment
        pcl::PointCloud<PointRGBNT>::Ptr object_pre_icp (new pcl::PointCloud<PointRGBNT>);
        pcl::SampleConsensusPrerejective<PointRGBNT, PointRGBNT, FeatureT> align;
        //pcl::SampleConsensusInitialAlignment<PointRGBNT, PointRGBNT, FeatureT> align;

        align.setInputSource (object_with_normals);
        align.setSourceFeatures (object_features);
        align.setInputTarget (model_with_normals);
        align.setTargetFeatures (model_features);
        align.setMaximumIterations (50000);
        align.setNumberOfSamples (3);
        align.setCorrespondenceRandomness (5);
        align.setSimilarityThreshold (0.9f);
        align.setMaxCorrespondenceDistance (2.5f * leaf);
        align.setInlierFraction (0.25f);
        align.align (*object_pre_icp);

        keypoints_pub.publish (object_pre_icp);

        // Try Transformation Estimation Symmetric Point To Plane LLS
        pcl::IterativeClosestPointNonLinear<PointRGBNT, PointRGBNT> reg;
        pcl::PointCloud<PointRGBNT>::Ptr object_registered (new pcl::PointCloud<PointRGBNT>);

        reg.setInputSource (object_pre_icp);
        reg.setInputTarget (model_with_normals);

        reg.setMaximumIterations (5000);
        reg.setTransformationEpsilon (1e-4);
        reg.setEuclideanFitnessEpsilon (1e-8);
        reg.setMaxCorrespondenceDistance (0.1);

        model_pub.publish (model);
        target_pub.publish (object_with_normals);

        reg.align (*object_registered);
        Eigen::Matrix4f transformation = reg.getFinalTransformation ();

        if (reg.hasConverged ()) {
            const Eigen::IOFormat fmt(4, 0, ", ", "\n", "[", "]");
            cout << "Transformation Matrix: \n" << transformation.format(fmt) << endl;
            transform_pub.publish (object_registered);
        } else {
            cout << "ICP has not converged" << endl;
        }
        cout << "Fitness Score: " << reg.getFitnessScore () << endl;
    }

    // Convert to ROS data type
    pcl::PCLPointCloud2* cloud_filtered2 = new pcl::PCLPointCloud2;
    pcl::toPCLPointCloud2(*cloud_filtered, *cloud_filtered2);
    sensor_msgs::PointCloud2 output;
    pcl_conversions::fromPCL(*cloud_filtered2, output);
    output.header.frame_id = FRAME_ID;
    output.header.stamp = ros::Time::now();

    // Publish the data
    cloud_pub.publish (output);
}

void image_cb (const sensor_msgs::ImageConstPtr& image_msg) {

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    string homedir;
    if ((homedir = getenv("HOME")) == "") {
        homedir = getpwuid(getuid())->pw_dir;
    }
    string weights = homedir + "/catkin_ws/src/tests/yolov4-custom_last.weights";
    string cfg = homedir + "/catkin_ws/src/tests/yolov4-custom.cfg";
    string classesFile = homedir + "/catkin_ws/src/tests/objects.names";

    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    Mat blob;

    dnn::Net net = dnn::readNetFromDarknet(cfg, weights);
    dnn::blobFromImage(cv_ptr->image, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);

    // Set the input
    net.setInput(blob);
    
    // Get the output
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Get confident boxes
    tuple<vector<int>, vector<float>, vector<Rect>> confident_boxes;
    confident_boxes = getConfidentBoxes(cv_ptr->image, outs, classes);
    
    vector<Rect> boxes = get<2>(confident_boxes);
    vector<float> confidences = get<1>(confident_boxes);

#pragma omp parallel for
    for (int i = 0; i < boxes.size(); i++)
        cout << "BOX" << i << "::x:" << boxes[i].x
            << " y:" << boxes[i].y << " ";
    if (boxes.size() > 0) 
        cout << '\n';
    
    // show the image
    //imshow("image", cv_ptr->image);  // show the image inside of it
    //waitKey(1);
    global_boxes = boxes;
    image_pub.publish(cv_ptr->toImageMsg());
}

int main (int argc, char** argv) {
    // Initialize ROS
    ros::init (argc, argv, "tests");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub_cloud = nh.subscribe ("cloud", 1, cloud_cb);
    ros::Subscriber sub_image = nh.subscribe ("image", 1, image_cb);

    // Create a ROS publisher for the output point cloud
    cout << "Advertising output \n";
    cloud_pub = nh.advertise<sensor_msgs::PointCloud2> ("output/cloud", 1);
    target_pub = nh.advertise<sensor_msgs::PointCloud2> ("output/target", 1);
    transform_pub = nh.advertise<sensor_msgs::PointCloud2> ("output/transformation", 1);
    model_pub = nh.advertise<sensor_msgs::PointCloud2> ("output/model", 1);
    keypoints_pub = nh.advertise<sensor_msgs::PointCloud2> ("output/keypoints", 1);
    image_pub = it.advertise("/image_converter/output_video", 1);
    
    // Create marker publisher node
    marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);

    // Spin
    //ros::spin ();
    ros::AsyncSpinner spinner(4);
    spinner.start();
    ros::waitForShutdown();

    // Close everything and exit
    destroyAllWindows();
    return(0);
}
