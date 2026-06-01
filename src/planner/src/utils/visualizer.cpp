#include <fstream>
#include <string>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include "map/grid_map.h"
#include "utils/data.hpp"
#include "utils/random_pc_gen.hpp"

#include <Eigen/Eigen>

#include <pcl/point_cloud.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

using namespace nmoma_planner;

MomaParam moma_param;

void vis_path(const std::vector<Eigen::VectorXd>& path, ros::Publisher& puber, vector<int> ids = vector<int>())
{
    if (path.empty())
        return;

    visualization_msgs::Marker line_strip, arrow, text;
    arrow.header.frame_id = line_strip.header.frame_id = text.header.frame_id = "world";
    arrow.header.stamp = line_strip.header.stamp = text.header.stamp = ros::Time::now();
    line_strip.type = visualization_msgs::Marker::LINE_STRIP;
    arrow.type = visualization_msgs::Marker::ARROW;
    text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text.id = 1886666;
    text.action = visualization_msgs::Marker::ADD;
    text.scale.z = 0.1;
    text.color.a = 1.0;
    line_strip.id = 10086;
    arrow.scale.x = 0.03;
    arrow.scale.y = 0.05;
    arrow.color.a = 1.0;
    arrow.color.r = 1.0;
    arrow.color.g = 0.0;
    arrow.color.b = 0.0;
    arrow.pose.orientation.w = 1.0;
    line_strip.pose.orientation.w = 1.0;
    line_strip.scale.x = 0.03;
    line_strip.scale.y = 0.03;
    line_strip.scale.z = 0.03;
    line_strip.color.a = 1.0;
    line_strip.color.r = 0.0;
    line_strip.color.g = 1.0;
    line_strip.color.b = 0.0;

    visualization_msgs::MarkerArray array_msg;
    visualization_msgs::Marker p;
    p.action = visualization_msgs::Marker::DELETEALL;
    p.id = 0;
    array_msg.markers.push_back(p);
    for (size_t i=0; i<path.size(); i++)
    {
        visualization_msgs::MarkerArray node_array = moma_param.getColliCylinderArray(path[i]);
        size_t array_size = node_array.markers.size();
        for (size_t j=0; j<array_size; j++)
        {
            node_array.markers[j].id = i*array_size+j;
            node_array.markers[j].color.a = 0.3;
            node_array.markers[j].color.r = 0.5;
            node_array.markers[j].color.g = 0.5;
            node_array.markers[j].color.b = 0.5;
            array_msg.markers.push_back(node_array.markers[j]);
        }
        geometry_msgs::Point pt;
        pt.x = path[i].x();
        pt.y = path[i].y();
        pt.z = 0.0;
        line_strip.points.push_back(pt);
        geometry_msgs::Point pt_arrow;
        pt_arrow.x = path[i].x() + moma_param.chassis_colli_radius*cos(path[i].z());
        pt_arrow.y = path[i].y() + moma_param.chassis_colli_radius*sin(path[i].z());
        arrow.points.clear();
        arrow.points.push_back(pt);
        arrow.points.push_back(pt_arrow);
        arrow.id = line_strip.id + i + 1;
        array_msg.markers.push_back(arrow);
        text.color.r = 0.0;
        arrow.color.b = 0.0;
        for (size_t j=0; j<ids.size(); j++)
        {
            if (ids[j] == (int)i)
            {
                text.color.r = 1.0;
                arrow.color.b = 1.0;
                break;
            }
        }
        text.text = std::to_string(i);
        text.id = text.id + 1;
        text.pose.orientation.w = 1.0;
        text.pose.position = node_array.markers.back().pose.position;
        text.pose.position.z = text.pose.position.z + 0.1;
        array_msg.markers.push_back(text);
    }
    array_msg.markers.push_back(line_strip);
    puber.publish(array_msg);
    return;
}

pcl::PointCloud<pcl::PointXYZ> getPointCloud(GridMap::Ptr pGridMap) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    const std::vector<char>& occ3d = pGridMap->getOccBuffer3d();
    std::cout << "Occupancy grid size: " << pGridMap->voxel_num.transpose() << std::endl;
    for (int i = 0; i < pGridMap->voxel_num(0); i++)
        for (int j = 0; j < pGridMap->voxel_num(1); j++)
            for (int k = 0; k < pGridMap->voxel_num(2); k++)
            {
                if (occ3d[pGridMap->toAddress3d(i, j, k)]) {
                    Eigen::Vector3d pos;
                    pGridMap->indexToPos3d(Eigen::Vector3i(i, j, k), pos);

                    pcl::PointXYZ point;
                    point.x = pos(0);
                    point.y = pos(1);
                    point.z = pos(2);
                    cloud.points.push_back(point);
                }
            }
    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    std::cout << "Point cloud size: " << cloud.points.size() << std::endl;
    return cloud;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "verifier_node");
    ros::NodeHandle nh("~");


    std::shared_ptr<GridMap> pGridMap;
    pGridMap.reset(new GridMap());
    pGridMap->init(nh);

    std::string data_path;
    nh.getParam("path", data_path);

    // RandomPCGenerator pc_gen;
    // pc_gen.init(nh);
    
    // ifstream ifs(data_path);
    // if (!ifs.is_open()) throw std::runtime_error("Failed to open file: " + data_path);

    // boost::archive::binary_iarchive ia(ifs);

    ros::Publisher pub_cloud = nh.advertise<sensor_msgs::PointCloud2>("/global_map", 1);
    ros::Publisher pub_front = nh.advertise<visualization_msgs::MarkerArray>("/front_path", 1);

    data::DataLoader<10> data_loader(data_path);

    try {
        int i = 0;
        while (data_loader.has_next()) {
            std::cin.get();
            std::cout << "Processing data point " << i << std::endl;
            data::DataPoint<10> datapoint = data_loader.next();
            pGridMap->loadMap(datapoint.getOcc2D(), datapoint.getOcc3D());
            pcl::PointCloud<pcl::PointXYZ> cloud = getPointCloud(pGridMap);
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(cloud, cloud_msg);
            cloud_msg.header.frame_id = "world";
            cloud_msg.header.stamp = ros::Time::now();
            pub_cloud.publish(cloud_msg);

            std::vector<Eigen::VectorXd> path;

            for (const auto& wp : datapoint.getTraj()) {
                Eigen::VectorXd state = Eigen::VectorXd::Zero(10);
                for (int i = 0; i < 10; i++) state(i) = wp[i];
                path.push_back(state);
            }

            vis_path(path, pub_front);

            i++;
        }
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }

    std::cout << "Verification complete." << std::endl;

    return 0;
}