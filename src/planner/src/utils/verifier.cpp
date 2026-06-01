#include <fstream>
#include <string>
#include <ros/ros.h>
#include "map/grid_map.h"
#include "utils/data.hpp"

#include <Eigen/Eigen>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

using namespace nmoma_planner;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "verifier_node");
    ros::NodeHandle nh("~");


    std::shared_ptr<GridMap> pGridMap;
    pGridMap.reset(new GridMap());
    pGridMap->init(nh);

    std::string data_path;
    nh.getParam("path", data_path);
    
    // ifstream ifs(data_path);
    // if (!ifs.is_open()) throw std::runtime_error("Failed to open file: " + data_path);

    // boost::archive::binary_iarchive ia(ifs);

    data::DataLoader<10> data_loader(data_path);

    try {
        int i = 0;
        while (data_loader.has_next()) {
            data::DataPoint<10> datapoint = data_loader.next();
            pGridMap->loadMap(datapoint.getOcc2D(), datapoint.getOcc3D());

            bool hash_match = (data::hash_value(datapoint) == datapoint.getHash());
            bool collision = false;
            for (const auto& wp : datapoint.getTraj()) {
                Eigen::VectorXd state = Eigen::VectorXd::Zero(10);
                for (int i = 0; i < 10; i++) state(i) = wp[i];
                if (pGridMap->isWholeBodyCollision(state)) { collision = true; break;}
            }
            if (!hash_match) std::cout << "Hash value mismatch for data point " << i << std::endl;
            if (collision) std::cout << "Collision detected for data point " << i << std::endl;
            if (hash_match && !collision) std::cout << "Data point " << i << " verified." << std::endl;
            
            i++;
        }
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }

    std::cout << "Verification complete." << std::endl;

    return 0;
}