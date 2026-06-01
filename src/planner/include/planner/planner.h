#pragma once

// #include "planner/pinocchio_ik.h"
// #include <torch/torch.h>
// #include <torch/script.h>
// #include <unsupported/Eigen/CXX11/Tensor>
// #include <c10/cuda/CUDAStream.h>
// #include <ATen/cuda/CUDAContext.h>
#include <mutex>
#include <stdlib.h>

#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <sys/stat.h>

#include <string.h>
#include <iostream>
#include <random>
// #include <thread>
#include <time.h>
#include <Eigen/Eigen>
#include <utility>
#include <future>
#include <chrono>

#include <ros/ros.h>
#include <ros/package.h>
#include <visualization_msgs/Marker.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float64MultiArray.h>

#include "map/grid_map.h"
#include "planner/graph_search.h"
#include "planner/birrts.h"
#include "planner/mcrrts.h"
#include "planner/ompls.h"
#include "planner/topo_prm.h"
#include "planner/moma_traj_opt.h"
#include "planner/mpc.h"
#include "planner/ompc.h"
#include "planner/Plan.h"
#include "planner/Task.h"
#include "utils/moma_py2cpp.hpp"
#include "utils/data.hpp"
#include "utils/shared_data.hpp"

#include <fake_moma/moma_param.h>
#include "fake_moma/MomaState.h"
#include "fake_moma/MomaCmd.h"

// #include <tbb/parallel_for.h>
// #include <tbb/blocked_range.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>

#include "planner/MeshPart.h"
#include "planner/MeshState.h"
#include "planner/MeshTraj.h"
#include "planner/MeshPCD.h"

#define BOOST_THREAD_PROVIDES_FUTURE
#include <boost/thread.hpp>
// #include <boost/thread/future.hpp>

namespace nmoma_planner
{
    class Planner
    {
        private:
            // mesh sequence number
            unsigned int _sequence = 0;

            bool use_nn = false;
            bool has_odom = false;

            // data
            Eigen::Vector3d se2_set;
            Eigen::VectorXd now_state;
            Eigen::VectorXd now_dstate; // v, w, 0, q1-q7
            default_random_engine eng;

            // trajs
            MomaTraj end_traj;
            MomaTraj topay_traj;
            MomaTraj ddim_traj;
            std::vector<Eigen::VectorXd> front_path;
            
            // members
            GridMap::Ptr grid_map;
            
            JPS::GraphSearch::Ptr graph_search;
            
            OMPLPlanner::Ptr ompl_planner;
            BiRRTs::Ptr birrts;
            MCRRTs::Ptr mcrrts;
            OMPC::Ptr mpc;
            // MPC::Ptr mpc;
            MomaParam moma_param;
            MomaTrajOpt::Ptr traj_opter;
            std::vector<Eigen::MatrixXd> primitives;

            // parallel
            unique_ptr<TopologyPRM> topo_prm;
            vector<vector<Eigen::Vector3d>> topo_select_paths;
            vector<MCRRTs::UniPtr> mc_rrtsers;
            vector<MomaTrajOpt::UniPtr> traj_opters;
            vector<MomaTraj> parallel_ends;
            
            std::vector<ros::Publisher> opt_traj_pub_list;
            std::vector<ros::Publisher> ddim_pub_list;
            ros::Publisher front_pub;
            ros::Publisher ompl_pub;
            ros::Publisher end_pub;
            ros::Publisher trad_pub;

            ros::Publisher car_traj_pub;
            ros::Publisher moma_cmd_pub;
            ros::Publisher car_target_pub;
            ros::Subscriber state_sub;
            ros::Subscriber statistics_sub;
            ros::Timer replan_timer;
            
            // mesh
            ros::Publisher anime_pub;
            ros::Publisher mesh_traj_pub;
            ros::Publisher plot_traj_ee;

            ros::Publisher plot_trad_traj_ee;

            // text
            ros::Publisher text_pub;
            ros::Timer anim_timer;

            bool end_traj_available = false;

            int scene_seed =0;
            bool auto_mode = false;

            // replan variables
            bool local_mode = false;
            bool has_goal = false;
            bool has_traj = false;
            bool in_plan = false;
            bool is_safe = true;
            double planning_budget = 0.0;
            double replan_interval = 1.0;
            double planning_horizon = 6.0;
            Eigen::VectorXd global_goal;
            MomaTraj global_traj;
            ros::Time begin_time; 
            ros::Time last_gplan_time;
            ros::Time last_replan_time;
            
            // task
            bool gripper_open = true;
            std::vector<Eigen::VectorXd> wps_list;
            Eigen::VectorXd pick_vec;
            Eigen::VectorXd place_vec;

            //statistics

            std::string mode;
            bool stat_mode;
            std::string shm_name;


            bool fixed_sequence;
            int stat_num;
            int traj_num_per_env;
            string stat_number;
            string stat_file;
            string planner_type;
            int ddim_path_num;
            bool only_front;
            bool fixed_startgoal;
            std::vector<MCRRTs> stat_searchs;
            std::vector<MomaTrajOpt> stat_opters;
            std::vector<nmoma_planner::MomaTraj> prim_trajs;
            Eigen::VectorXi prim_status;
            Eigen::VectorXd prim_cost;
            Eigen::VectorXd prim_time_consume;

            //bk ros
            ros::Publisher bk_front_pub;
            ros::Publisher bk_end_pub;

            // debug
            ros::Publisher init_end_pub;
            ros::Publisher afirst_end_pub;
            ros::Publisher prm_pub;

            ros::ServiceClient plannerClient;
            ros::ServiceClient taskClient;

        public:
            void init(ros::NodeHandle& nh);
            void shmCallback(const geometry_msgs::PoseStamped msg);
            void replicashmCallback(const geometry_msgs::PoseStamped msg);
            void rcvStaCallBack(const geometry_msgs::PoseStamped msg);
            void planCallBack(const geometry_msgs::PoseStamped msg);
            void fixedPlanCallBack(const geometry_msgs::PoseStamped msg);

            void benchmarkCallBack(const geometry_msgs::PoseStamped msg);
            void benchmarkReplicaCallBack(const geometry_msgs::PoseStamped msg);

            void rcvStateCallBack(const fake_moma::MomaStatePtr msg);
            static void cmdCallback(void *obj);
            static void replanCallback(void *obj);
            static void safeCallback(void *obj);
            std::vector<Eigen::VectorXd> planOmpls(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Eigen::VectorXd& start_v) const;
            bool staRemani(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Eigen::VectorXd& start_v);
            bool planRemani(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Eigen::VectorXd& start_v);
            bool staMomaParallel(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Eigen::VectorXd& start_v);

            std::pair<bool, float> planMomaParallel(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Eigen::VectorXd& start_v);
            std::pair<float, float> planDDIM(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Eigen::VectorXd& start_v, int path_num=1);
            bool optMomaOnce(const std::vector<Eigen::Vector3d>& topo_path, int idx, 
                            const Eigen::VectorXd& start, const Eigen::VectorXd& end, 
                            const Eigen::VectorXd& start_v);
            bool optDenseOnce(const std::vector<Eigen::VectorXd>& topo_path, int idx, 
                            const Eigen::VectorXd& start, const Eigen::VectorXd& end, 
                            const Eigen::VectorXd& start_v);
            
            void vis_path(const std::vector<Eigen::VectorXd>& path, ros::Publisher& pub, vector<float> rgba = {0.5, 0.5, 0.5, 0.3}, vector<int> ids = {});
            void vis_whole_path(ros::Publisher& pub);
            std::vector<Eigen::Vector4d> vis_prm_paths();

            void vis_path_mesh(const MomaTraj& traj, int nsample, ros::Publisher& pub, vector<float> rgba = {0.5, 0.5, 0.5, 1.0}, int id = 451, bool fade=false);
            void vis_path_mesh(const std::vector<Eigen::VectorXd>& path, ros::Publisher& pub, vector<float> rgba = {0.5, 0.5, 0.5, 1.0}, int id = 451, bool fade=false);
            void vis_ee_traj(const MomaTraj& traj, ros::Publisher& pub, vector<float> rgba = {0.5, 0.5, 0.5, 1.0}) const;
            
            planner::MeshTraj toMeshMsg(const MomaTraj& traj) const;
            void publishBenchmarkText(int ddim_succ, int moma_succ, int total);

            // void animate_trajectory(const std::vector<Eigen::VectorXd>& path,
            //     ros::Publisher& marker_pub,
            //     const std::vector<float>& rgba,
            //     int id,
            //     double period_sec = 0.1);
            
            void animate_trajectory_timer_cb(const ros::TimerEvent&);
    };
}
