#include "planner/planner.h"

namespace nmoma_planner
{
    void Planner::init(ros::NodeHandle& nh)
    {
        GET_PARAM_OR_THROW(nh, "agent/local_mode", local_mode);
        GET_PARAM_OR_THROW(nh, "agent/replan_interval", replan_interval);
        GET_PARAM_OR_THROW(nh, "agent/planning_horizon", planning_horizon);
        GET_PARAM_OR_THROW(nh, "agent/mode", mode);
        GET_PARAM_OR_THROW(nh, "agent/stat_num", stat_num);
        GET_PARAM_OR_THROW(nh, "agent/stat_file", stat_file);
        GET_PARAM_OR_THROW(nh, "agent/only_front", only_front);
        GET_PARAM_OR_THROW(nh, "agent/fixed_startgoal", fixed_startgoal);
        
        GET_PARAM_OR_THROW(nh, "agent/planner", planner_type);
        GET_PARAM_OR_THROW(nh, "agent/ddim_path_num", ddim_path_num);
        GET_PARAM_OR_THROW(nh, "agent/fixed_sequence", fixed_sequence);
        int number;
        GET_PARAM_OR_THROW(nh, "agent/stat_number", number);
        GET_PARAM_OR_THROW(nh, "agent/auto_mode", auto_mode);

        stat_number = std::to_string(number);
        bool random_ee = true;
        nh.getParam("agent/random_ee", random_ee);
        if (!random_ee)
        {
            std::vector<double> pick, mid, place;
            nh.param<std::vector<double>>("agent/pick_state", pick, std::vector<double>());
            nh.param<std::vector<double>>("agent/mid_state", mid, std::vector<double>());
            nh.param<std::vector<double>>("agent/place_state", place, std::vector<double>());
            pick_vec = Eigen::Map<Eigen::VectorXd>(pick.data(), pick.size());
            Eigen::VectorXd mid_vec = Eigen::Map<Eigen::VectorXd>(mid.data(), mid.size());
            place_vec = Eigen::Map<Eigen::VectorXd>(place.data(), place.size());
            wps_list.push_back(pick_vec);
            wps_list.push_back(mid_vec);
            wps_list.push_back(place_vec);
            wps_list.push_back(mid_vec);
            wps_list.push_back(Eigen::VectorXd::Zero(pick_vec.size()));
        }
        nh.getParam("agent/shm_name", shm_name);

        grid_map.reset(new GridMap);
        grid_map->init(nh);
        
        graph_search = std::make_shared<JPS::GraphSearch>(grid_map, moma_param.chassis_colli_radius);
        birrts = std::make_shared<BiRRTs>(grid_map);
        birrts->init(nh);
        topo_prm.reset(new TopologyPRM);
        topo_prm->setEnv(grid_map);
        topo_prm->init(nh);
        mcrrts = std::make_shared<MCRRTs>(grid_map);
        mcrrts->init(nh);
        ompl_planner = std::make_shared<OMPLPlanner>(grid_map);
        ompl_planner->init(nh);
        traj_opter = std::make_shared<MomaTrajOpt>(grid_map);
        traj_opter->init(nh);
        mpc.reset(new OMPC);
        // mpc.reset(new MPC);
        mpc->init(nh);

        traj_opters.resize(12);
        mc_rrtsers.resize(12);
        for (int i = 0; i < 12; i++)
        {
            traj_opters[i] = std::make_unique<MomaTrajOpt>(grid_map);
            traj_opters[i]->init(nh);
            mc_rrtsers[i].reset(new MCRRTs(grid_map));
            mc_rrtsers[i]->init(nh);
            opt_traj_pub_list.push_back(
                nh.advertise<visualization_msgs::MarkerArray>("/opt_traj_" + std::to_string(i + 1), 1)
            );
        }
        
        for (int i = 0; i < ddim_path_num; i++)
        {
            ddim_pub_list.push_back(
                nh.advertise<visualization_msgs::MarkerArray>("/ddim_path_" + std::to_string(i + 1), 1)
            );
        }

        front_pub = nh.advertise<visualization_msgs::MarkerArray>("/front_path", 1);
        ompl_pub = nh.advertise<visualization_msgs::MarkerArray>("/ompl_path", 1);
        end_pub = nh.advertise<visualization_msgs::MarkerArray>("/end_path", 1);
        trad_pub = nh.advertise<visualization_msgs::MarkerArray>("/trad_path", 1);
        anime_pub = nh.advertise<visualization_msgs::MarkerArray>("/anime_path", 1);
        car_traj_pub = nh.advertise<nav_msgs::Path>("/car_traj", 1);
        car_target_pub = nh.advertise<visualization_msgs::Marker>("/car_target", 1);

        bk_front_pub = nh.advertise<visualization_msgs::MarkerArray>("/bk_front_path", 1);
        bk_end_pub = nh.advertise<visualization_msgs::MarkerArray>("/bk_end_path", 1);

        init_end_pub = nh.advertise<visualization_msgs::MarkerArray>("/init_end_path", 1);
        afirst_end_pub = nh.advertise<visualization_msgs::MarkerArray>("/afirst_end_path", 1);

        prm_pub = nh.advertise<visualization_msgs::MarkerArray>("/prm_path", 1);

        moma_cmd_pub = nh.advertise<fake_moma::MomaCmd>("cmd", 1);
        text_pub = nh.advertise<visualization_msgs::Marker>("/benchmark_text", 1);

        mesh_traj_pub= nh.advertise<planner::MeshPCD>("/mesh_traj", 1, true);
        plot_traj_ee = nh.advertise<visualization_msgs::Marker>("/plot_traj_ee", 1);
        plot_trad_traj_ee = nh.advertise<visualization_msgs::Marker>("/plot_trad_traj_ee", 1);

        // anim_timer = nh.createTimer(ros::Duration(0.1),
        //     &Planner::animate_trajectory_timer_cb, this);

        // wps_sub = nh.subscribe<geometry_msgs::Pose>("/manual_target", 1, &Planner::rcvWpsCallBack, this);
        state_sub = nh.subscribe("state", 1, &Planner::rcvStateCallBack, this);
        if (mode.compare("stat") == 0) {
            // for data collection
            statistics_sub = nh.subscribe("/move_base_simple/goal", 1, &Planner::rcvStaCallBack, this);
        } else if (mode.compare("bk") == 0) {
            statistics_sub = nh.subscribe("/move_base_simple/goal", 1, &Planner::fixedPlanCallBack, this);
        } else if (mode.compare("planner") == 0) 
        {
            if (random_ee)
                statistics_sub = nh.subscribe("/move_base_simple/goal", 1, &Planner::planCallBack, this);
            else
                statistics_sub = nh.subscribe("/manual_target", 1, &Planner::planCallBack, this);
        } else if (mode.compare("benchmark") == 0) {
            statistics_sub = nh.subscribe("/move_base_simple/goal", 1, &Planner::benchmarkCallBack, this);
        } else if (mode.compare("benchmark_replica") == 0) {
            statistics_sub = nh.subscribe("/move_base_simple/goal", 1, &Planner::benchmarkReplicaCallBack, this);
            taskClient = nh.serviceClient<planner::Task>("/replica_task");
        } else if (mode.compare("shm") == 0) {
            statistics_sub = nh.subscribe("/move_base_simple/goal", 1, &Planner::shmCallback, this);
        // } else if (mode.compare("benchmark_remani") == 0) {
        //     statistics_sub = nh.subscribe("/move_base_simple/goal", 1, &Planner::benchmarkRemaniCallBack, this);
        } else if (mode.compare("replica_shm") == 0) {
            statistics_sub = nh.subscribe("/move_base_simple/goal", 1, &Planner::replicashmCallback, this);
            taskClient = nh.serviceClient<planner::Task>("/replica_task");
        } else {
            throw std::runtime_error("Unidentified mode");
        }

        plannerClient = nh.serviceClient<planner::Plan>("/ddim_plan");
        
        // if (planner_type.compare("ddim") == 0) {
        //     // plannerClient = nh.serviceClient<planner::Plan>("/ddim_plan");
        // } else if (planner_type.compare("moma") == 0) {

        // } else {
        //     throw std::runtime_error("Unidentified planner type");
        // }
        

        if (local_mode)
        {
            std::thread cmd_thread(Planner::cmdCallback, this);
            cmd_thread.detach();
            std::thread replan_thread(Planner::replanCallback, this);
            replan_thread.detach();
            std::thread safe_thread(Planner::safeCallback, this);
            safe_thread.detach();
        }
        
        bool scene_seeded; nh.getParam("agent/scene_seeded", scene_seeded);
        scene_seed; nh.getParam("agent/scene_seed", scene_seed);
        if (scene_seeded) {
            eng =default_random_engine(scene_seed);
        }
        else if (fixed_sequence) {
            eng = default_random_engine(42);
        } else {
            random_device rd;
            eng = default_random_engine(rd());

        }

        se2_set.setZero();
        front_path.clear();
        now_state.resize(3+moma_param.dof_num);
        now_state.setZero();
        now_dstate = now_state;
        begin_time = ros::Time::now();

        // === for fig type 2 ===
        // grid_map->regenerateMap(scene_seed);

        return;
    }

    void Planner::cmdCallback(void *obj)
    {
        Planner *tsvr = reinterpret_cast<Planner *>(obj);
        while (true)
        {
            ros::Time start_time = ros::Time::now();
            if (tsvr->mpc->hasTraj())
            {
                // tsvr->moma_cmd_pub.publish(tsvr->mpc->getCmd(tsvr->now_state));
                tsvr->mpc->pubCmd(tsvr->now_state, tsvr->moma_cmd_pub, tsvr->gripper_open);
                double t_mpc = (ros::Time::now() - start_time).toSec() * 1000.0;
                if (t_mpc > 1000.0 / tsvr->mpc->ctrl_freq)
                    PRINT_YELLOW("[Planner] MPC time too long: " << t_mpc << " ms");
            }

            int cmd_mm_num = 1000.0 / tsvr->mpc->ctrl_freq;
            std::chrono::milliseconds dura(max(cmd_mm_num - (int)((ros::Time::now() - start_time).toSec() * 1000), 1));
            std::this_thread::sleep_for(dura);
        }
        return;
    }

    void Planner::planCallBack(const geometry_msgs::PoseStamped msg)
    {
        PRINT_GREEN("Get new goal.");
        Eigen::VectorXd end_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);

        if (msg.header.frame_id.compare("target") == 0)
        {
            if (!wps_list.empty())
                end_state = wps_list[0];
            // Eigen::VectorXd moma_set = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            // moma_set.head(3) = Eigen::Vector3d(msg.pose.position.x, msg.pose.position.y, 0.0);
            // Eigen::VectorXd ee_set = Eigen::VectorXd::Zero(9);
            // ee_set.head(3) = Eigen::Vector3d(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);
            // Eigen::Matrix3d R = Eigen::Quaterniond(msg.pose.orientation.w, msg.pose.orientation.x, 
            //                                      msg.pose.orientation.y, msg.pose.orientation.z).toRotationMatrix();
            // ee_set.segment(3, 3) = R.row(0);
            // ee_set.tail(3) = R.row(1);
            // if (traj_opter->optimizeEE(moma_set, ee_set))
            //     end_state = moma_set;
            // else
            //     return;
        }else
        {
            se2_set(0) = msg.pose.position.x;
            se2_set(1) = msg.pose.position.y;
            double dist;
            grid_map->getDistance2d(se2_set.head(2), dist);
            if (dist < moma_param.chassis_colli_radius)
                return;

            se2_set(2) = atan2(2.0*msg.pose.orientation.z*msg.pose.orientation.w, 
                                2.0*msg.pose.orientation.w*msg.pose.orientation.w-1.0);
            visualization_msgs::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = ros::Time::now();
            marker.id = 10086;
            marker.action = visualization_msgs::Marker::ADD;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.scale.x = moma_param.chassis_colli_radius * 2.0;
            marker.scale.y = moma_param.chassis_colli_radius * 2.0;
            marker.scale.z = moma_param.chassis_height;
            marker.pose.position.x = se2_set[0];
            marker.pose.position.y = se2_set[1];
            marker.pose.position.z = moma_param.chassis_height / 2.0;
            marker.pose.orientation.w = 1.0;
            marker.color.a = 0.5;
            marker.color.r = 0.5;
            marker.color.g = 0.5;
            marker.color.b = 0.5;
            car_target_pub.publish(marker);

            uniform_real_distribution<double> rand_q(0.0, 1.0);
            end_state.resize(3+moma_param.dof_num);
            end_state.setZero();
            end_state.head(3) = se2_set;
            
            //! xulong for fix end for huawei
            // Eigen::VectorXd set_mani = Eigen::VectorXd::Zero(moma_param.dof_num);
            // set_mani << 1.0259568691253662, \
            //             0.7873354554176331, \
            //             -0.037315141409635544, \
            //             0.024556782096624374, \
            //             0.09346238523721695, \
            //             0.7401068210601807, \
            //             -1.14198637008667;
            // end_state.tail(moma_param.dof_num) = set_mani;
            //! xulong for fix end for huawei
            
            ros::Time find_time_start = ros::Time::now();
            bool timeout;
            do
            {
                for (size_t i=0; i<moma_param.dof_num; i++)
                    end_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                    moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                    + moma_param.joint_pos_limit_min(i);
            } while (grid_map->isWholeBodyCollision(end_state)
                    && !(timeout = (ros::Time::now() - find_time_start).toSec() > 1.0));

            if(timeout) return;
        }

        global_goal = end_state;
        has_goal = true;
        Eigen::VectorXd local_start = now_state;
        Eigen::VectorXd local_v = now_dstate;
        if (has_traj && local_mode)
        {
            double t = (ros::Time::now() - last_replan_time).toSec() +planning_budget;
            local_start = end_traj.getState(t);
            local_v = end_traj.getDState(t);
        }

        //! xulong: choose your planner here
        // planOmpls(local_start, end_state, local_v);
        // planRemani(local_start, end_state, local_v);
        // planDDIM(local_start, end_state, local_v);

        ros::Time start_time = ros::Time::now();
        bool succ = false;
        if (planner_type.compare("ddim") == 0)
            succ = planDDIM(local_start, end_state, local_v, ddim_path_num).first > 0.0;
        else if (planner_type.compare("moma") == 0)
            succ = planMomaParallel(local_start, end_state, local_v).first;
        
        if (succ)
        {
            global_traj = end_traj;
            // mpc->setTraj(end_traj, 0.0);
            begin_time = ros::Time::now();
            has_traj = true;
        }

        return;
    }

    void Planner::shmCallback(const geometry_msgs::PoseStamped msg)
    {
        Eigen::Vector3d min_bound = grid_map->min_boundary;
        Eigen::Vector3d max_bound = grid_map->max_boundary;

        uniform_real_distribution<double> rand_x(min_bound(0)+2.0, max_bound(0)-2.0);
        uniform_real_distribution<double> rand_y(min_bound(1)+2.0, max_bound(1)-2.0);
        uniform_real_distribution<double> rand_theta(-M_PI, M_PI);
        
        //TODO: add to param
        traj_num_per_env = 1;
        int sum = stat_num;
        int cnt = 0;
        double all_time = 0.0;
        
        shared_data::SharedDataProducer producer(shm_name.c_str(), 32);

        while (cnt < sum && ros::ok())
        {
            grid_map->regenerateMap(scene_seed+cnt);
            Eigen::Vector3d start = Eigen::Vector3d::Zero(3);
            Eigen::Vector3d end;
            {
                if (fixed_startgoal)
                    end << 3.0, 3.0, 0.0;
                else
                    end << rand_x(eng), rand_y(eng), rand_theta(eng);
            }
            
            // Ensure chassis is not in collision
            double start_distance, end_distance;
            grid_map->getDistance2d(start.head(2), start_distance);
            grid_map->getDistance2d(end.head(2), end_distance);
            if (start_distance < moma_param.chassis_colli_radius 
                || end_distance < moma_param.chassis_colli_radius)
                continue;
            // Ensure minimum distance is within range
            float distance = (end - start).head(2).norm();
            if (!fixed_startgoal && (distance > 8.0 || distance < 3.0)) continue;

            bool direct = false;
            {

                list<GraphNode::Ptr> graph;
                vector<vector<Eigen::Vector3d>> raw_paths, filtered_paths, select_paths;
                Eigen::Vector3d topo_start(start(0), start(1), 0.0);
                Eigen::Vector3d topo_end(end(0), end(1), 0.0);
                std::vector<Eigen::Vector3d> start_pts, end_pts;
                start_pts.push_back(topo_start);
                end_pts.push_back(topo_end);
                topo_prm->findTopoPaths(topo_start, topo_end, start_pts, end_pts, graph,
                    raw_paths, filtered_paths, select_paths);

                auto jps_result = graph_search->plan2dJPS(start.head(2), end.head(2), moma_param.chassis_colli_radius+0.1);

                double jps_length = 0.0;
                for (size_t i = 0; i < jps_result.size() - 1; ++i)
                    jps_length += (jps_result[i] - jps_result[i+1]).norm();
                direct = jps_length - (start-end).head(2).norm() < 0.1;
            }
            if (!direct) continue;
            
            Eigen::VectorXd start_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            start_state.head(3) = start;
            Eigen::VectorXd end_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            end_state.head(3) = end;
            // Generate random start and end state
            if (!fixed_startgoal)
            {
                
                // randomly generate start and end state until no collision or maximum try count reached
                uniform_real_distribution<double> rand_q(0.0, 1.0);
                bool collision;
                ros::Time start_time = ros::Time::now();
                do {
                    for (size_t i=0; i<moma_param.dof_num; i++)
                        start_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                         moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                         + moma_param.joint_pos_limit_min(i);
                } while((collision = grid_map->isWholeBodyCollision(start_state)) &&
                        (ros::Time::now() - start_time).toSec() < 2.0 &&
                        ros::ok());
                if (collision) {continue;}
                
                start_time = ros::Time::now();
                do {
                    for (size_t i=0; i<moma_param.dof_num; i++)
                        end_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                         moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                         + moma_param.joint_pos_limit_min(i);
                } while((collision = grid_map->isWholeBodyCollision(end_state)) && 
                        (ros::Time::now() - start_time).toSec() < 1.0 && 
                        ros::ok());
                if (collision) {continue;}
            }
            Eigen::VectorXd v = Eigen::VectorXd::Zero(3+moma_param.dof_num);

            if (grid_map->isWholeBodyCollision(start_state) || grid_map->isWholeBodyCollision(end_state))
                continue;                

            ros::Time start_time = ros::Time::now();
            //! xulong: choose your planner here
            if (staMomaParallel(start_state, end_state, v))
            // if (staRemani(start_state, end_state, v))
            {
                PRINT_GREEN("[Planner] plan success: " << cnt);

                if(!producer.produce(
                    shared_data::DataPoint(
                        grid_map->getESDFBuffer3d(),
                        front_path
                    )
                )) break;
                PRINT_YELLOW("[Planner] data point produced from node id: " << stat_number);
                all_time += (ros::Time::now() - start_time).toSec() * 1000.0;
                cnt ++;
            }
        }
        PRINT_YELLOW("[Planner] Producer exited, node id: " << stat_number);
        return;
    }
    
    void Planner::replicashmCallback(const geometry_msgs::PoseStamped msg)
    {
        int sum = stat_num;
        int cnt = 0;
        double all_time = 0.0;
        
        shared_data::SharedDataProducer producer(shm_name.c_str(), 32);
        while (cnt < sum && ros::ok())
        {
            planner::Task srv;
            taskClient.call(srv);
            grid_map->updateMap(srv.response.xs, srv.response.ys, srv.response.zs);

            Eigen::Vector3d start = Eigen::Vector3d::Zero(3);
            Eigen::Vector3d end(srv.response.goal[0], srv.response.goal[1], 0);

            // Ensure chassis is not in collision
            double start_distance, end_distance;
            grid_map->getDistance2d(start.head(2), start_distance);
            grid_map->getDistance2d(end.head(2), end_distance);
            if (start_distance < moma_param.chassis_colli_radius 
                || end_distance < moma_param.chassis_colli_radius)
                continue;

            // Ensure minimum distance is within range
            float distance = (end - start).head(2).norm();
            if (!fixed_startgoal && (distance > 8.0 || distance < 3.0)) continue;
            
            Eigen::VectorXd start_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            start_state.head(3) = start;
            
            Eigen::VectorXd end_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            end_state.head(3) = end;
            
            uniform_real_distribution<double> rand_q(0.0, 1.0);
            {   // generate start pose
                bool collision;
                ros::Time start_time = ros::Time::now();
                do {
                    for (size_t i=0; i<moma_param.dof_num; i++)
                        start_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                            moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                            + moma_param.joint_pos_limit_min(i);
                } while((collision = grid_map->isWholeBodyCollision(start_state)) &&
                        (ros::Time::now() - start_time).toSec() < 2.0 &&
                        ros::ok());
                if (collision) {continue;}
            }
            
            {   // generate end pose
                uniform_real_distribution<double> rand_theta(-M_PI, M_PI);
                bool collision;
                ros::Time start_time = ros::Time::now();
                do {
                    end_state(2) = rand_theta(eng);
                    for (size_t i=0; i<moma_param.dof_num; i++)
                        end_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                            moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                            + moma_param.joint_pos_limit_min(i);
                } while((collision = grid_map->isWholeBodyCollision(end_state)) && 
                        (ros::Time::now() - start_time).toSec() < 1.0 && 
                        ros::ok());
                if (collision) {continue;}
            }    
            

            Eigen::VectorXd v = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            if (grid_map->isWholeBodyCollision(start_state) || grid_map->isWholeBodyCollision(end_state))
                continue;                

            ros::Time start_time = ros::Time::now();
            if (staMomaParallel(start_state, end_state, v))
            {
                PRINT_GREEN("[Planner] plan success: " << cnt);

                if(!producer.produce(
                    shared_data::DataPoint(
                        grid_map->getESDFBuffer3d(),
                        front_path
                    )
                )) break;
                PRINT_GREEN("[Planner] data point produced from node id: " << stat_number);
                all_time += (ros::Time::now() - start_time).toSec() * 1000.0;
                cnt ++;
            }
        }
        PRINT_YELLOW("[Planner] Producer exited, node id: " << stat_number);
        return;
    }

    void Planner::rcvStaCallBack(const geometry_msgs::PoseStamped msg)
    {
        Eigen::Vector3d min_bound = grid_map->min_boundary;
        Eigen::Vector3d max_bound = grid_map->max_boundary;

        uniform_real_distribution<double> rand_x(min_bound(0)+2.0, max_bound(0)-2.0);
        uniform_real_distribution<double> rand_y(min_bound(1)+2.0, max_bound(1)-2.0);
        uniform_real_distribution<double> rand_theta(-M_PI, M_PI);
        
        //TODO: add to param
        traj_num_per_env = 1;
        int sum = stat_num;
        int cnt = 0;
        double all_time = 0.0;
        
        // std::ofstream ofs;
        // ofs.open(ros::package::getPath("planner") + stat_file + "data_" + stat_number, std::ofstream::app);
        // boost::archive::text_oarchive oa(ofs);

        std::ofstream ofsb;
        ofsb.open(ros::package::getPath("planner") + stat_file + "data_bin_" + stat_number, std::ofstream::binary);
        boost::archive::binary_oarchive oab(ofsb);
        oab << stat_num; // IMPORTANT: write the number of data points into archive

        while (cnt < sum && ros::ok())
        {
            grid_map->regenerateMap();
            Eigen::Vector3d start = Eigen::Vector3d::Zero(3);
            Eigen::Vector3d end;
            {
                if (fixed_startgoal)
                    end << 3.0, 3.0, 0.0;
                else
                    end << rand_x(eng), rand_y(eng), rand_theta(eng);
            }
            
            // Ensure chassis is not in collision
            double start_distance, end_distance;
            grid_map->getDistance2d(start.head(2), start_distance);
            grid_map->getDistance2d(end.head(2), end_distance);
            if (start_distance < moma_param.chassis_colli_radius 
                || end_distance < moma_param.chassis_colli_radius)
                continue;
            
            // Ensure minimum distance is within range
            float distance = (end - start).head(2).norm();
            if (!fixed_startgoal && (distance > 8.0 || distance < 3.0)) continue;
            
            Eigen::VectorXd start_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            start_state.head(3) = start;
            Eigen::VectorXd end_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            end_state.head(3) = end;
            if (!fixed_startgoal)
            {
                
                // randomly generate start and end state until no collision or maximum try count reached
                uniform_real_distribution<double> rand_q(0.0, 1.0);
                bool collision;
                ros::Time start_time = ros::Time::now();
                do {
                    for (size_t i=0; i<moma_param.dof_num; i++)
                        start_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                         moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                         + moma_param.joint_pos_limit_min(i);
                } while((collision = grid_map->isWholeBodyCollision(start_state)) &&
                        (ros::Time::now() - start_time).toSec() < 2.0 &&
                        ros::ok());
                if (collision) {continue;}
                
                start_time = ros::Time::now();
                do {
                    for (size_t i=0; i<moma_param.dof_num; i++)
                        end_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                         moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                         + moma_param.joint_pos_limit_min(i);
                } while((collision = grid_map->isWholeBodyCollision(end_state)) && 
                        (ros::Time::now() - start_time).toSec() < 1.0 && 
                        ros::ok());
                if (collision) {continue;}
            }
            Eigen::VectorXd v = Eigen::VectorXd::Zero(3+moma_param.dof_num);

            if (grid_map->isWholeBodyCollision(start_state) || grid_map->isWholeBodyCollision(end_state))
                continue;                

            ros::Time start_time = ros::Time::now();
            //! xulong: choose your planner here
            if (staMomaParallel(start_state, end_state, v))
            // if (staRemani(start_state, end_state, v))
            {
                PRINT_GREEN("[Planner] plan success: " << cnt);

                data::DataPoint<10> data_point(
                    grid_map->getESDFBuffer3d(),
                    front_path
                );
                data_point.setHash(data::hash_value(data_point));

                // PRINT_GREEN("[Planner] Hash: " << data_point.getHash());

                // oa << data_point;
                oab << data_point;
                
                all_time += (ros::Time::now() - start_time).toSec() * 1000.0;
                cnt ++;
            }
        }
        
        // ofs.close();
        ofsb.flush();
        ofsb.close();

        PRINT_GREEN("[Planner] average plan time: " << all_time/sum << " ms");
        PRINT_GREEN("[Planner] statistice done.");
        return;
    }

    void Planner::fixedPlanCallBack(const geometry_msgs::PoseStamped msg)
    {
        Eigen::Vector3d start = Eigen::Vector3d::Zero(3);
        Eigen::Vector3d end;
        end << 3.0, 3.0, 0.0;
        
        Eigen::VectorXd start_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
        start_state.head(3) = start;
        Eigen::VectorXd end_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
        end_state.head(3) = end;

        PRINT_GREEN("[Planner] generating random start and end state");
        int i;
        do {
            i++;
            grid_map->regenerateMap(scene_seed + i); 
            
            
            if (!fixed_startgoal) {
                Eigen::Vector3d min_bound = grid_map->min_boundary;
                Eigen::Vector3d max_bound = grid_map->max_boundary;

                uniform_real_distribution<double> rand_x(min_bound(0)+2.0, max_bound(0)-2.0);
                uniform_real_distribution<double> rand_y(min_bound(1)+2.0, max_bound(1)-2.0);
                uniform_real_distribution<double> rand_theta(-M_PI, M_PI);
                uniform_real_distribution<double> rand_q(0.0, 1.0);

                start << 0.0, 0.0, 0.0;
                
                double distance = (end - start).head(2).norm();
                if (distance > 8.0 || distance < 3.0) continue;
                
                double start_distance, end_distance;
                grid_map->getDistance2d(start.head(2), start_distance);
                grid_map->getDistance2d(end.head(2), end_distance);

                if (start_distance < moma_param.chassis_colli_radius 
                    || end_distance < moma_param.chassis_colli_radius)
                    continue;
                    
                    
                    ros::Time time_start;
                    bool timeout, start_state_collision, end_state_collision;
                    time_start = ros::Time::now();
                do
                {
                    end << rand_x(eng), rand_y(eng), rand_theta(eng);
                    end_state.head(3) = end;

                    for (size_t i=0; i<moma_param.dof_num; i++)
                        end_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                        moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                        + moma_param.joint_pos_limit_min(i);
                } while (grid_map->isWholeBodyCollision(end_state)
                        && !(timeout = (ros::Time::now() - time_start).toSec() > 1.0) && ros::ok());
                if(!ros::ok()) break;
                if(timeout || end_state_collision) continue;


                Eigen::VectorXd start_state;
                start_state.resize(3+moma_param.dof_num);
                start_state.setZero();

                time_start = ros::Time::now();
                do
                {
                    for (size_t i=0; i<moma_param.dof_num; i++)
                        start_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                        moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                        + moma_param.joint_pos_limit_min(i);
                } while ((start_state_collision = grid_map->isWholeBodyCollision(start_state))
                        && !(timeout = (ros::Time::now() - time_start).toSec() > 1.0) && ros::ok());
                if(!ros::ok()) break;
                if(timeout || start_state_collision) continue;
                break;
            } else if (fixed_startgoal) {
                if(grid_map->isWholeBodyCollision(start_state) || grid_map->isWholeBodyCollision(end_state))
                    continue;
                break;
            }
        } while (ros::ok());
        PRINT_GREEN("[Planner] generated random start and end state");
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time::now();
        marker.id = 10086;
        marker.action = visualization_msgs::Marker::ADD;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.scale.x = moma_param.chassis_colli_radius * 2.0;
        marker.scale.y = moma_param.chassis_colli_radius * 2.0;
        marker.scale.z = moma_param.chassis_height;
        marker.pose.position.x = end_state[0];
        marker.pose.position.y = end_state[1];
        marker.pose.position.z = moma_param.chassis_height / 2.0;
        marker.pose.orientation.w = 1.0;
        marker.color.a = 0.5;
        marker.color.r = 0.5;
        marker.color.g = 0.5;
        marker.color.b = 0.5;
        car_target_pub.publish(marker);

        Eigen::VectorXd v = Eigen::VectorXd::Zero(3+moma_param.dof_num);
        
        //! xulong test ompl
        // auto ompl_path = planOmpls(start_state, end_state, v);
        // Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(10, 2);
        // Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(10, 2);
        // boundary_vel.col(0) = v;
        // if(ompl_path.empty()) {
        //     PRINT_RED("[Planner] OMPL failed.\n");
        //     return;
        // }
        // vis_path(ompl_path, ompl_pub);

        // for(auto& state : ompl_path) {
        //     if(grid_map->isWholeBodyCollision(state)) {
        //         PRINT_RED("[Planner] OMPL failed due to collision.\n");
        //     }
        // }
        
        // if (!this->traj_opters[0]->optimizeTraj(ompl_path, boundary_vel, boundary_acc)
        //     || !this->traj_opters[0]->printConstraintsSituations(traj_opters[0]->getTraj())
        //     || !this->traj_opters[0]->getTraj().is_init
        // ){
        //     PRINT_RED("[Planner] OMPL optimize failed.\n");
        //     return;
        // }
        // end_traj = traj_opters[0]->getTraj();
        // vis_whole_path(end_pub);
        //! xulong test ompl
        
        ros::Time start_time = ros::Time::now();
        bool succ =false;
        if (planner_type.compare("ddim") == 0)
            succ = planDDIM(start_state, end_state, v, ddim_path_num).first > 0.0;
        else if (planner_type.compare("moma") == 0)
            succ = planMomaParallel(start_state, end_state, v).first;
        if (succ) PRINT_GREEN("[Planner] Moma success: " << (ros::Time::now() - start_time).toSec() << " s");

        // if(!succ)
        // {
        //     PRINT_RED("[Planner] Moma failed.\n");
        //     auto ompl_path = planOmpls(start_state, end_state, v);
        //     Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(10, 2);
        //     Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(10, 2);
        //     boundary_vel.col(0) = v;
        //     if (ompl_path.empty()) {
        //         PRINT_RED("[Planner] OMPL failed.\n");
        //         return;
        //     }
        //     vis_path(ompl_path, ompl_pub);
        //     if (!this->traj_opters[0]->optimizeTraj(ompl_path, boundary_vel, boundary_acc)
        //         || !this->traj_opters[0]->printConstraintsSituations(traj_opters[0]->getTraj())
        //         || !this->traj_opters[0]->getTraj().is_init
        //     ){
        //         PRINT_RED("[Planner] OMPL optimize failed.\n");
        //         return;
        //     }
        //     end_traj = traj_opters[0]->getTraj();
        //     vis_whole_path(end_pub);
        // }

        return;
    }

    void Planner::benchmarkCallBack(const geometry_msgs::PoseStamped msg)
    {
        Eigen::Vector3d start = Eigen::Vector3d::Zero(3);
        Eigen::Vector3d end;
        end << 3.0, 3.0, 0.0;
        
        Eigen::VectorXd start_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
        start_state.head(3) = start;
        Eigen::VectorXd end_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
        end_state.head(3) = end;
        
        int comparison_num = 0;
        double mean_time_ratio = 0.0;
        double mean_path_ratio = 0.0;
        
        int num_ddim_succ = 0;
        double mean_ddim_duration = 0.0;
        float mean_ddim_optim_time = 0.0;
        float mean_ddim_sample_time = 0.0;

        int num_moma_succ = 0;
        float mean_moma_optim_time = 0.0;
        double mean_moma_duration = 0.0;

        size_t num_plan = 0;

        static unsigned int seed_offset = 0;

        for (; num_plan<stat_num && ros::ok(); num_plan++) {

            // Randomly generate map, start and end state
            do {
                // comment out for fig type 2
                grid_map->regenerateMap(scene_seed+seed_offset);

                // PRINT_GREEN("SEED : " << scene_seed+seed_offset);e
                // seed_offset++;
                
                if (!fixed_startgoal) {
                    Eigen::Vector3d min_bound = grid_map->min_boundary;
                    Eigen::Vector3d max_bound = grid_map->max_boundary;
    
                    uniform_real_distribution<double> rand_x(min_bound(0)+2.0, max_bound(0)-2.0);
                    uniform_real_distribution<double> rand_y(min_bound(1)+2.0, max_bound(1)-2.0);
                    uniform_real_distribution<double> rand_theta(-M_PI, M_PI);
                    uniform_real_distribution<double> rand_q(0.0, 1.0);

                    end << rand_x(eng), rand_y(eng), rand_theta(eng);
                    end_state.head(3) = end;

                    start_state(2) = 0;

                    // bool direct = false;
                    // {

                    //     list<GraphNode::Ptr> graph;
                    //     vector<vector<Eigen::Vector3d>> raw_paths, filtered_paths, select_paths;
                    //     Eigen::Vector3d topo_start(start(0), start(1), 0.0);
                    //     Eigen::Vector3d topo_end(end(0), end(1), 0.0);
                    //     std::vector<Eigen::Vector3d> start_pts, end_pts;
                    //     start_pts.push_back(topo_start);
                    //     end_pts.push_back(topo_end);
                    //     topo_prm->findTopoPaths(topo_start, topo_end, start_pts, end_pts, graph,
                    //         raw_paths, filtered_paths, select_paths);

                    //     auto jps_result = graph_search->plan2dJPS(start.head(2), end.head(2), moma_param.chassis_colli_radius+0.1);
                    //     double jps_length = 0.0;
                    //     for (size_t i = 0; (i+1) < jps_result.size(); ++i)
                    //         jps_length += (jps_result[i] - jps_result[i+1]).norm();
                    //     direct = jps_length - (start-end).head(2).norm() < 0.1;
                    // }
                    // if (direct) continue;

                    int max_trial = 2000;
                    int n_trial = 0;
                    ros::Time time_start;
                    bool timeout, start_state_collision, end_state_collision;
                    time_start = ros::Time::now();
                    do
                    {
                        for (size_t i=0; i<moma_param.dof_num; i++)
                            end_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                            moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                            + moma_param.joint_pos_limit_min(i);
                    } while ((start_state_collision = grid_map->isWholeBodyCollision(end_state))
                            && !(timeout = (n_trial++ > max_trial)));
                    if(timeout || start_state_collision) continue;
    
                    time_start = ros::Time::now();
                    n_trial = 0;
                    do
                    {
                        for (size_t i=0; i<moma_param.dof_num; i++)
                            start_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                            moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                            + moma_param.joint_pos_limit_min(i);
                    } while ((end_state_collision = grid_map->isWholeBodyCollision(start_state))
                            && !(timeout = (n_trial++ > max_trial)));
                    if(timeout || end_state_collision) continue;
                    break;
                } else if (fixed_startgoal) {
                    uniform_real_distribution<double> rand_q(0.0, 1.0);

                    int max_trial = 2000;
                    int n_trial = 0;
                    ros::Time time_start;
                    bool timeout, start_state_collision, end_state_collision;
                    time_start = ros::Time::now();
                    do
                    {
                        for (size_t i=0; i<moma_param.dof_num; i++)
                            end_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                            moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                            + moma_param.joint_pos_limit_min(i);
                    } while ((start_state_collision = grid_map->isWholeBodyCollision(end_state))
                            && !(timeout = (n_trial++ > max_trial)));
                    if(timeout || start_state_collision) continue;
    
                    time_start = ros::Time::now();
                    n_trial = 0;
                    do
                    {
                        for (size_t i=0; i<moma_param.dof_num; i++)
                            start_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                            moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                            + moma_param.joint_pos_limit_min(i);
                    } while ((end_state_collision = grid_map->isWholeBodyCollision(start_state))
                            && !(timeout = (n_trial++ > max_trial)));
                    if(timeout || end_state_collision) continue;
                    if(grid_map->isWholeBodyCollision(start_state) || grid_map->isWholeBodyCollision(end_state))
                        continue;
                    break;
                }
            } while (true);

            if (grid_map->isWholeBodyCollision(start_state) || grid_map->isWholeBodyCollision(end_state))
                PRINT_RED("Collision detected, skip this data point.");
            
            Eigen::VectorXd v = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            
            bool ddim_succ = false;
            double ddim_path = 0.0;
            double ddim_duration = 0.0;
            float ddim_optim_time = 0.0;
            float ddim_sample_time;
            {
                ros::Time start_time = ros::Time::now();
                std::tie(ddim_optim_time, ddim_sample_time) = planDDIM(start_state, end_state, v, ddim_path_num);
                ddim_succ = ddim_optim_time > 0.0;
                ddim_duration = (ros::Time::now() - start_time).toSec();
                if(ddim_succ) {
                    ddim_path = end_traj.getTotalDuration();
                }
            }

            bool moma_succ = false;
            double moma_path = 0.0;
            double moma_duration = 0.0;
            float moma_optim_time = 0.0;
            {
                ros::Time start_time = ros::Time::now();
                std::tie(moma_succ, moma_optim_time) = planMomaParallel(start_state, end_state, v);
                moma_duration = (ros::Time::now() - start_time).toSec() * 1000.0;
                if(moma_succ) {
                    moma_path = end_traj.getTotalDuration();
                }
            }
            
            if (ddim_succ) {
                num_ddim_succ++;
                mean_ddim_duration += (ddim_duration - mean_ddim_duration) / num_ddim_succ;
                mean_ddim_optim_time += (ddim_optim_time - mean_ddim_optim_time) / num_ddim_succ;
                mean_ddim_sample_time += (ddim_sample_time - mean_ddim_sample_time) / num_ddim_succ;

                vis_path_mesh(ddim_traj, 8, 
                    end_pub,
                    {
                        38.0/255.0,
                        70.0/255.0,
                        83.0/255,
                        0.2
                    },
                    4322
                );

                vis_ee_traj(ddim_traj, plot_traj_ee, 
                    {
                        38.0/255.0,
                        70.0/255.0,
                        83.0/255,
                        0.2
                    }
                );
            }
            
            if (moma_succ) {
                num_moma_succ++;
                mean_moma_duration += (moma_duration - mean_moma_duration) / num_moma_succ;
                mean_moma_optim_time += (moma_optim_time - mean_moma_optim_time) / num_moma_succ;

                vis_path_mesh(topay_traj, 8, 
                    trad_pub,
                    {
                        230.0/255.0,
                        111.0/255.0,
                        81.0/255,
                        0.2
                    },
                    4322
                );

                vis_ee_traj(topay_traj, plot_trad_traj_ee, 
                    {
                        230.0/255.0,
                        111.0/255.0,
                        81.0/255,
                        0.2
                    }
                );
            }

            publishBenchmarkText(num_ddim_succ, num_moma_succ, num_plan+1);

            // visualization_msgs::Marker marker;
            // marker.action = visualization_msgs::Marker::DELETEALL;

            // plot_traj_ee.publish(marker);
            // plot_trad_traj_ee.publish(marker);

            // visualization_msgs::MarkerArray marker_array;
            // end_pub.publish(marker_array);
            // trad_pub.publish(marker_array);


            if (ddim_succ && moma_succ) {
                comparison_num++;
                mean_path_ratio += (ddim_path / moma_path - mean_path_ratio) / comparison_num;
                mean_time_ratio += (ddim_optim_time / moma_optim_time - mean_time_ratio) / comparison_num;
            }
        }

        if (!auto_mode) {
            PRINT_GREEN("[Planner] benchmark done.");
            PRINT_GREEN("DDIM:\t" << num_ddim_succ << "/" << num_plan  << "\tAvg. Plan time: " << mean_ddim_duration<< "ms\t Avg. optim time: " << mean_ddim_optim_time << "ms\t Avg. sample time: " << mean_ddim_sample_time << "ms" << std::endl);
            PRINT_GREEN("MOMA:\t" << num_moma_succ << "/" << num_plan  << "\tAvg. Plan time: " << mean_moma_duration<< "ms\t Avg. optim time: " << mean_moma_optim_time << "ms" << std::endl);
            
            PRINT_GREEN("Comparison:\tAvg. Path ratio: " << mean_path_ratio << "\t Avg. Time ratio: " << mean_time_ratio << std::endl);
        } else
        {
            // Generate filename based on current time
            std::time_t t = std::time(nullptr);
            std::tm tm = *std::localtime(&t);
    
            std::ostringstream filename;
            filename << "benchmark_"
                    << std::put_time(&tm, "%Y%m%d_%H%M%S")
                    << ".txt";

            std::string pkg_path = ros::package::getPath("planner");

            std::string log_dir = pkg_path + "/logs/";
            mkdir(log_dir.c_str(), 0777);

            std::string full_path = log_dir + filename.str();

            std::ofstream outfile(full_path);
    
            if (!outfile.is_open()) {
                std::cerr << "Failed to open file!" << std::endl;
                return;
            }
    
            // Write results
            outfile << "[Planner] benchmark done.\n";
            outfile << "ddim_path_num: " << ddim_path_num << "\n";
    
            outfile << "DDIM:\t" << num_ddim_succ << "/" << num_plan
                    << "\tAvg. Plan time: " << mean_ddim_duration << "ms"
                    << "\tAvg. optim time: " << mean_ddim_optim_time << "ms"
                    << "\tAvg. sample time: " << mean_ddim_sample_time << "ms\n";
    
            outfile << "MOMA:\t" << num_moma_succ << "/" << num_plan
                    << "\tAvg. Plan time: " << mean_moma_duration << "ms"
                    << "\tAvg. optim time: " << mean_moma_optim_time << "ms\n";
    
            outfile << "Comparison:\tAvg. Path ratio: " << mean_path_ratio
                    << "\tAvg. Time ratio: " << mean_time_ratio << "\n";
    
            outfile.close();
    
            std::cout << "Benchmark saved to " << full_path << std::endl;
            ros::shutdown();
        }

        

        
        return;
    }

    void Planner::benchmarkReplicaCallBack(const geometry_msgs::PoseStamped msg)
    {
        Eigen::Vector3d start = Eigen::Vector3d::Zero(3);
        Eigen::Vector3d end;
        end << 3.0, 3.0, 0.0;
        
        Eigen::VectorXd start_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
        start_state.head(3) = start;
        Eigen::VectorXd end_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
        end_state.head(3) = end;
        
        int comparison_num = 0;
        double mean_time_ratio = 0.0;
        double mean_path_ratio = 0.0;
        
        int num_ddim_succ = 0;
        double mean_ddim_duration = 0.0;
        float mean_ddim_optim_time = 0.0;
        float mean_ddim_sample_time = 0.0;

        int num_moma_succ = 0;
        float mean_moma_optim_time = 0.0;
        double mean_moma_duration = 0.0;

        size_t num_plan = 0;
        for (; num_plan<stat_num && ros::ok(); num_plan++) {
            // Randomly generate map, start and end state
            do {
                planner::Task srv;
                taskClient.call(srv);
                grid_map->updateMap(srv.response.xs, srv.response.ys, srv.response.zs);
                
                Eigen::Vector3d min_bound = grid_map->min_boundary;
                Eigen::Vector3d max_bound = grid_map->max_boundary;

                uniform_real_distribution<double> rand_x(min_bound(0)+2.0, max_bound(0)-2.0);
                uniform_real_distribution<double> rand_y(min_bound(1)+2.0, max_bound(1)-2.0);
                uniform_real_distribution<double> rand_theta(-M_PI, M_PI);
                uniform_real_distribution<double> rand_q(0.0, 1.0);

                end << srv.response.goal[0], srv.response.goal[1], rand_theta(eng);
                end_state.head(3) = end;

                start_state(2) = 0;
                
                int max_trial = 2000;
                int n_trial = 0;
                ros::Time time_start;
                bool timeout, start_state_collision, end_state_collision;
                time_start = ros::Time::now();
                do
                {
                    for (size_t i=0; i<moma_param.dof_num; i++)
                        end_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                        moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                        + moma_param.joint_pos_limit_min(i);
                } while ((start_state_collision = grid_map->isWholeBodyCollision(end_state))
                        && !(timeout = (ros::Time::now() - time_start).toSec() > 1.0));
                if(timeout || start_state_collision) continue;

                n_trial = 0;
                time_start = ros::Time::now();
                do
                {
                    for (size_t i=0; i<moma_param.dof_num; i++)
                        start_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                        moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                        + moma_param.joint_pos_limit_min(i);
                } while ((end_state_collision = grid_map->isWholeBodyCollision(start_state))
                        && !(timeout = (ros::Time::now() - time_start).toSec() > 1.0));
                if(timeout || end_state_collision) continue;
                break;
            } while (true);

            Eigen::VectorXd v = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            
            bool ddim_succ = false;
            double ddim_path = 0.0;
            double ddim_duration = 0.0;
            float ddim_optim_time = 0.0;
            float ddim_sample_time;
            {
                ros::Time start_time = ros::Time::now();
                std::tie(ddim_optim_time, ddim_sample_time) = planDDIM(start_state, end_state, v, ddim_path_num);
                ddim_succ = ddim_optim_time > 0.0;
                ddim_duration = (ros::Time::now() - start_time).toSec();
                if(ddim_succ) {
                    ddim_path = end_traj.getTotalDuration();
                }
            }

            bool moma_succ = false;
            double moma_path = 0.0;
            double moma_duration = 0.0;
            float moma_optim_time = 0.0;
            {
                ros::Time start_time = ros::Time::now();
                std::tie(moma_succ, moma_optim_time) = planMomaParallel(start_state, end_state, v);
                moma_duration = (ros::Time::now() - start_time).toSec();
                if(moma_succ) {
                    moma_path = end_traj.getTotalDuration();
                }
            }
            
            if (ddim_succ) {
                num_ddim_succ++;
                mean_ddim_duration += (ddim_duration - mean_ddim_duration) / num_ddim_succ;
                mean_ddim_optim_time += (ddim_optim_time - mean_ddim_optim_time) / num_ddim_succ;
                mean_ddim_sample_time += (ddim_sample_time - mean_ddim_sample_time) / num_ddim_succ;
                vis_path_mesh(ddim_traj, 8, 
                    end_pub,
                    {
                        38.0/255.0,
                        70.0/255.0,
                        83.0/255,
                        0.2
                    },
                    4322
                );

                vis_ee_traj(ddim_traj, plot_traj_ee, 
                    {
                        38.0/255.0,
                        70.0/255.0,
                        83.0/255,
                        0.2
                    }
                );
            }
            
            if (moma_succ) {
                num_moma_succ++;
                mean_moma_duration += (moma_duration - mean_moma_duration) / num_moma_succ;
                mean_moma_optim_time += (moma_optim_time - mean_moma_optim_time) / num_moma_succ;
                vis_path_mesh(topay_traj, 8, 
                    trad_pub,
                    {
                        230.0/255.0,
                        111.0/255.0,
                        81.0/255,
                        0.2
                    },
                    4322
                );

                vis_ee_traj(topay_traj, plot_trad_traj_ee, 
                    {
                        230.0/255.0,
                        111.0/255.0,
                        81.0/255,
                        0.2
                    }
                );
            }

            publishBenchmarkText(num_ddim_succ, num_moma_succ, num_plan+1);

            if (ddim_succ && moma_succ) {
                comparison_num++;
                mean_path_ratio += (ddim_path / moma_path - mean_path_ratio) / comparison_num;
                mean_time_ratio += (ddim_optim_time / moma_optim_time - mean_time_ratio) / comparison_num;
            }

            // if(ddim_succ || moma_succ) {ros::Duration(0.25).sleep();}
            // visualization_msgs::Marker marker;
            // marker.action = visualization_msgs::Marker::DELETEALL;

            // plot_traj_ee.publish(marker);
            // plot_trad_traj_ee.publish(marker);

            // visualization_msgs::MarkerArray marker_array;
            // marker_array.markers.push_back(marker);
            // end_pub.publish(marker_array);
            // trad_pub.publish(marker_array);
        }
        if (!auto_mode) {
            PRINT_GREEN("[Planner] benchmark done.");
            PRINT_GREEN("DDIM:\t" << num_ddim_succ << "/" << num_plan  << "\tAvg. Plan time: " << mean_ddim_duration<< "ms\t Avg. optim time: " << mean_ddim_optim_time << "ms\t Avg. sample time: " << mean_ddim_sample_time << "ms" << std::endl);
            PRINT_GREEN("MOMA:\t" << num_moma_succ << "/" << num_plan  << "\tAvg. Plan time: " << mean_moma_duration<< "ms\t Avg. optim time: " << mean_moma_optim_time << "ms" << std::endl);
            
            PRINT_GREEN("Comparison:\tAvg. Path ratio: " << mean_path_ratio << "\t Avg. Time ratio: " << mean_time_ratio << std::endl);
        } else
        {
            // Generate filename based on current time
            std::time_t t = std::time(nullptr);
            std::tm tm = *std::localtime(&t);
    
            std::ostringstream filename;
            filename << "benchmark_"
                    << std::put_time(&tm, "%Y%m%d_%H%M%S")
                    << ".txt";

            std::string pkg_path = ros::package::getPath("planner");

            std::string log_dir = pkg_path + "/logs/";
            mkdir(log_dir.c_str(), 0777);

            std::string full_path = log_dir + filename.str();

            std::ofstream outfile(full_path);
    
            if (!outfile.is_open()) {
                std::cerr << "Failed to open file!" << std::endl;
                return;
            }
    
            // Write results
            outfile << "[Planner] benchmark done.\n";
            outfile << "ddim_path_num: " << ddim_path_num << "\n";
    
            outfile << "DDIM:\t" << num_ddim_succ << "/" << num_plan
                    << "\tAvg. Plan time: " << mean_ddim_duration << "ms"
                    << "\tAvg. optim time: " << mean_ddim_optim_time << "ms"
                    << "\tAvg. sample time: " << mean_ddim_sample_time << "ms\n";
    
            outfile << "MOMA:\t" << num_moma_succ << "/" << num_plan
                    << "\tAvg. Plan time: " << mean_moma_duration << "ms"
                    << "\tAvg. optim time: " << mean_moma_optim_time << "ms\n";
    
            outfile << "Comparison:\tAvg. Path ratio: " << mean_path_ratio
                    << "\tAvg. Time ratio: " << mean_time_ratio << "\n";
    
            outfile.close();
    
            std::cout << "Benchmark saved to " << full_path << std::endl;
            ros::shutdown();
        }

        // PRINT_GREEN("[Planner] benchmark done.");
        // PRINT_GREEN("DDIM:\t" << num_ddim_succ << "/" << num_plan  << "\tAvg. Plan time: " << mean_ddim_duration<< "ms\t Avg. optim time: " << mean_ddim_optim_time << "ms\t Avg. sample time: " << mean_ddim_sample_time << "ms" << std::endl);
        // PRINT_GREEN("MOMA:\t" << num_moma_succ << "/" << num_plan  << "\tAvg. Plan time: " << mean_moma_duration<< "ms\t Avg. optim time: " << mean_moma_optim_time << "ms" << std::endl);

        // PRINT_GREEN("Comparison:\tAvg. Path ratio: " << mean_path_ratio << "\t Avg. Time ratio: " << mean_time_ratio << std::endl);
        return;
    }

    void Planner::safeCallback(void *obj)
    {
        Planner *tsvr = reinterpret_cast<Planner *>(obj);
        while (true)
        {
            ros::Time start_time = ros::Time::now();
            if (tsvr->has_goal && tsvr->has_traj && tsvr->is_safe && (!tsvr->in_plan) )
            {
                Eigen::VectorXd temp_state = Eigen::VectorXd::Zero(10);
                std::vector<Eigen::Vector4d> min_dist_mani = tsvr->moma_param.getColliPts(temp_state);
                double res = 0.01;
                for (double t=0.0; t<tsvr->end_traj.getTotalDuration(); t+=res)
                {
                    Eigen::VectorXd state = tsvr->end_traj.getState(t);
        
                    double d = 0.0;
                    tsvr->grid_map->getDistance2d(state.head(2), d);
                    if (d < tsvr->moma_param.chassis_colli_radius * 0.99)
                    {
                        tsvr->is_safe = false;
                        break;
                    }
                    std::vector<Eigen::Vector4d> mani_pts = tsvr->moma_param.getColliPts(state);
                    for (size_t i=0; i<mani_pts.size(); i++)
                    {
                        double d = 0.0;
                        tsvr->grid_map->getDistance3d(mani_pts[i].head(3), d);
                        if (d < min_dist_mani[i].w() * 0.99)
                        {
                            tsvr->is_safe = false;
                            break;
                        }
                    }
                    if (!tsvr->is_safe) break;
                }
            }

            int cmd_mm_num = 1000.0 / tsvr->mpc->ctrl_freq;
            std::chrono::milliseconds dura(max(cmd_mm_num - (int)((ros::Time::now() - start_time).toSec() * 1000), 1));
            std::this_thread::sleep_for(dura);
        }
    }

    void Planner::replanCallback(void *obj)
    {
        Planner *tsvr = reinterpret_cast<Planner *>(obj);
        while (true)
        {
            ros::Time start_time = ros::Time::now();
            
            if (tsvr->has_goal && tsvr->has_traj)
            {
                if ((tsvr->now_state.head(2)-tsvr->global_goal.head(2)).norm () < 0.5)
                {
                    tsvr->has_goal = false;
                    tsvr->has_traj = false;
                    if (tsvr->wps_list.size() > 1)
                    {
                        if ((tsvr->global_goal-tsvr->place_vec).norm() < 0.1 ||
                            (tsvr->global_goal-tsvr->pick_vec).norm() < 0.1)
                        {
                            while (!tsvr->mpc->atGoal())
                                ROS_DEBUG("Waiting reaching goal...");
                            // in
                            Eigen::Vector3d direct;
                            direct.head(2) = tsvr->now_state.head(2) + \
                                            0.1 * Eigen::Vector2d(cos(tsvr->now_state(2)), sin(tsvr->now_state(2)));
                            tsvr->mpc->setDirect(direct);
                            while (!tsvr->mpc->atGoal())
                                ROS_DEBUG("Waiting reaching goal...");
                            // pick or place
                            tsvr->gripper_open = !tsvr->gripper_open;
                            this_thread::sleep_for(chrono::milliseconds(1000));
                            // out
                            direct.head(2) = tsvr->now_state.head(2) - \
                                            1.0 * Eigen::Vector2d(cos(tsvr->now_state(2)), sin(tsvr->now_state(2)));
                            tsvr->mpc->setDirect(direct);
                            while (!tsvr->mpc->atGoal())
                                ROS_DEBUG("Waiting reaching goal...");
                            this_thread::sleep_for(chrono::milliseconds(1000));
                        }
                        
                        tsvr->wps_list.erase(tsvr->wps_list.begin());
                        tsvr->global_goal = tsvr->wps_list.front();
                        PRINT_GREEN("New goal: " << tsvr->global_goal.transpose());
                        tsvr->has_goal = true;
                        bool succ;
                        Eigen::VectorXd local_start = tsvr->now_state;
                        Eigen::VectorXd local_v = tsvr->now_dstate;
                        if (tsvr->planner_type.compare("ddim") == 0)
                            succ = tsvr->planDDIM(local_start, tsvr->global_goal, local_v, tsvr->ddim_path_num).first > 0.0;
                        else if (tsvr->planner_type.compare("moma") == 0)
                        {
                            tsvr->in_plan = true;
                            succ = tsvr->planMomaParallel(local_start, tsvr->global_goal, local_v).first;
                            tsvr->in_plan = false;
                        }
                        if (succ)
                        {
                            tsvr->global_traj = tsvr->end_traj;
                            tsvr->mpc->setTraj(tsvr->end_traj, 0.0);
                            tsvr->begin_time = ros::Time::now();
                            tsvr->has_traj = true;
                            tsvr->is_safe = true;
                        }
                    }
                }
                else
                {
                    if ((ros::Time::now() - tsvr->last_replan_time).toSec() > tsvr->replan_interval || \
                        ! tsvr->is_safe)
                    {
                        ros::Time plan_start_time = ros::Time::now();
                        Eigen::VectorXd local_goal;
                        Eigen::VectorXd local_start = tsvr->now_state;
                        Eigen::VectorXd local_v = tsvr->now_dstate;
                        {
                            double t = (ros::Time::now() - tsvr->last_replan_time).toSec() + tsvr->planning_budget;
                            local_start = tsvr->end_traj.getState(t);
                            local_v = tsvr->end_traj.getDState(t);
                        }
                        {
                            double t = (ros::Time::now() - tsvr->begin_time).toSec();
                            bool found = false;
                            for (; t<tsvr->global_traj.getTotalDuration(); t+=0.1)
                            {
                                Eigen::VectorXd state = tsvr->global_traj.getState(t);
                                if ((state.head(2)-local_start.head(2)).norm() > tsvr->planning_horizon)
                                {
                                    local_goal = state;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found)
                                local_goal = tsvr->global_goal;
                        }

                        PRINT_GREEN("local goal: " << local_goal.transpose());

                        tsvr->in_plan = true;
                        if (tsvr->planMomaParallel(local_start, local_goal, local_v).first)
                        {
                            tsvr->is_safe = true;
                            while (ros::Time::now() - plan_start_time < ros::Duration(tsvr->planning_budget)) {;}
                            // planning_budget = (ros::Time::now() - start_time).toSec();
                            double wait_time = (ros::Time::now() - plan_start_time).toSec() - tsvr->planning_budget;
                            PRINT_GREEN("Wait time = "<<wait_time<<" s.");
                            tsvr->mpc->setTraj(tsvr->end_traj, std::max(0.0, wait_time));
                            tsvr->begin_time = ros::Time::now();
                            tsvr->last_replan_time = ros::Time::now();
                            tsvr->has_traj = true;
                        }
                        tsvr->in_plan = false;
                    }
                }
            }

            int cmd_mm_num = 1000.0 / tsvr->mpc->ctrl_freq;
            std::chrono::milliseconds dura(max(cmd_mm_num - (int)((ros::Time::now() - start_time).toSec() * 1000), 1));
            std::this_thread::sleep_for(dura);
        }
        return;
    }

    void Planner::rcvStateCallBack(const fake_moma::MomaStatePtr msg)
    {
        has_odom = true;
        now_state[0] = msg->chassis_odom.pose.pose.position.x;
        now_state[1] = msg->chassis_odom.pose.pose.position.y;
        double ori_z = msg->chassis_odom.pose.pose.orientation.z;
        double ori_w = msg->chassis_odom.pose.pose.orientation.w;
        now_state[2] = atan2(2.0*ori_z*ori_w, 
                             2.0*ori_w*ori_w-1.0);
        now_dstate[0] = msg->chassis_odom.twist.twist.linear.x;
        now_dstate[1] = msg->chassis_odom.twist.twist.angular.z;
        // now_dstate[0] = 0.0;
        for (size_t i=0; i<moma_param.dof_num; i++)
        {
            now_state[3+i] = msg->arm_odom[i].twist.twist.linear.x;
            now_dstate[3+i] = msg->arm_odom[i].twist.twist.angular.z;
        }
        return;
    }

    std::vector<Eigen::VectorXd> Planner::planOmpls(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Eigen::VectorXd& start_v) const
    {
        // front end
        std::vector<Eigen::VectorXd> path;
        // ompls
        PRINT_GREEN("\n[Planner] Begin OMPL planning...");
        ompl::msg::setLogLevel(ompl::msg::LOG_NONE);
        ompl_planner->planRRT(start, end, path);

        return path;
    }

    bool Planner::staMomaParallel(const Eigen::VectorXd& start, 
                                  const Eigen::VectorXd& end, 
                                  const Eigen::VectorXd& start_v)
    {
        front_path.clear();
        list<GraphNode::Ptr> graph;
        vector<vector<Eigen::Vector3d>> raw_paths, filtered_paths;
        Eigen::Vector3d topo_start(start(0), start(1), 0.0);
        Eigen::Vector3d topo_end(end(0), end(1), 0.0);
        std::vector<Eigen::Vector3d> start_pts, end_pts;
        start_pts.push_back(topo_start);
        end_pts.push_back(topo_end);
        topo_select_paths.clear();
        topo_prm->findTopoPaths(topo_start, topo_end, start_pts, end_pts, graph,
                               raw_paths, filtered_paths, topo_select_paths);

        auto jps_result = graph_search->plan2dJPS(start.head(2), end.head(2), moma_param.chassis_colli_radius+0.1);
        if (!jps_result.empty())
        {
            //! xulong delete direct traj
            // double jps_length = 0.0;
            // for (size_t i = 0; i < jps_result.size() - 1; ++i)
            //     jps_length += (jps_result[i] - jps_result[i+1]).norm();
            // if ( jps_length - (start-end).head(2).norm() < 0.1)
            //     return false;

            std::vector<Eigen::Vector3d> jps3_res;
            for (size_t i = 0; i < jps_result.size(); ++i)
                jps3_res.push_back(Eigen::Vector3d(jps_result[i].x(), jps_result[i].y(), 0.0));
            topo_select_paths.push_back(jps3_res);
        }
        
        if (topo_select_paths.empty()){
            PRINT_RED("[Planner] TOPO failed");
            return false;
        }

        while (topo_select_paths.size() > 3)
            topo_select_paths.pop_back();

        ros::Time t1 = ros::Time::now();
        vector<thread> optimize_threads;
        parallel_ends.clear();
        parallel_ends.resize(topo_select_paths.size());

        for (size_t i = 0; i < topo_select_paths.size(); ++i)
            optimize_threads.emplace_back(&Planner::optMomaOnce, this, topo_select_paths[i], i, start, end, start_v);
        for (size_t i = 0; i < topo_select_paths.size(); ++i) optimize_threads[i].join();

        bool succ = false;
        for (size_t i = 0; i < parallel_ends.size(); ++i)
        {
            if (parallel_ends[i].is_init)
            {
                succ = true;
                break;
            }
        }

        if (succ)
        {
            sort(parallel_ends.begin(), parallel_ends.end(),
            [&](MomaTraj& tj1, MomaTraj& tj2) 
            {   
                if (tj1.is_init && tj2.is_init)
                    return tj1.getTotalDuration() < tj2.getTotalDuration(); 
                else if (tj1.is_init)
                    return true;
                else
                    return false;
            });
            end_traj = parallel_ends[0];
        }
        else
        {
            topo_select_paths.clear();
            topo_prm->findTopoPaths(topo_start, topo_end, start_pts, end_pts, graph,
                                    raw_paths, filtered_paths, topo_select_paths, true);
                                    
            while (topo_select_paths.size() > 8)
                topo_select_paths.pop_back();

            if (!topo_select_paths.empty())
            {
                ros::Time t1 = ros::Time::now();
                vector<thread> optimize_threads;
                parallel_ends.clear();
                parallel_ends.resize(topo_select_paths.size());

                for (size_t i = 0; i < topo_select_paths.size(); ++i) 
                    optimize_threads.emplace_back(&Planner::optMomaOnce, this, topo_select_paths[i], i, start, end, start_v);
                for (size_t i = 0; i < topo_select_paths.size(); ++i) optimize_threads[i].join();
                for (size_t i = 0; i < parallel_ends.size(); ++i)
                {
                    if (parallel_ends[i].is_init)
                    {
                        succ = true;
                        break;
                    }
                }

                if (succ)
                {
                    sort(parallel_ends.begin(), parallel_ends.end(),
                    [&](MomaTraj& tj1, MomaTraj& tj2) 
                    { 
                        if (tj1.is_init && tj2.is_init)
                            return tj1.getTotalDuration() < tj2.getTotalDuration(); 
                        else
                            return false;
                    });
                    end_traj = parallel_ends[0];
                }
            }
        }

        // fallback to ompl
        if (!succ)
        {
            PRINT_RED("[Planner] TOPO failed, fallback to ompl");
            auto ompl_path = planOmpls(start, end, start_v);
            if (ompl_path.empty()) {
                PRINT_RED("[Planner] OMPL failed");
                return false;
            }
    
            Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(10, 2);
            Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(10, 2);
            Eigen::VectorXd v = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            boundary_vel.col(0) = v;
    
            if (!this->traj_opters[0]->optimizeTraj(ompl_path, boundary_vel, boundary_acc)
                || !this->traj_opters[0]->printConstraintsSituations(traj_opters[0]->getTraj())
                || !this->traj_opters[0]->getTraj().is_init
            ){
                return false;
            }
    
            end_traj = this->traj_opters[0]->getTraj();
            succ = true;
        }

        if (succ)
        {
            RowMatrixXd sampled_traj = end_traj.sampleArcPoints(PNUM);
            // RowMatrixXd sampled_traj = end_traj.sampleTimePoints(PNUM);
            front_path.clear();
            if (sampled_traj.rows() < 2)
            {
                succ = false;
                PRINT_RED("[Planner] Sampled traj less than 2, failed");
                return succ;
            }

            for (size_t i=0; i<sampled_traj.rows(); i++)
                front_path.push_back(sampled_traj.row(i));
            for (size_t i=0; i<front_path.size(); i++)
            {
                if (grid_map->isWholeBodyCollision(front_path[i]))
                {
                    succ = false;
                    break;
                }
            }
        }

        return succ;
    }

    bool Planner::staRemani(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Eigen::VectorXd& start_v)
    {
        // front end
        front_path.clear();
        ros::Time start_time = ros::Time::now();

        // JPS
        auto jps_result = graph_search->plan2dJPS(start.head(2), end.head(2), moma_param.chassis_colli_radius+0.1);
        if (jps_result.empty())
            return false;

        //! xulong delete direct traj
        double jps_length = 0.0;
        for (size_t i = 0; i < jps_result.size() - 1; ++i)
            jps_length += (jps_result[i] - jps_result[i+1]).norm();
        if ( jps_length - (start-end).head(2).norm() < 0.1)
            return false;
        
        // MCRRTs
        auto dense_result = graph_search->getDensePath(jps_result, 1.414, start(2), end(2), 
                                                       moma_param.max_v, moma_param.max_w);

        std::pair<int, int> layer(-1, -1);
        if (!mcrrts->plan(start, end, dense_result, front_path))
            return false;
        
        if (front_path.empty())
            return false;

        Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(3+moma_param.dof_num, 2);
        Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(3+moma_param.dof_num, 2);
        boundary_vel.col(0) = start_v;
        start_time = ros::Time::now();
        if (!traj_opter->optimizeTraj(front_path, boundary_vel, boundary_acc)
            || !traj_opter->checkFeasible(traj_opter->getTraj()) 
            )
            return false;
        else
        {
            // PRINT_YELLOW("[Planner] TrajOpt time: " << (ros::Time::now() - start_time).toSec() * 1000.0 << " ms");

            end_traj = traj_opter->getTraj();
            // vis_whole_path(end_pub);
            RowMatrixXd sampled_traj = end_traj.sampleTimePoints(PNUM);
            front_path.clear();
            for (size_t i=0; i<sampled_traj.rows(); i++)
                front_path.push_back(sampled_traj.row(i));
            bool flag = true;
            for (size_t i=0; i<front_path.size(); i++)
            {
                if (grid_map->isWholeBodyCollision(front_path[i]))
                {
                    flag = false;
                    break;
                }
            }
            return flag;
        }

        return true;
    }

    std::pair<bool,float> Planner::planMomaParallel(const Eigen::VectorXd& start, 
                                   const Eigen::VectorXd& end, 
                                   const Eigen::VectorXd& start_v)
    {
        bool succ = false;                              // flag for successful optimization
        bool _critical = false;                         // whether to use critical map
        std::vector<Eigen::Vector4d> colors;            // colors of prm paths
        std::vector<std::pair<bool, MomaTraj>> results; // storing results of optimization
        std::vector<std::vector<Eigen::VectorXd>> front_paths;       // storing pre-optimized paths
        MomaTraj ret_traj;
        do {
            // start first trial with non-critical
            topo_select_paths.clear();
            {
                list<GraphNode::Ptr> graph;
                vector<vector<Eigen::Vector3d>> raw_paths, filtered_paths;
                Eigen::Vector3d topo_start(start(0), start(1), 0.0);
                Eigen::Vector3d topo_end(end(0), end(1), 0.0);
                std::vector<Eigen::Vector3d> start_pts, end_pts;
                start_pts.push_back(topo_start);
                end_pts.push_back(topo_end);
                topo_prm->findTopoPaths(topo_start, topo_end, start_pts, end_pts, graph,
                                       raw_paths, filtered_paths, topo_select_paths, _critical);
                if (!_critical) {
                    auto jps_result = graph_search->plan2dJPS(start.head(2), end.head(2), moma_param.chassis_colli_radius+0.1);
                    if (!jps_result.empty())
                    {
                        std::vector<Eigen::Vector3d> jps3_res;
                        for (size_t i = 0; i < jps_result.size(); ++i)
                            jps3_res.push_back(Eigen::Vector3d(jps_result[i].x(), jps_result[i].y(), 0.0));
                        topo_select_paths.push_back(jps3_res);
                    }
                }
                
                if (topo_select_paths.empty())
                    return std::make_pair(false, 0.0);
        
                if(topo_select_paths.size() > traj_opters.size()) throw std::runtime_error("Too many paths to optimize");
                
                colors = vis_prm_paths();
        
                PRINT_GREEN("[MOMA] Start optimization");
                // std::promise<MomaTraj> promise_traj;
                // auto future_traj = promise_traj.get_future();
                // std::atomic_flag ready_flag = ATOMIC_FLAG_INIT;
        
                // Eigen::VectorXd test;
                // test.resize(10);
                // test.setZero();
        
                // bool coll = grid_map->isWholeBodyCollision(test);
                // if (coll) std::cout << "Collision!" << std::endl;
        
                results.resize(topo_select_paths.size());
                front_paths.resize(topo_select_paths.size());
                for (auto res : results) { res.first = false; }
                
                std::promise<bool> promise_succ;
                auto future_succ = promise_succ.get_future();

                boost::mutex mtx;
                boost::condition_variable cv_first; // for the first successful thread
                boost::condition_variable cv_all;   // for all threads to finish
                std::atomic_flag rdy_flag = ATOMIC_FLAG_INIT;
                std::atomic<int> completed_threads{0};
                auto worker = [this, &results, &front_paths, &mtx, &promise_succ, &cv_first, &cv_all, &completed_threads, &rdy_flag] (
                    int idx, 
                    std::vector<Eigen::Vector3d>& topo_path, 
                    const Eigen::VectorXd& start, 
                    const Eigen::VectorXd& end, 
                    const Eigen::VectorXd& start_v)
                {
                    ros::Time start_time = ros::Time::now();
                    std::vector<Eigen::Vector2d> in_path;
                    for(auto &wp : topo_path)
                        in_path.push_back(wp.head(2));
                    auto dense_result = graph_search->getDensePath(in_path, 1.414, start(2), end(2), 
                        moma_param.max_v, moma_param.max_w);
                        
                    boost::this_thread::interruption_point();

                    bool _succ = false;
                    do
                    {
                        if (!mc_rrtsers[idx]->plan(start, end, dense_result, front_paths[idx]) || front_paths[idx].empty())
                        {
                            _succ = false;
                            PRINT_RED("MCRRT fail.");
                            break;
                        }
                        boost::this_thread::interruption_point();
                        Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(10, 2);
                        Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(10, 2);
                        boundary_vel.col(0) = start_v;
    
                        _succ = 
                            this->traj_opters[idx]->optimizeTraj(front_paths[idx], boundary_vel, boundary_acc)
                            && this->traj_opters[idx]->printConstraintsSituations(traj_opters[idx]->getTraj())
                            && this->traj_opters[idx]->getTraj().is_init;
                        
                    } while(false);
                    
                    results[idx].first = _succ;
                    if(_succ) results[idx].second = this->traj_opters[idx]->getTraj();
                    // results[idx] = std::make_pair(_succ, this->traj_opters[idx]->getTraj());
                    
                    if (_succ && !rdy_flag.test_and_set()) {
                        // boost::lock_guard<boost::mutex> lock(mtx);
                        // if (!first_success.exchange(true)) {  // Atomic check-and-set
                        //     cv_first.notify_all();  // Notify all waiters
                        // }
                        promise_succ.set_value(true);
                    }
                    // MomaTraj traj = this->traj_opters[idx]->getTraj();
                    // PRINT_RED("[Thread] Successful optimization with duration: " << traj.getTotalDuration() << std::endl);
                    // promise_traj.set_value(traj);
                    // promise_traj.set_value(this->traj_opters[idx]->getTraj());
                    // optim_time = (ros::Time::now() - start_time).toSec() * 1000.0;
                    // try {
                    // } catch (boost::thread_interrupted&) {
        
                    // } catch (...) {
        
                    // }
                    ros::Time end_time = ros::Time::now();
                    PRINT_GREEN("[Thread] ID: " << idx << " Optimization time: " << (end_time - start_time).toSec() * 1000.0 << " ms");
                    {
                        PRINT_GREEN("[Threads] Thread " << completed_threads+1 << " / " << topo_select_paths.size() << " completed");
                        boost::lock_guard<boost::mutex> lock(mtx);
                        if(++completed_threads == topo_select_paths.size()){
                            if(!rdy_flag.test_and_set()) promise_succ.set_value(true);
                            PRINT_GREEN("[Threads] All threads completed");
                            cv_all.notify_all();
                        }
                    }
                    
                };
                
                PRINT_YELLOW("[Threads] Starting " << topo_select_paths.size() << " threads");
                boost::thread_group threads;
                for (size_t i = 0; i < topo_select_paths.size(); ++i)
                    threads.create_thread(std::bind(
                        worker, i, topo_select_paths[i], start, end, start_v
                    ));
                // bool optSucc = future_traj.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
                // threads.interrupt_all();
                // threads.join_all();
                
                // === wait for first successful thread ===
                future_succ.wait();
                
                bool timeout; // indicate early termination of threads
                {
                    boost::unique_lock<boost::mutex> lock(mtx);
                    
                    PRINT_RED("[Threads] Waiting for First Successful Optimization");
                    // cv_first.wait(lock, [&]() { return first_success.load(); } );
                    // while (!first_success) {
                    //     cv_first.wait(lock);
                
                    // === wait additional 100ms for other threads to finish ===
                    while (completed_threads < topo_select_paths.size()){
                        PRINT_RED("[Threads] Waiting for All Threads");
                        if(timeout = 
                            boost::cv_status::timeout == cv_all.wait_for(lock, boost::chrono::milliseconds(1000))
                        )
                            break;
                    }
                }
                threads.interrupt_all();
                threads.join_all();
                
                if(timeout) {
                    PRINT_YELLOW("[Threads] Timeout in waiting threads");
                    PRINT_YELLOW("[Threads] " << completed_threads << " / " << topo_select_paths.size() << " completed");
                }
                
                for(auto &res : results) succ = succ || res.first;
            }
            //! xulong for topay
            // _critical = true;
            //! xulong for topay
            if (!succ) PRINT_YELLOW("Non-Critical optimization failed, try critical optimization");
        } while(!succ && !_critical && (_critical = true));
        
        // ros::Time t1 = ros::Time::now();
        // vector<thread> optimize_threads;
        // parallel_ends.clear();
        // parallel_ends.resize(topo_select_paths.size());
        // for (size_t i = 0; i < topo_select_paths.size(); ++i) 
        //     optimize_threads.emplace_back(&Planner::optMomaOnce, this, topo_select_paths[i], i, start, end, start_v);
        // for (size_t i = 0; i < topo_select_paths.size(); ++i) optimize_threads[i].join();
        // optim_time = (ros::Time::now() - t1).toSec() * 1000.0;
        // PRINT_GREEN("[MOMA] End optimization");
        bool ompl_succ = false; // flag for OMPL optimization success
        //! xulong for Topay:
        // if (!succ) {
        //     do {
        //         auto ompl_path = planOmpls(start, end, start_v);
        //         if (ompl_path.empty()) break; // OMPL failed
        //         Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(10, 2);
        //         Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(10, 2);
        //         boundary_vel.col(0) = start_v;
        //         if (!this->traj_opters[0]->optimizeTraj(ompl_path, boundary_vel, boundary_acc)
        //             || !this->traj_opters[0]->printConstraintsSituations(traj_opters[0]->getTraj())
        //             || !this->traj_opters[0]->getTraj().is_init
        //         ) break; // OMPL optimization failed
        //         end_traj = traj_opters[0]->getTraj();
        //         succ = true;
        //         results.resize(1);
        //         results[0].first = true;
        //         results[0].second = end_traj;
        //         ompl_succ = true;
        //     } while (false);
        // }
        //! xulong for Topay.

        if (succ)
        {
            PRINT_GREEN("[planner]: First successful optimization!");

            int shortest_idx = -1;
            int idx = -1;
            for(auto &res : results) {
                idx++;
                if(!res.first) continue;
                if(shortest_idx == -1) shortest_idx = idx;
                double traj_duration = res.second.getTotalDuration();
                if(traj_duration < results[shortest_idx].second.getTotalDuration())
                    shortest_idx = idx;
            }
            PRINT_YELLOW("Shortest path index: " << shortest_idx+1);
            topay_traj = ret_traj = end_traj = results[shortest_idx].second;
            // vis_whole_path(end_pub);
            // vis_prm = topo_select_paths[shortest_idx];
            // vis_prm_color = colors[shortest_idx];

            PRINT_YELLOW("Publishing MeshTraj");
            // auto mesh_traj = toMeshMsg(end_traj);
            // mesh_traj_pub.publish(mesh_traj);
            last_replan_time = ros::Time::now();

            // PRINT_YELLOW("Publishing PlotTraj");
            // planner::PlotTraj msg = toPlotMsg(end_traj);
            // plot_traj_pub.publish(msg);

            // vis_ee_traj(end_traj, plot_traj_ee, {226.0/255.0, 145.0/255.0, 53.0/255.0, 0.5});

            // vis_path_mesh(topay_traj, 8, 
            //     trad_pub,
            //     {
            //         230.0/255.0,
            //         111.0/255.0,
            //         81.0/255,
            //         0.2
            //     },
            //     4322
            // );

            // vis_ee_traj(topay_traj, plot_trad_traj_ee, 
            //     {
            //         230.0/255.0,
            //         111.0/255.0,
            //         81.0/255,
            //         0.2
            //     }
            // );

            int res_idx = -1;
            for (auto &res : results) {
                res_idx++;
                if(!res.first) continue;
                
                bool shortest = res_idx == shortest_idx;
                float r,g,b,a;
                r = shortest? 1.0 : 0.5;
                g = shortest? 0.0 : 0.5;
                b = shortest? 0.0 : 0.5;
                a = shortest? 1.0 : 0.2;
                int nsample = shortest ? 16 : 16;
                MomaTraj traj = res.second;
                
                if(!ompl_succ) {
                    // vis_isAvailable[res_idx] = true;
                    // vis_front_paths[res_idx] = front_paths[res_idx];
                    // vis_opt_paths[res_idx]   = traj;

                    // vis_path_mesh(traj, nsample, 
                    //     opt_traj_pub_list[res_idx], 
                    //     {r, g, b, a}, 
                    //     res_idx*1000 + 800);
                    
                    // vis_path_mesh(sparsifyPath(front_paths[res_idx], 0.5), 
                    //     front_traj_pub_list[res_idx], 
                    //     {0.0, 0.0, 1.0, 0.2}, 
                    //     res_idx*1000 + 900);
                }
            }
            if(ompl_succ) {
                // for (size_t i = 0; i < vis_isAvailable.size(); ++i) vis_isAvailable[i] = false;
            }
            
            // end_traj = future_traj.get();
            // last_replan_time = ros::Time::now();
            // vis_whole_path(end_pub);
        }

        return std::make_pair(succ, 0.0);
    }

    /*
    std::pair<float, float> Planner::planDDIM(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Eigen::VectorXd& start_v, int path_num) 
    {
        front_path.clear();

        planner::Plan srv;
        srv.request.path_num = path_num;
        srv.request.start = std::vector<float>(start.data(), start.data() + start.size());
        srv.request.goal = std::vector<float>(end.data(), end.data() + end.size());
        
        std::vector<float> serialized_boxes;
        serialized_boxes.resize(grid_map->getBoxes().size() * 8);
        
        for (size_t i = 0; i < grid_map->getBoxes().size(); ++i)
        for (size_t j = 0; j < 8; ++j)
            serialized_boxes[i*8+j] = grid_map->getBoxes()[i][j];
        srv.request.boxes = serialized_boxes;
        
        pcl::PointCloud<pcl::PointXYZ> cloud;
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(cloud, cloud_msg);
        cloud_msg = grid_map->getSurround();
        srv.request.input_cloud = cloud_msg;

        // a collection of paths, number specified by function parameter 'path_num'
        
        float sample_time = 0.0;
        auto sample_start_time = ros::Time::now();
        if(!plannerClient.call(srv) || !srv.response.success)
        {
            PRINT_RED("Failed plan by diffusion");
            return std::make_pair(-1.0, 0.0);
        }
        sample_time = srv.response.sample_time;
        
        std::vector<std::vector<Eigen::VectorXd>> all_paths; 
        int path_length = srv.response.path_length;
        for (int p = 0; p < path_num; p++){
            std::vector<Eigen::VectorXd> _path;
            for (int i = 0; i < path_length; i++) 
            {
                Eigen::VectorXd wp = Eigen::VectorXd::Zero(3+moma_param.dof_num);
                int offset = 10 * i + p * 640;
                for(int j = 0; j < 3+moma_param.dof_num; j++) 
                {
                    wp(j) = srv.response.path[j+offset];
                }
                _path.push_back(wp);
            }
            all_paths.push_back(_path);
        }
        // front_path = all_paths[0];
        // vis_path(front_path, front_pub);
        for(int i = 0; i < ddim_path_num; i++) {
            // ompl_planner->reduceVertices(all_paths[i]);
            vis_path(all_paths[i], ddim_pub_list[i], {0.5, 0.5, 0.5, 0.15});
        }


        // PRINT_GREEN("[DDIM] Start optimization");
        // vector<thread> optThreads;
        // parallel_ends.clear();
        // parallel_ends.resize(all_paths.size());
        // ros::Time start_time = ros::Time::now();
        // for (size_t i = 0; i < all_paths.size(); ++i) 
        //     optThreads.emplace_back(&Planner::optDenseOnce, this, all_paths[i], i, start, end, start_v);
        // for (size_t i = 0; i < all_paths.size(); ++i) optThreads[i].join();
        // float optim_time = (ros::Time::now() - start_time).toSec() * 1000.0;
        // PRINT_GREEN("[DDIM] End optimization");

        std::promise<MomaTraj> promise_traj;
        auto future_traj = promise_traj.get_future();
        std::atomic_flag ready_flag = ATOMIC_FLAG_INIT;

        auto worker = [this, &promise_traj, &ready_flag] (
            int idx, 
            std::vector<Eigen::VectorXd>& path, 
            const Eigen::VectorXd& start, 
            const Eigen::VectorXd& end, 
            const Eigen::VectorXd& start_v)
        {
            try {
                boost::this_thread::interruption_point();
                Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(10, 2);
                Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(10, 2);
                boundary_vel.col(0) = start_v;
                if (!this->traj_opters[idx]->optimizeTrajNN(path, boundary_vel, boundary_acc)
                    || !this->traj_opters[idx]->printConstraintsSituations(traj_opters[idx]->getTraj())
                    || !this->traj_opters[idx]->getTraj().is_init
                    || ready_flag.test_and_set()) 
                    return;
                // MomaTraj traj = this->traj_opters[idx]->getTraj();
                // PRINT_RED("[Thread] Successful optimization with duration: " << traj.getTotalDuration() << std::endl);
                // promise_traj.set_value(traj);
                promise_traj.set_value(this->traj_opters[idx]->getTraj());
            } catch (boost::thread_interrupted&) {

            } catch (...) {

            }
        };

        
        ros::Time start_time = ros::Time::now();
        boost::thread_group threads;
        for (size_t i = 0; i < all_paths.size(); ++i)
            threads.create_thread(std::bind(worker,
                i, all_paths[i], start, end, start_v
            ));
        bool optSucc = future_traj.wait_for (std::chrono::seconds(2)) == std::future_status::ready;
        ros::Time first_succ_time = ros::Time::now();
        threads.interrupt_all();
        threads.join_all();
        float cleanup_time = (ros::Time::now() - first_succ_time).toSec() * 1000.0;
        float optim_time = (ros::Time::now() - start_time).toSec() * 1000.0;
        PRINT_YELLOW("[DDIM] Cleanup time: " << cleanup_time << " ms" << std::endl);
        // ros::Time start_time = ros::Time::now();
        // std::vector<boost::future<MomaTraj>> futures;
        // int idx = 0;
        // for (auto path : all_paths) {
        //     futures.push_back(boost::async(
        //         [this, idx, path, start, end, start_v] () {
        //             Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(10, 2);
        //             Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(10, 2);
        //             boundary_vel.col(0) = start_v;
        //             if (!this->traj_opters[idx]->optimizeTraj(path, boundary_vel, boundary_acc)
        //                 || !this->traj_opters[idx]->printConstraintsSituations(traj_opters[idx]->getTraj()) 
        //                 )
        //                 throw std::runtime_error("Failed optimization");
        //             MomaTraj traj = this->traj_opters[idx]->getTraj();
        //             return traj;
        //         }
        //     ));
        //     idx++;
        // }
        // std::vector<MomaTraj> ready_trajs;

        // bool optSucc = false;
        // float optim_time;
        // while (!futures.empty() && !optSucc) {
        //     auto it = boost::wait_for_any(futures.begin(), futures.end());
        //     try {
        //         MomaTraj traj = it->get();
        //         if (!traj.is_init) throw std::runtime_error("Failed optimization");
        //         // PRINT_RED("[Thread] Successful optimization");
        //         // PRINT_RED("Duration: " << traj.getTotalDuration() << std::endl);
        //         ready_trajs.push_back(traj);
        //         optSucc = true;
        //         optim_time = (ros::Time::now() - start_time).toSec() * 1000.0;
        //     } catch (const std::exception& e) {
        //         // PRINT_RED("[Thread] Failed optimization");
        //         futures.erase(it);
        //     }

        // }

        // boost::wait_for_all(futures.begin(), futures.end());
        
        if (optSucc) {

            PRINT_RED("[planner]: Successful critical optimization!");
            // double duration = std::numeric_limits<double>::max();
            // int idx = -1;
            // for (int i = 0; i < ready_trajs.size(); ++i) {
            //     auto traj = ready_trajs[i];
            //     if (traj.is_init && traj.getTotalDuration() < duration) {
            //         duration = traj.getTotalDuration();
            //         idx = i;
            //     }
            // }
            // end_traj = ready_trajs[idx];

            end_traj = future_traj.get();
            PRINT_RED("Duration: " << end_traj.getTotalDuration() << std::endl);

            last_replan_time = ros::Time::now();
            // front_path = all_paths[idx];
            // vis_path(front_path, front_pub);
            
            // vis_path_mesh(end_traj, 24, 
            //     end_pub,
            //     {226.0/255.0, 145.0/255.0, 53.0/255.0, 0.5},
            //     4322
            // );
            // vis_whole_path(end_pub);
            // sort(parallel_ends.begin(), parallel_ends.end(),
            // [&](MomaTraj& tj1, MomaTraj& tj2) 
            // { 
            //     if (tj1.is_init && tj2.is_init)
            //         return tj1.getTotalDuration() < tj2.getTotalDuration(); 
            //     else
            //         return false;
            // });
            // end_traj = parallel_ends[0];
        }

        if (!optSucc) optim_time = -1.0;
        return std::make_pair(optim_time, sample_time);
    }
    */

    
    std::pair<float, float> Planner::planDDIM(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Eigen::VectorXd& start_v, int path_num) 
    {
        front_path.clear();

        auto prep_start_time = ros::Time::now();

        planner::Plan srv;
        srv.request.path_num = path_num;
        srv.request.start = std::vector<float>(start.data(), start.data() + start.size());
        srv.request.goal = std::vector<float>(end.data(), end.data() + end.size());
        srv.request.seq = _sequence;
        
        std::vector<float> serialized_boxes;
        serialized_boxes.resize(grid_map->getBoxes().size() * 8);
        
        for (size_t i = 0; i < grid_map->getBoxes().size(); ++i)
        for (size_t j = 0; j < 8; ++j)
            serialized_boxes[i*8+j] = grid_map->getBoxes()[i][j];
        srv.request.boxes = serialized_boxes;
        
        pcl::PointCloud<pcl::PointXYZ> cloud;
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(cloud, cloud_msg);
        cloud_msg = grid_map->getSurround();
        srv.request.input_cloud = cloud_msg;

        float prep_time = (ros::Time::now() - prep_start_time).toSec() * 1000;
        std::cout << "prep time: " << prep_time << std::endl;


        // a collection of paths, number specified by function parameter 'path_num'
        
        auto call_start_time = ros::Time::now();
        auto sample_start_time = ros::Time::now();
        if(!plannerClient.call(srv) || !srv.response.success)
        {
            PRINT_RED("Failed plan by diffusion");
            return std::make_pair(-1.0, 0.0);
        }
        float sample_time = srv.response.sample_time;
        float call_time = (ros::Time::now() - call_start_time).toSec() * 1000;
        PRINT_GREEN("CALL time " << call_time);
        
        std::vector<std::vector<Eigen::VectorXd>> all_paths; 
        int path_length = srv.response.path_length;
        for (int p = 0; p < path_num; p++){
            std::vector<Eigen::VectorXd> _path;
            for (int i = 0; i < path_length; i++) 
            {
                Eigen::VectorXd wp = Eigen::VectorXd::Zero(3+moma_param.dof_num);
                int offset = 10 * i + p * 640;
                for(int j = 0; j < 3+moma_param.dof_num; j++) 
                {
                    wp(j) = srv.response.path[j+offset];
                }
                _path.push_back(wp);
            }
            all_paths.push_back(_path);
        }
        for(int i = 0; i < ddim_path_num; i++) {
            vis_path(all_paths[i], ddim_pub_list[i], {0.5, 0.5, 0.5, 0.1});
            // ompl_planner->reduceVertices(all_paths[i]);
        }

        std::vector<std::vector<Eigen::VectorXd>> down_sampled_paths; 
        {
            for (auto path : all_paths)
            {
                std::vector<Eigen::VectorXd> _sampled1, _sampled2;
                _sampled1.push_back(path[0]);
                _sampled2.push_back(path[0]);
                for (size_t i = 1; i < path.size(); i++) {
                    Eigen::VectorXd curr, prev1, prev2;
                    prev1 = _sampled1.back();
                    prev2 = _sampled2.back();
                    curr = path[i];
                    double arc_len1 = (curr.head(2) - prev1.head(2)).norm();
                    double arc_len2 = (curr.head(2) - prev2.head(2)).norm();
                    if (arc_len1 > 0.5) _sampled1.push_back(curr);
                    if (arc_len2 > 0.2) _sampled2.push_back(curr);
                }
                down_sampled_paths.push_back(_sampled1);
                down_sampled_paths.push_back(_sampled2);
            }
        }

        std::vector<std::pair<bool, MomaTraj>> results;
        results.resize(down_sampled_paths.size());
        for (auto res : results) { res.first = false; }

        std::promise<bool> promise_succ;
        auto future_succ = promise_succ.get_future();

        boost::mutex mtx;
        boost::condition_variable cv_first;
        boost::condition_variable cv_all;
        std::atomic_flag rdy_flag = ATOMIC_FLAG_INIT;
        std::atomic<int> completed_threads(0);
        int sample_num = down_sampled_paths.size();
        
        auto worker = [this, sample_num, &results, &mtx, &promise_succ, &cv_first, &cv_all, &completed_threads, &rdy_flag] (
            int idx, 
            std::vector<Eigen::VectorXd>& path, 
            const Eigen::VectorXd& start, 
            const Eigen::VectorXd& end, 
            const Eigen::VectorXd& start_v)
        {
            try {

                ros::Time begin = ros::Time::now();
                Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(10, 2);
                Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(10, 2);
                boundary_vel.col(0) = start_v;
                boost::this_thread::interruption_point();
    
                bool _succ = 
                    this->traj_opters[idx]->optimizeTrajNN(path, boundary_vel, boundary_acc)
                    && this->traj_opters[idx]->printConstraintsSituations(traj_opters[idx]->getTraj())
                    && this->traj_opters[idx]->getTraj().is_init;
                
                results[idx].first = _succ;
                if(_succ) results[idx].second = this->traj_opters[idx]->getTraj();
                // // if(_succ) PRINT_GREEN("[Threads] Succ time" << (ros::Time::now() - begin).toSec() * 1000);
    
                if (_succ && !rdy_flag.test_and_set()) {
                    promise_succ.set_value(true);
                    PRINT_GREEN("[Threads] Succ time" << (ros::Time::now() - begin).toSec() * 1000);
                }
    
                {
                    boost::lock_guard<boost::mutex> lock(mtx);
                    if(++completed_threads == sample_num){
                        if(!rdy_flag.test_and_set()) promise_succ.set_value(true);
                        cv_all.notify_all();
                    }
                }
            } catch (boost::thread_interrupted&) {

            } catch (...) {

            }

        };

        
        ros::Time start_time = ros::Time::now();
        boost::thread_group threads;
        for (size_t i = 0; i < down_sampled_paths.size(); ++i)
            threads.create_thread(std::bind(worker,
                i, down_sampled_paths[i], start, end, start_v
            ));

        
        future_succ.wait();
        int dur = (ros::Time::now() - start_time).toSec() * 1000;
        PRINT_GREEN("[Threads] First Succ " << dur);
        

        ros::Time wait_time = ros::Time::now();
        bool timeout;
        {
            boost::unique_lock<boost::mutex> lock(mtx);
                    
            // === wait additional 100ms for other threads to finish ===
            while (completed_threads < down_sampled_paths.size()){
                PRINT_RED("[Threads] Waiting for All Threads");
                if(timeout = 
                    boost::cv_status::timeout == cv_all.wait_for(lock, boost::chrono::milliseconds(20))
                )
                    break;
            }
        }
        threads.interrupt_all();
        threads.join_all();

        int sec = (ros::Time::now() - wait_time).toSec() * 1000;
        PRINT_GREEN("[Threads] All terminate " << sec);


        bool optSucc = false;
        for(auto &res : results) optSucc = optSucc || res.first;

        float optim_time = (ros::Time::now() - start_time).toSec() * 1000.0;
        if (optSucc) {

            int shortest_idx = -1;
            int idx = -1;
            for(auto &res : results) {
                idx++;
                if(!res.first) continue;
                if(shortest_idx == -1) shortest_idx = idx;
                double traj_duration = res.second.getTotalDuration();
                if(traj_duration < results[shortest_idx].second.getTotalDuration())
                    shortest_idx = idx;
            }
            ddim_traj = end_traj = results[shortest_idx].second;
            PRINT_GREEN("[DDIM Opt] Duration: " << end_traj.getTotalDuration() <<"s");

            last_replan_time = ros::Time::now();

            // end_traj_available = true;
            
            // vis_path_mesh(ddim_traj, 8, 
            //     end_pub,
            //     {
            //         38.0/255.0,
            //         70.0/255.0,
            //         83.0/255,
            //         0.2
            //     },
            //     4322
            // );

            // vis_ee_traj(ddim_traj, plot_traj_ee, 
            //     {
            //         38.0/255.0,
            //         70.0/255.0,
            //         83.0/255,
            //         0.2
            //     }
            // );

            // vis_whole_path(end_pub);

            // mesh
            planner::MeshPCD mpcd;
            mpcd.seq = _sequence++;
            mpcd.pid = shortest_idx % 4;
            mpcd.meshTraj = toMeshMsg(end_traj);
            mesh_traj_pub.publish(mpcd);
        }

        if (!optSucc) optim_time = -1.0;
        return std::make_pair(optim_time, sample_time);
    } 

    bool Planner::optMomaOnce(const std::vector<Eigen::Vector3d>& topo_path, int idx, 
                            const Eigen::VectorXd& start, const Eigen::VectorXd& end,
                            const Eigen::VectorXd& start_v)
    {
        std::vector<Eigen::Vector2d> in_path;
        for (size_t i = 0; i < topo_path.size(); ++i)
            in_path.push_back(topo_path[i].head(2));
        auto dense_result = graph_search->getDensePath(in_path, 1.414, start(2), end(2), 
                                                       moma_param.max_v, moma_param.max_w);
        std::vector<Eigen::VectorXd> full_path;
        if (!mc_rrtsers[idx]->plan(start, end, dense_result, full_path))
        {
            PRINT_RED("MCRRTs fail, idx = "<<idx);
            return false;
        }
        if (full_path.empty())
            return false;
        Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(3+moma_param.dof_num, 2);
        Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(3+moma_param.dof_num, 2);
        boundary_vel.col(0) = start_v;
        if (!traj_opters[idx]->optimizeTraj(full_path, boundary_vel, boundary_acc)
            || !traj_opters[idx]->printConstraintsSituations(traj_opters[idx]->getTraj()) 
            )
            return false;
        else
        {
            parallel_ends[idx] = traj_opters[idx]->getTraj();
            return true;
        }
        return true;
    }

    bool Planner::optDenseOnce(const std::vector<Eigen::VectorXd>& full_path, int idx, 
                            const Eigen::VectorXd& start, const Eigen::VectorXd& end,
                            const Eigen::VectorXd& start_v)
    {
        Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(3+moma_param.dof_num, 2);
        Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(3+moma_param.dof_num, 2);
        boundary_vel.col(0) = start_v;
        if (!traj_opters[idx]->optimizeTraj(full_path, boundary_vel, boundary_acc)
            || !traj_opters[idx]->printConstraintsSituations(traj_opters[idx]->getTraj()) 
            )
            return false;
        else
        {
            parallel_ends[idx] = traj_opters[idx]->getTraj();
            return true;
        }
        return true;
    }

    //TODO: change to real
    bool Planner::planRemani(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Eigen::VectorXd& start_v)
    {
        list<GraphNode::Ptr> graph;
        vector<vector<Eigen::Vector3d>> raw_paths, filtered_paths;
        Eigen::Vector3d topo_start(start(0), start(1), 0.0);
        Eigen::Vector3d topo_end(end(0), end(1), 0.0);
        std::vector<Eigen::Vector3d> start_pts, end_pts;
        start_pts.push_back(topo_start);
        end_pts.push_back(topo_end);
        topo_select_paths.clear();
        topo_prm->findTopoPaths(topo_start, topo_end, start_pts, end_pts, graph,
                               raw_paths, filtered_paths, topo_select_paths);
                            //    raw_paths, filtered_paths, topo_select_paths);
        vis_prm_paths();

        // front end
        front_path.clear();
        ros::Time plan_start = ros::Time::now();
        ros::Time start_time = ros::Time::now();
        // auto result = grid_map->astarPlan2d(start.head(2), end.head(2), moma_param.chassis_colli_radius+0.1);
        // PRINT_YELLOW("astar time: " << (ros::Time::now() - start_time).toSec() * 1000.0 << " ms");

        // JPS
        // PRINT_GREEN("[Planner] Begin JPS + MCRRTs + BiRRTs hole filling...");
        start_time = ros::Time::now();
        auto jps_result = graph_search->plan2dJPS(start.head(2), end.head(2), moma_param.chassis_colli_radius+0.1);
        if (jps_result.empty())
        {
            // PRINT_RED("[Planner] JPS fail.");
            return false;
        }
        // PRINT_GREEN("[MCRRTs] JPS time: " << (ros::Time::now() - start_time).toSec() * 1000.0 << " ms");
        
        // MCRRTs
        auto dense_result = graph_search->getDensePath(jps_result, 1.414, start(2), end(2), 
                                                       moma_param.max_v, moma_param.max_w);

#ifdef PUB_DEBUG
        std::vector<Eigen::VectorXd> ppp;
        for (size_t i=0; i<dense_result.size(); i++)
        {
            Eigen::VectorXd temp = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            temp.head(3) = dense_result[i].head(3);
            if (i==0) temp.tail(moma_param.dof_num) = start.tail(moma_param.dof_num);
            if (i==dense_result.size()-1) temp.tail(moma_param.dof_num) = end.tail(moma_param.dof_num);
            ppp.push_back(temp);
        }
        PRINT_GREEN("dense_result path:");
        for (size_t i=0; i<dense_result.size(); i++)
            PRINT_YELLOW(dense_result[i].transpose());
        vis_path(ppp, front_pub);
#endif

        std::pair<int, int> layer(-1, -1);
        if (!mcrrts->plan(start, end, dense_result, front_path))
        {
            return false;
            PRINT_YELLOW("[Planner] Try fill hole by BiRRTs...");
            std::vector<double> t_list;
            std::vector<Eigen::VectorXd> starts, ends, hole_path;
            std::vector<double> start_costs, end_costs;
            std::vector<int> start_idx, end_idx;
            mcrrts->getHoleNodes(starts, ends, start_costs, end_costs, start_idx, end_idx);
            ros::Time hole_time = ros::Time::now();
            if (birrts->plan(starts, ends, start_costs, end_costs, start_idx, end_idx, hole_path, t_list, layer))
                mcrrts->fillHole(hole_path, layer, front_path);
            else
                PRINT_RED("[Planner] BiRRTs fill hole fail.");
            layer.second = layer.first + hole_path.size() - 1;
            PRINT_GREEN("[Planner] BiRRTs fill hole time: " << (ros::Time::now() - hole_time).toSec() * 1000.0 << " ms");
        }
        // PRINT_GREEN("[Planner] MCRRTs all time: " << (ros::Time::now() - start_time).toSec() * 1000.0 << " ms");
        
        // birrts
        // PRINT_GREEN("\n[Planner] Begin BiRRTs planning...");
        // start_time = ros::Time::now();
        // std::vector<double> t_list;
        // birrts->plan(start, end, front_path, t_list);
        // PRINT_YELLOW("[Planner] BiRRTs time: " << (ros::Time::now() - start_time).toSec() << " s");

        if (front_path.empty())
        {
            // PRINT_RED("[Planner] Front end fail.");
            return false;
        }

        // std::vector<int> mc_ids;
        // mc_ids.push_back(layer.first);
        // mc_ids.push_back(layer.second);
        // vis_path(front_path, front_pub, mc_ids);

#ifdef PUB_DEBUG
        PRINT_GREEN("front_path: ");
        for (size_t i=0; i<front_path.size(); i++)
            PRINT_GREEN(front_path[i].transpose());
#endif

        Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(3+moma_param.dof_num, 2);
        Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(3+moma_param.dof_num, 2);
        boundary_vel.col(0) = start_v;
        start_time = ros::Time::now();
        if (!traj_opter->optimizeTraj(front_path, boundary_vel, boundary_acc)
            || !traj_opter->printConstraintsSituations(traj_opter->getTraj()) 
            )
            return false;
        else
        {
            // PRINT_YELLOW("[Planner] TrajOpt time: " << (ros::Time::now() - start_time).toSec() * 1000.0 << " ms");

            end_traj = traj_opter->getTraj();
            // mpc->setTraj(end_traj, (ros::Time::now() - plan_start).toSec());
            // vis_whole_path(end_pub);
            if (local_mode)
            {
                begin_time = ros::Time::now();
                last_gplan_time = ros::Time::now();
                global_traj = end_traj;
                has_traj = true;
                global_goal = end;
            }

            return true;
        }

        return true;
    }

    std::vector<Eigen::Vector4d> Planner::vis_prm_paths()
    {
        visualization_msgs::MarkerArray markers;
        visualization_msgs::Marker line_strip, delet_p;

        delet_p.action = visualization_msgs::Marker::DELETEALL;
        delet_p.id = 0;
        markers.markers.push_back(delet_p);

        line_strip.type = visualization_msgs::Marker::LINE_STRIP;
        line_strip.header.frame_id = "world";
        line_strip.pose.orientation.w = 1.0;
        line_strip.scale.x = 0.10;
        line_strip.scale.y = 0.10;
        line_strip.scale.z = 0.10;
        line_strip.color.a = 1.0;

        std::vector<Eigen::Vector4d> colors;

        for (size_t i=0; i<topo_select_paths.size(); i++)
        {
            Eigen::Vector4d color = {
                1.0 * (rand() % 1000) / 1000.0, 
                1.0 * (rand() % 1000) / 1000.0, 
                1.0 * (rand() % 1000) / 1000.0, 
                1.0};
            colors.push_back(color);
            line_strip.header.stamp = ros::Time::now();
            line_strip.id = i + 1;
            line_strip.color.r = color[0];
            line_strip.color.g = color[1];
            line_strip.color.b = color[2];
            line_strip.points.clear();
            for (size_t j=0; j<topo_select_paths[i].size(); j++)
            {
                geometry_msgs::Point pt;
                pt.x = topo_select_paths[i][j].x();
                pt.y = topo_select_paths[i][j].y();
                pt.z = 0.0;
                line_strip.points.push_back(pt);
            }
            markers.markers.push_back(line_strip);
        }
        prm_pub.publish(markers);
        return colors;
    }

    void Planner::vis_path(const std::vector<Eigen::VectorXd>& path, ros::Publisher& puber, vector<float> rgba, vector<int> ids)
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
        for (size_t i=0; i<path.size(); i+=4)
        {
            visualization_msgs::MarkerArray node_array = moma_param.getColliCylinderArray(path[i]);
            size_t array_size = node_array.markers.size();
            for (size_t j=0; j<array_size; j++)
            {
                node_array.markers[j].id = i*array_size+j;
                node_array.markers[j].color.a = rgba[3];
                node_array.markers[j].color.r = rgba[0];
                node_array.markers[j].color.g = rgba[1];
                node_array.markers[j].color.b = rgba[2];
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
            // array_msg.markers.push_back(arrow);
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
            // array_msg.markers.push_back(text);
        }
        // array_msg.markers.push_back(line_strip);
        puber.publish(array_msg);
        return;
    }

    void Planner::vis_whole_path(ros::Publisher& pub)
    {
        std::vector<Eigen::VectorXd> end_path;
        nav_msgs::Path car_traj;
        for (double t=0.0; t<end_traj.getTotalDuration(); t+=0.1)
        {
            Eigen::VectorXd state = end_traj.getState(t);
            end_path.push_back(state);
            car_traj.header.frame_id = "world";
            car_traj.header.stamp = ros::Time::now();
            geometry_msgs::PoseStamped gp;
            gp.pose.position.x = state.x();
            gp.pose.position.y = state.y();
            gp.pose.position.z = 0.0;
            gp.pose.orientation.w = cos(state.z()/2.0);
            gp.pose.orientation.x = 0.0;
            gp.pose.orientation.y = 0.0;
            gp.pose.orientation.z = sin(state.z()/2.0);
            car_traj.poses.push_back(gp);
        }
        if (pub == end_pub)
            car_traj_pub.publish(car_traj);
        vis_path(end_path, pub, {1.0, 0.0, 0.0, 0.15});
    }

    Eigen::Quaterniond euler2rotation(double r, double p, double y)
    {
        return Eigen::AngleAxisd(r, Eigen::Vector3d::UnitX()) 
                * Eigen::AngleAxisd(p, Eigen::Vector3d::UnitY()) 
                * Eigen::AngleAxisd(y, Eigen::Vector3d::UnitZ());
    }

    Eigen::Quaterniond euler2rotation(Eigen::Vector3d rpy)
    {
        return Eigen::AngleAxisd(rpy(0), Eigen::Vector3d::UnitX()) 
                * Eigen::AngleAxisd(rpy(1), Eigen::Vector3d::UnitY()) 
                * Eigen::AngleAxisd(rpy(2), Eigen::Vector3d::UnitZ());
    }

    void Planner::vis_path_mesh(const std::vector<Eigen::VectorXd>& path, ros::Publisher& pub, vector<float> rgba, int id, bool fade)
    {
        if(path.empty()) return;

        visualization_msgs::MarkerArray moma_marker;
        
        visualization_msgs::Marker delet_p;
        delet_p.action = visualization_msgs::Marker::DELETEALL;
        delet_p.id = 9871;
        moma_marker.markers.push_back(delet_p);
        pub.publish(moma_marker);

        for(auto wp : path) {
            visualization_msgs::Marker diff_marker;
            {
                diff_marker.header.frame_id = "world";
                diff_marker.id = id++;
                diff_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
                diff_marker.action = visualization_msgs::Marker::ADD;
                diff_marker.mesh_resource = "package://fake_moma/meshes/tracer.dae";
                // pose
                diff_marker.pose.position.x = wp[0];
                diff_marker.pose.position.y = wp[1];
                diff_marker.pose.position.z = moma_param.chassis_height;
                // orientation
                Eigen::Quaterniond q = euler2rotation(M_PI_2, wp[2], 0.0);
                diff_marker.pose.orientation.w = q.w();
                diff_marker.pose.orientation.x = q.x();
                diff_marker.pose.orientation.y = q.y();
                diff_marker.pose.orientation.z = q.z();
                // color
                diff_marker.color.a = rgba[3];
                diff_marker.color.r = rgba[0];
                diff_marker.color.g = rgba[1];
                diff_marker.color.b = rgba[2];
                // scale
                diff_marker.scale.x = 1.0;
                diff_marker.scale.y = 1.0;
                diff_marker.scale.z = 1.0;
                if(fade) diff_marker.lifetime = ros::Duration(0.1);
            }
            moma_marker.markers.push_back(diff_marker);

            Eigen::Vector3d ap(
                diff_marker.pose.position.x, 
                diff_marker.pose.position.y, 
                diff_marker.pose.position.z
            );

            Eigen::Quaterniond aq(cos(wp[2]/2.0), 0.0, 0.0, sin(wp[2]/2.0));
            ap += aq.matrix() * moma_param.relative_t;
            aq = aq.matrix() * moma_param.relative_R;

            //link0
            visualization_msgs::Marker link_marker;
            {
                link_marker.header.frame_id = "world";
                link_marker.id = id++;
                link_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
                link_marker.action = visualization_msgs::Marker::ADD;
                link_marker.pose.position.x = ap.x();
                link_marker.pose.position.y = ap.y();
                link_marker.pose.position.z = ap.z();
                link_marker.pose.orientation.w = aq.w();
                link_marker.pose.orientation.x = aq.x();
                link_marker.pose.orientation.y = aq.y();
                link_marker.pose.orientation.z = aq.z();
                link_marker.color.a = rgba[3];
                link_marker.color.r = rgba[0];
                link_marker.color.g = rgba[1];
                link_marker.color.b = rgba[2];
                link_marker.scale.x = 1.0;
                link_marker.scale.y = 1.0;
                link_marker.scale.z = 1.0;
                link_marker.mesh_resource = "package://fake_moma/meshes/link0.STL";
                if(fade) link_marker.lifetime = ros::Duration(0.1);
            }
            moma_marker.markers.push_back(link_marker);

            //link1-7
            for (size_t i = 0; i < moma_param.dof_num; i++)
            {
                visualization_msgs::Marker link_marker;
                {
                    link_marker.header.frame_id = "world";
                    link_marker.id = id++;
                    link_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
                    link_marker.action = visualization_msgs::Marker::ADD;
                    ap += aq.matrix() * Eigen::Vector3d(0.0, 0.0, moma_param.link_length[i]);
                    link_marker.pose.position.x = ap.x();
                    link_marker.pose.position.y = ap.y();
                    link_marker.pose.position.z = ap.z();
                    aq = aq.matrix() * euler2rotation(moma_param.joint_offset.row(i))
                            * euler2rotation(moma_param.joint_dof_axis.row(i)*wp[i+3]);
                    link_marker.pose.orientation.w = aq.w();
                    link_marker.pose.orientation.x = aq.x();
                    link_marker.pose.orientation.y = aq.y();
                    link_marker.pose.orientation.z = aq.z();
                    link_marker.color.a = rgba[3];
                    link_marker.color.r = rgba[0];
                    link_marker.color.g = rgba[1];
                    link_marker.color.b = rgba[2];
                    link_marker.scale.x = 1.0;
                    link_marker.scale.y = 1.0;
                    link_marker.scale.z = 1.0;
                    link_marker.mesh_resource = "package://fake_moma/meshes/link"+std::to_string(i+1)+".STL";
                    if(fade) link_marker.lifetime = ros::Duration(0.1);
                }
                moma_marker.markers.push_back(link_marker);
            }

            //gripper
            visualization_msgs::Marker gripper_marker;
            {
                gripper_marker.header.frame_id = "world";
                gripper_marker.id = id++;
                gripper_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
                gripper_marker.action = visualization_msgs::Marker::ADD;
                gripper_marker.pose = moma_marker.markers.back().pose;
                gripper_marker.color.a = rgba[3];
                gripper_marker.color.r = rgba[0];
                gripper_marker.color.g = rgba[1];
                gripper_marker.color.b = rgba[2];
                gripper_marker.scale.x = 1.0;
                gripper_marker.scale.y = 1.0;
                gripper_marker.scale.z = 1.0;
                gripper_marker.mesh_resource = "package://fake_moma/meshes/gripper.dae";
                if(fade) gripper_marker.lifetime = ros::Duration(0.1);
            }
            moma_marker.markers.push_back(gripper_marker);
        }

        pub.publish(moma_marker);
    }


    void Planner::vis_path_mesh(const MomaTraj& traj, int nsample, ros::Publisher& pub, vector<float> rgba, int id, bool fade) {
        RowMatrixXd path_m = traj.sampleTimePoints(nsample);
        std::vector<Eigen::VectorXd> path;
        for (int i = 0; i < path_m.rows(); i++) {
            path.push_back(path_m.row(i).head(10));
        }

        vis_path_mesh(path, pub, rgba, id, fade);
    }

    void Planner::vis_ee_traj(const MomaTraj& traj, ros::Publisher& pub, vector<float> rgba) const {
        const int res = 200;
        const double intvl = traj.getTotalDuration() / res;
        double t = 0.0;

        visualization_msgs::Marker line_strip;
        {
            line_strip.header.frame_id = "world";
            line_strip.header.stamp = ros::Time::now();
            line_strip.ns = "velocity_trajectory";
            line_strip.action = visualization_msgs::Marker::ADD;
            line_strip.pose.orientation.w = 1.0;
            line_strip.id = 2077;
            line_strip.type = visualization_msgs::Marker::LINE_STRIP;
            line_strip.scale.x = 0.10;
            line_strip.scale.y = 0.10;
            line_strip.scale.z = 0.10;
            // line_strip.lifetime = ros::Duration(10.0);
        }

        std::vector<double> velocities;
        Eigen::Vector4d gripper_prev = moma_param.getColliPts(    
            traj.getState(0)
        ).back();

        for (size_t i = 0; i < res; i++) {
            Eigen::VectorXd state = traj.getState(t);
            Eigen::Vector4d gripper;
            gripper = moma_param.getColliPts(state).back();

            geometry_msgs::Point pt;
            pt.x = gripper (0);
            pt.y = gripper (1);
            pt.z = gripper (2);
            line_strip.points.push_back(pt);

            velocities.push_back((gripper.head(3) - gripper_prev.head(3)).norm() / intvl);
            t += intvl;
            gripper_prev = gripper;
        }
        velocities[0] = velocities[1]; // because the first velocity is not reliable

        double max_vel = *std::max_element(velocities.begin(), velocities.end());
        double min_vel = *std::min_element(velocities.begin(), velocities.end());
        // double avg_vel = std::accumulate(velocities.begin(), velocities.end(), 0.0) / velocities.size();


        std::cout << "max vel: " << max_vel << " min vel: " << min_vel << std::endl;
        for (size_t i = 0; i < res; i++){
            double vel = velocities[i];
            double r = (vel - min_vel) / (max_vel - min_vel);

            std_msgs::ColorRGBA color;
            {
                // Viridis
                // color.r = 0.267004 + 0.031242 * r - 1.17733 * pow(r, 2) + 
                //         0.781638 * pow(r, 3) + 0.46992 * pow(r, 4);
                // color.g = 0.004874 + 1.05819 * r - 0.218094 * pow(r, 2) - 
                //         1.52621 * pow(r, 3) + 1.80664 * pow(r, 4);
                // color.b = 0.329415 - 0.197112 * r - 5.8219 * pow(r, 3) + 
                //         5.30237 * pow(r, 4);
                // color.a = 1.0;
            }

            {
                // inferno
                // r = r > 0.9 ? 0.9 : r;
                // if (r < 0.25) {
                //     color.r = 0.2 * r * 4.0;
                //     color.g = 0.0;
                //     color.b = 0.3 + 0.7 * r * 4.0;
                // } 
                // else if (r < 0.5) {
                //     color.r = 0.2 + 0.8 * (r-0.25)*4.0;
                //     color.g = 0.1 * (r-0.25)*4.0;
                //     color.b = 1.0 - 0.8 * (r-0.25)*4.0;
                // }
                // else if (r < 0.75) {
                //     color.r = 1.0;
                //     color.g = 0.1 + 0.9 * (r-0.5)*4.0;
                //     color.b = 0.2 - 0.2 * (r-0.5)*4.0;
                // }
                // else {
                //     color.r = 1.0;
                //     color.g = 1.0;
                //     color.b = 0.0 + 1.0 * (r-0.75)*4.0;
                // }
                
                // color.a = 0.5;
            }
            {
                color.r = rgba[0];
                color.g = rgba[1];
                color.b = rgba[2];
                color.a = rgba[3];
            }
            line_strip.colors.push_back(color);
        }

        pub.publish(line_strip);
    }
    
    planner::MeshTraj Planner::toMeshMsg(const MomaTraj& traj) const {
        double traj_duration = traj.getTotalDuration();
        const int res = 100;
        double intvl = traj_duration / res;


        planner::MeshTraj ret;
        std::vector<planner::MeshState> states;
        std::vector<double> arc_lengths;
        std::vector<double> yaws;

        double acc_arc_length = 0.0;
        Eigen::VectorXd prev_state = traj.getState(0);
        
        for (double t = 0.0; t < traj_duration; t += intvl) {
            Eigen::VectorXd state = traj.getState(t);
            std::vector<Eigen::VectorXd> mesh_poses = moma_param.getMeshPose(state);

            planner::MeshState mesh_state;
            std::vector<planner::MeshPart> mesh_poses_msg;

            for (Eigen::VectorXd mesh_pose : mesh_poses) {
                planner::MeshPart mesh_pose_msg;
                

                mesh_pose_msg.pos_x = mesh_pose(0);
                mesh_pose_msg.pos_y = mesh_pose(1);
                mesh_pose_msg.pos_z = mesh_pose(2);
                mesh_pose_msg.orient_w = mesh_pose(3);
                mesh_pose_msg.orient_x = mesh_pose(4);
                mesh_pose_msg.orient_y = mesh_pose(5);
                mesh_pose_msg.orient_z = mesh_pose(6);

                mesh_poses_msg.push_back(mesh_pose_msg);
            }


            mesh_state.parts = mesh_poses_msg;
            states.push_back(mesh_state);

            acc_arc_length += (state.head(2) - prev_state.head(2)).norm();

            arc_lengths.push_back(acc_arc_length);
            yaws.push_back(state(2));


            prev_state = state;
        }

        // for (int i = 0; i < 10; i++){

        //     std::cout << prev_state[i] << std::endl;
        // }

        ret.states = states;
        ret.arc_lengths = arc_lengths;
        ret.yaws = yaws;
        return ret;
    }

    void Planner::publishBenchmarkText(int ddim_succ, int moma_succ, int total)
    {
        std::vector<std::string> lines;
        // lines.push_back("Succ. Counter");
        lines.push_back("PTDM:  " + std::to_string(ddim_succ) + "/" + std::to_string(total));
        // lines.push_back("DDIM:  ");

        lines.push_back("TopAY: " + std::to_string(moma_succ) + "/" + std::to_string(total));

        for (size_t i = 0; i < lines.size(); i++)
        {
            visualization_msgs::Marker marker;

            marker.header.frame_id = "world";
            marker.header.stamp = ros::Time::now();

            marker.ns = "benchmark_text";
            marker.id = i+706;

            marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            marker.action = visualization_msgs::Marker::ADD;

            marker.pose.position.x = 0.0;
            marker.pose.position.y = 0.0;
            marker.pose.position.z = 6.0 - i * 0.5;  // stack vertically

            marker.pose.orientation.w = 1.0;

            marker.scale.z = 0.4;

            // Different colors
            if (i == 0)
            {
                marker.color.r = 38.0/255.0;
                marker.color.g = 70.0/255.0;
                marker.color.b = 83.0/255.0;
            }
            else if (i == 1) {
                
                marker.color.r = 230.0/255.0;
                marker.color.g = 111.0/255.0;
                marker.color.b = 81.0/255;
            }
            else if (i == 2) {
                // marker.color.r = 0.9647058823529412;
                // marker.color.g = 0.5803921568627451;
                // marker.color.b = 0.29411764705882354;
                
            }
            marker.color.a = 1.0;

            marker.text = lines[i];

            text_pub.publish(marker);
        }
    }
    void Planner::animate_trajectory_timer_cb(const ros::TimerEvent&)
    {
        static size_t idx = 0;
        if(!end_traj_available) { idx=0; return; }



        int nsample = int(ddim_traj.getTotalDuration() / 0.1) + 1;
        RowMatrixXd path_m = ddim_traj.sampleTimePoints(nsample);
            

        int path_size = path_m.rows();

        // play only once
        if ((++idx) >= path_size){
            idx = 0;
            end_traj_available = false;
        }

        std::vector<Eigen::VectorXd> path;
        path.push_back(path_m.row(idx).head(10));

        vis_path_mesh(path, anime_pub, 
        {
            38.0/255.0,
            70.0/255.0,
            83.0/255,
            1.0
        },
        4322,
        true);
    }
}