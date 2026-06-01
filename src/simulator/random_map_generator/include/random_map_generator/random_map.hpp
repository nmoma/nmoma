#pragma once

#include <iostream>
#include <math.h>
#include <random>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <ros/package.h>
#include <vector>
#include <utility>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>

using namespace std;

namespace nmoma_planner {
namespace random_map {
    struct Box {
        typedef std::array<float, 8> array_repr;

        Eigen::Vector3d pos;
        Eigen::Vector3d size;
        double theta;
        std::array<Eigen::Vector3d, 8> corners;

        Box(const Eigen::Vector3d& pos, const Eigen::Vector3d& size, double theta = 0.0) : pos(pos), size(size), theta(theta) {
            Eigen::Matrix3d rotation_matrix;

            rotation_matrix << 
                cos(theta), -sin(theta), 0, 
                sin(theta),  cos(theta), 0, 
                0, 0, 1;
            
            corners[0] = pos;
            corners[1] = pos + rotation_matrix * Eigen::Vector3d(size.x(), 0, 0);
            corners[2] = pos + rotation_matrix * Eigen::Vector3d(0, size.y(), 0);
            corners[3] = pos + rotation_matrix * Eigen::Vector3d(size.x(), size.y(), 0);
            
            corners[4] = corners[0] + size.z() * Eigen::Vector3d (0, 0, 1.0);
            corners[5] = corners[1] + size.z() * Eigen::Vector3d (0, 0, 1.0);
            corners[6] = corners[2] + size.z() * Eigen::Vector3d (0, 0, 1.0);
            corners[7] = corners[3] + size.z() * Eigen::Vector3d (0, 0, 1.0);
        }

        pcl::PointCloud<pcl::PointXYZ> generatePCL(double resolution) const;

        inline array_repr toArray() const {
            return { pos.x(), pos.y(), pos.z(), size.x(), size.y(), size.z(), cos(theta), sin(theta) };
        }

        inline bool overlap2d(const Box& other) const {
            // Project all corners onto axes of both boxes
            std::array<Eigen::Vector2d, 4> axes = {
                (corners[1] - corners[0]).head<2>().normalized(),
                (corners[2] - corners[0]).head<2>().normalized(),
                (other.corners[1] - other.corners[0]).head<2>().normalized(),
                (other.corners[2] - other.corners[0]).head<2>().normalized()
            };
        
            for (const auto& axis : axes) {
                double min1 = INFINITY, max1 = -INFINITY;
                double min2 = INFINITY, max2 = -INFINITY;
                for (int i = 0; i < 4; i++) {
                    double proj1 = corners[i].head<2>().dot(axis);
                    double proj2 = other.corners[i].head<2>().dot(axis);
                    min1 = std::min(min1, proj1);
                    max1 = std::max(max1, proj1);
                    min2 = std::min(min2, proj2);
                    max2 = std::max(max2, proj2);
                }
                if (max1 < min2 || max2 < min1) return false;
            }
            return true;
        }

        inline bool overlap(const Box& other) const {
            return overlap2d(other) && 
                (pos.z() + size.z() > other.pos.z() && 
                pos.z() < other.pos.z() + other.size.z());
        }
    };

    struct Sphere {
        typedef std::array<float, 4> array_repr;
    
        Eigen::Vector3d pos;   // center
        double radius;
    
        Sphere(const Eigen::Vector3d& pos, double radius)
            : pos(pos), radius(radius) {}
    
        inline array_repr toArray() const {
            return {
                static_cast<float>(pos.x()),
                static_cast<float>(pos.y()),
                static_cast<float>(pos.z()),
                static_cast<float>(radius)
            };
        }
    
        pcl::PointCloud<pcl::PointXYZ> generatePCL(double resolution) const;
    };

    struct Cylinder {
        typedef std::array<float, 7> array_repr;
    
        // Base center (z = 0 by construction)
        Eigen::Vector3d pos;     // (x, y, 0)
        double radius;
        double height;
    
        // Orientation
        double theta;  // yaw (around Z)
        double gamma;  // tilt angle (pitch)
    
        // Unit axis direction of the cylinder
        Eigen::Vector3d axis;
    
        // Optional: sample points on bottom & top rim
        std::vector<Eigen::Vector3d> rim_bottom;
        std::vector<Eigen::Vector3d> rim_top;
    
        Cylinder(
            const Eigen::Vector2d& pos2d,
            double radius,
            double height,
            double theta = 0.0,
            double gamma = 0.0
        )
            : pos(pos2d.x(), pos2d.y(), 0.0),
              radius(radius),
              height(height),
              theta(theta),
              gamma(gamma)
        {
            // Rotation: yaw (Z) then tilt (X')
            Eigen::Matrix3d Rz, Rx;
            Rz <<
                cos(theta), -sin(theta), 0,
                sin(theta),  cos(theta), 0,
                0,           0,          1;
    
            Rx <<
                1, 0,           0,
                0, cos(gamma), -sin(gamma),
                0, sin(gamma),  cos(gamma);
    
            Eigen::Matrix3d R = Rz * Rx;
    
            // Cylinder axis (originally Z)
            axis = R * Eigen::Vector3d(0, 0, 1);
        }
    
        inline Eigen::Vector3d topCenter() const {
            return pos + height * axis;
        }
    
        inline array_repr toArray() const {
            return {
                pos.x(), pos.y(),
                static_cast<float>(radius),
                static_cast<float>(height),
                static_cast<float>(cos(theta)),
                static_cast<float>(sin(theta)),
                static_cast<float>(gamma)
            };
        }
    
        pcl::PointCloud<pcl::PointXYZ> generatePCL(double resolution) const;
        
        bool overlap(const Cylinder& other) const {
            Eigen::Vector3d p1 = pos;
            Eigen::Vector3d p2 = pos + axis * height;
    
            Eigen::Vector3d q1 = other.pos;
            Eigen::Vector3d q2 = other.pos + other.axis * other.height;
    
            double dist2 = segmentDistanceSquared(p1, p2, q1, q2);
            double radiusSum = radius + other.radius;
    
            return dist2 <= radiusSum * radiusSum;
        }
        
        bool overlap(const Sphere& sphere) const {
            // Cylinder axis segment
            Eigen::Vector3d a = pos;
            Eigen::Vector3d b = pos + axis * height;
        
            // Vector from a to sphere center
            Eigen::Vector3d ap = sphere.pos - a;
            Eigen::Vector3d ab = b - a;
        
            double ab_len2 = ab.squaredNorm();
        
            // Project sphere center onto axis segment
            double t = ap.dot(ab) / ab_len2;
            t = std::max(0.0, std::min(1.0, t));
        
            // Closest point on cylinder axis
            Eigen::Vector3d closest = a + t * ab;
        
            // Distance from sphere center to axis
            double dist2 = (sphere.pos - closest).squaredNorm();
        
            double r_sum = radius + sphere.radius;
            return dist2 <= r_sum * r_sum;
        }

        bool overlap(const Box& box) const {
            // Cylinder axis segment
            Eigen::Vector3d a = pos;
            Eigen::Vector3d b = pos + axis * height;
        
            // Sample closest approach of segment to box
            const int samples = 8;  // small, fixed
            double minDist2 = std::numeric_limits<double>::infinity();
        
            for (int i = 0; i <= samples; ++i) {
                double t = static_cast<double>(i) / samples;
                Eigen::Vector3d p = a + t * (b - a);
        
                Eigen::Vector3d q = closestPointOnBox(box, p);
                double d2 = (p - q).squaredNorm();
        
                minDist2 = std::min(minDist2, d2);
                if (minDist2 <= radius * radius)
                    return true;
            }
        
            return false;
        }        
    
        private:
        // Compute squared distance between two line segments
        static double segmentDistanceSquared(
            const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
            const Eigen::Vector3d& q1, const Eigen::Vector3d& q2
        ) {
            Eigen::Vector3d u = p2 - p1;
            Eigen::Vector3d v = q2 - q1;
            Eigen::Vector3d w = p1 - q1;
    
            double a = u.dot(u);
            double b = u.dot(v);
            double c = v.dot(v);
            double d = u.dot(w);
            double e = v.dot(w);
    
            double denom = a * c - b * b;
            double sN, sD = denom;
            double tN, tD = denom;
    
            const double SMALL_NUM = 1e-8;
    
            if (denom < SMALL_NUM) { // parallel
                sN = 0.0; sD = 1.0;
                tN = e; tD = c;
            } else {
                sN = b*e - c*d;
                tN = a*e - b*d;
            }
    
            if (sN < 0.0) sN = 0.0; else if (sN > sD) sN = sD;
            if (tN < 0.0) tN = 0.0; else if (tN > tD) tN = tD;
    
            double sc = (std::abs(sN) < SMALL_NUM ? 0.0 : sN / sD);
            double tc = (std::abs(tN) < SMALL_NUM ? 0.0 : tN / tD);
    
            Eigen::Vector3d dP = w + (sc * u) - (tc * v);
            return dP.squaredNorm();
        }

        static Eigen::Vector3d closestPointOnBox(
            const Box& box,
            const Eigen::Vector3d& p
        ) {
            // Box local frame
            Eigen::Matrix3d R;
            R <<
                cos(box.theta), -sin(box.theta), 0,
                sin(box.theta),  cos(box.theta), 0,
                0,               0,              1;
        
            Eigen::Vector3d local = R.transpose() * (p - box.pos);
        
            Eigen::Vector3d half = box.size * 0.5;
        
            Eigen::Vector3d clamped;
            clamped.x() = std::max(0.0, std::min(box.size.x(), local.x()));
            clamped.y() = std::max(0.0, std::min(box.size.y(), local.y()));
            clamped.z() = std::max(0.0, std::min(box.size.z(), local.z()));
        
            return R * clamped + box.pos;
        }
    };    

    struct RandomPCGenerator {
        // random
        default_random_engine eng;
        uniform_real_distribution<double> rand_x;
        uniform_real_distribution<double> rand_y;
        uniform_real_distribution<double> rand_z;
        uniform_real_distribution<double> rand_wall_size;
        uniform_real_distribution<double> rand_wall_height;
        uniform_real_distribution<double> rand_float_size;
        uniform_real_distribution<double> rand_float_height;
        uniform_real_distribution<double> rand_theta;

        uniform_real_distribution<double> rand_desk_width;
        uniform_real_distribution<double> rand_desk_length;
        uniform_real_distribution<double> rand_desk_height;

        uniform_real_distribution<double> rand_cylinder_yaw;
        uniform_real_distribution<double> rand_cylinder_tilting;
        uniform_real_distribution<double> rand_cylinder_radius;
        uniform_real_distribution<double> rand_sphere_radius;


        uniform_int_distribution<int> rand_arragement;

        // params
        std::string scene;
        vector<int> obs_num = {1, 1};
        vector<double> wall_size_range = {0.1, 0.1};
        vector<double> wall_height_range = {0.1, 0.1};
        vector<double> float_size_range = {0.1, 0.1};
        vector<double> float_height_range = {0.1, 0.1};
        // vector<double> theta_range = {-M_PI, M_PI};
        double resolution = 0.1;
        double size_x = 30.0;
        double size_y = 30.0;
        double size_z = 30.0;
        double min_obs_dis = 1.0;

        vector<double> desk_width_range = {0.75, 1.25};
        vector<double> desk_length_range= {0.75, 1.5};
        vector<double> desk_height_range={0.75, 1.5};
        vector<int> desk_arrangement_range = {1, 2};

        vector<double> cylinder_tilting_range = {-M_PI/4, M_PI/4};
        vector<double> cylinder_radius_range = {0.1, 0.4};
        vector<double> sphere_radius_range = {0.2, 0.4};

        void init(
            const vector<int>& obs_num,
            const vector<double>& wall_size_range,
            const vector<double>& wall_height_range,
            const vector<double>& float_size_range,
            const vector<double>& float_height_range,
            double resolution,
            double size_x,
            double size_y,
            double min_obs_dis
        );
        void init(ros::NodeHandle& nh);


        pcl::PointCloud<pcl::PointXYZ> generateBox(const Eigen::Vector3d& size);

        std::pair<pcl::PointCloud<pcl::PointXYZ>, std::vector<Box::array_repr>> generateDesk(
            const Eigen::Vector3d& pos,
            const Eigen::Vector3d& size,
            double theta);
        
        std::pair<pcl::PointCloud<pcl::PointXYZ>, std::vector<Box::array_repr>> generateDesk(
            const Eigen::Vector3d& pos,
            const Eigen::Vector3d& size,
            double theta, 
            const Eigen::Vector2i& arrangement);
            
        std::pair<pcl::PointCloud<pcl::PointXYZ>, std::vector<Box::array_repr>> generateDeskCase();
        std::pair<pcl::PointCloud<pcl::PointXYZ>, std::vector<Box::array_repr>> generateDeskCase(std::vector<Box> obs);


        std::pair<pcl::PointCloud<pcl::PointXYZ>, std::vector<Box::array_repr>> generateHybridCaseAux();
        std::pair<pcl::PointCloud<pcl::PointXYZ>, std::vector<Box::array_repr>> generateHybridCase();
        std::pair<pcl::PointCloud<pcl::PointXYZ>, std::vector<Box::array_repr>> generateHybridCase(unsigned int seed);

        std::pair<pcl::PointCloud<pcl::PointXYZ>, std::vector<Box::array_repr>> generateRandomCaseAux();
        std::pair<pcl::PointCloud<pcl::PointXYZ>, std::vector<Box::array_repr>> generateRandomCase();
        std::pair<pcl::PointCloud<pcl::PointXYZ>, std::vector<Box::array_repr>> generateRandomCase(unsigned int seed);

        std::pair<pcl::PointCloud<pcl::PointXYZ>, std::vector<Box::array_repr>> generateMap(unsigned int);
    };
}
}