#ifndef PROXEMICS_LAYER_ROS1__PROXEMIC_LAYER_HPP_
#define PROXEMICS_LAYER_ROS1__PROXEMIC_LAYER_HPP_

#include <mutex>
#include <string>

#include <ros/ros.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <tf2_ros/buffer.h>

#include <affective_msgs/ProxemicCircleArray.h>

namespace proxemics_layer
{

class ProxemicLayer : public costmap_2d::Layer
{
public:
  ProxemicLayer() = default;

  void onInitialize() override;
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(
    costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;

private:
  void circlesCallback(
    const affective_msgs::ProxemicCircleArray::ConstPtr & msg);

  ros::Subscriber sub_;
  std::mutex mutex_;
  affective_msgs::ProxemicCircleArray latest_;

  std::string topic_{"/proxemics/circles"};
  double cutoff_factor_{2.0};
  double stale_timeout_{1.0};
  bool enabled_{true};

  double last_min_x_{0.0}, last_min_y_{0.0};
  double last_max_x_{0.0}, last_max_y_{0.0};
};

}  // namespace proxemics_layer

#endif  // PROXEMICS_LAYER_ROS1__PROXEMIC_LAYER_HPP_
