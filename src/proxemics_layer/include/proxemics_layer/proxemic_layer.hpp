#ifndef PROXEMICS_LAYER__PROXEMIC_LAYER_HPP_
#define PROXEMICS_LAYER__PROXEMIC_LAYER_HPP_

#include <algorithm>
#include <mutex>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"

#include "affective_msgs/msg/proxemic_circle_array.hpp"

namespace proxemics_layer
{

class ProxemicLayer : public nav2_costmap_2d::Layer
{
public:
  ProxemicLayer() = default;

  void onInitialize() override;
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;
  void reset() override;
  bool isClearable() override {return true;}

private:
  void circlesCallback(
    affective_msgs::msg::ProxemicCircleArray::SharedPtr msg);

  rclcpp::Subscription<affective_msgs::msg::ProxemicCircleArray>::SharedPtr sub_;
  std::mutex mutex_;
  affective_msgs::msg::ProxemicCircleArray latest_;

  std::string topic_{"/proxemics/circles"};
  double cutoff_factor_{2.0};
  double stale_timeout_{1.0};
  bool enabled_{true};

  double last_min_x_{0.0}, last_min_y_{0.0};
  double last_max_x_{0.0}, last_max_y_{0.0};
};

}  // namespace proxemics_layer

#endif  // PROXEMICS_LAYER__PROXEMIC_LAYER_HPP_
