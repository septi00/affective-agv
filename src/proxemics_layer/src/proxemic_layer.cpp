#include "proxemics_layer/proxemic_layer.hpp"

#include <string>

#include "proxemics_layer/proxemic_stamp.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2/time.h"
#include "tf2/exceptions.h"
#include "pluginlib/class_list_macros.hpp"

namespace proxemics_layer
{

void ProxemicLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"ProxemicLayer: could not lock parent node"};
  }

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("topic", rclcpp::ParameterValue(topic_));
  declareParameter("cutoff_factor", rclcpp::ParameterValue(cutoff_factor_));
  declareParameter("stale_timeout", rclcpp::ParameterValue(stale_timeout_));

  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".topic", topic_);
  node->get_parameter(name_ + ".cutoff_factor", cutoff_factor_);
  node->get_parameter(name_ + ".stale_timeout", stale_timeout_);

  sub_ = node->create_subscription<affective_msgs::msg::ProxemicCircleArray>(
    topic_, rclcpp::SensorDataQoS(),
    std::bind(&ProxemicLayer::circlesCallback, this, std::placeholders::_1));

  current_ = true;
  RCLCPP_INFO(
    node->get_logger(), "ProxemicLayer active, subscribed to '%s'",
    topic_.c_str());
}

void ProxemicLayer::circlesCallback(
  affective_msgs::msg::ProxemicCircleArray::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  latest_ = *msg;
}

void ProxemicLayer::updateBounds(
  double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) {return;}

  std::lock_guard<std::mutex> lock(mutex_);

  *min_x = std::min(*min_x, last_min_x_);
  *min_y = std::min(*min_y, last_min_y_);
  *max_x = std::max(*max_x, last_max_x_);
  *max_y = std::max(*max_y, last_max_y_);

  // Bounds are widened generously; the exact transformed centres are
  // resolved in updateCosts. Half a metre of slack covers robot motion
  // between the two calls.
  for (const auto & c : latest_.circles) {
    const double reach = cutoff_factor_ * c.radius + 0.5;
    *min_x = std::min(*min_x, static_cast<double>(c.x) - reach);
    *min_y = std::min(*min_y, static_cast<double>(c.y) - reach);
    *max_x = std::max(*max_x, static_cast<double>(c.x) + reach);
    *max_y = std::max(*max_y, static_cast<double>(c.y) + reach);
  }

  last_min_x_ = *min_x; last_min_y_ = *min_y;
  last_max_x_ = *max_x; last_max_y_ = *max_y;
}

void ProxemicLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int /*min_i*/, int /*min_j*/, int /*max_i*/, int /*max_j*/)
{
  if (!enabled_) {return;}

  affective_msgs::msg::ProxemicCircleArray circles;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    circles = latest_;
  }
  if (circles.circles.empty()) {return;}

  // If perception has gone quiet, stop stamping cost.
  if (auto node = node_.lock()) {
    const rclcpp::Time stamp(circles.header.stamp);
    if (stamp.nanoseconds() > 0 &&
      (node->now() - stamp).seconds() > stale_timeout_)
    {
      return;
    }
  }

  // Transform circle centres from their own frame (base_link on the robot)
  // into the costmap's global frame. Radius is a scalar distance, so it is
  // unaffected by translation/rotation and carries over unchanged.
  const std::string target = layered_costmap_->getGlobalFrameID();
  const std::string source =
    circles.header.frame_id.empty() ? target : circles.header.frame_id;

  geometry_msgs::msg::TransformStamped tf;
  try {
    // TimePointZero = latest available transform; robust against small
    // stamp/extrapolation mismatches between perception and TF.
    tf = tf_->lookupTransform(target, source, tf2::TimePointZero);
  } catch (const tf2::TransformException & ex) {
    if (auto node = node_.lock()) {
      RCLCPP_WARN_THROTTLE(
        node->get_logger(), *node->get_clock(), 2000,
        "ProxemicLayer: no transform %s -> %s (%s)",
        source.c_str(), target.c_str(), ex.what());
    }
    return;
  }

  for (const auto & c : circles.circles) {
    geometry_msgs::msg::PointStamped in, out;
    in.header = circles.header;
    in.point.x = c.x;
    in.point.y = c.y;
    in.point.z = 0.0;
    tf2::doTransform(in, out, tf);
    stampCircle(
      master_grid, out.point.x, out.point.y,
      c.radius, c.peak_cost, cutoff_factor_);
  }
}

void ProxemicLayer::reset()
{
  std::lock_guard<std::mutex> lock(mutex_);
  latest_.circles.clear();
}

}  // namespace proxemics_layer

PLUGINLIB_EXPORT_CLASS(proxemics_layer::ProxemicLayer, nav2_costmap_2d::Layer)
