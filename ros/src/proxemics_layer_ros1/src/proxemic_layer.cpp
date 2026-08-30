#include "proxemics_layer_ros1/proxemic_layer.hpp"

#include <algorithm>
#include <cmath>

#include "proxemics_layer_ros1/proxemic_cost.hpp"
#include <costmap_2d/cost_values.h>
#include <pluginlib/class_list_macros.h>
#include <tf2/utils.h>
#include <geometry_msgs/PointStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace proxemics_layer
{

void ProxemicLayer::onInitialize()
{
  ros::NodeHandle nh("~" + name_);

  nh.param("enabled", enabled_, true);
  nh.param("topic", topic_, topic_);
  nh.param("cutoff_factor", cutoff_factor_, cutoff_factor_);
  nh.param("stale_timeout", stale_timeout_, stale_timeout_);

  ros::NodeHandle g;
  sub_ = g.subscribe(topic_, 10, &ProxemicLayer::circlesCallback, this);

  current_ = true;
  ROS_INFO("ProxemicLayer active, subscribed to '%s'", topic_.c_str());
}

void ProxemicLayer::circlesCallback(
  const affective_msgs::ProxemicCircleArray::ConstPtr & msg)
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
  costmap_2d::Costmap2D & master_grid,
  int /*min_i*/, int /*min_j*/, int /*max_i*/, int /*max_j*/)
{
  if (!enabled_) {return;}

  affective_msgs::ProxemicCircleArray circles;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    circles = latest_;
  }
  if (circles.circles.empty()) {return;}

  // If perception has gone quiet, stop stamping cost.
  if (circles.header.stamp.toSec() > 0.0 &&
    (ros::Time::now() - circles.header.stamp).toSec() > stale_timeout_)
  {
    return;
  }

  // Transform circle centres from their own frame (base_footprint on the
  // robot) into the costmap's global frame. Radius is a scalar distance,
  // unaffected by translation/rotation, so it carries over unchanged.
  const std::string target = layered_costmap_->getGlobalFrameID();
  const std::string source =
    circles.header.frame_id.empty() ? target : circles.header.frame_id;

  geometry_msgs::TransformStamped tf;
  try {
    tf = tf_->lookupTransform(target, source, ros::Time(0));
  } catch (const tf2::TransformException & ex) {
    ROS_WARN_THROTTLE(2.0, "ProxemicLayer: no transform %s -> %s (%s)",
      source.c_str(), target.c_str(), ex.what());
    return;
  }

  for (const auto & c : circles.circles) {
    geometry_msgs::PointStamped in, out;
    in.header = circles.header;
    in.point.x = c.x;
    in.point.y = c.y;
    in.point.z = 0.0;
    tf2::doTransform(in, out, tf);

    const double cx = out.point.x, cy = out.point.y;
    const double reach = cutoff_factor_ * std::max<double>(c.radius, 1e-3);

    unsigned int mx0, my0, mx1, my1;
    if (!master_grid.worldToMap(cx - reach, cy - reach, mx0, my0)) {continue;}
    if (!master_grid.worldToMap(cx + reach, cy + reach, mx1, my1)) {continue;}

    for (unsigned int j = my0; j <= my1; ++j) {
      for (unsigned int i = mx0; i <= mx1; ++i) {
        double wx, wy;
        master_grid.mapToWorld(i, j, wx, wy);
        const double d = std::hypot(wx - cx, wy - cy);
        if (d > reach) {continue;}

        const unsigned char cost = proxemic_cost_cell(d, c.radius, c.peak_cost);
        const unsigned char old = master_grid.getCost(i, j);
        const unsigned char base =
          (old == costmap_2d::NO_INFORMATION) ? 0 : old;
        if (cost > base) {
          master_grid.setCost(i, j, cost);
        }
      }
    }
  }
}

}  // namespace proxemics_layer

PLUGINLIB_EXPORT_CLASS(proxemics_layer::ProxemicLayer, costmap_2d::Layer)
