#ifndef PROXEMICS_LAYER_ROS1__PROXEMIC_COST_HPP_
#define PROXEMICS_LAYER_ROS1__PROXEMIC_COST_HPP_

#include <algorithm>
#include <cmath>

namespace proxemics_layer
{

// Soft-cost cap. Deliberately BELOW the costmap_2d special values
// INSCRIBED_INFLATED_OBSTACLE (253) and LETHAL_OBSTACLE (254) so a
// proxemic bump is always *soft*: the planner strongly prefers to avoid a
// person's space but may enter it if there is no other way through.
constexpr unsigned char kMaxProxemicCost = 250;

// Normalized proxemic cost in [0, 1] at distance `d` (metres) from a person,
// given their affect-modulated personal-space `radius` (metres) and a
// per-person `peak` weight in [0, 1]. Gaussian bump, sigma = radius / 2.
inline double proxemic_cost(double d, double radius, double peak)
{
  const double r = std::max(radius, 1e-3);
  const double sigma = r / 2.0;
  const double g = std::exp(-(d * d) / (2.0 * sigma * sigma));
  return std::clamp(peak, 0.0, 1.0) * g;
}

inline unsigned char proxemic_cost_cell(double d, double radius, double peak)
{
  const double c = proxemic_cost(d, radius, peak);
  return static_cast<unsigned char>(c * kMaxProxemicCost + 0.5);
}

}  // namespace proxemics_layer

#endif  // PROXEMICS_LAYER_ROS1__PROXEMIC_COST_HPP_
