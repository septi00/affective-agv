#ifndef PROXEMICS_LAYER__PROXEMIC_COST_HPP_
#define PROXEMICS_LAYER__PROXEMIC_COST_HPP_

#include <algorithm>
#include <cmath>

namespace proxemics_layer
{

// Soft-cost cap. Deliberately BELOW the Nav2 special values
// INSCRIBED_INFLATED_OBSTACLE (253) and LETHAL_OBSTACLE (254) so a
// proxemic bump is always *soft*: the planner strongly prefers to avoid a
// person's space but may enter it if there is no other way through.
constexpr unsigned char kMaxProxemicCost = 250;

// Normalized proxemic cost in [0, 1] at distance `d` (metres) from a
// person, given their affect-modulated personal-space `radius` (metres)
// and a per-person `peak` weight in [0, 1].
//
// Model: an isotropic Gaussian bump centred on the person. `radius` is the
// 2-sigma extent (sigma = radius / 2), so most of the cost mass sits inside
// `radius`; at exactly `radius` the cost has decayed to e^-2 (~13.5%) of
// `peak`. This keeps the affect->radius policy legible: a larger radius
// (person wants more space) widens the bump, a smaller one tightens it,
// with no hard discontinuity for the planner to chatter on.
inline double proxemic_cost(double d, double radius, double peak)
{
  const double r = std::max(radius, 1e-3);
  const double sigma = r / 2.0;
  const double g = std::exp(-(d * d) / (2.0 * sigma * sigma));
  return std::clamp(peak, 0.0, 1.0) * g;
}

// Map the normalized cost to a Nav2 cell value, capped so it stays soft.
inline unsigned char proxemic_cost_cell(double d, double radius, double peak)
{
  const double c = proxemic_cost(d, radius, peak);
  return static_cast<unsigned char>(c * kMaxProxemicCost + 0.5);
}

}  // namespace proxemics_layer

#endif  // PROXEMICS_LAYER__PROXEMIC_COST_HPP_
