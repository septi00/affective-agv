#ifndef PROXEMICS_LAYER__PROXEMIC_STAMP_HPP_
#define PROXEMICS_LAYER__PROXEMIC_STAMP_HPP_

#include "nav2_costmap_2d/costmap_2d.hpp"

namespace proxemics_layer
{

// Stamp one person's Gaussian proxemic bump into `grid`.
// Centre (cx, cy) and `radius` are in the grid's world frame (metres);
// `peak` is the per-person weight in [0, 1]; cost is rasterized out to
// cutoff_factor * radius. Existing higher costs are preserved (max-combine),
// and unknown cells are treated as free before combining.
void stampCircle(
  nav2_costmap_2d::Costmap2D & grid,
  double cx, double cy, double radius, double peak,
  double cutoff_factor);

}  // namespace proxemics_layer

#endif  // PROXEMICS_LAYER__PROXEMIC_STAMP_HPP_
