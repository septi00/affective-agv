#include "proxemics_layer/proxemic_stamp.hpp"

#include <algorithm>
#include <cmath>

#include "proxemics_layer/proxemic_cost.hpp"
#include "nav2_costmap_2d/cost_values.hpp"

namespace proxemics_layer
{

void stampCircle(
  nav2_costmap_2d::Costmap2D & grid,
  double cx, double cy, double radius, double peak,
  double cutoff_factor)
{
  const double reach = cutoff_factor * std::max(radius, 1e-3);

  unsigned int mx0, my0, mx1, my1;
  if (!grid.worldToMap(cx - reach, cy - reach, mx0, my0)) {return;}
  if (!grid.worldToMap(cx + reach, cy + reach, mx1, my1)) {return;}

  for (unsigned int j = my0; j <= my1; ++j) {
    for (unsigned int i = mx0; i <= mx1; ++i) {
      double wx, wy;
      grid.mapToWorld(i, j, wx, wy);
      const double d = std::hypot(wx - cx, wy - cy);
      if (d > reach) {continue;}

      const unsigned char cost = proxemic_cost_cell(d, radius, peak);
      const unsigned char old = grid.getCost(i, j);
      const unsigned char base =
        (old == nav2_costmap_2d::NO_INFORMATION) ? 0 : old;
      if (cost > base) {
        grid.setCost(i, j, cost);
      }
    }
  }
}

}  // namespace proxemics_layer
