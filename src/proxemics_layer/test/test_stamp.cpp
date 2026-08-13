#include <gtest/gtest.h>

#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/cost_values.hpp"

#include "proxemics_layer/proxemic_stamp.hpp"
#include "proxemics_layer/proxemic_cost.hpp"

using nav2_costmap_2d::Costmap2D;
using nav2_costmap_2d::FREE_SPACE;
using nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;

namespace
{
// 10 x 10 m grid, 0.05 m cells, origin (0,0), initialized free.
Costmap2D makeGrid()
{
  return Costmap2D(200, 200, 0.05, 0.0, 0.0, FREE_SPACE);
}
}  // namespace

TEST(StampCircle, PeakAtCentre)
{
  auto grid = makeGrid();
  proxemics_layer::stampCircle(grid, 5.0, 5.0, 0.8, 1.0, 2.0);

  unsigned int mx, my;
  ASSERT_TRUE(grid.worldToMap(5.0, 5.0, mx, my));
  EXPECT_GE(grid.getCost(mx, my), proxemics_layer::kMaxProxemicCost - 3);
}

TEST(StampCircle, DecaysWithDistance)
{
  auto grid = makeGrid();
  proxemics_layer::stampCircle(grid, 5.0, 5.0, 0.8, 1.0, 2.0);

  unsigned int cx, cy, ex, ey;
  grid.worldToMap(5.0, 5.0, cx, cy);
  grid.worldToMap(5.5, 5.0, ex, ey);   // 0.5 m from centre
  EXPECT_GT(grid.getCost(cx, cy), grid.getCost(ex, ey));
}

TEST(StampCircle, StaysSoft)
{
  auto grid = makeGrid();
  proxemics_layer::stampCircle(grid, 5.0, 5.0, 0.8, 1.0, 2.0);

  for (unsigned int j = 0; j < grid.getSizeInCellsY(); ++j) {
    for (unsigned int i = 0; i < grid.getSizeInCellsX(); ++i) {
      EXPECT_LT(grid.getCost(i, j), INSCRIBED_INFLATED_OBSTACLE);
    }
  }
}

TEST(StampCircle, FarFieldUntouched)
{
  auto grid = makeGrid();
  proxemics_layer::stampCircle(grid, 5.0, 5.0, 0.8, 1.0, 2.0);

  unsigned int fx, fy;
  grid.worldToMap(1.0, 1.0, fx, fy);   // well beyond reach
  EXPECT_EQ(grid.getCost(fx, fy), FREE_SPACE);
}

TEST(StampCircle, HigherPeakStampsMore)
{
  auto lo = makeGrid();
  auto hi = makeGrid();
  proxemics_layer::stampCircle(lo, 5.0, 5.0, 0.8, 0.4, 2.0);
  proxemics_layer::stampCircle(hi, 5.0, 5.0, 0.8, 0.9, 2.0);

  unsigned int mx, my;
  lo.worldToMap(5.0, 5.0, mx, my);
  EXPECT_GT(hi.getCost(mx, my), lo.getCost(mx, my));
}
