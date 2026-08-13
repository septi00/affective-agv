#include "proxemics_layer/proxemic_cost.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

using proxemics_layer::proxemic_cost;
using proxemics_layer::proxemic_cost_cell;
using proxemics_layer::kMaxProxemicCost;

int main()
{
  // Peak at the centre.
  assert(std::abs(proxemic_cost(0.0, 0.8, 1.0) - 1.0) < 1e-9);

  // Monotonically non-increasing with distance.
  double prev = 1.0;
  for (double d = 0.0; d <= 2.0; d += 0.1) {
    double c = proxemic_cost(d, 0.8, 1.0);
    assert(c <= prev + 1e-9);
    prev = c;
  }

  // At d == radius, cost ~= e^-2 of peak.
  assert(std::abs(proxemic_cost(0.8, 0.8, 1.0) - std::exp(-2.0)) < 1e-6);

  // peak weight scales the bump linearly.
  assert(std::abs(proxemic_cost(0.5, 1.0, 0.6)
                  - 0.6 * proxemic_cost(0.5, 1.0, 1.0)) < 1e-9);

  // Larger radius -> higher cost at the same distance (wider bump).
  assert(proxemic_cost(0.8, 1.6, 1.0) > proxemic_cost(0.8, 0.8, 1.0));

  // Centre cell hits the soft cap, and the cap stays below Nav2's
  // inscribed (253) / lethal (254) values.
  assert(proxemic_cost_cell(0.0, 0.8, 1.0) == kMaxProxemicCost);
  assert(kMaxProxemicCost < 253);

  // Print the curve so you can eyeball the falloff.
  std::printf("person: radius=0.8 m, peak=1.0\n");
  std::printf("  dist     norm-cost   cell\n");
  for (double d = 0.0; d <= 2.0; d += 0.2) {
    std::printf("  %.1f m      %.3f      %3u\n",
                d, proxemic_cost(d, 0.8, 1.0),
                proxemic_cost_cell(d, 0.8, 1.0));
  }
  std::printf("\nall assertions passed\n");
  return 0;
}
