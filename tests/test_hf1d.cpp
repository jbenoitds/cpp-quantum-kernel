#include <cmath>
#include <numeric>

#include "qk/hf1d.hpp"
#include "qk_test.hpp"

using namespace qk::hf1d;

// Reference energies from the independent NumPy implementation in
// tools/hf1d_reference.py (identical grids).

QK_TEST("hf1d: soft-Coulomb He, H2 and Be match the independent reference") {
  System he;
  he.nuclei = {{2.0, 0.0}};
  he.n_electrons = 2;
  const Result r = solve(he);
  QK_CHECK(r.converged);
  QK_CHECK_NEAR(r.energy, -2.224209553009, 1e-9);

  System h2;
  h2.nuclei = {{1.0, -0.8}, {1.0, 0.8}};
  h2.n_electrons = 2;
  QK_CHECK_NEAR(solve(h2).energy, -1.425372462179, 1e-9);

  System be;
  be.nuclei = {{4.0, 0.0}};
  be.n_electrons = 4;
  Grid g;
  g.x_min = -25;
  g.x_max = 25;
  g.n_points = 301;
  QK_CHECK_NEAR(solve(be, g).energy, -6.739449622423, 1e-9);
}

QK_TEST("hf1d: non-interacting electrons in a harmonic trap are exact") {
  // 6 electrons, omega = 1: E = 2 * (0.5 + 1.5 + 2.5) = 9.
  System s;
  s.n_electrons = 6;
  s.harmonic_omega = 1.0;
  s.interaction = 0.0;
  Grid g;
  g.x_min = -10;
  g.x_max = 10;
  g.n_points = 161;
  const Result r = solve(s, g);
  QK_CHECK(r.converged);
  QK_CHECK_NEAR(r.energy, 9.0, 1e-10);
  for (int k = 0; k < 5; ++k) QK_CHECK_NEAR(r.orbital_energies[static_cast<std::size_t>(k)], k + 0.5, 1e-10);
  // Virial theorem for the harmonic oscillator: <T> = E / 2.
  QK_CHECK_NEAR(r.kinetic_energy, 4.5, 1e-9);
}

QK_TEST("hf1d: density normalisation and grid convergence") {
  System he;
  he.nuclei = {{2.0, 0.0}};
  const Result coarse = solve(he);
  const double dx = coarse.grid[1] - coarse.grid[0];
  const double N = std::accumulate(coarse.density.begin(), coarse.density.end(), 0.0) * dx;
  QK_CHECK_NEAR(N, 2.0, 1e-12);
  Grid fine;
  fine.x_min = -24;
  fine.x_max = 24;
  fine.n_points = 321;  // larger box, finer spacing
  QK_CHECK_NEAR(solve(he, fine).energy, coarse.energy, 1e-9);  // sinc-DVR converges spectrally
}

QK_TEST("hf1d: symmetric molecule has a symmetric density; invalid input throws") {
  System h2;
  h2.nuclei = {{1.0, -0.7}, {1.0, 0.7}};
  const Result r = solve(h2);
  const std::size_t n = r.density.size();
  double asym = 0.0;
  for (std::size_t i = 0; i < n; ++i) asym = std::max(asym, std::abs(r.density[i] - r.density[n - 1 - i]));
  QK_CHECK(asym < 1e-8);
  System odd;
  odd.nuclei = {{1.0, 0.0}};
  odd.n_electrons = 1;
  QK_CHECK_THROWS(solve(odd));
}
