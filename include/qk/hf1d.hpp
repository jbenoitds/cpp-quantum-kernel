// cpp-quantum-kernel - closed-shell Hartree-Fock for 1D model systems on a grid.
//
// Electrons on a line interact through the soft-Coulomb potential
//   w(x) = lambda / sqrt(x^2 + a^2),
// the standard regularised model for 1D atoms and molecules (Eberly et al.).
// The kinetic energy uses the Colbert-Miller sinc discrete variable
// representation [J. Chem. Phys. 96, 1982 (1992)], which is spectrally
// accurate; Hartree and exchange are evaluated exactly on the grid.  Because
// the interaction is translation invariant only the N-vector w(|i-j| dx) is
// stored, not an N x N kernel.
#pragma once

#include <cstddef>
#include <iosfwd>
#include <vector>

namespace qk::hf1d {

struct Nucleus {
  double charge;
  double position;
};

struct System {
  std::vector<Nucleus> nuclei;
  int n_electrons = 2;          ///< must be even (closed shell)
  double softening = 1.0;       ///< a in 1/sqrt(x^2 + a^2)
  double interaction = 1.0;     ///< lambda scaling of e-e repulsion (0 = non-interacting)
  double harmonic_omega = 0.0;  ///< optional confinement 1/2 omega^2 x^2
};

struct Grid {
  double x_min = -20.0;
  double x_max = 20.0;
  std::size_t n_points = 201;
  double spacing() const { return (x_max - x_min) / static_cast<double>(n_points - 1); }
};

struct Controls {
  int max_iterations = 200;
  double energy_tol = 1e-11;
  double error_tol = 1e-8;  ///< rms of [F, D]
  int diis_subspace = 6;
  std::ostream* log = nullptr;
};

struct Result {
  bool converged = false;
  int iterations = 0;
  double energy = 0.0;               ///< total energy incl. nuclear repulsion
  double nuclear_repulsion = 0.0;
  double kinetic_energy = 0.0;
  std::vector<double> orbital_energies;  ///< all occupied + first few virtual
  std::vector<double> grid;              ///< x_i
  std::vector<std::vector<double>> orbitals;  ///< occupied phi_k(x_i), int phi^2 dx = 1
  std::vector<double> density;           ///< n(x_i), int n dx = N
};

Result solve(const System& system, const Grid& grid = {}, const Controls& controls = {});

}  // namespace qk::hf1d
