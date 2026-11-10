// cpp-quantum-kernel - closed-shell restricted Hartree-Fock and MP2.
#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>
#include <vector>

#include "qk/basis.hpp"
#include "qk/fock.hpp"
#include "qk/linalg.hpp"
#include "qk/molecule.hpp"

namespace qk {

enum class EriStrategy {
  Auto,    ///< in-core if the packed tensor fits in max_memory_mb, else direct
  InCore,  ///< compute and store all unique integrals once
  Direct,  ///< recompute integrals each iteration (O(N^2) memory)
};

struct ScfOptions {
  int max_iterations = 100;
  double energy_tol = 1e-10;    ///< |E_n - E_{n-1}|
  double gradient_tol = 1e-8;   ///< rms of the orthogonalised commutator X^T(FDS - SDF)X
  int diis_subspace = 8;        ///< 0 disables DIIS
  EriStrategy eri = EriStrategy::Auto;
  double max_memory_mb = 2048;  ///< limit for the in-core tensor under Auto
  double schwarz_tol = 1e-12;
  double lindep_tol = 1e-8;     ///< overlap eigenvalues below this are dropped
  bool incremental_fock = true; ///< direct only: build G from the density change
  int full_rebuild_period = 8;  ///< direct + incremental: periodic full rebuild
  std::ostream* log = nullptr;  ///< iteration log (nullptr = silent)
};

struct ScfIteration {
  int iteration;
  double energy;
  double delta_e;
  double error;
  std::size_t quartets;  ///< shell quartets computed (direct mode)
};

struct ScfResult {
  bool converged = false;
  int iterations = 0;
  double energy = 0.0;             ///< total energy (Eh)
  double electronic_energy = 0.0;
  double nuclear_repulsion = 0.0;
  std::size_t n_occupied = 0;
  std::size_t n_dropped = 0;       ///< linearly dependent combinations removed
  bool used_incore = false;
  std::vector<double> orbital_energies;
  Matrix C;  ///< MO coefficients (AO x MO)
  Matrix D;  ///< C_occ C_occ^T  (the total density is 2 D)
  Matrix F, H, S;
  std::vector<ScfIteration> history;
  double seconds_integrals = 0.0;
  double seconds_total = 0.0;
};

/// Run RHF. If `eri` is supplied it is used (and EriStrategy is ignored).
ScfResult run_rhf(const Molecule& mol, const BasisSet& basis, const ScfOptions& options = {},
                  const EriTensor* eri = nullptr);

/// Total dipole moment (nuclear + electronic) about the origin, atomic units.
std::array<double, 3> dipole_moment(const Molecule& mol, const BasisSet& basis, const Matrix& D);

/// Mulliken atomic charges.
std::vector<double> mulliken_charges(const Molecule& mol, const BasisSet& basis, const Matrix& D, const Matrix& S);

struct Mp2Result {
  double correlation_energy = 0.0;
  double same_spin = 0.0;
  double opposite_spin = 0.0;
};

/// Closed-shell MP2 correlation energy from converged RHF orbitals.
Mp2Result mp2_energy(const ScfResult& scf, const EriTensor& eri, std::size_t n_frozen_core = 0);

}  // namespace qk
