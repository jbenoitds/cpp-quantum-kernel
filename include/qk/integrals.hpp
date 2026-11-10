// cpp-quantum-kernel - one-electron integrals over contracted Cartesian Gaussians.
#pragma once

#include <array>

#include "qk/basis.hpp"
#include "qk/linalg.hpp"
#include "qk/molecule.hpp"

namespace qk {

struct PointCharge {
  double charge;
  std::array<double, 3> xyz;  ///< bohr
};

/// S_{mu nu} = <mu|nu>
Matrix overlap_matrix(const BasisSet& basis);
/// T_{mu nu} = <mu| -1/2 nabla^2 |nu>
Matrix kinetic_matrix(const BasisSet& basis);
/// V_{mu nu} = <mu| -sum_C Z_C / |r - R_C| |nu>
Matrix nuclear_attraction_matrix(const BasisSet& basis, const std::vector<PointCharge>& charges);
Matrix nuclear_attraction_matrix(const BasisSet& basis, const Molecule& mol);
/// H = T + V
Matrix core_hamiltonian(const BasisSet& basis, const Molecule& mol);
/// Electric dipole integrals <mu| (r - origin)_k |nu>, k = x, y, z.
std::array<Matrix, 3> dipole_matrices(const BasisSet& basis, const std::array<double, 3>& origin = {0.0, 0.0, 0.0});

}  // namespace qk
