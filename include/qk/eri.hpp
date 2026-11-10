// cpp-quantum-kernel - two-electron repulsion integrals (mu nu | lambda sigma).
//
// Algorithm: McMurchie-Davidson in "matrix form".  For every shell pair the
// Hermite expansion coefficients of all primitive pairs are precomputed once
// (ShellPair), with contraction coefficients, Cartesian normalisation and the
// 1/p factor folded in.  A shell quartet is then
//
//   (ab|cd) = sum_{ij in bra prims} E^{ab}_i . [ sum_{kl in ket prims} R(i,kl) . E^{cd}_kl^T ]
//
// i.e. two small dense contractions per primitive quartet, with the ket
// primitive sum hoisted before the bra contraction.
#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "qk/basis.hpp"

namespace qk {

/// Precomputed data for a shell pair (P, Q) with P >= Q.
struct ShellPair {
  std::size_t P = 0, Q = 0;
  int la = 0, lb = 0, L = 0;
  std::size_t nfa = 0, nfb = 0, nherm = 0;
  std::vector<double> p;                      ///< exponent sum per primitive pair
  std::vector<std::array<double, 3>> center;  ///< Gaussian product centre
  /// E[prim][fa*nfb+fb][hermite] scaled by c_a c_b K_ab N_a N_b / p.
  std::vector<double> E;
  double schwarz = 0.0;  ///< sqrt(max |(PQ|PQ)|)

  std::size_t n_primitive_pairs() const noexcept { return p.size(); }
  std::size_t n_functions() const noexcept { return nfa * nfb; }
};

inline std::size_t pair_index(std::size_t i, std::size_t j) noexcept {
  return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
}

class EriEngine {
 public:
  /// `primitive_tol`: primitive pairs whose overlap prefactor is below this
  /// are dropped (they cannot contribute above ~1e-15 Eh).
  explicit EriEngine(const BasisSet& basis, double primitive_tol = 1e-15);

  const BasisSet& basis() const noexcept { return *basis_; }

  /// All shell pairs P >= Q, stored at index pair_index(P, Q).
  const std::vector<ShellPair>& shell_pairs() const noexcept { return pairs_; }
  const ShellPair& pair(std::size_t P, std::size_t Q) const { return pairs_[pair_index(P, Q)]; }

  /// Largest block size nfa*nfb*nfc*nfd (for sizing caller buffers).
  std::size_t max_block_size() const noexcept { return max_block_; }
  double max_schwarz() const noexcept { return max_schwarz_; }

  /// Compute the (PQ|RS) block into out[a][b][c][d] (row-major,
  /// a in P, b in Q, c in R, d in S).  Thread-safe (thread-local scratch).
  void compute(const ShellPair& bra, const ShellPair& ket, double* out) const;

  /// Convenience: (PQ|RS) for arbitrary shell indices, any order.
  std::vector<double> compute_shells(std::size_t P, std::size_t Q, std::size_t R, std::size_t S) const;

 private:
  const BasisSet* basis_;
  std::vector<ShellPair> pairs_;
  std::size_t max_block_ = 0;
  double max_schwarz_ = 0.0;
};

}  // namespace qk
