// cpp-quantum-kernel - Coulomb/exchange (J, K) builds.
//
// Two strategies with identical results:
//   * EriTensor   : in-core, 8-fold packed storage (N^4/8 doubles), built once.
//   * direct      : integrals recomputed every iteration, never stored.
//                   Schwarz + density-weighted screening, O(N^2) memory.
// Both loop over symmetry-unique quartets only and are OpenMP-parallel with
// thread-private J/K accumulators (no atomics).
#pragma once

#include <cstddef>
#include <vector>

#include "qk/eri.hpp"
#include "qk/linalg.hpp"

namespace qk {

class EriTensor {
 public:
  /// Compute all unique AO integrals; quartets with Q_PQ Q_RS < schwarz_tol
  /// are skipped (stored as zero).
  explicit EriTensor(const EriEngine& engine, double schwarz_tol = 1e-12);

  static std::size_t packed_size(std::size_t nbf) noexcept {
    const std::size_t np = nbf * (nbf + 1) / 2;
    return np * (np + 1) / 2;
  }
  static double memory_mb(std::size_t nbf) noexcept { return static_cast<double>(packed_size(nbf) * sizeof(double)) / (1024.0 * 1024.0); }

  std::size_t n_functions() const noexcept { return nbf_; }
  double operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const noexcept {
    return data_[pair_index(pair_index(i, j), pair_index(k, l))];
  }
  const std::vector<double>& data() const noexcept { return data_; }
  std::size_t quartets_computed() const noexcept { return computed_; }
  std::size_t quartets_screened() const noexcept { return screened_; }

 private:
  std::size_t nbf_ = 0;
  std::vector<double> data_;
  std::size_t computed_ = 0;
  std::size_t screened_ = 0;
};

struct JK {
  Matrix J;  ///< J_{mn} = sum_{ls} (mn|ls) D_{ls}
  Matrix K;  ///< K_{mn} = sum_{ls} (ml|ns) D_{ls}
};

/// J and K from stored integrals. D must be symmetric.
JK build_jk(const EriTensor& eri, const Matrix& D);

struct DirectStats {
  std::size_t quartets_computed = 0;
  std::size_t quartets_screened = 0;
};

/// Integral-direct J and K.  A quartet is skipped when
///   Q_PQ * Q_RS * max(relevant |D| shell blocks) < tol.
JK build_jk_direct(const EriEngine& engine, const Matrix& D, double tol = 1e-12, DirectStats* stats = nullptr);

}  // namespace qk
