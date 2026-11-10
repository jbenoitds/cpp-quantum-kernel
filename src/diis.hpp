// Internal: Pulay DIIS extrapolation shared by the 3D and 1D SCF solvers.
#pragma once

#include <algorithm>
#include <deque>
#include <stdexcept>
#include <vector>

#include "qk/linalg.hpp"

namespace qk::detail {

/// Pulay's direct inversion in the iterative subspace.
class Diis {
 public:
  explicit Diis(int max_size) : max_(static_cast<std::size_t>(max_size)) {}

  /// Store (F, e) and return the extrapolated Fock matrix.
  Matrix extrapolate(const Matrix& F, const Matrix& err) {
    if (max_ == 0) return F;
    focks_.push_back(F);
    errors_.push_back(err);
    if (focks_.size() > max_) {
      focks_.pop_front();
      errors_.pop_front();
    }
    while (focks_.size() >= 2) {
      const std::size_t m = focks_.size();
      Matrix B(m + 1, m + 1);
      double scale = 0.0;
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
          B(i, j) = B(j, i) = frobenius_dot(errors_[i], errors_[j]);
          scale = std::max(scale, B(i, i));
        }
      if (scale <= 0.0) return F;
      for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) B(i, j) /= scale;
        B(i, m) = B(m, i) = -1.0;
      }
      std::vector<double> rhs(m + 1, 0.0);
      rhs[m] = -1.0;
      try {
        const std::vector<double> c = solve_linear(B, rhs);
        Matrix out(F.rows(), F.cols());
        for (std::size_t i = 0; i < m; ++i) {
          const double* f = focks_[i].data();
          double* o = out.data();
          for (std::size_t k = 0; k < out.size(); ++k) o[k] += c[i] * f[k];
        }
        return out;
      } catch (const std::runtime_error&) {
        focks_.pop_front();  // ill-conditioned: drop the oldest vector
        errors_.pop_front();
      }
    }
    return F;
  }

 private:
  std::size_t max_;
  std::deque<Matrix> focks_, errors_;
};

}  // namespace qk::detail
