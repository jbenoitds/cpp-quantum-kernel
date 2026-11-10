// cpp-quantum-kernel - small dense linear-algebra layer.
//
// A deliberately dependency-free, row-major dense matrix plus the handful of
// kernels a Hartree-Fock code needs: GEMM variants, a symmetric eigensolver
// (Householder tridiagonalisation + implicit QL), the orthogonaliser X with
// X^T S X = 1, and a pivoted linear solve for DIIS.
#pragma once

#include <cstddef>
#include <vector>

namespace qk {

class Matrix {
 public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t cols, double value = 0.0)
      : rows_(rows), cols_(cols), data_(rows * cols, value) {}

  static Matrix identity(std::size_t n);

  std::size_t rows() const noexcept { return rows_; }
  std::size_t cols() const noexcept { return cols_; }
  std::size_t size() const noexcept { return data_.size(); }
  bool empty() const noexcept { return data_.empty(); }

  double& operator()(std::size_t i, std::size_t j) noexcept { return data_[i * cols_ + j]; }
  double operator()(std::size_t i, std::size_t j) const noexcept { return data_[i * cols_ + j]; }

  double* data() noexcept { return data_.data(); }
  const double* data() const noexcept { return data_.data(); }
  double* row(std::size_t i) noexcept { return data_.data() + i * cols_; }
  const double* row(std::size_t i) const noexcept { return data_.data() + i * cols_; }

  void fill(double value);

  Matrix& operator+=(const Matrix& other);
  Matrix& operator-=(const Matrix& other);
  Matrix& operator*=(double s);

 private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<double> data_;
};

Matrix operator+(Matrix a, const Matrix& b);
Matrix operator-(Matrix a, const Matrix& b);
Matrix operator*(double s, Matrix a);

Matrix transpose(const Matrix& a);
/// C = A * B
Matrix matmul(const Matrix& a, const Matrix& b);
/// C = A^T * B
Matrix matmul_tn(const Matrix& a, const Matrix& b);
/// C = A * B^T
Matrix matmul_nt(const Matrix& a, const Matrix& b);
/// sum_ij A_ij B_ij
double frobenius_dot(const Matrix& a, const Matrix& b);
double max_abs(const Matrix& a);
double rms(const Matrix& a);

struct EigenSystem {
  std::vector<double> values;  ///< ascending
  Matrix vectors;              ///< column k is the eigenvector of values[k]
};

/// Eigen-decomposition of a real symmetric matrix (only symmetry is assumed;
/// the full matrix is read). O(n^3), numerically stable.
EigenSystem eigh(const Matrix& a);

/// Orthogonaliser X with X^T S X = 1.  Uses symmetric (Loewdin) S^{-1/2} when
/// S is well conditioned, otherwise canonical orthogonalisation dropping
/// eigenvectors with eigenvalue < lindep_tol (X is then n x m, m < n).
Matrix orthogonalizer(const Matrix& s, double lindep_tol = 1e-8, std::size_t* n_dropped = nullptr);

/// Solve A x = b with partial pivoting. Throws std::runtime_error if singular.
std::vector<double> solve_linear(Matrix a, std::vector<double> b);

}  // namespace qk
