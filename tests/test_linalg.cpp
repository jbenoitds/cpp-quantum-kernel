#include <cmath>
#include <random>

#include "qk/linalg.hpp"
#include "qk_test.hpp"

using qk::Matrix;

namespace {
Matrix random_symmetric(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Matrix a(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j <= i; ++j) a(i, j) = a(j, i) = u(rng);
  return a;
}

void check_eigensystem(const Matrix& a, const qk::EigenSystem& es, double tol) {
  const std::size_t n = a.rows();
  // Residual A v = lambda v, orthonormality, ascending order.
  const Matrix av = qk::matmul(a, es.vectors);
  double res = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < n; ++k) res = std::max(res, std::abs(av(i, k) - es.values[k] * es.vectors(i, k)));
  QK_CHECK(res < tol);
  const Matrix vtv = qk::matmul_tn(es.vectors, es.vectors);
  QK_CHECK(qk::max_abs(vtv - Matrix::identity(n)) < tol);
  for (std::size_t k = 1; k < n; ++k) QK_CHECK(es.values[k - 1] <= es.values[k]);
}
}  // namespace

QK_TEST("linalg: eigh on random symmetric matrices") {
  for (std::size_t n : {1u, 2u, 3u, 7u, 40u, 120u}) {
    const Matrix a = random_symmetric(n, static_cast<unsigned>(n) * 17u + 3u);
    check_eigensystem(a, qk::eigh(a), 1e-11 * static_cast<double>(n));
  }
}

QK_TEST("linalg: eigh with degenerate and analytic spectra") {
  // Identity (fully degenerate) and a block-diagonal matrix with repeated eigenvalues.
  check_eigensystem(Matrix::identity(6), qk::eigh(Matrix::identity(6)), 1e-13);
  Matrix b(4, 4);
  b(0, 0) = b(1, 1) = 2.0;
  b(0, 1) = b(1, 0) = 1.0;  // eigenvalues 1, 3
  b(2, 2) = b(3, 3) = 2.0;
  b(2, 3) = b(3, 2) = 1.0;  // eigenvalues 1, 3
  const auto es = qk::eigh(b);
  QK_CHECK_NEAR(es.values[0], 1.0, 1e-14);
  QK_CHECK_NEAR(es.values[1], 1.0, 1e-14);
  QK_CHECK_NEAR(es.values[2], 3.0, 1e-14);
  QK_CHECK_NEAR(es.values[3], 3.0, 1e-14);
  check_eigensystem(b, es, 1e-13);
  // 1D Laplacian: eigenvalues 2 - 2 cos(k pi / (n+1)).
  const std::size_t n = 50;
  Matrix lap(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    lap(i, i) = 2.0;
    if (i + 1 < n) lap(i, i + 1) = lap(i + 1, i) = -1.0;
  }
  const auto el = qk::eigh(lap);
  for (std::size_t k = 0; k < n; ++k)
    QK_CHECK_NEAR(el.values[k], 2.0 - 2.0 * std::cos((k + 1) * 3.14159265358979323846 / (n + 1)), 1e-12);
}

QK_TEST("linalg: GEMM variants agree") {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Matrix a(5, 3), b(3, 4), c(5, 4);
  for (std::size_t k = 0; k < a.size(); ++k) a.data()[k] = u(rng);
  for (std::size_t k = 0; k < b.size(); ++k) b.data()[k] = u(rng);
  for (std::size_t i = 0; i < 5; ++i)
    for (std::size_t j = 0; j < 4; ++j)
      for (std::size_t k = 0; k < 3; ++k) c(i, j) += a(i, k) * b(k, j);
  QK_CHECK(qk::max_abs(qk::matmul(a, b) - c) < 1e-14);
  QK_CHECK(qk::max_abs(qk::matmul_tn(qk::transpose(a), b) - c) < 1e-14);
  QK_CHECK(qk::max_abs(qk::matmul_nt(a, qk::transpose(b)) - c) < 1e-14);
  QK_CHECK_THROWS(qk::matmul(a, a));
}

QK_TEST("linalg: orthogonalizer gives X^T S X = 1, also for near-singular S") {
  Matrix s = random_symmetric(10, 99);
  s = qk::matmul_tn(s, s);
  for (std::size_t i = 0; i < 10; ++i) s(i, i) += 0.5;  // SPD
  std::size_t dropped = 99;
  Matrix x = qk::orthogonalizer(s, 1e-8, &dropped);
  QK_CHECK(dropped == 0);
  QK_CHECK(qk::max_abs(qk::matmul_tn(x, qk::matmul(s, x)) - Matrix::identity(10)) < 1e-11);

  // Two identical basis functions: one combination must be dropped.
  Matrix sing(3, 3);
  sing(0, 0) = sing(1, 1) = sing(2, 2) = 1.0;
  sing(0, 1) = sing(1, 0) = 1.0;
  sing(0, 2) = sing(2, 0) = sing(1, 2) = sing(2, 1) = 0.3;
  x = qk::orthogonalizer(sing, 1e-8, &dropped);
  QK_CHECK(dropped == 1);
  QK_CHECK(x.cols() == 2);
  QK_CHECK(qk::max_abs(qk::matmul_tn(x, qk::matmul(sing, x)) - Matrix::identity(2)) < 1e-12);
}

QK_TEST("linalg: pivoted linear solve") {
  Matrix a(3, 3);
  a(0, 0) = 0.0; a(0, 1) = 2.0; a(0, 2) = 1.0;  // zero pivot forces a row swap
  a(1, 0) = 1.0; a(1, 1) = -1.0; a(1, 2) = 0.0;
  a(2, 0) = 3.0; a(2, 1) = 0.0; a(2, 2) = 1.0;
  const std::vector<double> x = qk::solve_linear(a, {5.0, -1.0, 4.0});
  QK_CHECK_NEAR(x[0], 1.0, 1e-14);
  QK_CHECK_NEAR(x[1], 2.0, 1e-14);
  QK_CHECK_NEAR(x[2], 1.0, 1e-14);
  QK_CHECK_THROWS(qk::solve_linear(Matrix(2, 2, 1.0), {1.0, 1.0}));
}
