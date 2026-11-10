#include "qk/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace qk {

namespace {
void check_same_shape(const Matrix& a, const Matrix& b, const char* what) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    throw std::invalid_argument(std::string("qk::Matrix shape mismatch in ") + what);
  }
}
}  // namespace

Matrix Matrix::identity(std::size_t n) {
  Matrix m(n, n);
  for (std::size_t i = 0; i < n; ++i) m(i, i) = 1.0;
  return m;
}

void Matrix::fill(double value) { std::fill(data_.begin(), data_.end(), value); }

Matrix& Matrix::operator+=(const Matrix& other) {
  check_same_shape(*this, other, "operator+=");
  for (std::size_t k = 0; k < data_.size(); ++k) data_[k] += other.data_[k];
  return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
  check_same_shape(*this, other, "operator-=");
  for (std::size_t k = 0; k < data_.size(); ++k) data_[k] -= other.data_[k];
  return *this;
}

Matrix& Matrix::operator*=(double s) {
  for (double& x : data_) x *= s;
  return *this;
}

Matrix operator+(Matrix a, const Matrix& b) { return a += b; }
Matrix operator-(Matrix a, const Matrix& b) { return a -= b; }
Matrix operator*(double s, Matrix a) { return a *= s; }

Matrix transpose(const Matrix& a) {
  Matrix t(a.cols(), a.rows());
  for (std::size_t i = 0; i < a.rows(); ++i)
    for (std::size_t j = 0; j < a.cols(); ++j) t(j, i) = a(i, j);
  return t;
}

Matrix matmul(const Matrix& a, const Matrix& b) {
  if (a.cols() != b.rows()) throw std::invalid_argument("qk::matmul: inner dimension mismatch");
  Matrix c(a.rows(), b.cols());
  const std::size_t n = b.cols();
  // i-k-j ordering: the innermost loop streams contiguous rows of B and C.
  for (std::size_t i = 0; i < a.rows(); ++i) {
    double* ci = c.row(i);
    const double* ai = a.row(i);
    for (std::size_t k = 0; k < a.cols(); ++k) {
      const double aik = ai[k];
      if (aik == 0.0) continue;
      const double* bk = b.row(k);
      for (std::size_t j = 0; j < n; ++j) ci[j] += aik * bk[j];
    }
  }
  return c;
}

Matrix matmul_tn(const Matrix& a, const Matrix& b) {
  if (a.rows() != b.rows()) throw std::invalid_argument("qk::matmul_tn: inner dimension mismatch");
  Matrix c(a.cols(), b.cols());
  const std::size_t n = b.cols();
  for (std::size_t k = 0; k < a.rows(); ++k) {
    const double* ak = a.row(k);
    const double* bk = b.row(k);
    for (std::size_t i = 0; i < a.cols(); ++i) {
      const double aki = ak[i];
      if (aki == 0.0) continue;
      double* ci = c.row(i);
      for (std::size_t j = 0; j < n; ++j) ci[j] += aki * bk[j];
    }
  }
  return c;
}

Matrix matmul_nt(const Matrix& a, const Matrix& b) {
  if (a.cols() != b.cols()) throw std::invalid_argument("qk::matmul_nt: inner dimension mismatch");
  Matrix c(a.rows(), b.rows());
  const std::size_t n = a.cols();
  for (std::size_t i = 0; i < a.rows(); ++i) {
    const double* ai = a.row(i);
    for (std::size_t j = 0; j < b.rows(); ++j) {
      const double* bj = b.row(j);
      double s = 0.0;
      for (std::size_t k = 0; k < n; ++k) s += ai[k] * bj[k];
      c(i, j) = s;
    }
  }
  return c;
}

double frobenius_dot(const Matrix& a, const Matrix& b) {
  check_same_shape(a, b, "frobenius_dot");
  const double* x = a.data();
  const double* y = b.data();
  double s = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) s += x[k] * y[k];
  return s;
}

double max_abs(const Matrix& a) {
  double m = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) m = std::max(m, std::abs(a.data()[k]));
  return m;
}

double rms(const Matrix& a) {
  if (a.size() == 0) return 0.0;
  return std::sqrt(frobenius_dot(a, a) / static_cast<double>(a.size()));
}

// ---------------------------------------------------------------------------
// Symmetric eigensolver: Householder reduction to tridiagonal form followed by
// the implicit QL algorithm with Wilkinson-type shifts (the classic
// EISPACK tred2/tql2 pair, in the public-domain form popularised by JAMA).
// The routines operate on the *transpose* of the textbook working matrix so
// that every O(n^3) inner loop walks contiguous memory in our row-major
// storage; on exit row k of V holds eigenvector k.
// ---------------------------------------------------------------------------
namespace {

void tridiagonalize(std::size_t n, Matrix& V, std::vector<double>& d, std::vector<double>& e) {
  for (std::size_t j = 0; j < n; ++j) d[j] = V(j, n - 1);

  for (std::size_t i = n - 1; i > 0; --i) {
    double scale = 0.0;
    double h = 0.0;
    for (std::size_t k = 0; k < i; ++k) scale += std::abs(d[k]);
    if (scale == 0.0) {
      e[i] = d[i - 1];
      for (std::size_t j = 0; j < i; ++j) {
        d[j] = V(j, i - 1);
        V(j, i) = 0.0;
        V(i, j) = 0.0;
      }
    } else {
      for (std::size_t k = 0; k < i; ++k) {
        d[k] /= scale;
        h += d[k] * d[k];
      }
      double f = d[i - 1];
      double g = std::sqrt(h);
      if (f > 0) g = -g;
      e[i] = scale * g;
      h -= f * g;
      d[i - 1] = f - g;
      for (std::size_t j = 0; j < i; ++j) e[j] = 0.0;

      for (std::size_t j = 0; j < i; ++j) {
        f = d[j];
        V(i, j) = f;
        g = e[j] + V(j, j) * f;
        for (std::size_t k = j + 1; k <= i - 1; ++k) {
          g += V(j, k) * d[k];
          e[k] += V(j, k) * f;
        }
        e[j] = g;
      }
      f = 0.0;
      for (std::size_t j = 0; j < i; ++j) {
        e[j] /= h;
        f += e[j] * d[j];
      }
      const double hh = f / (h + h);
      for (std::size_t j = 0; j < i; ++j) e[j] -= hh * d[j];
      for (std::size_t j = 0; j < i; ++j) {
        f = d[j];
        g = e[j];
        for (std::size_t k = j; k <= i - 1; ++k) V(j, k) -= (f * e[k] + g * d[k]);
        d[j] = V(j, i - 1);
        V(j, i) = 0.0;
      }
    }
    d[i] = h;
  }

  // Accumulate the Householder transformations.
  for (std::size_t i = 0; i + 1 < n; ++i) {
    V(i, n - 1) = V(i, i);
    V(i, i) = 1.0;
    const double h = d[i + 1];
    if (h != 0.0) {
      for (std::size_t k = 0; k <= i; ++k) d[k] = V(i + 1, k) / h;
      for (std::size_t j = 0; j <= i; ++j) {
        double g = 0.0;
        for (std::size_t k = 0; k <= i; ++k) g += V(i + 1, k) * V(j, k);
        for (std::size_t k = 0; k <= i; ++k) V(j, k) -= g * d[k];
      }
    }
    for (std::size_t k = 0; k <= i; ++k) V(i + 1, k) = 0.0;
  }
  for (std::size_t j = 0; j < n; ++j) {
    d[j] = V(j, n - 1);
    V(j, n - 1) = 0.0;
  }
  V(n - 1, n - 1) = 1.0;
  e[0] = 0.0;
}

void tridiagonal_ql(std::size_t n, Matrix& V, std::vector<double>& d, std::vector<double>& e) {
  for (std::size_t i = 1; i < n; ++i) e[i - 1] = e[i];
  e[n - 1] = 0.0;

  double f = 0.0;
  double tst1 = 0.0;
  const double eps = std::ldexp(1.0, -52);
  for (std::size_t l = 0; l < n; ++l) {
    tst1 = std::max(tst1, std::abs(d[l]) + std::abs(e[l]));
    std::size_t m = l;
    while (m < n) {
      if (std::abs(e[m]) <= eps * tst1) break;
      ++m;
    }
    if (m == n) m = n - 1;  // e[n-1] == 0 guarantees termination; defensive.

    if (m > l) {
      int iter = 0;
      do {
        if (++iter > 200) throw std::runtime_error("qk::eigh: QL iteration did not converge");
        double g = d[l];
        double p = (d[l + 1] - g) / (2.0 * e[l]);
        double r = std::hypot(p, 1.0);
        if (p < 0) r = -r;
        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        const double dl1 = d[l + 1];
        double h = g - d[l];
        for (std::size_t i = l + 2; i < n; ++i) d[i] -= h;
        f += h;

        p = d[m];
        double c = 1.0, c2 = 1.0, c3 = 1.0;
        const double el1 = e[l + 1];
        double s = 0.0, s2 = 0.0;
        for (std::size_t ii = m; ii-- > l;) {  // ii = m-1 down to l
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[ii];
          h = c * p;
          r = std::hypot(p, e[ii]);
          e[ii + 1] = s * r;
          s = e[ii] / r;
          c = p / r;
          p = c * d[ii] - s * g;
          d[ii + 1] = h + s * (c * g + s * d[ii]);
          for (std::size_t k = 0; k < n; ++k) {
            h = V(ii + 1, k);
            V(ii + 1, k) = s * V(ii, k) + c * h;
            V(ii, k) = c * V(ii, k) - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (std::abs(e[l]) > eps * tst1);
    }
    d[l] += f;
    e[l] = 0.0;
  }
}

}  // namespace

EigenSystem eigh(const Matrix& a) {
  if (a.rows() != a.cols()) throw std::invalid_argument("qk::eigh: matrix must be square");
  const std::size_t n = a.rows();
  EigenSystem result;
  if (n == 0) return result;
  if (n == 1) {
    result.values = {a(0, 0)};
    result.vectors = Matrix::identity(1);
    return result;
  }
  // Symmetrise defensively: rounding noise in Fock builds is not exactly symmetric.
  Matrix V(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) V(i, j) = 0.5 * (a(i, j) + a(j, i));

  std::vector<double> d(n), e(n);
  tridiagonalize(n, V, d, e);
  tridiagonal_ql(n, V, d, e);

  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return d[x] < d[y]; });

  result.values.resize(n);
  result.vectors = Matrix(n, n);
  for (std::size_t k = 0; k < n; ++k) {
    result.values[k] = d[order[k]];
    for (std::size_t i = 0; i < n; ++i) result.vectors(i, k) = V(order[k], i);
  }
  return result;
}

Matrix orthogonalizer(const Matrix& s, double lindep_tol, std::size_t* n_dropped) {
  const EigenSystem es = eigh(s);
  const std::size_t n = s.rows();
  std::size_t keep = 0;
  for (double v : es.values)
    if (v >= lindep_tol) ++keep;
  if (keep == 0) throw std::runtime_error("qk::orthogonalizer: overlap matrix has no usable eigenvalues");
  if (n_dropped) *n_dropped = n - keep;

  if (keep == n) {
    // Symmetric (Loewdin) orthogonalisation: X = U s^{-1/2} U^T.
    Matrix us(n, n);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t k = 0; k < n; ++k) us(i, k) = es.vectors(i, k) / std::sqrt(es.values[k]);
    return matmul_nt(us, es.vectors);
  }
  // Canonical orthogonalisation, discarding near-null eigenvectors.
  Matrix x(n, keep);
  const std::size_t first = n - keep;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < keep; ++k)
      x(i, k) = es.vectors(i, first + k) / std::sqrt(es.values[first + k]);
  return x;
}

std::vector<double> solve_linear(Matrix a, std::vector<double> b) {
  const std::size_t n = a.rows();
  if (a.cols() != n || b.size() != n) throw std::invalid_argument("qk::solve_linear: shape mismatch");
  double scale = max_abs(a);
  if (scale == 0.0) throw std::runtime_error("qk::solve_linear: singular matrix");
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t piv = col;
    for (std::size_t r = col + 1; r < n; ++r)
      if (std::abs(a(r, col)) > std::abs(a(piv, col))) piv = r;
    if (std::abs(a(piv, col)) <= 1e-14 * scale) throw std::runtime_error("qk::solve_linear: singular matrix");
    if (piv != col) {
      for (std::size_t j = 0; j < n; ++j) std::swap(a(col, j), a(piv, j));
      std::swap(b[col], b[piv]);
    }
    for (std::size_t r = col + 1; r < n; ++r) {
      const double factor = a(r, col) / a(col, col);
      if (factor == 0.0) continue;
      for (std::size_t j = col; j < n; ++j) a(r, j) -= factor * a(col, j);
      b[r] -= factor * b[col];
    }
  }
  std::vector<double> x(n);
  for (std::size_t i = n; i-- > 0;) {
    double s = b[i];
    for (std::size_t j = i + 1; j < n; ++j) s -= a(i, j) * x[j];
    x[i] = s / a(i, i);
  }
  return x;
}

}  // namespace qk
