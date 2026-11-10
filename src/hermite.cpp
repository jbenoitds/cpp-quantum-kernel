#include "qk/hermite.hpp"

#include <algorithm>
#include <stdexcept>

#include "qk/boys.hpp"

namespace qk {

const std::vector<HermiteIndex>& hermite_indices(int L) {
  constexpr int kMaxL = kBoysMaxOrder;
  static const std::vector<std::vector<HermiteIndex>> table = [] {
    std::vector<std::vector<HermiteIndex>> t(kMaxL + 1);
    for (int lmax = 0; lmax <= kMaxL; ++lmax) {
      for (int n = 0; n <= lmax; ++n)
        for (int a = n; a >= 0; --a)
          for (int b = n - a; b >= 0; --b) t[static_cast<std::size_t>(lmax)].push_back({a, b, n - a - b});
    }
    return t;
  }();
  if (L < 0 || L > kMaxL) throw std::out_of_range("qk::hermite_indices: L out of range");
  return table[static_cast<std::size_t>(L)];
}

void hermite_expansion_1d(int la, int lb, double p, double XPA, double XPB, double* E) {
  const int L = la + lb;
  const int nt = L + 1;
  const int nj = lb + 1;
  const std::size_t total = static_cast<std::size_t>((la + 1) * nj * nt);
  std::fill(E, E + total, 0.0);
  auto at = [&](int i, int j, int t) -> double& { return E[(i * nj + j) * nt + t]; };
  auto get = [&](int i, int j, int t) -> double {
    return (t < 0 || t > i + j) ? 0.0 : E[(i * nj + j) * nt + t];
  };
  const double oo2p = 0.5 / p;
  at(0, 0, 0) = 1.0;
  // Raise i with j = 0.
  for (int i = 0; i < la; ++i)
    for (int t = 0; t <= i + 1; ++t)
      at(i + 1, 0, t) = oo2p * get(i, 0, t - 1) + XPA * get(i, 0, t) + (t + 1) * get(i, 0, t + 1);
  // Raise j for every i.
  for (int i = 0; i <= la; ++i)
    for (int j = 0; j < lb; ++j)
      for (int t = 0; t <= i + j + 1; ++t)
        at(i, j + 1, t) = oo2p * get(i, j, t - 1) + XPB * get(i, j, t) + (t + 1) * get(i, j, t + 1);
}

void hermite_coulomb(int L, double alpha, double X, double Y, double Z, double prefactor, double* R, double* work) {
  const int d = L + 1;
  const std::size_t vol = static_cast<std::size_t>(d) * d * d;
  auto idx = [d](int t, int u, int v) { return (static_cast<std::size_t>(t) * d + u) * d + v; };

  double F[kBoysMaxOrder + 1];
  boys_function(L, alpha * (X * X + Y * Y + Z * Z), F);

  double* prev = work;
  double* cur = work + vol;
  // F[n] <- prefactor * (-2 alpha)^n * F_n
  double pw = prefactor;
  for (int n = 0; n <= L; ++n) {
    F[n] *= pw;
    pw *= -2.0 * alpha;
  }
  // Level n = L only has R^L_000.
  prev[0] = F[L];

  for (int n = L - 1; n >= 0; --n) {
    double* out = (n == 0) ? R : cur;
    const int top = L - n;
    for (int t = 0; t <= top; ++t) {
      for (int u = 0; u <= top - t; ++u) {
        for (int v = 0; v <= top - t - u; ++v) {
          double val;
          if (t > 0) {
            val = X * prev[idx(t - 1, u, v)];
            if (t > 1) val += (t - 1) * prev[idx(t - 2, u, v)];
          } else if (u > 0) {
            val = Y * prev[idx(0, u - 1, v)];
            if (u > 1) val += (u - 1) * prev[idx(0, u - 2, v)];
          } else if (v > 0) {
            val = Z * prev[idx(0, 0, v - 1)];
            if (v > 1) val += (v - 1) * prev[idx(0, 0, v - 2)];
          } else {
            val = F[n];
          }
          out[idx(t, u, v)] = val;
        }
      }
    }
    if (n > 0) std::swap(prev, cur);
  }
  if (L == 0) R[0] = prev[0];
}

}  // namespace qk
