#include "qk/boys.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "qk/constants.hpp"

namespace qk {

namespace {

constexpr double kGridStep = 0.1;
constexpr double kInvGridStep = 10.0;
constexpr double kTaylorCutoff = 40.0;
constexpr int kTaylorTerms = 8;  // k = 0..7
constexpr int kTableOrders = kBoysMaxOrder + kTaylorTerms;
constexpr int kGridPoints = static_cast<int>(kTaylorCutoff * kInvGridStep) + 2;

struct BoysTable {
  std::vector<double> values;  // [grid point][order], order-contiguous

  BoysTable() : values(static_cast<std::size_t>(kGridPoints) * kTableOrders) {
    for (int g = 0; g < kGridPoints; ++g) {
      const double T = g * kGridStep;
      double* row = &values[static_cast<std::size_t>(g) * kTableOrders];
      // Highest order from the series, the rest by downward recursion
      // (which is numerically stable).
      row[kTableOrders - 1] = boys_reference(kTableOrders - 1, T);
      const double e = std::exp(-T);
      for (int n = kTableOrders - 2; n >= 0; --n) row[n] = (2.0 * T * row[n + 1] + e) / (2 * n + 1);
    }
  }
};

const BoysTable& table() {
  static const BoysTable t;  // thread-safe initialisation (C++11 magic statics)
  return t;
}

}  // namespace

double boys_reference(int n, double T) {
  if (n < 0) throw std::invalid_argument("qk::boys_reference: negative order");
  if (T < 0) throw std::invalid_argument("qk::boys_reference: negative argument");
  if (T > 100.0) {
    // Asymptotic form; the exp(-T) correction is below double precision.
    double f = 0.5 * std::sqrt(kPi / T);
    for (int k = 0; k < n; ++k) f *= (2 * k + 1) / (2.0 * T);
    return f;
  }
  // F_n(T) = exp(-T) * sum_i (2T)^i / [(2n+1)(2n+3)...(2n+2i+1)]
  // All terms are positive, so the series is free of cancellation.
  long double term = 1.0L / (2 * n + 1);
  long double sum = term;
  const long double twoT = 2.0L * T;
  for (int i = 1; i < 10000; ++i) {
    term *= twoT / (2 * n + 2 * i + 1);
    sum += term;
    if (term < 1e-21L * sum) break;
  }
  return static_cast<double>(std::exp(-static_cast<long double>(T)) * sum);
}

void boys_function(int nmax, double T, double* F) {
  if (nmax < 0 || nmax > kBoysMaxOrder) throw std::out_of_range("qk::boys_function: order out of range");
  if (T < kTaylorCutoff) {
    const int g = static_cast<int>(T * kInvGridStep + 0.5);
    const double dT = g * kGridStep - T;  // |dT| <= 0.05
    const double* row = &table().values[static_cast<std::size_t>(g) * kTableOrders];
    // F_n(T) = sum_k F_{n+k}(T0) dT^k / k!   (Horner form)
    double s = row[nmax + kTaylorTerms - 1];
    for (int k = kTaylorTerms - 1; k >= 1; --k) s = row[nmax + k - 1] + s * dT / k;
    F[nmax] = s;
    if (nmax > 0) {
      const double e = std::exp(-T);
      for (int n = nmax - 1; n >= 0; --n) F[n] = (2.0 * T * F[n + 1] + e) / (2 * n + 1);
    }
  } else {
    const double e = std::exp(-T);
    F[0] = 0.5 * std::sqrt(kPi / T) * std::erf(std::sqrt(T));
    const double inv2T = 0.5 / T;
    for (int n = 0; n < nmax; ++n) F[n + 1] = ((2 * n + 1) * F[n] - e) * inv2T;
  }
}

}  // namespace qk
