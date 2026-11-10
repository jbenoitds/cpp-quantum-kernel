#include <cmath>
#include <vector>

#include "qk/boys.hpp"
#include "qk_test.hpp"

namespace {
// Independent check: F_n(T) = int_0^1 t^{2n} exp(-T t^2) dt by composite
// 10-point Gauss-Legendre on 200 panels (long double).
double boys_quadrature(int n, double T) {
  static const long double xg[5] = {0.1488743389816312108848260L, 0.4333953941292471907992659L,
                                    0.6794095682990244062343274L, 0.8650633666889845107320967L,
                                    0.9739065285171717200779640L};
  static const long double wg[5] = {0.2955242247147528701738930L, 0.2692667193099963550912269L,
                                    0.2190863625159820439955349L, 0.1494513491505805931457763L,
                                    0.0666713443086881375935688L};
  const int panels = 200;
  const long double h = 1.0L / panels;
  long double sum = 0.0L;
  for (int p = 0; p < panels; ++p) {
    const long double mid = (p + 0.5L) * h;
    for (int k = 0; k < 5; ++k)
      for (int s = -1; s <= 1; s += 2) {
        const long double t = mid + s * xg[k] * h / 2;
        sum += wg[k] * std::pow(t, 2 * n) * std::exp(-T * t * t);
      }
  }
  return static_cast<double>(sum * h / 2);
}
}  // namespace

QK_TEST("boys: F_n(0) = 1/(2n+1)") {
  std::vector<double> F(qk::kBoysMaxOrder + 1);
  qk::boys_function(qk::kBoysMaxOrder, 0.0, F.data());
  for (int n = 0; n <= qk::kBoysMaxOrder; ++n) QK_CHECK_NEAR(F[n], 1.0 / (2 * n + 1), 1e-15);
}

QK_TEST("boys: agrees with Gauss-Legendre quadrature") {
  std::vector<double> F(qk::kBoysMaxOrder + 1);
  for (double T : {1e-9, 0.013, 0.37, 1.0, 2.55, 7.77, 15.0, 29.96, 39.99, 40.0, 41.3, 60.0, 120.0}) {
    qk::boys_function(qk::kBoysMaxOrder, T, F.data());
    for (int n = 0; n <= qk::kBoysMaxOrder; n += 3) {
      const double ref = boys_quadrature(n, T);
      QK_CHECK(std::abs(F[n] - ref) <= 2e-14 * ref + 1e-300);
    }
  }
}

QK_TEST("boys: tabulated path vs series reference on a dense sweep") {
  std::vector<double> F(qk::kBoysMaxOrder + 1);
  double worst = 0.0;
  for (int k = 0; k <= 4400; ++k) {
    const double T = 0.0113 * k;  // crosses every grid cell and the T = 40 switch
    qk::boys_function(qk::kBoysMaxOrder, T, F.data());
    for (int n = 0; n <= qk::kBoysMaxOrder; ++n) {
      const double ref = qk::boys_reference(n, T);
      worst = std::max(worst, std::abs(F[n] - ref) / ref);
    }
  }
  QK_CHECK(worst < 1e-13);
}

QK_TEST("boys: argument validation") {
  double F[4];
  QK_CHECK_THROWS(qk::boys_function(qk::kBoysMaxOrder + 1, 1.0, F));
  QK_CHECK_THROWS(qk::boys_reference(-1, 1.0));
}
