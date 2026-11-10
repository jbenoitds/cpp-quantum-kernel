#include <cmath>
#include <random>
#include <sstream>
#include <string>

#include "qk/qk.hpp"
#include "qk_test.hpp"

#ifndef QK_TEST_DATA_DIR
#define QK_TEST_DATA_DIR "tests/data"
#endif

using namespace qk;

namespace {
Molecule water() {
  return Molecule::from_string("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692");
}
}  // namespace

QK_TEST("integrals: Szabo & Ostlund H2/STO-3G integrals (Table 3.6, R = 1.4 bohr)") {
  const Molecule h2 = Molecule::from_string("H 0 0 0; H 0 0 1.4", /*in_bohr=*/true);
  const BasisSet b(h2, "sto-3g");
  const Matrix S = overlap_matrix(b), T = kinetic_matrix(b);
  QK_CHECK_NEAR(S(0, 0), 1.0, 1e-12);
  QK_CHECK_NEAR(S(0, 1), 0.6593, 5e-5);
  QK_CHECK_NEAR(T(0, 0), 0.7600, 5e-5);
  QK_CHECK_NEAR(T(0, 1), 0.2365, 5e-5);
  const Matrix V1 = nuclear_attraction_matrix(b, std::vector<PointCharge>{{1.0, {0, 0, 0}}});
  QK_CHECK_NEAR(V1(0, 0), -1.2266, 5e-5);
  QK_CHECK_NEAR(V1(0, 1), -0.5974, 5e-5);
  QK_CHECK_NEAR(V1(1, 1), -0.6538, 5e-5);
  EriEngine eng(b);
  EriTensor eri(eng);
  QK_CHECK_NEAR(eri(0, 0, 0, 0), 0.7746, 5e-5);
  QK_CHECK_NEAR(eri(0, 0, 1, 1), 0.5697, 5e-5);
  QK_CHECK_NEAR(eri(1, 0, 0, 0), 0.4441, 5e-5);
  QK_CHECK_NEAR(eri(1, 0, 1, 0), 0.2970, 5e-5);
}

QK_TEST("integrals: every Cartesian component is normalised (s..g)") {
  const Molecule w = water();
  for (const BasisSet& b : {BasisSet(w, "6-31g*"), BasisSet::from_g94_file(w, QK_TEST_DATA_DIR "/spdf.g94")}) {
    const Matrix S = overlap_matrix(b);
    for (std::size_t i = 0; i < b.n_functions(); ++i) QK_CHECK_NEAR(S(i, i), 1.0, 1e-13);
  }
}

QK_TEST("integrals: ERI 8-fold permutational symmetry up to (gg|gg)") {
  const Molecule w = water();
  const BasisSet b = BasisSet::from_g94_file(w, QK_TEST_DATA_DIR "/spdf.g94");
  const EriEngine eng(b);
  std::mt19937 rng(1234);
  std::uniform_int_distribution<std::size_t> pick(0, b.n_shells() - 1);
  for (int trial = 0; trial < 25; ++trial) {
    const std::size_t P = pick(rng), Q = pick(rng), R = pick(rng), S = pick(rng);
    const auto pqrs = eng.compute_shells(P, Q, R, S);
    const auto qprs = eng.compute_shells(Q, P, R, S);
    const auto rspq = eng.compute_shells(R, S, P, Q);
    const auto srqp = eng.compute_shells(S, R, Q, P);
    const std::size_t np = b.shell(P).size(), nq = b.shell(Q).size(), nr = b.shell(R).size(), ns = b.shell(S).size();
    double worst = 0.0;
    for (std::size_t a = 0; a < np; ++a)
      for (std::size_t q = 0; q < nq; ++q)
        for (std::size_t r = 0; r < nr; ++r)
          for (std::size_t s = 0; s < ns; ++s) {
            const double v = pqrs[((a * nq + q) * nr + r) * ns + s];
            worst = std::max(worst, std::abs(v - qprs[((q * np + a) * nr + r) * ns + s]));
            worst = std::max(worst, std::abs(v - rspq[((r * ns + s) * np + a) * nq + q]));
            worst = std::max(worst, std::abs(v - srqp[((s * nr + r) * nq + q) * np + a]));
          }
    QK_CHECK(worst < 1e-12);
  }
}

QK_TEST("integrals: p and d ERIs equal derivatives of s ERIs w.r.t. the centre") {
  // x_A exp(-a r_A^2) = (1/2a) d/dA_x exp(-a r_A^2);  x_A y_A exp(..) = (1/4a^2) d2/dA_x dA_y exp(..)
  const double a = 0.83;
  const std::array<double, 3> B{0.3, -0.4, 1.1}, C{-0.7, 0.2, 0.5}, D{0.1, 0.9, -0.6};
  auto ssss = [&](double ax, double ay) {
    BasisSet bs;
    bs.add_shell(make_shell(0, {ax, ay, 0.2}, {a}, {1.0}));
    bs.add_shell(make_shell(0, B, {1.3}, {1.0}));
    bs.add_shell(make_shell(0, C, {0.6}, {1.0}));
    bs.add_shell(make_shell(0, D, {2.1}, {1.0}));
    EriEngine e(bs);
    return e.compute_shells(0, 1, 2, 3)[0];
  };
  auto first = [&](int l) {
    BasisSet bs;
    bs.add_shell(make_shell(l, {0.0, 0.0, 0.2}, {a}, {1.0}));
    bs.add_shell(make_shell(0, B, {1.3}, {1.0}));
    bs.add_shell(make_shell(0, C, {0.6}, {1.0}));
    bs.add_shell(make_shell(0, D, {2.1}, {1.0}));
    EriEngine e(bs);
    return e.compute_shells(0, 1, 2, 3);
  };
  const double h = 1e-4;
  const double dsdx = (ssss(h, 0) - ssss(-h, 0)) / (2 * h);
  const double px = first(1)[0];  // component order x, y, z
  QK_CHECK_NEAR(px, primitive_norm(a, 1, 0, 0) / primitive_norm(a, 0, 0, 0) * dsdx / (2 * a), 1e-8);

  const double k = 1e-3;
  const double d2 = (ssss(k, k) - ssss(k, -k) - ssss(-k, k) + ssss(-k, -k)) / (4 * k * k);
  const double dxy = first(2)[1];  // order xx, xy, xz, yy, yz, zz
  QK_CHECK_NEAR(dxy, primitive_norm(a, 1, 1, 0) / primitive_norm(a, 0, 0, 0) * d2 / (4 * a * a), 1e-7);
}

QK_TEST("integrals: translation invariance and Schwarz inequality") {
  const Molecule w = water();
  std::vector<Atom> shifted = w.atoms();
  for (auto& at : shifted) {
    at.xyz[0] += 3.7;
    at.xyz[1] -= 1.25;
    at.xyz[2] += 0.5;
  }
  const BasisSet b1(w, "6-31g*"), b2(Molecule(shifted), "6-31g*");
  const EriEngine e1(b1), e2(b2);
  const EriTensor t1(e1, 0.0), t2(e2, 0.0);
  double worst = 0.0;
  for (std::size_t k = 0; k < t1.data().size(); ++k) worst = std::max(worst, std::abs(t1.data()[k] - t2.data()[k]));
  QK_CHECK(worst < 1e-12);
  QK_CHECK(max_abs(overlap_matrix(b1) - overlap_matrix(b2)) < 1e-14);
  QK_CHECK(max_abs(kinetic_matrix(b1) - kinetic_matrix(b2)) < 1e-13);

  const std::size_t n = b1.n_functions();
  bool ok = true;
  for (std::size_t i = 0; i < n && ok; ++i)
    for (std::size_t j = 0; j <= i; ++j)
      for (std::size_t k = 0; k < n; ++k)
        for (std::size_t l = 0; l <= k; ++l)
          if (std::abs(t1(i, j, k, l)) > std::sqrt(t1(i, j, i, j) * t1(k, l, k, l)) * (1 + 1e-12) + 1e-15) ok = false;
  QK_CHECK(ok);
}

QK_TEST("integrals: dipole integrals are origin-consistent") {
  const Molecule w = water();
  const BasisSet b(w, "6-31g");
  const Matrix S = overlap_matrix(b);
  const auto m0 = dipole_matrices(b);
  const auto m1 = dipole_matrices(b, {0.5, -1.0, 2.0});
  // <mu| x - Ox |nu> = <mu|x|nu> - Ox <mu|nu>
  QK_CHECK(max_abs(m1[0] - (m0[0] - 0.5 * S)) < 1e-13);
  QK_CHECK(max_abs(m1[1] - (m0[1] + 1.0 * S)) < 1e-13);
  QK_CHECK(max_abs(m1[2] - (m0[2] - 2.0 * S)) < 1e-13);
}

QK_TEST("basis: Gaussian94 parser (SP shells, D exponents, scale factors, comments)") {
  const std::string g94 =
      "! comment line\n"
      "****\n"
      "H     0\n"
      "S   1   1.24\n"           // scale factor: exponent * 1.24^2
      "  1.0D+00   1.0\n"
      "****\n"
      "C     0\n"
      "SP  2   1.00\n"
      "  2.9412494D+00  -0.09996723   0.15591627\n"
      "  0.6834831      0.39951283   0.60768372\n"
      "****\n";
  std::istringstream in(g94);
  const Molecule m = Molecule::from_string("C 0 0 0; H 0 0 1.1");
  const BasisSet b = BasisSet::from_g94(m, in);
  QK_CHECK(b.n_shells() == 3);
  QK_CHECK(b.n_functions() == 1 + 3 + 1);
  QK_CHECK(b.shell(0).l == 0 && b.shell(1).l == 1 && b.shell(2).l == 0);
  QK_CHECK_NEAR(b.shell(2).exponents[0], 1.24 * 1.24, 1e-14);
  QK_CHECK_NEAR(b.shell(1).exponents[0], 2.9412494, 1e-14);
  QK_CHECK(b.function_atom(0) == 0 && b.function_atom(3) == 0 && b.function_atom(4) == 1);

  std::istringstream in2(g94);
  QK_CHECK_THROWS(BasisSet::from_g94(Molecule::from_string("N 0 0 0"), in2));  // element missing
  QK_CHECK_THROWS(BasisSet(m, "cc-pv5z-not-built-in"));
  QK_CHECK(BasisSet(m, "6-31G(d)").n_functions() == BasisSet(m, "6-31g*").n_functions());
  QK_CHECK(BasisSet(m, "6-31g*").n_functions() == 15 + 2);  // Cartesian d: 6 functions
}

QK_TEST("molecule: XYZ parsing, units and nuclear repulsion") {
  std::istringstream xyz("3\nwater\nO 0 0 0\nH 0 0 1\nH 0 1 0\n");
  const Molecule m = Molecule::from_xyz(xyz);
  QK_CHECK(m.size() == 3);
  QK_CHECK(m.n_electrons() == 10);
  QK_CHECK_NEAR(m.atoms()[1].xyz[2], kAngstromToBohr, 1e-14);
  const double r = kAngstromToBohr;
  QK_CHECK_NEAR(m.nuclear_repulsion(), 8.0 / r + 8.0 / r + 1.0 / (std::sqrt(2.0) * r), 1e-12);
  QK_CHECK(atomic_number("cl") == 17);
  QK_CHECK(atomic_number("He2") == 2);
  QK_CHECK_THROWS(atomic_number("Xx"));
  std::istringstream bad("2\nshort\nH 0 0 0\n");
  QK_CHECK_THROWS(Molecule::from_xyz(bad));
}
