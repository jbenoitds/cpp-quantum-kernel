#include <algorithm>
#include <cmath>

#include "qk/qk.hpp"
#include "qk_test.hpp"

#ifndef QK_TEST_DATA_DIR
#define QK_TEST_DATA_DIR "tests/data"
#endif

using namespace qk;

namespace {
// Geometry of T. D. Crawford's programming projects #3/#4 (bohr).
Molecule crawford_water() {
  return Molecule::from_string(
      "O 0.000000000000 -0.143225816552 0.000000000000;"
      "H 1.638036840407 1.136548822547 -0.000000000000;"
      "H -1.638036840407 1.136548822547 -0.000000000000",
      /*in_bohr=*/true);
}
Molecule water() { return Molecule::from_string("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"); }
}  // namespace

QK_TEST("scf: H2/STO-3G reproduces Szabo & Ostlund") {
  const Molecule h2 = Molecule::from_string("H 0 0 0; H 0 0 1.4", true);
  const ScfResult r = run_rhf(h2, BasisSet(h2, "sto-3g"));
  QK_CHECK(r.converged);
  QK_CHECK_NEAR(r.energy, -1.1167, 5e-5);
  QK_CHECK_NEAR(r.electronic_energy, -1.8310, 5e-5);
  QK_CHECK_NEAR(r.orbital_energies[0], -0.5782, 5e-5);
  QK_CHECK_NEAR(r.orbital_energies[1], 0.6703, 5e-5);
}

QK_TEST("scf: HeH+ with Szabo's custom STO-3G exponents (zeta_He = 2.0925)") {
  // Szabo & Ostlund section 3.5.2: R = 1.4632 bohr, E = -2.860662 Eh.
  const Molecule m({{2, {0, 0, 0}}, {1, {0, 0, 1.4632}}}, /*charge=*/1);
  const std::vector<double> e1 = {2.227660, 0.405771, 0.109818};
  const std::vector<double> c = {0.154329, 0.535328, 0.444635};
  auto scaled = [&](double zeta) {
    std::vector<double> e;
    for (double x : e1) e.push_back(x * zeta * zeta);
    return e;
  };
  BasisSet b;
  b.add_shell(make_shell(0, m.atoms()[0].xyz, scaled(2.0925), c, 0));
  b.add_shell(make_shell(0, m.atoms()[1].xyz, scaled(1.24), c, 1));
  const ScfResult r = run_rhf(m, b);
  QK_CHECK(r.converged);
  // Szabo's Appendix-B program evaluates erf with the Abramowitz-Stegun 7.1.26
  // rational approximation (|error| ~ 1.5e-7), which shifts its published
  // energies by ~3e-6 Eh; exact integrals give -2.8606587.
  QK_CHECK_NEAR(r.energy, -2.860662, 1e-5);
  QK_CHECK_NEAR(r.electronic_energy, -4.227529, 1e-5);
}

QK_TEST("scf: Crawford water/STO-3G energy, MP2, dipole and Mulliken charges") {
  const Molecule m = crawford_water();
  const BasisSet b(m, "sto-3g");
  QK_CHECK_NEAR(m.nuclear_repulsion(), 8.002367061810450, 1e-12);
  const EriEngine eng(b);
  const EriTensor eri(eng);
  const ScfResult r = run_rhf(m, b, {}, &eri);
  QK_CHECK(r.converged);
  QK_CHECK_NEAR(r.energy, -74.942079928192, 1e-9);
  QK_CHECK_NEAR(mp2_energy(r, eri).correlation_energy, -0.049149636120, 1e-9);
  const auto mu = dipole_moment(m, b, r.D);
  QK_CHECK_NEAR(mu[0], 0.0, 1e-8);
  QK_CHECK_NEAR(mu[1], 0.603521296525, 1e-7);
  const auto q = mulliken_charges(m, b, r.D, r.S);
  QK_CHECK_NEAR(q[0], -0.253146052405, 1e-7);
  QK_CHECK_NEAR(q[1], 0.126573026202, 1e-7);
}

QK_TEST("scf: energies match PySCF 2.6.2 (cart=True) for 6-31G and 6-31G*") {
  struct Case {
    const char* geom;
    const char* basis;
    double energy;
  };
  const Case cases[] = {
      {"O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", "6-31g*", -76.0105049883},
      {"N 0 0 0.1173; H 0 0.9377 -0.2737; H 0.8121 -0.4689 -0.2737; H -0.8121 -0.4689 -0.2737", "6-31g", -56.1603459272},
      {"C 0 0 0; H 0.6287 0.6287 0.6287; H -0.6287 -0.6287 0.6287; H -0.6287 0.6287 -0.6287; H 0.6287 -0.6287 -0.6287",
       "6-31g*", -40.1950949408},
      {"H 0 0 0; Cl 0 0 1.2746", "6-31g*", -460.0599300282},
  };
  for (const Case& c : cases) {
    const Molecule m = Molecule::from_string(c.geom);
    const ScfResult r = run_rhf(m, BasisSet(m, c.basis));
    QK_CHECK(r.converged);
    QK_CHECK_NEAR(r.energy, c.energy, 1e-8);
  }
}

QK_TEST("scf: high angular momentum basis (s..g) matches PySCF") {
  const Molecule m = water();
  const BasisSet b = BasisSet::from_g94_file(m, QK_TEST_DATA_DIR "/spdf.g94");
  const ScfResult r = run_rhf(m, b);
  QK_CHECK(r.converged);
  QK_CHECK_NEAR(r.energy, -74.6176064385, 1e-8);
}

QK_TEST("scf: MP2 all-electron and frozen-core match PySCF") {
  const Molecule m = water();
  const BasisSet b(m, "6-31g*");
  const EriEngine eng(b);
  const EriTensor eri(eng);
  const ScfResult r = run_rhf(m, b, {}, &eri);
  QK_CHECK_NEAR(mp2_energy(r, eri).correlation_energy, -0.188521600925, 1e-9);
  const Mp2Result fc = mp2_energy(r, eri, 1);
  QK_CHECK_NEAR(fc.correlation_energy, -0.186119540118, 1e-9);
  QK_CHECK_NEAR(fc.same_spin + fc.opposite_spin, fc.correlation_energy, 1e-14);
  QK_CHECK_THROWS(mp2_energy(r, eri, 5));
}

QK_TEST("scf: direct, incremental-direct and in-core give identical energies") {
  const Molecule m = water();
  const BasisSet b(m, "6-31g*");
  ScfOptions o;
  o.eri = EriStrategy::InCore;
  const ScfResult incore = run_rhf(m, b, o);
  o.eri = EriStrategy::Direct;
  o.incremental_fock = false;
  const ScfResult direct = run_rhf(m, b, o);
  o.incremental_fock = true;
  const ScfResult incr = run_rhf(m, b, o);
  QK_CHECK(incore.used_incore && !direct.used_incore);
  QK_CHECK_NEAR(direct.energy, incore.energy, 1e-10);
  QK_CHECK_NEAR(incr.energy, incore.energy, 1e-10);
}

QK_TEST("scf: incremental Fock builds screen more quartets as SCF converges") {
  const Molecule m = Molecule::from_string(
      "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692;"
      "O 6 0 0.1173; H 6 0.7572 -0.4692; H 6 -0.7572 -0.4692;"
      "O 0 6 0.1173; H 0 6.7572 -0.4692; H 0 5.2428 -0.4692");
  ScfOptions o;
  o.eri = EriStrategy::Direct;
  const ScfResult r = run_rhf(m, BasisSet(m, "6-31g*"), o);
  QK_CHECK(r.converged);
  std::size_t fewest = r.history.front().quartets;
  for (const auto& h : r.history) fewest = std::min(fewest, h.quartets);
  QK_CHECK(static_cast<double>(fewest) < 0.75 * static_cast<double>(r.history.front().quartets));
}

QK_TEST("scf: J/K builders agree (in-core vs direct) for an arbitrary density") {
  const Molecule m = water();
  const BasisSet b(m, "6-31g*");
  const EriEngine eng(b);
  const EriTensor eri(eng, 0.0);
  const std::size_t n = b.n_functions();
  Matrix D(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j <= i; ++j) D(i, j) = D(j, i) = std::sin(1.0 + i + 2.0 * j) / (1.0 + i + j);
  const JK a = build_jk(eri, D);
  const JK c = build_jk_direct(eng, D, 0.0);
  QK_CHECK(max_abs(a.J - c.J) < 1e-12);
  QK_CHECK(max_abs(a.K - c.K) < 1e-12);
  // Brute-force reference from the definitions.
  double worst = 0.0;
  for (std::size_t mu = 0; mu < n; mu += 3)
    for (std::size_t nu = 0; nu < n; nu += 2) {
      double j = 0.0, k = 0.0;
      for (std::size_t l = 0; l < n; ++l)
        for (std::size_t s = 0; s < n; ++s) {
          j += eri(mu, nu, l, s) * D(l, s);
          k += eri(mu, l, nu, s) * D(l, s);
        }
      worst = std::max({worst, std::abs(j - a.J(mu, nu)), std::abs(k - a.K(mu, nu))});
    }
  QK_CHECK(worst < 1e-12);
}

QK_TEST("scf: thread count does not change the result") {
  const Molecule m = water();
  const BasisSet b(m, "6-31g");
  const int nt = max_threads();
  set_num_threads(1);
  const double e1 = run_rhf(m, b).energy;
  set_num_threads(nt > 1 ? nt : 2);
  const double e2 = run_rhf(m, b).energy;
  set_num_threads(nt);
  QK_CHECK_NEAR(e1, e2, 1e-11);
}

QK_TEST("scf: input validation") {
  const Molecule oh = Molecule::from_string("O 0 0 0; H 0 0 0.97");  // 9 electrons
  QK_CHECK_THROWS(run_rhf(oh, BasisSet(oh, "sto-3g")));
  Molecule ohm = oh;
  ohm.set_charge(-1);
  QK_CHECK(run_rhf(ohm, BasisSet(ohm, "sto-3g")).converged);
}
