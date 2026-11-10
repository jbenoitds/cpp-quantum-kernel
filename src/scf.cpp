#include "qk/scf.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>

#include "diis.hpp"
#include "qk/eri.hpp"
#include "qk/integrals.hpp"
#include "qk/parallel.hpp"

namespace qk {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

/// Diagonalise F in the orthogonal basis; returns (eps, C) with C = X C'.
void diagonalize_fock(const Matrix& F, const Matrix& X, std::vector<double>& eps, Matrix& C) {
  const Matrix Fp = matmul_tn(X, matmul(F, X));
  EigenSystem es = eigh(Fp);
  eps = std::move(es.values);
  C = matmul(X, es.vectors);
}

Matrix density_from(const Matrix& C, std::size_t nocc) {
  const std::size_t n = C.rows();
  Matrix D(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j <= i; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < nocc; ++k) s += C(i, k) * C(j, k);
      D(i, j) = D(j, i) = s;
    }
  return D;
}

Matrix two_electron_part(const JK& jk) {
  Matrix G = jk.J;
  G *= 2.0;
  G -= jk.K;
  return G;
}

}  // namespace

ScfResult run_rhf(const Molecule& mol, const BasisSet& basis, const ScfOptions& opt, const EriTensor* eri_in) {
  const auto t_start = Clock::now();
  ScfResult res;
  const int nelec = mol.n_electrons();
  if (nelec <= 0) throw std::invalid_argument("qk::run_rhf: no electrons");
  if (nelec % 2 != 0) throw std::invalid_argument("qk::run_rhf: RHF requires an even number of electrons");
  res.n_occupied = static_cast<std::size_t>(nelec / 2);
  res.nuclear_repulsion = mol.nuclear_repulsion();

  const std::size_t n = basis.n_functions();
  res.S = overlap_matrix(basis);
  res.H = core_hamiltonian(basis, mol);
  const Matrix X = orthogonalizer(res.S, opt.lindep_tol, &res.n_dropped);
  if (res.n_occupied > X.cols()) throw std::invalid_argument("qk::run_rhf: basis too small for the electron count");

  // Two-electron integral strategy.
  const auto t_int = Clock::now();
  std::unique_ptr<EriEngine> engine;
  std::unique_ptr<EriTensor> own_tensor;
  const EriTensor* eri = eri_in;
  if (!eri) {
    engine = std::make_unique<EriEngine>(basis);
    const bool incore = opt.eri == EriStrategy::InCore ||
                        (opt.eri == EriStrategy::Auto && EriTensor::memory_mb(n) <= opt.max_memory_mb);
    if (incore) {
      own_tensor = std::make_unique<EriTensor>(*engine, opt.schwarz_tol);
      eri = own_tensor.get();
    }
  }
  res.used_incore = (eri != nullptr);
  res.seconds_integrals = seconds_since(t_int);

  std::ostream* log = opt.log;
  if (log) {
    *log << "RHF: " << n << " basis functions, " << res.n_occupied << " doubly occupied orbitals, "
         << (res.used_incore ? "in-core" : "direct") << " ERIs, " << max_threads() << " thread(s)\n";
    if (res.n_dropped) *log << "     removed " << res.n_dropped << " near-linear-dependent combinations\n";
    *log << "  iter        energy (Eh)          dE        rms[F,D]\n";
  }

  // Core-Hamiltonian guess.
  std::vector<double> eps;
  Matrix C;
  diagonalize_fock(res.H, X, eps, C);
  Matrix D = density_from(C, res.n_occupied);

  detail::Diis diis(opt.diis_subspace);
  Matrix G(n, n), D_last(n, n);
  double e_old = 0.0;
  Matrix F;

  for (int it = 1; it <= opt.max_iterations; ++it) {
    std::size_t quartets = 0;
    if (eri) {
      G = two_electron_part(build_jk(*eri, D));
    } else {
      DirectStats st;
      const bool full = !opt.incremental_fock || it == 1 ||
                        (opt.full_rebuild_period > 0 && (it - 1) % opt.full_rebuild_period == 0);
      if (full) {
        G = two_electron_part(build_jk_direct(*engine, D, opt.schwarz_tol, &st));
      } else {
        // G is linear in D: G(D_n) = G(D_{n-1}) + G(D_n - D_{n-1}); the
        // density change shrinks as SCF converges, so screening bites harder.
        G += two_electron_part(build_jk_direct(*engine, D - D_last, opt.schwarz_tol, &st));
      }
      quartets = st.quartets_computed;
      D_last = D;
    }
    F = res.H + G;
    const double e_elec = frobenius_dot(D, res.H + F);
    const double energy = e_elec + res.nuclear_repulsion;

    // Orbital gradient: X^T (F D S - S D F) X
    const Matrix FDS = matmul(F, matmul(D, res.S));
    const Matrix err = matmul_tn(X, matmul(FDS - transpose(FDS), X));
    const double err_rms = rms(err);
    const double de = energy - e_old;

    res.history.push_back({it, energy, de, err_rms, quartets});
    if (log) {
      std::ostringstream line;  // format locally: never disturb the caller's stream state
      line << std::setw(6) << it << std::setw(22) << std::fixed << std::setprecision(12) << energy << std::setw(12)
           << std::scientific << std::setprecision(2) << de << std::setw(12) << err_rms << "\n";
      *log << line.str();
    }

    res.iterations = it;
    res.electronic_energy = e_elec;
    res.energy = energy;
    if (it > 1 && std::abs(de) < opt.energy_tol && err_rms < opt.gradient_tol) {
      res.converged = true;
      break;
    }
    e_old = energy;

    const Matrix Fx = diis.extrapolate(F, err);
    diagonalize_fock(Fx, X, eps, C);
    D = density_from(C, res.n_occupied);
  }

  // Canonical orbitals of the final (non-extrapolated) Fock matrix.
  diagonalize_fock(F, X, eps, C);
  res.orbital_energies = std::move(eps);
  res.C = std::move(C);
  res.D = density_from(res.C, res.n_occupied);
  res.F = std::move(F);
  res.seconds_total = seconds_since(t_start);
  if (log) *log << (res.converged ? "RHF converged" : "RHF NOT converged") << " in " << res.iterations << " iterations\n";
  return res;
}

std::array<double, 3> dipole_moment(const Molecule& mol, const BasisSet& basis, const Matrix& D) {
  const auto M = dipole_matrices(basis);
  std::array<double, 3> mu{};
  for (int k = 0; k < 3; ++k) {
    double nuc = 0.0;
    for (const Atom& a : mol.atoms()) nuc += a.Z * a.xyz[static_cast<std::size_t>(k)];
    mu[static_cast<std::size_t>(k)] = nuc - 2.0 * frobenius_dot(D, M[static_cast<std::size_t>(k)]);
  }
  return mu;
}

std::vector<double> mulliken_charges(const Molecule& mol, const BasisSet& basis, const Matrix& D, const Matrix& S) {
  std::vector<double> q(mol.size());
  for (std::size_t a = 0; a < mol.size(); ++a) q[a] = mol.atoms()[a].Z;
  const Matrix DS = matmul(D, S);
  for (std::size_t mu = 0; mu < basis.n_functions(); ++mu) q[basis.function_atom(mu)] -= 2.0 * DS(mu, mu);
  return q;
}

// ---------------------------------------------------------------------------
// MP2: (ia|jb) by two half-transformations, O(N^5).
// ---------------------------------------------------------------------------

Mp2Result mp2_energy(const ScfResult& scf, const EriTensor& eri, std::size_t n_frozen) {
  const std::size_t n = eri.n_functions();
  const std::size_t nmo = scf.C.cols();
  const std::size_t nocc = scf.n_occupied;
  if (n_frozen >= nocc) throw std::invalid_argument("qk::mp2_energy: too many frozen orbitals");
  const std::size_t no = nocc - n_frozen;
  const std::size_t nv = nmo - nocc;
  Mp2Result r;
  if (nv == 0) return r;

  Matrix Co(n, no), Cv(n, nv);
  for (std::size_t mu = 0; mu < n; ++mu) {
    for (std::size_t i = 0; i < no; ++i) Co(mu, i) = scf.C(mu, n_frozen + i);
    for (std::size_t a = 0; a < nv; ++a) Cv(mu, a) = scf.C(mu, nocc + a);
  }

  // Step 1: half[ls][i][a] = sum_{mn} C_mi (mn|ls) C_na  for every AO pair l>=s.
  const std::size_t npair = n * (n + 1) / 2;
  const std::size_t ov = no * nv;
  std::vector<double> half(npair * ov);
  const int np = static_cast<int>(npair);
#pragma omp parallel
  {
    Matrix M(n, n);
#pragma omp for schedule(dynamic, 4)
    for (int ils = 0; ils < np; ++ils) {
      const std::size_t ls = static_cast<std::size_t>(ils);
      std::size_t l = static_cast<std::size_t>((std::sqrt(8.0 * static_cast<double>(ls) + 1.0) - 1.0) / 2.0);
      while (l * (l + 1) / 2 > ls) --l;
      while ((l + 1) * (l + 2) / 2 <= ls) ++l;
      const std::size_t s = ls - l * (l + 1) / 2;
      for (std::size_t m = 0; m < n; ++m)
        for (std::size_t nn = 0; nn <= m; ++nn) M(m, nn) = M(nn, m) = eri(m, nn, l, s);
      const Matrix h = matmul_tn(Co, matmul(M, Cv));
      std::copy(h.data(), h.data() + ov, half.data() + ls * ov);
    }
  }

  // Step 2: (ia|jb) = sum_{ls} C_lj half[ls][i][a] C_sb.
  std::vector<double> iajb(ov * ov);
  const int nov = static_cast<int>(ov);
#pragma omp parallel
  {
    Matrix M(n, n);
#pragma omp for schedule(dynamic, 4)
    for (int iia = 0; iia < nov; ++iia) {
      const std::size_t ia = static_cast<std::size_t>(iia);
      for (std::size_t l = 0, ls = 0; l < n; ++l)
        for (std::size_t s = 0; s <= l; ++s, ++ls) M(l, s) = M(s, l) = half[ls * ov + ia];
      const Matrix t = matmul_tn(Co, matmul(M, Cv));
      std::copy(t.data(), t.data() + ov, iajb.data() + ia * ov);
    }
  }

  const auto& e = scf.orbital_energies;
  double os = 0.0, ss = 0.0;
  for (std::size_t i = 0; i < no; ++i)
    for (std::size_t a = 0; a < nv; ++a)
      for (std::size_t j = 0; j < no; ++j)
        for (std::size_t b = 0; b < nv; ++b) {
          const double v = iajb[(i * nv + a) * ov + j * nv + b];
          const double x = iajb[(i * nv + b) * ov + j * nv + a];
          const double denom = e[n_frozen + i] + e[n_frozen + j] - e[nocc + a] - e[nocc + b];
          os += v * v / denom;
          ss += v * (v - x) / denom;
        }
  r.opposite_spin = os;
  r.same_spin = ss;
  r.correlation_energy = os + ss;
  return r;
}

}  // namespace qk
