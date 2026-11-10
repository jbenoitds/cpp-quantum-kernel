#include "qk/hf1d.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>

#include "diis.hpp"
#include "qk/constants.hpp"
#include "qk/linalg.hpp"

namespace qk::hf1d {

namespace {

Matrix occupied_density(const Matrix& C, std::size_t nocc) {
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

std::size_t absdiff(std::size_t i, std::size_t j) { return i > j ? i - j : j - i; }

}  // namespace

Result solve(const System& sys, const Grid& grid, const Controls& ctl) {
  if (sys.n_electrons <= 0 || sys.n_electrons % 2 != 0)
    throw std::invalid_argument("qk::hf1d::solve: closed shell requires a positive even electron count");
  if (grid.n_points < 3 || !(grid.x_max > grid.x_min)) throw std::invalid_argument("qk::hf1d::solve: bad grid");
  if (!(sys.softening > 0.0)) throw std::invalid_argument("qk::hf1d::solve: softening must be positive");

  const std::size_t N = grid.n_points;
  const std::size_t nocc = static_cast<std::size_t>(sys.n_electrons / 2);
  if (nocc > N) throw std::invalid_argument("qk::hf1d::solve: grid too small");
  const double dx = grid.spacing();
  const double a2 = sys.softening * sys.softening;

  Result res;
  res.grid.resize(N);
  for (std::size_t i = 0; i < N; ++i) res.grid[i] = grid.x_min + static_cast<double>(i) * dx;

  // Translation-invariant interaction: only w(k dx), k = 0..N-1, is stored.
  std::vector<double> w(N);
  for (std::size_t k = 0; k < N; ++k) {
    const double r = static_cast<double>(k) * dx;
    w[k] = sys.interaction / std::sqrt(r * r + a2);
  }

  // One-electron Hamiltonian: sinc-DVR kinetic energy + local external potential.
  Matrix T(N, N), h(N, N);
  const double tpref = 1.0 / (2.0 * dx * dx);
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      if (i == j) {
        T(i, j) = tpref * kPi * kPi / 3.0;
      } else {
        const double d = static_cast<double>(absdiff(i, j));
        T(i, j) = tpref * 2.0 * (((i + j) % 2 == 0) ? 1.0 : -1.0) / (d * d);
      }
    }
  h = T;
  for (std::size_t i = 0; i < N; ++i) {
    double v = 0.5 * sys.harmonic_omega * sys.harmonic_omega * res.grid[i] * res.grid[i];
    for (const Nucleus& nuc : sys.nuclei) {
      const double r = res.grid[i] - nuc.position;
      v -= nuc.charge / std::sqrt(r * r + a2);
    }
    h(i, i) += v;
  }
  for (std::size_t A = 0; A < sys.nuclei.size(); ++A)
    for (std::size_t B = 0; B < A; ++B) {
      const double r = sys.nuclei[A].position - sys.nuclei[B].position;
      res.nuclear_repulsion += sys.nuclei[A].charge * sys.nuclei[B].charge / std::sqrt(r * r + a2);
    }

  EigenSystem es = eigh(h);
  Matrix D = occupied_density(es.vectors, nocc);
  detail::Diis diis(ctl.diis_subspace);
  Matrix F(N, N);
  double e_old = 0.0;

  if (ctl.log) *ctl.log << "1D-HF: " << N << " grid points, dx = " << dx << ", " << nocc << " occupied orbitals\n";

  for (int it = 1; it <= ctl.max_iterations; ++it) {
    // F = h + v_H - K,   v_H(x_i) = 2 sum_j D_jj w_ij,   K_ij = D_ij w_ij
    F = h;
    for (std::size_t i = 0; i < N; ++i) {
      double vh = 0.0;
      for (std::size_t j = 0; j < N; ++j) vh += D(j, j) * w[absdiff(i, j)];
      F(i, i) += 2.0 * vh;
      for (std::size_t j = 0; j < N; ++j) F(i, j) -= D(i, j) * w[absdiff(i, j)];
    }
    const double energy = frobenius_dot(D, h + F) + res.nuclear_repulsion;
    const Matrix FD = matmul(F, D);
    const Matrix err = FD - transpose(FD);
    const double err_rms = rms(err);
    const double de = energy - e_old;
    if (ctl.log) {
      std::ostringstream line;
      line << std::setw(5) << it << std::setw(22) << std::fixed << std::setprecision(12) << energy << std::setw(12)
           << std::scientific << std::setprecision(2) << de << std::setw(12) << err_rms << "\n";
      *ctl.log << line.str();
    }
    res.energy = energy;
    res.iterations = it;
    if (it > 1 && std::abs(de) < ctl.energy_tol && err_rms < ctl.error_tol) {
      res.converged = true;
      break;
    }
    e_old = energy;
    es = eigh(diis.extrapolate(F, err));
    D = occupied_density(es.vectors, nocc);
  }

  es = eigh(F);
  const std::size_t nkeep = std::min(N, nocc + 4);
  res.orbital_energies.assign(es.values.begin(), es.values.begin() + static_cast<std::ptrdiff_t>(nkeep));
  const double inv_sqrt_dx = 1.0 / std::sqrt(dx);
  res.density.assign(N, 0.0);
  res.orbitals.assign(nocc, std::vector<double>(N));
  for (std::size_t k = 0; k < nocc; ++k) {
    for (std::size_t i = 0; i < N; ++i) {
      const double phi = es.vectors(i, k) * inv_sqrt_dx;
      res.orbitals[k][i] = phi;
      res.density[i] += 2.0 * phi * phi;
    }
  }
  const Matrix Dfin = occupied_density(es.vectors, nocc);
  res.kinetic_energy = 2.0 * frobenius_dot(Dfin, T);
  return res;
}

}  // namespace qk::hf1d
