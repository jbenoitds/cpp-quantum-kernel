// qk-bench: throughput of the ERI engine and Fock builds.
//
//   qk-bench [--basis 6-31g*] [--waters 1,2,4,8] [--xyz file.xyz] [--threads N] [--scf]
//
// For each system it reports: shell-pair setup, in-core tensor build
// (unique integrals per second), one integral-direct J/K build at 1 thread
// and at all threads, and optionally a full RHF run.
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "qk/qk.hpp"

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

/// n water molecules on a cubic lattice with 2.9 Angstrom spacing.
qk::Molecule water_cluster(int n) {
  std::ostringstream s;
  const int side = static_cast<int>(std::ceil(std::cbrt(static_cast<double>(n))));
  int placed = 0;
  for (int i = 0; i < side && placed < n; ++i)
    for (int j = 0; j < side && placed < n; ++j)
      for (int k = 0; k < side && placed < n; ++k, ++placed) {
        const double x = 2.9 * i, y = 2.9 * j, z = 2.9 * k;
        s << "O " << x << " " << y << " " << z + 0.1173 << ";"
          << "H " << x << " " << y + 0.7572 << " " << z - 0.4692 << ";"
          << "H " << x << " " << y - 0.7572 << " " << z - 0.4692 << ";";
      }
  return qk::Molecule::from_string(s.str());
}

void run(const std::string& label, const qk::Molecule& mol, const std::string& basis_name, bool do_scf) {
  const qk::BasisSet basis(mol, basis_name);
  const std::size_t n = basis.n_functions();
  const int nt = qk::max_threads();

  auto t0 = Clock::now();
  const qk::EriEngine engine(basis);
  const double t_setup = since(t0);

  double t_incore = -1.0, rate = 0.0;
  if (qk::EriTensor::memory_mb(n) < 4096.0) {
    t0 = Clock::now();
    const qk::EriTensor eri(engine);
    t_incore = since(t0);
    rate = static_cast<double>(qk::EriTensor::packed_size(n)) / t_incore;
  }

  // A realistic density: core-Hamiltonian guess.
  const qk::Matrix S = qk::overlap_matrix(basis);
  const qk::Matrix H = qk::core_hamiltonian(basis, mol);
  const qk::Matrix X = qk::orthogonalizer(S);
  const auto es = qk::eigh(qk::matmul_tn(X, qk::matmul(H, X)));
  const qk::Matrix C = qk::matmul(X, es.vectors);
  const std::size_t nocc = static_cast<std::size_t>(mol.n_electrons() / 2);
  qk::Matrix D(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t k = 0; k < nocc; ++k) D(i, j) += C(i, k) * C(j, k);

  qk::set_num_threads(1);
  t0 = Clock::now();
  qk::DirectStats st;
  (void)qk::build_jk_direct(engine, D, 1e-12, &st);
  const double t_jk1 = since(t0);
  qk::set_num_threads(nt);
  t0 = Clock::now();
  (void)qk::build_jk_direct(engine, D, 1e-12);
  const double t_jkn = since(t0);

  std::cout << std::left << std::setw(14) << label << std::right << std::setw(6) << n << std::fixed
            << std::setprecision(3) << std::setw(10) << t_setup;
  if (t_incore >= 0)
    std::cout << std::setw(10) << t_incore << std::setw(10) << std::setprecision(1) << rate / 1e6;
  else
    std::cout << std::setw(10) << "-" << std::setw(10) << "-";
  const double frac = 100.0 * static_cast<double>(st.quartets_screened) /
                      static_cast<double>(st.quartets_screened + st.quartets_computed);
  std::cout << std::setprecision(3) << std::setw(10) << t_jk1 << std::setw(10) << t_jkn << std::setprecision(2)
            << std::setw(8) << t_jk1 / t_jkn << std::setprecision(1) << std::setw(9) << frac;

  if (do_scf) {
    qk::ScfOptions o;
    o.eri = qk::EriStrategy::Direct;
    t0 = Clock::now();
    const auto r = qk::run_rhf(mol, basis, o);
    std::cout << std::setprecision(3) << std::setw(9) << since(t0) << std::setw(5) << r.iterations
              << std::setprecision(8) << std::setw(18) << r.energy;
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string basis = "6-31g*", xyz;
  std::vector<int> waters = {1, 2, 4, 8};
  bool scf = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--basis" && i + 1 < argc) {
      basis = argv[++i];
    } else if (a == "--waters" && i + 1 < argc) {
      waters.clear();
      std::stringstream ss(argv[++i]);
      std::string tok;
      while (std::getline(ss, tok, ',')) waters.push_back(std::stoi(tok));
    } else if (a == "--xyz" && i + 1 < argc) {
      xyz = argv[++i];
    } else if (a == "--threads" && i + 1 < argc) {
      qk::set_num_threads(std::stoi(argv[++i]));
    } else if (a == "--scf") {
      scf = true;
    } else {
      std::cout << "usage: qk-bench [--basis NAME] [--waters 1,2,4] [--xyz FILE] [--threads N] [--scf]\n";
      return a == "-h" || a == "--help" ? 0 : 1;
    }
  }
  std::cout << "cpp-quantum-kernel " << qk::kVersion << " benchmark | basis " << basis << " | " << qk::max_threads()
            << " thread(s), OpenMP " << (qk::openmp_enabled() ? "on" : "off") << "\n\n"
            << "system           nbf  pairs(s) incore(s)  Mint/s   JK 1t(s)  JK Nt(s) speedup  screen%";
  if (scf) std::cout << "   SCF(s)  it      E_RHF (Eh)";
  std::cout << "\n";
  if (!xyz.empty()) run(xyz, qk::Molecule::from_xyz_file(xyz), basis, scf);
  for (int w : waters) run("(H2O)" + std::to_string(w), water_cluster(w), basis, scf);
  return 0;
}
