// qk-scf: RHF (+ optional MP2) single-point energies from an XYZ file.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "qk/qk.hpp"

namespace {

void usage() {
  std::cout << R"(usage: qk-scf <molecule.xyz> [options]

  -b, --basis NAME      built-in basis: sto-3g | 6-31g | 6-31g*   [sto-3g]
      --basis-file F    Gaussian94-format basis file (e.g. from basissetexchange.org)
  -c, --charge Q        molecular charge                            [0]
      --bohr            XYZ coordinates are in bohr (default Angstrom)
      --incore          force in-core integrals
      --direct          force integral-direct SCF
      --mp2             MP2 correlation energy (uses in-core integrals)
      --frozen-core N   number of frozen core orbitals for MP2      [0]
  -t, --threads N       OpenMP threads                              [all]
      --memory MB       in-core memory limit                        [2048]
      --conv E          energy convergence threshold                [1e-10]
      --dump DIR        write S, T, V, ERI (.npy) and basis.json to DIR
  -q, --quiet           no iteration log
  -h, --help
)";
}

// Minimal NumPy .npy (v1.0) writer: little-endian float64, C order.
void write_npy(const std::string& path, const double* data, const std::vector<std::size_t>& shape) {
  std::ostringstream hdr;
  hdr << "{'descr': '<f8', 'fortran_order': False, 'shape': (";
  std::size_t count = 1;
  for (std::size_t s : shape) {
    hdr << s << ",";
    count *= s;
  }
  hdr << "), }";
  std::string h = hdr.str();
  const std::size_t total = 10 + h.size() + 1;
  h.append((64 - total % 64) % 64, ' ');
  h += '\n';
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  const char magic[] = "\x93NUMPY\x01\x00";
  f.write(magic, 8);
  const std::uint16_t len = static_cast<std::uint16_t>(h.size());
  const char lenb[2] = {static_cast<char>(len & 0xff), static_cast<char>(len >> 8)};
  f.write(lenb, 2);
  f.write(h.data(), static_cast<std::streamsize>(h.size()));
  f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(double)));
}

void dump(const std::string& dir, const qk::Molecule& mol, const qk::BasisSet& basis) {
  const std::size_t n = basis.n_functions();
  const qk::Matrix S = qk::overlap_matrix(basis), T = qk::kinetic_matrix(basis),
                   V = qk::nuclear_attraction_matrix(basis, mol);
  write_npy(dir + "/S.npy", S.data(), {n, n});
  write_npy(dir + "/T.npy", T.data(), {n, n});
  write_npy(dir + "/V.npy", V.data(), {n, n});
  if (n <= 80) {
    qk::EriEngine engine(basis);
    qk::EriTensor eri(engine, 0.0);
    std::vector<double> full(n * n * n * n);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        for (std::size_t k = 0; k < n; ++k)
          for (std::size_t l = 0; l < n; ++l) full[((i * n + j) * n + k) * n + l] = eri(i, j, k, l);
    write_npy(dir + "/ERI.npy", full.data(), {n, n, n, n});
  } else {
    std::cerr << "note: ERI.npy skipped (more than 80 basis functions)\n";
  }
  std::ofstream js(dir + "/basis.json");
  js << std::setprecision(17) << "{\n  \"atoms\": [";
  for (std::size_t a = 0; a < mol.size(); ++a) {
    const auto& at = mol.atoms()[a];
    js << (a ? ", " : "") << "{\"Z\": " << at.Z << ", \"xyz\": [" << at.xyz[0] << ", " << at.xyz[1] << ", "
       << at.xyz[2] << "]}";
  }
  js << "],\n  \"charge\": " << mol.charge() << ",\n  \"shells\": [\n";
  for (std::size_t s = 0; s < basis.n_shells(); ++s) {
    const auto& sh = basis.shell(s);
    js << "    {\"atom\": " << sh.atom << ", \"l\": " << sh.l << ", \"exps\": [";
    for (std::size_t p = 0; p < sh.n_primitives(); ++p) js << (p ? ", " : "") << sh.exponents[p];
    js << "], \"coefs\": [";  // raw (unnormalised-primitive) contraction coefficients
    for (std::size_t p = 0; p < sh.n_primitives(); ++p)
      js << (p ? ", " : "") << sh.coefficients[p] / qk::primitive_norm(sh.exponents[p], sh.l, 0, 0);
    js << "]}" << (s + 1 < basis.n_shells() ? "," : "") << "\n";
  }
  js << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string xyz, basis_name = "sto-3g", basis_file, dump_dir;
  int charge = 0, threads = 0;
  std::size_t frozen = 0;
  bool bohr = false, mp2 = false, quiet = false;
  qk::ScfOptions opt;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
        return argv[++i];
      };
      if (a == "-h" || a == "--help") {
        usage();
        return 0;
      } else if (a == "-b" || a == "--basis") {
        basis_name = next();
      } else if (a == "--basis-file") {
        basis_file = next();
      } else if (a == "-c" || a == "--charge") {
        charge = std::stoi(next());
      } else if (a == "--bohr") {
        bohr = true;
      } else if (a == "--incore") {
        opt.eri = qk::EriStrategy::InCore;
      } else if (a == "--direct") {
        opt.eri = qk::EriStrategy::Direct;
      } else if (a == "--mp2") {
        mp2 = true;
      } else if (a == "--frozen-core") {
        frozen = static_cast<std::size_t>(std::stoul(next()));
      } else if (a == "-t" || a == "--threads") {
        threads = std::stoi(next());
      } else if (a == "--memory") {
        opt.max_memory_mb = std::stod(next());
      } else if (a == "--conv") {
        opt.energy_tol = std::stod(next());
      } else if (a == "--dump") {
        dump_dir = next();
      } else if (a == "-q" || a == "--quiet") {
        quiet = true;
      } else if (!a.empty() && a[0] == '-') {
        throw std::invalid_argument("unknown option " + a);
      } else {
        xyz = a;
      }
    }
    if (xyz.empty()) {
      usage();
      return 1;
    }
    if (threads > 0) qk::set_num_threads(threads);

    qk::Molecule mol = qk::Molecule::from_xyz_file(xyz, bohr);
    mol.set_charge(charge);
    const qk::BasisSet basis = basis_file.empty() ? qk::BasisSet(mol, basis_name)
                                                  : qk::BasisSet::from_g94_file(mol, basis_file);
    std::cout << "cpp-quantum-kernel " << qk::kVersion << "  |  " << xyz << "  |  "
              << (basis_file.empty() ? basis_name : basis_file) << "\n"
              << mol.size() << " atoms, " << mol.n_electrons() << " electrons, " << basis.n_shells() << " shells, "
              << basis.n_functions() << " Cartesian basis functions (max l = " << basis.max_l() << ")\n";

    if (!dump_dir.empty()) {
      dump(dump_dir, mol, basis);
      std::cout << "integrals written to " << dump_dir << "\n";
    }

    std::unique_ptr<qk::EriEngine> engine;
    std::unique_ptr<qk::EriTensor> eri;
    double t_eri = 0.0;
    if (mp2) {
      if (opt.eri == qk::EriStrategy::Direct) std::cerr << "note: --mp2 needs stored integrals; using in-core\n";
      const auto t0 = std::chrono::steady_clock::now();
      engine = std::make_unique<qk::EriEngine>(basis);
      eri = std::make_unique<qk::EriTensor>(*engine, opt.schwarz_tol);
      t_eri = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    if (!quiet) opt.log = &std::cout;
    const qk::ScfResult r = qk::run_rhf(mol, basis, opt, eri.get());

    std::cout << std::fixed << std::setprecision(10) << "\nNuclear repulsion   " << std::setw(20) << r.nuclear_repulsion
              << "\nElectronic energy   " << std::setw(20) << r.electronic_energy << "\nRHF total energy    "
              << std::setw(20) << r.energy << "  Eh" << (r.converged ? "" : "   (NOT CONVERGED)") << "\n";

    std::cout << "\nOrbital energies (Eh):\n" << std::setprecision(6);
    for (std::size_t k = 0; k < r.orbital_energies.size(); ++k) {
      std::cout << std::setw(12) << r.orbital_energies[k] << (k < r.n_occupied ? "*" : " ");
      if (k % 6 == 5 || k + 1 == r.orbital_energies.size()) std::cout << "\n";
    }
    const auto mu = qk::dipole_moment(mol, basis, r.D);
    const double mu_tot = std::sqrt(mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2]);
    std::cout << "\nDipole moment (Debye): " << std::setprecision(4) << mu[0] * qk::kAuToDebye << " "
              << mu[1] * qk::kAuToDebye << " " << mu[2] * qk::kAuToDebye << "   |mu| = " << mu_tot * qk::kAuToDebye
              << "\nMulliken charges:";
    const auto q = qk::mulliken_charges(mol, basis, r.D, r.S);
    for (std::size_t a = 0; a < q.size(); ++a)
      std::cout << "  " << qk::element_symbol(mol.atoms()[a].Z) << a + 1 << " " << std::showpos << q[a]
                << std::noshowpos;
    std::cout << "\n";

    if (mp2) {
      const auto t0 = std::chrono::steady_clock::now();
      const qk::Mp2Result m = qk::mp2_energy(r, *eri, frozen);
      const double t_mp2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::cout << std::setprecision(10) << "\nMP2 correlation     " << std::setw(20) << m.correlation_energy
                << "\n  same-spin         " << std::setw(20) << m.same_spin << "\n  opposite-spin     " << std::setw(20)
                << m.opposite_spin << "\nMP2 total energy    " << std::setw(20) << r.energy + m.correlation_energy
                << "  Eh\n";
      std::cout << std::setprecision(3) << "\nTimings: ERI " << t_eri << " s, SCF " << r.seconds_total << " s, MP2 "
                << t_mp2 << " s\n";
    } else {
      std::cout << std::setprecision(3) << "\nTimings: integrals " << r.seconds_integrals << " s, total "
                << r.seconds_total << " s\n";
    }
    return r.converged ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "qk-scf: error: " << e.what() << "\n";
    return 1;
  }
}
