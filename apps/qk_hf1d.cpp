// qk-hf1d: closed-shell Hartree-Fock for 1D soft-Coulomb atoms and molecules.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "qk/hf1d.hpp"

namespace {
void usage() {
  std::cout << R"(usage: qk-hf1d [options]

  --nucleus Z:X        add a nucleus of charge Z at position X (repeatable) [2:0 = He]
  --electrons N        number of electrons (even)                          [2]
  --softening A        soft-Coulomb parameter a                            [1.0]
  --lambda L           electron-electron interaction scale                 [1.0]
  --omega W            harmonic confinement 1/2 W^2 x^2                    [0]
  --xmin X --xmax X    grid extent                                         [-20 20]
  --points N           number of grid points                               [201]
  --output FILE        write x, density and occupied orbitals as CSV
  -q, --quiet          no iteration log
)";
}
}  // namespace

int main(int argc, char** argv) {
  qk::hf1d::System sys;
  qk::hf1d::Grid grid;
  qk::hf1d::Controls ctl;
  std::string output;
  bool quiet = false;
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
      } else if (a == "--nucleus") {
        const std::string v = next();
        const auto colon = v.find(':');
        if (colon == std::string::npos) throw std::invalid_argument("--nucleus expects Z:X");
        sys.nuclei.push_back({std::stod(v.substr(0, colon)), std::stod(v.substr(colon + 1))});
      } else if (a == "--electrons") {
        sys.n_electrons = std::stoi(next());
      } else if (a == "--softening") {
        sys.softening = std::stod(next());
      } else if (a == "--lambda") {
        sys.interaction = std::stod(next());
      } else if (a == "--omega") {
        sys.harmonic_omega = std::stod(next());
      } else if (a == "--xmin") {
        grid.x_min = std::stod(next());
      } else if (a == "--xmax") {
        grid.x_max = std::stod(next());
      } else if (a == "--points") {
        grid.n_points = static_cast<std::size_t>(std::stoul(next()));
      } else if (a == "--output") {
        output = next();
      } else if (a == "-q" || a == "--quiet") {
        quiet = true;
      } else {
        throw std::invalid_argument("unknown option " + a);
      }
    }
    if (sys.nuclei.empty() && sys.harmonic_omega == 0.0) sys.nuclei.push_back({2.0, 0.0});
    if (!quiet) ctl.log = &std::cout;

    const auto r = qk::hf1d::solve(sys, grid, ctl);
    std::cout << std::fixed << std::setprecision(10) << "\nTotal energy        " << r.energy << " Eh"
              << (r.converged ? "" : "  (NOT CONVERGED)") << "\nNuclear repulsion   " << r.nuclear_repulsion
              << "\nKinetic energy      " << r.kinetic_energy << "\nOrbital energies:";
    for (double e : r.orbital_energies) std::cout << " " << std::setprecision(6) << e;
    std::cout << "\n";

    if (!output.empty()) {
      std::ofstream f(output);
      f << "x,density";
      for (std::size_t k = 0; k < r.orbitals.size(); ++k) f << ",phi" << k;
      f << "\n" << std::setprecision(12);
      for (std::size_t i = 0; i < r.grid.size(); ++i) {
        f << r.grid[i] << "," << r.density[i];
        for (const auto& phi : r.orbitals) f << "," << phi[i];
        f << "\n";
      }
      std::cout << "wrote " << output << "\n";
    }
    return r.converged ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "qk-hf1d: error: " << e.what() << "\n";
    return 1;
  }
}
