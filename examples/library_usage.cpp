// Minimal use of the qk library: RHF + MP2 for water, then a 1D model atom.
#include <cstdio>

#include "qk/qk.hpp"

int main() {
  const qk::Molecule water = qk::Molecule::from_string("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692");
  const qk::BasisSet basis(water, "6-31g*");

  const qk::EriEngine engine(basis);    // shell-pair data + Schwarz bounds
  const qk::EriTensor eri(engine);      // all unique (mn|ls), 8-fold packed
  const qk::ScfResult scf = qk::run_rhf(water, basis, {}, &eri);
  const qk::Mp2Result mp2 = qk::mp2_energy(scf, eri, /*n_frozen_core=*/1);
  std::printf("E(RHF) = %.10f  E(MP2) = %.10f Eh\n", scf.energy, scf.energy + mp2.correlation_energy);

  qk::hf1d::System helium;
  helium.nuclei = {{2.0, 0.0}};
  helium.n_electrons = 2;
  std::printf("1D He  = %.10f Eh\n", qk::hf1d::solve(helium).energy);
}
