// cpp-quantum-kernel - McMurchie-Davidson building blocks.
//
// A product of two Cartesian Gaussians is expanded in Hermite Gaussians
// Lambda_tuv centred at P (expansion coefficients E^{ij}_t), and Coulomb
// integrals over Hermite Gaussians reduce to the auxiliary integrals
// R_tuv(p, R_PC), generated from the Boys function by the recurrences of
// McMurchie & Davidson, J. Comput. Phys. 26, 218 (1978); see also
// Helgaker, Joergensen & Olsen, "Molecular Electronic-Structure Theory", ch. 9.
#pragma once

#include <cstddef>
#include <vector>

namespace qk {

struct HermiteIndex {
  int t, u, v;
};

/// All (t,u,v) with t+u+v <= L, ordered by increasing t+u+v.
const std::vector<HermiteIndex>& hermite_indices(int L);
inline constexpr std::size_t n_hermite(int L) {
  return static_cast<std::size_t>((L + 1) * (L + 2) * (L + 3) / 6);
}

/// One-dimensional Hermite expansion coefficients E^{ij}_t (without the
/// exp(-mu X_AB^2) factor) for 0<=i<=la, 0<=j<=lb, 0<=t<=i+j.
/// Layout: E[(i*(lb+1) + j)*(la+lb+1) + t]; `E` must hold
/// (la+1)(lb+1)(la+lb+1) doubles.
void hermite_expansion_1d(int la, int lb, double p, double XPA, double XPB, double* E);

/// Hermite Coulomb integrals R^0_tuv for t+u+v <= L:
///   R[(t*(L+1) + u)*(L+1) + v] = prefactor * R_tuv(alpha, (X,Y,Z)).
/// `work` must hold 2*(L+1)^3 doubles (ping-pong buffers for the auxiliary
/// index n). `R` and `work` must not alias.
void hermite_coulomb(int L, double alpha, double X, double Y, double Z, double prefactor, double* R, double* work);

}  // namespace qk
