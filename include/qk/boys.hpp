// cpp-quantum-kernel - the Boys function F_n(T) = \int_0^1 t^{2n} exp(-T t^2) dt.
#pragma once

namespace qk {

/// Largest order supported by boys_function (enough for (gg|gg) ERIs and
/// their first derivatives).
inline constexpr int kBoysMaxOrder = 32;

/// Fill F[0..nmax] with F_n(T). Thread-safe; ~1e-14 relative accuracy.
///
/// T < 40: 8-term Taylor expansion of F_nmax around the nearest point of a
///         precomputed grid (spacing 0.1), then stable downward recursion.
/// T >= 40: F_0 from erf, then upward recursion (stable because T > n).
void boys_function(int nmax, double T, double* F);

/// Slow but accurate reference (long-double power series). Intended for
/// table construction and testing only.
double boys_reference(int n, double T);

}  // namespace qk
