// cpp-quantum-kernel - physical and mathematical constants.
#pragma once

namespace qk {

/// pi to double precision (M_PI is not portable to MSVC without _USE_MATH_DEFINES).
inline constexpr double kPi = 3.14159265358979323846264338327950288;

/// Bohr radius in Angstrom (CODATA 2018).
inline constexpr double kBohrInAngstrom = 0.529177210903;

/// Conversion factor Angstrom -> bohr.
inline constexpr double kAngstromToBohr = 1.0 / kBohrInAngstrom;

/// Hartree in electron-volt (CODATA 2018).
inline constexpr double kHartreeInEv = 27.211386245988;

/// Atomic unit of electric dipole moment in Debye.
inline constexpr double kAuToDebye = 2.541746473;

}  // namespace qk
