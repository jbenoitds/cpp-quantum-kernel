// cpp-quantum-kernel - contracted Cartesian Gaussian basis sets.
//
// Conventions
//  * A shell of angular momentum l holds (l+1)(l+2)/2 Cartesian functions in
//    the canonical order  xx..x, xx..y, ..., zz..z  (for d: xx xy xz yy yz zz),
//    i.e. the same order as Gaussian, libint and PySCF (cart=True).
//  * Every Cartesian component is individually normalised to unity.
//  * Shell::coefficients already contain the primitive normalisation of the
//    x^l component and the contraction normalisation; the per-component ratio
//    is applied by the integral code via cartesian_norm_ratio().
#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include "qk/molecule.hpp"

namespace qk {

/// Highest angular momentum handled by the integral engine (i functions).
inline constexpr int kMaxAngularMomentum = 6;

struct CartesianExponents {
  int x, y, z;
};

/// Cartesian exponent triples of a shell in canonical order.
const std::vector<CartesianExponents>& cartesian_components(int l);
inline constexpr std::size_t n_cartesian(int l) { return static_cast<std::size_t>((l + 1) * (l + 2) / 2); }

/// (2n-1)!! with (-1)!! = 1.
double double_factorial_odd(int n);

/// Normalisation constant of the primitive x^lx y^ly z^lz exp(-a r^2).
double primitive_norm(double a, int lx, int ly, int lz);

/// Ratio N(lx,ly,lz) / N(l,0,0) - independent of the exponent.
double cartesian_norm_ratio(int lx, int ly, int lz);

struct Shell {
  int l = 0;
  std::size_t atom = 0;
  std::array<double, 3> center{};
  std::vector<double> exponents;
  std::vector<double> coefficients;  ///< normalised, see file comment

  std::size_t size() const noexcept { return n_cartesian(l); }
  std::size_t n_primitives() const noexcept { return exponents.size(); }
};

/// Build a normalised shell from raw (as-tabulated) contraction coefficients.
Shell make_shell(int l, const std::array<double, 3>& center, std::vector<double> exponents,
                 std::vector<double> raw_coefficients, std::size_t atom = 0);

class BasisSet {
 public:
  BasisSet() = default;

  /// Built-in basis by name ("sto-3g", "6-31g", "6-31g*"; case-insensitive,
  /// "6-31g(d)" accepted). Throws if an element is not covered.
  BasisSet(const Molecule& mol, const std::string& name);

  /// Gaussian94-format basis text (as exported by the Basis Set Exchange).
  static BasisSet from_g94(const Molecule& mol, std::istream& in);
  static BasisSet from_g94_file(const Molecule& mol, const std::string& path);

  static std::vector<std::string> builtin_names();

  void add_shell(Shell shell);

  const std::vector<Shell>& shells() const noexcept { return shells_; }
  const Shell& shell(std::size_t i) const { return shells_[i]; }
  std::size_t n_shells() const noexcept { return shells_.size(); }
  std::size_t n_functions() const noexcept { return nbf_; }
  std::size_t offset(std::size_t shell) const { return offsets_[shell]; }
  /// Index of the atom on which basis function mu sits.
  std::size_t function_atom(std::size_t mu) const;
  int max_l() const noexcept { return max_l_; }
  std::size_t max_primitives() const noexcept { return max_nprim_; }

 private:
  std::vector<Shell> shells_;
  std::vector<std::size_t> offsets_;
  std::size_t nbf_ = 0;
  int max_l_ = 0;
  std::size_t max_nprim_ = 0;
};

}  // namespace qk
