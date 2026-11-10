// cpp-quantum-kernel - molecular geometry.
#pragma once

#include <array>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace qk {

struct Atom {
  int Z = 0;                        ///< nuclear charge
  std::array<double, 3> xyz{};      ///< position in bohr
};

/// Atomic number from an element symbol (case-insensitive, H..Kr).
/// Throws std::invalid_argument for unknown symbols.
int atomic_number(const std::string& symbol);
/// Element symbol for Z = 1..36.
std::string element_symbol(int Z);

class Molecule {
 public:
  Molecule() = default;
  Molecule(std::vector<Atom> atoms, int charge = 0) : atoms_(std::move(atoms)), charge_(charge) {}

  /// Standard XYZ format: atom count, comment line, then "Sym x y z" lines.
  /// Coordinates are read in Angstrom unless `in_bohr` is true.
  static Molecule from_xyz(std::istream& in, bool in_bohr = false);
  static Molecule from_xyz_file(const std::string& path, bool in_bohr = false);
  /// Compact inline form: "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587".
  static Molecule from_string(const std::string& spec, bool in_bohr = false);

  const std::vector<Atom>& atoms() const noexcept { return atoms_; }
  std::size_t size() const noexcept { return atoms_.size(); }
  int charge() const noexcept { return charge_; }
  void set_charge(int q) noexcept { charge_ = q; }

  int n_electrons() const;
  double nuclear_repulsion() const;

 private:
  std::vector<Atom> atoms_;
  int charge_ = 0;
};

}  // namespace qk
