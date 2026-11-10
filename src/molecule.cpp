#include "qk/molecule.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "qk/constants.hpp"

namespace qk {

namespace {
constexpr const char* kSymbols[] = {"",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",
                                    "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",
                                    "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
                                    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"};
constexpr int kMaxZ = 36;

std::string normalise_symbol(std::string s) {
  // Strip trailing digits/labels such as "H1" or "O_a".
  std::size_t n = 0;
  while (n < s.size() && std::isalpha(static_cast<unsigned char>(s[n]))) ++n;
  s.resize(n);
  if (s.empty()) return s;
  s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
  for (std::size_t i = 1; i < s.size(); ++i) s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
  return s;
}

Atom parse_atom_line(const std::string& line, bool in_bohr) {
  std::istringstream ls(line);
  std::string sym;
  Atom a;
  if (!(ls >> sym >> a.xyz[0] >> a.xyz[1] >> a.xyz[2])) {
    throw std::runtime_error("qk::Molecule: cannot parse atom line: '" + line + "'");
  }
  const bool numeric = std::all_of(sym.begin(), sym.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
  a.Z = numeric ? std::stoi(sym) : atomic_number(sym);
  if (!in_bohr)
    for (double& c : a.xyz) c *= kAngstromToBohr;
  return a;
}
}  // namespace

int atomic_number(const std::string& symbol) {
  const std::string s = normalise_symbol(symbol);
  for (int z = 1; z <= kMaxZ; ++z)
    if (s == kSymbols[z]) return z;
  throw std::invalid_argument("qk: unknown element symbol '" + symbol + "'");
}

std::string element_symbol(int Z) {
  if (Z < 1 || Z > kMaxZ) throw std::out_of_range("qk: element Z out of range");
  return kSymbols[Z];
}

Molecule Molecule::from_xyz(std::istream& in, bool in_bohr) {
  std::string line;
  std::size_t count = 0;
  // Skip leading blank lines.
  while (std::getline(in, line)) {
    if (line.find_first_not_of(" \t\r") != std::string::npos) break;
  }
  try {
    count = static_cast<std::size_t>(std::stoul(line));
  } catch (const std::exception&) {
    throw std::runtime_error("qk::Molecule::from_xyz: first line must be the atom count");
  }
  std::getline(in, line);  // comment
  std::vector<Atom> atoms;
  while (atoms.size() < count && std::getline(in, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    atoms.push_back(parse_atom_line(line, in_bohr));
  }
  if (atoms.size() != count) throw std::runtime_error("qk::Molecule::from_xyz: fewer atoms than declared");
  return Molecule(std::move(atoms));
}

Molecule Molecule::from_xyz_file(const std::string& path, bool in_bohr) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("qk::Molecule: cannot open '" + path + "'");
  return from_xyz(f, in_bohr);
}

Molecule Molecule::from_string(const std::string& spec, bool in_bohr) {
  std::vector<Atom> atoms;
  std::string chunk;
  std::istringstream ss(spec);
  while (std::getline(ss, chunk, ';')) {
    std::replace(chunk.begin(), chunk.end(), '\n', ' ');
    if (chunk.find_first_not_of(" \t\r") == std::string::npos) continue;
    atoms.push_back(parse_atom_line(chunk, in_bohr));
  }
  return Molecule(std::move(atoms));
}

int Molecule::n_electrons() const {
  int n = -charge_;
  for (const Atom& a : atoms_) n += a.Z;
  return n;
}

double Molecule::nuclear_repulsion() const {
  double e = 0.0;
  for (std::size_t i = 0; i < atoms_.size(); ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      const double dx = atoms_[i].xyz[0] - atoms_[j].xyz[0];
      const double dy = atoms_[i].xyz[1] - atoms_[j].xyz[1];
      const double dz = atoms_[i].xyz[2] - atoms_[j].xyz[2];
      e += atoms_[i].Z * atoms_[j].Z / std::sqrt(dx * dx + dy * dy + dz * dz);
    }
  }
  return e;
}

}  // namespace qk
