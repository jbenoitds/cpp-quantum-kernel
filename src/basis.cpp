#include "qk/basis.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "basis_data.hpp"
#include "qk/constants.hpp"

namespace qk {

// ---------------------------------------------------------------------------
// Cartesian bookkeeping and normalisation
// ---------------------------------------------------------------------------

const std::vector<CartesianExponents>& cartesian_components(int l) {
  static const std::vector<std::vector<CartesianExponents>> table = [] {
    std::vector<std::vector<CartesianExponents>> t(kMaxAngularMomentum + 1);
    for (int L = 0; L <= kMaxAngularMomentum; ++L) {
      for (int x = L; x >= 0; --x)
        for (int y = L - x; y >= 0; --y) t[L].push_back({x, y, L - x - y});
    }
    return t;
  }();
  if (l < 0 || l > kMaxAngularMomentum) throw std::out_of_range("qk: angular momentum out of range");
  return table[static_cast<std::size_t>(l)];
}

double double_factorial_odd(int n) {
  double r = 1.0;
  for (int k = 2 * n - 1; k > 1; k -= 2) r *= k;
  return r;
}

double primitive_norm(double a, int lx, int ly, int lz) {
  const int L = lx + ly + lz;
  const double df = double_factorial_odd(lx) * double_factorial_odd(ly) * double_factorial_odd(lz);
  return std::pow(2.0 * a / kPi, 0.75) * std::pow(4.0 * a, 0.5 * L) / std::sqrt(df);
}

double cartesian_norm_ratio(int lx, int ly, int lz) {
  const int L = lx + ly + lz;
  return std::sqrt(double_factorial_odd(L) /
                   (double_factorial_odd(lx) * double_factorial_odd(ly) * double_factorial_odd(lz)));
}

Shell make_shell(int l, const std::array<double, 3>& center, std::vector<double> exponents,
                 std::vector<double> raw_coefficients, std::size_t atom) {
  if (l < 0 || l > kMaxAngularMomentum) throw std::invalid_argument("qk::make_shell: unsupported angular momentum");
  if (exponents.empty() || exponents.size() != raw_coefficients.size())
    throw std::invalid_argument("qk::make_shell: exponent/coefficient size mismatch");
  for (double a : exponents)
    if (!(a > 0.0)) throw std::invalid_argument("qk::make_shell: exponents must be positive");

  Shell s;
  s.l = l;
  s.atom = atom;
  s.center = center;
  s.exponents = std::move(exponents);
  s.coefficients = std::move(raw_coefficients);
  const std::size_t np = s.exponents.size();
  for (std::size_t i = 0; i < np; ++i) s.coefficients[i] *= primitive_norm(s.exponents[i], l, 0, 0);

  // <x^l g | x^l g> for the contracted function
  const double df = double_factorial_odd(l);
  double norm2 = 0.0;
  for (std::size_t i = 0; i < np; ++i) {
    for (std::size_t j = 0; j < np; ++j) {
      const double p = s.exponents[i] + s.exponents[j];
      norm2 += s.coefficients[i] * s.coefficients[j] * std::pow(kPi / p, 1.5) * df / std::pow(2.0 * p, l);
    }
  }
  const double scale = 1.0 / std::sqrt(norm2);
  for (double& c : s.coefficients) c *= scale;
  return s;
}

// ---------------------------------------------------------------------------
// Gaussian94 parser
// ---------------------------------------------------------------------------

namespace {

struct ShellSpec {
  int l;
  std::vector<double> exps;
  std::vector<double> coefs;
};
using ElementBasis = std::vector<ShellSpec>;

double parse_number(std::string tok) {
  std::replace(tok.begin(), tok.end(), 'D', 'E');
  std::replace(tok.begin(), tok.end(), 'd', 'e');
  std::size_t used = 0;
  const double v = std::stod(tok, &used);
  if (used != tok.size()) throw std::runtime_error("qk: malformed number '" + tok + "' in basis file");
  return v;
}

int letter_to_l(char c) {
  static const std::string letters = "SPDFGHI";
  const std::size_t pos = letters.find(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  if (pos == std::string::npos) throw std::runtime_error(std::string("qk: unknown shell type '") + c + "'");
  return static_cast<int>(pos);
}

bool is_blank_or_comment(const std::string& line) {
  const std::size_t p = line.find_first_not_of(" \t\r");
  return p == std::string::npos || line[p] == '!' || line[p] == '#';
}

std::map<int, ElementBasis> parse_g94(std::istream& in) {
  std::map<int, ElementBasis> result;
  std::string line;
  int current_Z = 0;
  while (std::getline(in, line)) {
    if (is_blank_or_comment(line)) continue;
    std::istringstream ls(line);
    std::string first;
    ls >> first;
    if (first == "****") {
      current_Z = 0;
      continue;
    }
    if (current_Z == 0) {
      // Element header: "Sym 0" (a leading '-' is sometimes used).
      if (!first.empty() && first[0] == '-') first.erase(0, 1);
      current_Z = atomic_number(first);
      result[current_Z].clear();
      continue;
    }
    // Shell header: TYPE nprim scale
    std::string type = first;
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    int nprim = 0;
    double scale = 1.0;
    if (!(ls >> nprim) || nprim <= 0) throw std::runtime_error("qk: malformed shell header '" + line + "'");
    std::string scale_tok;
    if (ls >> scale_tok) scale = parse_number(scale_tok);

    const bool sp = (type == "SP" || type == "L");
    if (!sp && type.size() != 1) throw std::runtime_error("qk: unsupported shell type '" + type + "'");
    ShellSpec s_part{sp ? 0 : letter_to_l(type[0]), {}, {}};
    ShellSpec p_part{1, {}, {}};
    for (int k = 0; k < nprim; ++k) {
      if (!std::getline(in, line)) throw std::runtime_error("qk: unexpected end of basis file");
      if (is_blank_or_comment(line)) {
        --k;
        continue;
      }
      std::istringstream ps(line);
      std::string a, c1, c2;
      if (!(ps >> a >> c1)) throw std::runtime_error("qk: malformed primitive line '" + line + "'");
      const double alpha = parse_number(a) * scale * scale;
      s_part.exps.push_back(alpha);
      s_part.coefs.push_back(parse_number(c1));
      if (sp) {
        if (!(ps >> c2)) throw std::runtime_error("qk: SP shell needs two coefficients: '" + line + "'");
        p_part.exps.push_back(alpha);
        p_part.coefs.push_back(parse_number(c2));
      }
    }
    result[current_Z].push_back(std::move(s_part));
    if (sp) result[current_Z].push_back(std::move(p_part));
  }
  return result;
}

std::string canonical_basis_name(std::string name) {
  std::string s;
  for (char c : name)
    if (!std::isspace(static_cast<unsigned char>(c))) s += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "sto3g") s = "sto-3g";
  if (s == "631g") s = "6-31g";
  if (s == "6-31g(d)" || s == "631g*" || s == "631g(d)") s = "6-31g*";
  return s;
}

BasisSet assemble(const Molecule& mol, const std::map<int, ElementBasis>& lib, const std::string& label) {
  BasisSet bs;
  for (std::size_t a = 0; a < mol.size(); ++a) {
    const Atom& atom = mol.atoms()[a];
    const auto it = lib.find(atom.Z);
    if (it == lib.end())
      throw std::runtime_error("qk: basis '" + label + "' has no entry for element " + element_symbol(atom.Z));
    for (const ShellSpec& spec : it->second) bs.add_shell(make_shell(spec.l, atom.xyz, spec.exps, spec.coefs, a));
  }
  return bs;
}

}  // namespace

BasisSet::BasisSet(const Molecule& mol, const std::string& name) {
  const std::string key = canonical_basis_name(name);
  std::map<int, ElementBasis> lib;
  bool known = false;
  for (const auto& entry : detail::builtin_basis_table()) {
    if (key != entry.basis) continue;
    known = true;
    std::istringstream in(entry.g94);
    for (auto& kv : parse_g94(in)) lib[kv.first] = std::move(kv.second);
  }
  if (!known) throw std::invalid_argument("qk: unknown built-in basis '" + name + "' (use a .g94 file instead)");
  *this = assemble(mol, lib, key);
}

BasisSet BasisSet::from_g94(const Molecule& mol, std::istream& in) { return assemble(mol, parse_g94(in), "g94 input"); }

BasisSet BasisSet::from_g94_file(const Molecule& mol, const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("qk: cannot open basis file '" + path + "'");
  return assemble(mol, parse_g94(f), path);
}

std::vector<std::string> BasisSet::builtin_names() { return {"sto-3g", "6-31g", "6-31g*"}; }

void BasisSet::add_shell(Shell shell) {
  offsets_.push_back(nbf_);
  nbf_ += shell.size();
  max_l_ = std::max(max_l_, shell.l);
  max_nprim_ = std::max(max_nprim_, shell.n_primitives());
  shells_.push_back(std::move(shell));
}

std::size_t BasisSet::function_atom(std::size_t mu) const {
  const auto it = std::upper_bound(offsets_.begin(), offsets_.end(), mu);
  return shells_[static_cast<std::size_t>(it - offsets_.begin()) - 1].atom;
}

}  // namespace qk
