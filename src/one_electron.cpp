#include <cmath>
#include <vector>

#include "qk/constants.hpp"
#include "qk/hermite.hpp"
#include "qk/integrals.hpp"

namespace qk {

namespace {

/// Everything a primitive-pair kernel needs.
struct PrimPairContext {
  int la, lb;  // shell angular momenta
  int LA, LB;  // dimensions used for the E tables (may exceed la, lb)
  double a, b, p;
  std::array<double, 3> P;
  double cc;  // c_a c_b exp(-mu |AB|^2)
  const double* E[3];

  double e(int dim, int i, int j, int t) const {
    if (t > i + j) return 0.0;
    return E[dim][(i * (LB + 1) + j) * (LA + LB + 1) + t];
  }
};

/// Generic driver: loops over unique shell pairs and primitive pairs, calls
/// `kernel(ctx, blocks)` which accumulates `ncomp` blocks of nfa*nfb values
/// (unnormalised Cartesian components), then applies the per-component
/// normalisation and scatters into symmetric matrices.
template <class Kernel>
std::vector<Matrix> one_electron_driver(const BasisSet& basis, int extra, std::size_t ncomp, Kernel&& kernel) {
  const std::size_t n = basis.n_functions();
  std::vector<Matrix> out(ncomp, Matrix(n, n));
  std::vector<double> block, Ebuf[3];

  for (std::size_t sa = 0; sa < basis.n_shells(); ++sa) {
    const Shell& A = basis.shell(sa);
    for (std::size_t sb = 0; sb <= sa; ++sb) {
      const Shell& B = basis.shell(sb);
      const std::size_t nfa = A.size(), nfb = B.size();
      block.assign(ncomp * nfa * nfb, 0.0);

      PrimPairContext ctx;
      ctx.la = A.l;
      ctx.lb = B.l;
      ctx.LA = A.l + extra;
      ctx.LB = B.l + extra;
      const std::size_t esize = static_cast<std::size_t>((ctx.LA + 1) * (ctx.LB + 1) * (ctx.LA + ctx.LB + 1));
      for (auto& e : Ebuf) e.resize(esize);

      double AB2 = 0.0;
      for (int k = 0; k < 3; ++k) AB2 += (A.center[k] - B.center[k]) * (A.center[k] - B.center[k]);

      for (std::size_t i = 0; i < A.n_primitives(); ++i) {
        for (std::size_t j = 0; j < B.n_primitives(); ++j) {
          ctx.a = A.exponents[i];
          ctx.b = B.exponents[j];
          ctx.p = ctx.a + ctx.b;
          const double mu = ctx.a * ctx.b / ctx.p;
          ctx.cc = A.coefficients[i] * B.coefficients[j] * std::exp(-mu * AB2);
          for (int k = 0; k < 3; ++k) {
            ctx.P[k] = (ctx.a * A.center[k] + ctx.b * B.center[k]) / ctx.p;
            hermite_expansion_1d(ctx.LA, ctx.LB, ctx.p, ctx.P[k] - A.center[k], ctx.P[k] - B.center[k], Ebuf[k].data());
            ctx.E[k] = Ebuf[k].data();
          }
          kernel(ctx, block.data());
        }
      }

      const auto& ca = cartesian_components(A.l);
      const auto& cb = cartesian_components(B.l);
      const std::size_t oa = basis.offset(sa), ob = basis.offset(sb);
      for (std::size_t c = 0; c < ncomp; ++c) {
        for (std::size_t fa = 0; fa < nfa; ++fa) {
          const double na = cartesian_norm_ratio(ca[fa].x, ca[fa].y, ca[fa].z);
          for (std::size_t fb = 0; fb < nfb; ++fb) {
            const double nb = cartesian_norm_ratio(cb[fb].x, cb[fb].y, cb[fb].z);
            const double v = block[(c * nfa + fa) * nfb + fb] * na * nb;
            out[c](oa + fa, ob + fb) = v;
            out[c](ob + fb, oa + fa) = v;
          }
        }
      }
    }
  }
  return out;
}

}  // namespace

Matrix overlap_matrix(const BasisSet& basis) {
  auto m = one_electron_driver(basis, 0, 1, [](const PrimPairContext& c, double* block) {
    const auto& ca = cartesian_components(c.la);
    const auto& cb = cartesian_components(c.lb);
    const double pref = c.cc * std::pow(kPi / c.p, 1.5);
    std::size_t k = 0;
    for (const auto& fa : ca)
      for (const auto& fb : cb)
        block[k++] += pref * c.e(0, fa.x, fb.x, 0) * c.e(1, fa.y, fb.y, 0) * c.e(2, fa.z, fb.z, 0);
  });
  return std::move(m[0]);
}

Matrix kinetic_matrix(const BasisSet& basis) {
  auto m = one_electron_driver(basis, 2, 1, [](const PrimPairContext& c, double* block) {
    const auto& ca = cartesian_components(c.la);
    const auto& cb = cartesian_components(c.lb);
    const double pref = c.cc * std::pow(kPi / c.p, 1.5);
    const double b = c.b;
    // 1D kinetic integral (without sqrt(pi/p)):
    //   T_ij = -2 b^2 S_{i,j+2} + b (2j+1) S_ij - j(j-1)/2 S_{i,j-2}
    auto t1d = [&](int dim, int i, int j) {
      double t = -2.0 * b * b * c.e(dim, i, j + 2, 0) + b * (2 * j + 1) * c.e(dim, i, j, 0);
      if (j >= 2) t -= 0.5 * j * (j - 1) * c.e(dim, i, j - 2, 0);
      return t;
    };
    std::size_t k = 0;
    for (const auto& fa : ca) {
      for (const auto& fb : cb) {
        const double sx = c.e(0, fa.x, fb.x, 0), sy = c.e(1, fa.y, fb.y, 0), sz = c.e(2, fa.z, fb.z, 0);
        const double tx = t1d(0, fa.x, fb.x), ty = t1d(1, fa.y, fb.y), tz = t1d(2, fa.z, fb.z);
        block[k++] += pref * (tx * sy * sz + sx * ty * sz + sx * sy * tz);
      }
    }
  });
  return std::move(m[0]);
}

Matrix nuclear_attraction_matrix(const BasisSet& basis, const std::vector<PointCharge>& charges) {
  std::vector<double> R, work;
  auto m = one_electron_driver(basis, 0, 1, [&](const PrimPairContext& c, double* block) {
    const int L = c.la + c.lb;
    const std::size_t d = static_cast<std::size_t>(L + 1);
    R.resize(d * d * d);
    work.resize(2 * d * d * d);
    const auto& ca = cartesian_components(c.la);
    const auto& cb = cartesian_components(c.lb);
    for (const PointCharge& q : charges) {
      const double pref = -q.charge * c.cc * 2.0 * kPi / c.p;
      hermite_coulomb(L, c.p, c.P[0] - q.xyz[0], c.P[1] - q.xyz[1], c.P[2] - q.xyz[2], pref, R.data(), work.data());
      std::size_t k = 0;
      for (const auto& fa : ca) {
        for (const auto& fb : cb) {
          double s = 0.0;
          for (int t = 0; t <= fa.x + fb.x; ++t) {
            const double ex = c.e(0, fa.x, fb.x, t);
            for (int u = 0; u <= fa.y + fb.y; ++u) {
              const double exy = ex * c.e(1, fa.y, fb.y, u);
              for (int v = 0; v <= fa.z + fb.z; ++v)
                s += exy * c.e(2, fa.z, fb.z, v) * R[(static_cast<std::size_t>(t) * d + u) * d + v];
            }
          }
          block[k++] += s;
        }
      }
    }
  });
  return std::move(m[0]);
}

Matrix nuclear_attraction_matrix(const BasisSet& basis, const Molecule& mol) {
  std::vector<PointCharge> q;
  q.reserve(mol.size());
  for (const Atom& a : mol.atoms()) q.push_back({static_cast<double>(a.Z), a.xyz});
  return nuclear_attraction_matrix(basis, q);
}

Matrix core_hamiltonian(const BasisSet& basis, const Molecule& mol) {
  return kinetic_matrix(basis) + nuclear_attraction_matrix(basis, mol);
}

std::array<Matrix, 3> dipole_matrices(const BasisSet& basis, const std::array<double, 3>& origin) {
  auto m = one_electron_driver(basis, 0, 3, [&](const PrimPairContext& c, double* block) {
    const auto& ca = cartesian_components(c.la);
    const auto& cb = cartesian_components(c.lb);
    const std::size_t nab = ca.size() * cb.size();
    const double pref = c.cc * std::pow(kPi / c.p, 1.5);
    // 1D first moment about the origin: E^{ij}_1 + X_PO E^{ij}_0
    auto m1d = [&](int dim, int i, int j) { return c.e(dim, i, j, 1) + (c.P[dim] - origin[dim]) * c.e(dim, i, j, 0); };
    std::size_t k = 0;
    for (const auto& fa : ca) {
      for (const auto& fb : cb) {
        const double sx = c.e(0, fa.x, fb.x, 0), sy = c.e(1, fa.y, fb.y, 0), sz = c.e(2, fa.z, fb.z, 0);
        block[k] += pref * m1d(0, fa.x, fb.x) * sy * sz;
        block[nab + k] += pref * sx * m1d(1, fa.y, fb.y) * sz;
        block[2 * nab + k] += pref * sx * sy * m1d(2, fa.z, fb.z);
        ++k;
      }
    }
  });
  return {std::move(m[0]), std::move(m[1]), std::move(m[2])};
}

}  // namespace qk
