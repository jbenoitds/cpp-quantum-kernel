#include "qk/eri.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "qk/boys.hpp"
#include "qk/constants.hpp"
#include "qk/hermite.hpp"
#include "qk/parallel.hpp"

namespace qk {

namespace {

ShellPair build_pair(const Shell& A, const Shell& B, std::size_t P, std::size_t Q, double tol) {
  ShellPair sp;
  sp.P = P;
  sp.Q = Q;
  sp.la = A.l;
  sp.lb = B.l;
  sp.L = A.l + B.l;
  sp.nfa = A.size();
  sp.nfb = B.size();
  sp.nherm = n_hermite(sp.L);

  const auto& herm = hermite_indices(sp.L);
  const auto& ca = cartesian_components(A.l);
  const auto& cb = cartesian_components(B.l);
  std::vector<double> na(sp.nfa), nb(sp.nfb);
  for (std::size_t f = 0; f < sp.nfa; ++f) na[f] = cartesian_norm_ratio(ca[f].x, ca[f].y, ca[f].z);
  for (std::size_t f = 0; f < sp.nfb; ++f) nb[f] = cartesian_norm_ratio(cb[f].x, cb[f].y, cb[f].z);

  double AB2 = 0.0;
  for (int k = 0; k < 3; ++k) AB2 += (A.center[k] - B.center[k]) * (A.center[k] - B.center[k]);

  const int nt = sp.L + 1;
  const int nj = B.l + 1;
  std::vector<double> Ex[3];
  for (auto& e : Ex) e.resize(static_cast<std::size_t>((A.l + 1) * nj * nt));
  auto e1 = [&](int dim, int i, int j, int t) { return Ex[dim][static_cast<std::size_t>((i * nj + j) * nt + t)]; };

  const std::size_t block = sp.nfa * sp.nfb * sp.nherm;
  for (std::size_t i = 0; i < A.n_primitives(); ++i) {
    for (std::size_t j = 0; j < B.n_primitives(); ++j) {
      const double a = A.exponents[i], b = B.exponents[j];
      const double p = a + b;
      const double cc = A.coefficients[i] * B.coefficients[j] * std::exp(-a * b / p * AB2);
      if (std::abs(cc) * std::pow(kPi / p, 1.5) < tol) continue;

      std::array<double, 3> Pc{};
      for (int k = 0; k < 3; ++k) {
        Pc[k] = (a * A.center[k] + b * B.center[k]) / p;
        hermite_expansion_1d(A.l, B.l, p, Pc[k] - A.center[k], Pc[k] - B.center[k], Ex[k].data());
      }
      sp.p.push_back(p);
      sp.center.push_back(Pc);
      const std::size_t base = sp.E.size();
      sp.E.resize(base + block, 0.0);
      double* E = sp.E.data() + base;
      for (std::size_t fa = 0; fa < sp.nfa; ++fa) {
        for (std::size_t fb = 0; fb < sp.nfb; ++fb) {
          const double f = cc * na[fa] * nb[fb] / p;
          const CartesianExponents& x = ca[fa];
          const CartesianExponents& y = cb[fb];
          double* row = E + (fa * sp.nfb + fb) * sp.nherm;
          for (std::size_t k = 0; k < sp.nherm; ++k) {
            const HermiteIndex& h = herm[k];
            if (h.t > x.x + y.x || h.u > x.y + y.y || h.v > x.z + y.z) continue;
            row[k] = f * e1(0, x.x, y.x, h.t) * e1(1, x.y, y.y, h.u) * e1(2, x.z, y.z, h.v);
          }
        }
      }
    }
  }
  return sp;
}

}  // namespace

EriEngine::EriEngine(const BasisSet& basis, double primitive_tol) : basis_(&basis) {
  if (basis.max_l() * 4 > kBoysMaxOrder) throw std::invalid_argument("qk::EriEngine: angular momentum too high");
  const std::size_t ns = basis.n_shells();
  pairs_.resize(ns * (ns + 1) / 2);
  for (std::size_t P = 0; P < ns; ++P) {
    for (std::size_t Q = 0; Q <= P; ++Q)
      pairs_[pair_index(P, Q)] = build_pair(basis.shell(P), basis.shell(Q), P, Q, primitive_tol);
  }
  std::size_t maxpair = 0;
  for (const auto& sp : pairs_) maxpair = std::max(maxpair, sp.n_functions());
  max_block_ = maxpair * maxpair;

  // Schwarz factors sqrt(max |(PQ|PQ)|).
  const int npairs = static_cast<int>(pairs_.size());
#pragma omp parallel
  {
    std::vector<double> buf(max_block_);
#pragma omp for schedule(dynamic, 4)
    for (int ip = 0; ip < npairs; ++ip) {
      ShellPair& sp = pairs_[static_cast<std::size_t>(ip)];
      compute(sp, sp, buf.data());
      const std::size_t n = sp.n_functions();
      double m = 0.0;
      for (std::size_t f = 0; f < n; ++f) m = std::max(m, std::abs(buf[f * n + f]));
      sp.schwarz = std::sqrt(m);
    }
  }
  for (const auto& sp : pairs_) max_schwarz_ = std::max(max_schwarz_, sp.schwarz);
}

void EriEngine::compute(const ShellPair& bra, const ShellPair& ket, double* out) const {
  const std::size_t nab = bra.n_functions();
  const std::size_t ncd = ket.n_functions();
  std::fill(out, out + nab * ncd, 0.0);
  if (bra.p.empty() || ket.p.empty()) return;

  const int L = bra.L + ket.L;
  const std::size_t d = static_cast<std::size_t>(L + 1);
  const std::size_t nhb = bra.nherm, nhk = ket.nherm;

  thread_local std::vector<double> R, work, G, tmp;
  thread_local std::vector<std::size_t> off_b, off_k;
  thread_local std::vector<double> sign_k;
  R.resize(d * d * d);
  work.resize(2 * d * d * d);
  G.resize(nhb * ncd);
  tmp.resize(nhk);
  off_b.resize(nhb);
  off_k.resize(nhk);
  sign_k.resize(nhk);

  // R index of (t+tau, u+nu, v+phi) is additive: off(t,u,v) + off(tau,nu,phi).
  const auto& hb = hermite_indices(bra.L);
  const auto& hk = hermite_indices(ket.L);
  for (std::size_t k = 0; k < nhb; ++k) off_b[k] = (static_cast<std::size_t>(hb[k].t) * d + hb[k].u) * d + hb[k].v;
  for (std::size_t k = 0; k < nhk; ++k) {
    off_k[k] = (static_cast<std::size_t>(hk[k].t) * d + hk[k].u) * d + hk[k].v;
    sign_k[k] = ((hk[k].t + hk[k].u + hk[k].v) & 1) ? -1.0 : 1.0;
  }

  const double two_pi_52 = 2.0 * std::pow(kPi, 2.5);

  for (std::size_t i = 0; i < bra.p.size(); ++i) {
    const double p = bra.p[i];
    const auto& P = bra.center[i];
    std::fill(G.begin(), G.end(), 0.0);

    for (std::size_t j = 0; j < ket.p.size(); ++j) {
      const double q = ket.p[j];
      const auto& Q = ket.center[j];
      const double pq = p + q;
      const double alpha = p * q / pq;
      hermite_coulomb(L, alpha, P[0] - Q[0], P[1] - Q[1], P[2] - Q[2], two_pi_52 / std::sqrt(pq), R.data(), work.data());

      const double* Ek = ket.E.data() + j * ncd * nhk;
      for (std::size_t kb = 0; kb < nhb; ++kb) {
        const double* Rb = R.data() + off_b[kb];
        for (std::size_t kk = 0; kk < nhk; ++kk) tmp[kk] = sign_k[kk] * Rb[off_k[kk]];
        double* Gk = G.data() + kb * ncd;
        for (std::size_t fcd = 0; fcd < ncd; ++fcd) {
          const double* e = Ek + fcd * nhk;
          double s = 0.0;
          for (std::size_t kk = 0; kk < nhk; ++kk) s += tmp[kk] * e[kk];
          Gk[fcd] += s;
        }
      }
    }

    const double* Eb = bra.E.data() + i * nab * nhb;
    for (std::size_t fab = 0; fab < nab; ++fab) {
      double* o = out + fab * ncd;
      const double* e = Eb + fab * nhb;
      for (std::size_t kb = 0; kb < nhb; ++kb) {
        const double c = e[kb];
        if (c == 0.0) continue;
        const double* Gk = G.data() + kb * ncd;
        for (std::size_t fcd = 0; fcd < ncd; ++fcd) o[fcd] += c * Gk[fcd];
      }
    }
  }
}

std::vector<double> EriEngine::compute_shells(std::size_t P, std::size_t Q, std::size_t R, std::size_t S) const {
  const BasisSet& b = *basis_;
  const ShellPair bra = build_pair(b.shell(P), b.shell(Q), P, Q, 0.0);
  const ShellPair ket = build_pair(b.shell(R), b.shell(S), R, S, 0.0);
  std::vector<double> out(bra.n_functions() * ket.n_functions());
  compute(bra, ket, out.data());
  return out;
}

}  // namespace qk
