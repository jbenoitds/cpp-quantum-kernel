#include "qk/fock.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "qk/parallel.hpp"

namespace qk {

namespace {

/// Accumulate one integral with degeneracy-weighted value into the raw
/// (unsymmetrised) J and K accumulators.  After all unique quartets,
///   J = (GJ + GJ^T) / 4,   K = (GK + GK^T) / 8.
inline void digest(double* GJ, double* GK, const double* D, std::size_t n, std::size_t i, std::size_t j,
                   std::size_t k, std::size_t l, double v) {
  GJ[i * n + j] += D[k * n + l] * v;
  GJ[k * n + l] += D[i * n + j] * v;
  GK[i * n + k] += D[j * n + l] * v;
  GK[j * n + l] += D[i * n + k] * v;
  GK[i * n + l] += D[j * n + k] * v;
  GK[j * n + k] += D[i * n + l] * v;
}

JK finalize(const std::vector<double>& GJ, const std::vector<double>& GK, std::size_t n) {
  JK jk{Matrix(n, n), Matrix(n, n)};
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      jk.J(i, j) = 0.25 * (GJ[i * n + j] + GJ[j * n + i]);
      jk.K(i, j) = 0.125 * (GK[i * n + j] + GK[j * n + i]);
    }
  }
  return jk;
}

/// Sum thread-private accumulators into acc[0] (pairwise loop, deterministic order).
void reduce(std::vector<std::vector<double>>& acc) {
  for (std::size_t t = 1; t < acc.size(); ++t) {
    if (acc[t].empty()) continue;  // thread did not take part in the region
    for (std::size_t k = 0; k < acc[0].size(); ++k) acc[0][k] += acc[t][k];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// In-core tensor
// ---------------------------------------------------------------------------

EriTensor::EriTensor(const EriEngine& engine, double schwarz_tol) {
  const BasisSet& basis = engine.basis();
  nbf_ = basis.n_functions();
  data_.assign(packed_size(nbf_), 0.0);

  const auto& pairs = engine.shell_pairs();
  const int npairs = static_cast<int>(pairs.size());
  long long computed = 0, screened = 0;

#pragma omp parallel reduction(+ : computed, screened)
  {
    std::vector<double> buf(engine.max_block_size());
#pragma omp for schedule(dynamic, 1)
    for (int ip = npairs - 1; ip >= 0; --ip) {
      const ShellPair& bra = pairs[static_cast<std::size_t>(ip)];
      const std::size_t oP = basis.offset(bra.P), oQ = basis.offset(bra.Q);
      for (std::size_t iq = 0; iq <= static_cast<std::size_t>(ip); ++iq) {
        const ShellPair& ket = pairs[iq];
        if (bra.schwarz * ket.schwarz < schwarz_tol) {
          ++screened;
          continue;
        }
        ++computed;
        engine.compute(bra, ket, buf.data());
        const std::size_t oR = basis.offset(ket.P), oS = basis.offset(ket.Q);
        std::size_t idx = 0;
        for (std::size_t a = 0; a < bra.nfa; ++a)
          for (std::size_t b = 0; b < bra.nfb; ++b) {
            const std::size_t ij = pair_index(oP + a, oQ + b);
            for (std::size_t c = 0; c < ket.nfa; ++c)
              for (std::size_t d = 0; d < ket.nfb; ++d, ++idx)
                data_[pair_index(ij, pair_index(oR + c, oS + d))] = buf[idx];
          }
      }
    }
  }
  computed_ = static_cast<std::size_t>(computed);
  screened_ = static_cast<std::size_t>(screened);
}

JK build_jk(const EriTensor& eri, const Matrix& D) {
  const std::size_t n = eri.n_functions();
  if (D.rows() != n || D.cols() != n) throw std::invalid_argument("qk::build_jk: density has wrong shape");
  const std::size_t npair = n * (n + 1) / 2;
  std::vector<std::size_t> pi(npair), pj(npair);
  for (std::size_t i = 0, ij = 0; i < n; ++i)
    for (std::size_t j = 0; j <= i; ++j, ++ij) {
      pi[ij] = i;
      pj[ij] = j;
    }

  const int nt = max_threads();
  std::vector<std::vector<double>> GJ(static_cast<std::size_t>(nt)), GK(static_cast<std::size_t>(nt));
  const double* Dp = D.data();
  const double* data = eri.data().data();
  const int np = static_cast<int>(npair);

#pragma omp parallel
  {
    const std::size_t t = static_cast<std::size_t>(thread_id());
    GJ[t].assign(n * n, 0.0);
    GK[t].assign(n * n, 0.0);
    double* gj = GJ[t].data();
    double* gk = GK[t].data();
#pragma omp for schedule(dynamic, 8)
    for (int sij = np - 1; sij >= 0; --sij) {
      const std::size_t ij = static_cast<std::size_t>(sij);
      const std::size_t i = pi[ij], j = pj[ij];
      const double dij = (i == j) ? 1.0 : 2.0;
      const double* row = data + ij * (ij + 1) / 2;
      for (std::size_t kl = 0; kl <= ij; ++kl) {
        const double v = row[kl];
        if (v == 0.0) continue;
        const std::size_t k = pi[kl], l = pj[kl];
        const double deg = dij * ((k == l) ? 1.0 : 2.0) * ((ij == kl) ? 1.0 : 2.0);
        digest(gj, gk, Dp, n, i, j, k, l, v * deg);
      }
    }
  }
  reduce(GJ);
  reduce(GK);
  return finalize(GJ[0], GK[0], n);
}

// ---------------------------------------------------------------------------
// Integral-direct build
// ---------------------------------------------------------------------------

JK build_jk_direct(const EriEngine& engine, const Matrix& D, double tol, DirectStats* stats) {
  const BasisSet& basis = engine.basis();
  const std::size_t n = basis.n_functions();
  const std::size_t ns = basis.n_shells();
  if (D.rows() != n || D.cols() != n) throw std::invalid_argument("qk::build_jk_direct: density has wrong shape");

  // Shell-block maxima of |D| for density screening.
  Matrix Dsh(ns, ns);
  for (std::size_t P = 0; P < ns; ++P)
    for (std::size_t Q = 0; Q < ns; ++Q) {
      double m = 0.0;
      for (std::size_t a = 0; a < basis.shell(P).size(); ++a)
        for (std::size_t b = 0; b < basis.shell(Q).size(); ++b)
          m = std::max(m, std::abs(D(basis.offset(P) + a, basis.offset(Q) + b)));
      Dsh(P, Q) = m;
    }
  const double dmax_global = max_abs(Dsh);

  // Significant shell pairs (canonical order preserved).
  const auto& pairs = engine.shell_pairs();
  std::vector<std::size_t> sig;
  sig.reserve(pairs.size());
  for (std::size_t k = 0; k < pairs.size(); ++k)
    if (pairs[k].schwarz * engine.max_schwarz() * dmax_global >= tol && !pairs[k].p.empty()) sig.push_back(k);

  const int nt = max_threads();
  std::vector<std::vector<double>> GJ(static_cast<std::size_t>(nt)), GK(static_cast<std::size_t>(nt));
  const double* Dp = D.data();
  const int nsig = static_cast<int>(sig.size());
  long long computed = 0, screened = 0;

#pragma omp parallel reduction(+ : computed, screened)
  {
    const std::size_t t = static_cast<std::size_t>(thread_id());
    GJ[t].assign(n * n, 0.0);
    GK[t].assign(n * n, 0.0);
    double* gj = GJ[t].data();
    double* gk = GK[t].data();
    std::vector<double> buf(engine.max_block_size());

#pragma omp for schedule(dynamic, 1)
    for (int si = nsig - 1; si >= 0; --si) {
      const ShellPair& bra = pairs[sig[static_cast<std::size_t>(si)]];
      const std::size_t P = bra.P, Q = bra.Q;
      const std::size_t oP = basis.offset(P), oQ = basis.offset(Q);
      const double dpq = Dsh(P, Q);
      for (int sj = 0; sj <= si; ++sj) {
        const ShellPair& ket = pairs[sig[static_cast<std::size_t>(sj)]];
        const std::size_t R = ket.P, S = ket.Q;
        const double dmax = std::max({2.0 * dpq, 2.0 * Dsh(R, S), Dsh(P, R), Dsh(P, S), Dsh(Q, R), Dsh(Q, S)});
        if (bra.schwarz * ket.schwarz * dmax < tol) {
          ++screened;
          continue;
        }
        ++computed;
        engine.compute(bra, ket, buf.data());
        const double deg = ((P == Q) ? 1.0 : 2.0) * ((R == S) ? 1.0 : 2.0) * ((si == sj) ? 1.0 : 2.0);
        const std::size_t oR = basis.offset(R), oS = basis.offset(S);
        std::size_t idx = 0;
        for (std::size_t a = 0; a < bra.nfa; ++a)
          for (std::size_t b = 0; b < bra.nfb; ++b)
            for (std::size_t c = 0; c < ket.nfa; ++c)
              for (std::size_t d = 0; d < ket.nfb; ++d, ++idx)
                digest(gj, gk, Dp, n, oP + a, oQ + b, oR + c, oS + d, buf[idx] * deg);
      }
    }
  }
  reduce(GJ);
  reduce(GK);
  if (stats) {
    stats->quartets_computed = static_cast<std::size_t>(computed);
    stats->quartets_screened = static_cast<std::size_t>(screened);
  }
  return finalize(GJ[0], GK[0], n);
}

}  // namespace qk
