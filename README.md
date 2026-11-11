# cpp-quantum-kernel

A small C++17 library for the core of ab initio quantum chemistry: Gaussian integrals, two-electron repulsion integrals, Hartree–Fock and MP2, plus a grid-based 1D Hartree–Fock solver. It has no dependencies beyond a C++17 compiler and CMake (OpenMP optional).

I wrote it in 2024, during my chemistry degree, because I wanted to understand what actually happens inside programs like PySCF or Gaussian when you press "run". So everything is written from scratch, including the eigensolver, the Boys function and the integral recurrences, and I checked it against PySCF/libcint until it agreed to machine precision (see [Validation](#validation)).

```text
$ ./build/qk-scf examples/h2o_crawford.xyz --bohr --mp2 --quiet
RHF total energy          -74.9420799282  Eh      <- same as the published reference to 12 digits
MP2 correlation            -0.0491496362
```

## Project history

I started it in February 2024. I wrote the code, and I used Claude (an AI assistant) to brainstorm the design of the integral engine and to review my code. The project lived on my laptop without version control until I published it in November 2026, which is also when I added the CI. This README was rewritten with Claude's help.

## What it does

| Area | Features |
|---|---|
| **Integrals** | Overlap, kinetic energy, nuclear attraction, dipole moments, and two-electron repulsion integrals (ERIs) over contracted Cartesian Gaussians, s through i shells (l ≤ 6) |
| **ERI handling** | In-core storage using all 8 index symmetries (about N⁴/8 doubles), or *integral-direct* evaluation (O(N²) memory) with Schwarz screening, density-weighted screening and incremental Fock builds |
| **Methods** | Closed-shell restricted Hartree–Fock (RHF) with DIIS; MP2 (with optional frozen core); dipole moment; Mulliken charges |
| **1D model systems** | Closed-shell Hartree–Fock for soft-Coulomb atoms and molecules on a grid, using a spectrally accurate sinc-DVR kinetic operator |
| **Basis sets** | STO-3G, 6-31G and 6-31G\* built in for H–Ar; any basis from the [Basis Set Exchange](https://www.basissetexchange.org) can be loaded in Gaussian94 format |
| **Parallelism** | OpenMP for integral, Fock and MP2 kernels, using per-thread accumulators so no atomics are needed. Works with GCC, Clang and MSVC (OpenMP 2.0), and builds serially without OpenMP |
| **Tooling** | Command-line programs, a benchmark, a test suite with no external dependencies, sanitizer builds, CI, and a script that cross-checks every integral against PySCF |

## Building

Requirements: CMake ≥ 3.16 and a C++17 compiler (GCC ≥ 9, Clang ≥ 10, or MSVC 2019/2022). OpenMP is used automatically if the compiler supports it.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure        # 7 test groups, ~5 s
```

On Windows with Visual Studio, run the same commands from a *Developer PowerShell*, and add `-C Release` to `ctest`.

| CMake option | Default | Meaning |
|---|---|---|
| `QK_ENABLE_OPENMP` | `ON` | Parallelise the integral, Fock and MP2 kernels |
| `QK_NATIVE_ARCH` | `OFF` | Compile with `-march=native` (GCC/Clang) |
| `QK_SANITIZE` | `OFF` | Build with AddressSanitizer + UndefinedBehaviorSanitizer |
| `QK_BUILD_TESTS` / `QK_BUILD_APPS` / `QK_BUILD_BENCH` | `ON` | Build the tests, the programs and the benchmark |

To use it from another CMake project, add `add_subdirectory(cpp-quantum-kernel)` and then `target_link_libraries(your_target PRIVATE qk::qk)`. Alternatively, run `cmake --install build` and use the exported `qk::qk` target.

## Using it

### Command line

```bash
# RHF + MP2 on water in 6-31G* (XYZ coordinates in Angstrom)
./build/qk-scf examples/h2o.xyz --basis 6-31g* --mp2 --frozen-core 1

# Any basis from basissetexchange.org (Gaussian94 format), integral-direct, 4 threads
./build/qk-scf examples/benzene.xyz --basis-file my-basis.g94 --direct --threads 4

# 1D soft-Coulomb H2 molecule at R = 1.6 bohr; write the density and orbitals to CSV
./build/qk-hf1d --nucleus 1:-0.8 --nucleus 1:0.8 --electrons 2 --output h2_density.csv

# Throughput benchmark on water clusters
./build/qk-bench --basis 6-31g* --waters 1,2,4,8 --scf
```

`qk-scf` prints the SCF iterations, the energy breakdown, orbital energies (occupied orbitals marked `*`), the dipole moment in Debye, Mulliken charges and timings. Run either program with `--help` for all options.

### As a library

```cpp
#include "qk/qk.hpp"

const qk::Molecule water = qk::Molecule::from_string("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692");
const qk::BasisSet basis(water, "6-31g*");

const qk::EriEngine engine(basis);   // per-shell-pair data + Schwarz bounds
const qk::EriTensor eri(engine);     // all unique (mn|ls), 8-fold packed
const qk::ScfResult scf = qk::run_rhf(water, basis, {}, &eri);
const qk::Mp2Result mp2 = qk::mp2_energy(scf, eri, /*n_frozen_core=*/1);
// scf.energy = -76.0105049883, scf.energy + mp2.correlation_energy = -76.1966245281

qk::hf1d::System helium;
helium.nuclei = {{2.0, 0.0}};
helium.n_electrons = 2;
double e1d = qk::hf1d::solve(helium).energy;   // -2.2242095530
```

The full program is [`examples/library_usage.cpp`](examples/library_usage.cpp) (built as `qk-example`). The lower-level pieces are public too: `boys_function`, `hermite_expansion_1d`, `hermite_coulomb`, `EriEngine::compute` (one shell quartet), `build_jk` / `build_jk_direct`, and `eigh`.

---

## How it works

### 1. Basis functions

A *shell* is a set of contracted Cartesian Gaussians that share a centre **A**, an angular momentum *l* and a list of primitive exponents:

$$
\phi_{l_x l_y l_z}(\mathbf r) = \sum_k c_k \, N_k \, x_A^{l_x} y_A^{l_y} z_A^{l_z}\, e^{-\alpha_k r_A^2}, \qquad l_x+l_y+l_z = l .
$$

A shell of angular momentum *l* holds (l+1)(l+2)/2 Cartesian components, in the order used by Gaussian, libint and PySCF (for d: xx, xy, xz, yy, yz, zz). Each component is normalised to one. Primitive and contraction normalisation are folded into the stored coefficients once, and the per-component factor $\sqrt{(2l-1)!!/[(2l_x-1)!!(2l_y-1)!!(2l_z-1)!!]}$ is applied inside the integral kernels. The Gaussian94 parser handles `SP` shells, Fortran `D` exponents and scale factors, and comments.

### 2. McMurchie–Davidson integrals

The product of two Gaussians on centres A and B is a single Gaussian on the weighted centre **P** = (αA + βB)/p, with p = α + β. Its polynomial part can be written exactly as a short sum of *Hermite Gaussians* Λ_tuv centred at P:

$$
G_a G_b = \sum_{t,u,v} E^{ab}_{t}\,E^{ab}_{u}\,E^{ab}_{v}\;\Lambda_{tuv}(p, \mathbf r_P).
$$

The 1D expansion coefficients come from a simple recurrence ([`hermite.cpp`](src/hermite.cpp)):

$$
E^{i+1,j}_t = \tfrac{1}{2p}E^{ij}_{t-1} + X_{PA}E^{ij}_t + (t+1)E^{ij}_{t+1}.
$$

Overlap and kinetic integrals follow directly from the t = 0 coefficients. Coulomb-type integrals reduce to the Hermite integrals $R_{tuv}$, built by downward recursion from the **Boys function** $F_n(T)=\int_0^1 t^{2n}e^{-Tt^2}\,dt$:

$$
R^{(n)}_{t+1,u,v} = t\,R^{(n+1)}_{t-1,u,v} + X_{PC}\,R^{(n+1)}_{tuv},
\qquad R^{(n)}_{000} = (-2\alpha)^n F_n(\alpha R_{PC}^2).
$$

Only two (L+1)³ buffers are kept, alternating between levels n and n+1. An ERI is then

$$
(ab|cd) = \frac{2\pi^{5/2}}{pq\sqrt{p+q}} \sum_{tuv}E^{ab}_{tuv}\sum_{\tau\nu\phi}(-1)^{\tau+\nu+\phi}E^{cd}_{\tau\nu\phi}\,R_{t+\tau,\,u+\nu,\,v+\phi}\!\left(\tfrac{pq}{p+q},\mathbf R_{PQ}\right).
$$

**Boys function** ([`boys.cpp`](src/boys.cpp)). For T < 40, it uses an 8-term Taylor expansion of the highest order needed around a precomputed grid (spacing 0.1, 128 KB), then the numerically stable downward recursion. For T ≥ 40, it starts from F₀ = ½√(π/T)·erf(√T) and recurses upward, which is stable there because T > n. The accuracy is better than 1e-13 relative, checked against Gauss–Legendre quadrature.

### 3. Engine design: precomputed shell pairs, small dense contractions

[`EriEngine`](include/qk/eri.hpp) does as much as possible once, at the shell-pair level. For every pair of shells (P ≥ Q), and every pair of primitives in them that survives screening, it stores p, **P**, and the complete Hermite coefficient matrix. That matrix has one row per Cartesian component pair and one column per Hermite index, with the contraction coefficients, normalisation and 1/p already folded in. The Schwarz bound $Q_{PQ}=\sqrt{\max|(PQ|PQ)|}$ is stored alongside.

Evaluating a shell quartet then becomes two small dense matrix products per primitive quartet, with the sum over ket primitives done before the bra contraction:

```
for each bra primitive pair i:
    G = 0
    for each ket primitive pair j:
        R  = hermite_coulomb(L, alpha_ij, P_i - Q_j)         # one Boys evaluation
        G += R[tuv + tau nu phi] * sign * E_ket_j^T          # (n_herm_bra x n_cd)
    block += E_bra_i * G                                     # (n_ab x n_cd)
```

The R index of a combined Hermite index (t+τ, u+ν, v+φ) is simply the sum of two precomputed offsets, so the inner loops contain no index arithmetic and read memory contiguously. Scratch buffers are `thread_local`, so `compute()` is thread-safe and does not allocate in the hot loop.

### 4. Symmetry and memory

Every ERI equals seven others under index swaps: (ij|kl) = (ji|kl) = (ij|lk) = (kl|ij) = …. [`EriTensor`](include/qk/fock.hpp) stores only the unique ones, at the packed index `ijkl = ij(ij+1)/2 + kl` with `ij = i(i+1)/2 + j`. That is about $\tfrac{1}{8}N^4$ doubles: 33 MB for N = 76 and 516 MB for N = 152.

Only symmetry-unique shell quartets are computed. Each one is *digested* into J and K with a degeneracy weight (1, 2, 4 or 8), and the result is symmetrised once at the end:

$$
J = \tfrac14\,(G_J + G_J^{\mathsf T}),\qquad K = \tfrac18\,(G_K + G_K^{\mathsf T}).
$$

### 5. Screening and the direct algorithm

`build_jk_direct` never stores integrals; its memory use is O(N²). It skips a quartet when

$$
Q_{PQ}\,Q_{RS}\,\max\left(2|D_{PQ}|,\,2|D_{RS}|,\,|D_{PR}|,\,|D_{PS}|,\,|D_{QR}|,\,|D_{QS}|\right) < \tau \quad (\tau = 10^{-12}),
$$

which bounds the quartet's contribution to F = H + 2J − K. In the SCF, the Fock matrix is built *incrementally*, $G(D_n) = G(D_{n-1}) + G(D_n - D_{n-1})$. As the density converges, ΔD shrinks and more quartets are screened out; for a water trimer the count falls from 30k to 15k per iteration. A full rebuild every 8 iterations stops screening errors from accumulating. `EriStrategy::Auto` chooses in-core storage when the packed tensor fits in `max_memory_mb`, and direct otherwise.

### 6. Parallelism

Each OpenMP thread accumulates into its own copy of the J/K arrays, and the copies are summed in a fixed order at the end. There are no atomics or critical sections in the hot loop. Outer loops run over shell pairs in reverse with `schedule(dynamic)`, so the expensive large-index pairs are handed out first. Loop indices are signed `int`, which MSVC's OpenMP 2.0 requires. Results do not depend on the thread count; a test checks this.

### 7. SCF and MP2

[`run_rhf`](src/scf.cpp) follows the standard Roothaan–Hall procedure:

1. Build S, H = T + V, and an orthogonaliser X with XᵀSX = 1. X is the symmetric (Löwdin) S^{-1/2}, or a canonical orthogonaliser that drops eigenvalues below 1e-8 if the basis is nearly linearly dependent.
2. Start from the core-Hamiltonian guess.
3. Iterate F = H + 2J − K, E = Σ D(H + F) + E_nuc.
4. Accelerate convergence with Pulay **DIIS**, using the orthogonalised commutator Xᵀ(FDS − SDF)X as the error vector. The DIIS equations are solved with partial pivoting, and the oldest vectors are dropped if they become ill-conditioned.
5. Stop when |ΔE| < 1e-10 and rms(error) < 1e-8.

[`mp2_energy`](src/scf.cpp) transforms the integrals to the MO basis in two half-transformations, O(N⁵) in total, and returns the same-spin and opposite-spin parts separately.

### 8. 1D Hartree–Fock on a grid

[`hf1d`](include/qk/hf1d.hpp) solves closed-shell HF for electrons on a line interacting through the soft-Coulomb potential $w(x)=\lambda/\sqrt{x^2+a^2}$, the standard regularised model for 1D atoms (Eberly et al.).

- **Kinetic energy.** The Colbert–Miller **sinc-DVR** matrix $T_{ij} = \frac{(-1)^{i-j}}{2\Delta x^2}\{\pi^2/3 \text{ if } i=j;\; 2/(i-j)^2 \text{ otherwise}\}$ converges spectrally: the He energy is already converged to 1e-12 at Δx = 0.2.
- **Hartree and exchange.** These are exact on the grid: $v_H(x_i) = 2\sum_j D_{jj}w_{ij}$ and $K_{ij} = D_{ij}w_{ij}$.
- **Memory.** The interaction depends only on |i − j|, so only the N-vector $w(k\Delta x)$ is stored, not an N × N kernel.
- **Convergence.** It uses the same DIIS as the 3D code.

### 9. Linear algebra

[`linalg.cpp`](src/linalg.cpp) provides a row-major `Matrix`, GEMM in cache-friendly i-k-j order, and a symmetric eigensolver. The eigensolver uses Householder tridiagonalisation followed by implicit QL (the classic EISPACK tred2/tql2 algorithms). It runs on the *transpose* of the textbook working matrix, so that every O(n³) inner loop reads memory contiguously. That change alone made the n = 800 eigensolve 2.6× faster (2.20 s → 0.86 s).

---

## Validation

[`tools/validate_pyscf.py`](tools/validate_pyscf.py) makes `qk-scf` write its integrals and its exact list of shells. It then rebuilds the identical basis in PySCF 2.6.2 (`cart=True`) and compares every matrix element.

| Check | Result |
|---|---|
| S, T, V, ERI vs PySCF — H₂O, NH₃, CH₄, HCl, benzene × STO-3G / 6-31G / 6-31G\* | max \|Δ\| ≤ 1.4e-13 (one-electron), ≤ 3e-14 (ERI)¹ |
| Synthetic s, p, d, f, g basis on H₂O — includes (gg\|gg) integrals | max \|ΔERI\| = 2.0e-14 |
| RHF total energies vs PySCF, all 16 cases above | \|ΔE\| ≤ 5e-11 Eh |
| H₂O/STO-3G, Crawford programming projects #3/#4 | E_RHF = −74.942079928192, E_MP2 = −0.049149636120, μ = 0.603521 a.u., q_O = −0.253146, all reproduced |
| H₂/STO-3G integrals and energy, Szabo & Ostlund Table 3.6 | all to the 4 published digits |
| MP2 (all-electron and frozen-core) vs PySCF, H₂O/6-31G\* | \|ΔE\| < 1e-9 |
| 1D He / H₂ / Be vs an independent NumPy implementation ([`tools/hf1d_reference.py`](tools/hf1d_reference.py)) | identical to 1e-12 |
| Physical and mathematical invariants | 8-fold ERI symmetry, translational invariance, Schwarz inequality, p and d ERIs equal derivatives of s ERIs with respect to the centre, harmonic oscillator exact, virial theorem |

¹ For benzene/6-31G\* (102 functions) the full N⁴ ERI array is too large to dump, so only S, T, V and the energy are compared there.

The test suite (`ctest`) contains all of these as regression tests with stored reference values, so PySCF is **not** needed to run it. It passes with GCC 13 (OpenMP, at 1–7 threads), Clang 18, and GCC with AddressSanitizer + UBSan.

## Performance

`qk-bench`, 6-31G\* (Cartesian), water clusters, Intel Xeon @ 2.1 GHz, **2 cores**, GCC 13 `-O3`:

| System | Basis functions | In-core ERIs | Unique integrals / s | Direct J+K (1 thread) | Direct J+K (2 threads) | Full RHF (direct) |
|---|---:|---:|---:|---:|---:|---:|
| (H₂O)₁ | 19 | 0.003 s | 6 M | 0.010 s | 0.003 s | 0.04 s |
| (H₂O)₂ | 38 | 0.03 s | 9 M | 0.051 s | 0.026 s | 0.35 s |
| (H₂O)₄ | 76 | 0.28 s | 15 M | 0.50 s | 0.26 s | 3.1 s |
| (H₂O)₈ | 152 | 2.97 s | 23 M | 4.69 s | 2.48 s | 27 s |
| Benzene | 102 | 1.19 s | 12 M | 2.41 s | 1.13 s | 10 s |

**Comparison with PySCF 2.6.2 (libcint), same machine, same 2 threads, (H₂O)₈/6-31G\*:**

| | qk | PySCF |
|---|---:|---:|
| One direct J+K build | 2.48 s | 2.39 s |
| Full in-core ERI tensor | 2.97 s | 1.80 s |

The direct Fock build is on par with libcint. The in-core tensor build is about 1.65× slower, because libcint uses Rys quadrature and heavily optimised, generated kernels. OpenMP speed-up on 2 cores is 1.9–2.0×. These timings come from a shared cloud VM and vary by roughly ±15% between runs.

---

## Project layout

```
include/qk/        public headers (qk.hpp includes everything)
  boys.hpp           Boys function
  hermite.hpp        McMurchie-Davidson E and R recurrences
  basis.hpp          shells, normalisation, built-in + Gaussian94 basis sets
  integrals.hpp      S, T, V, dipole
  eri.hpp            shell-pair data, ERI engine
  fock.hpp           packed ERI tensor, in-core and direct J/K builds
  scf.hpp            RHF, MP2, dipole, Mulliken
  hf1d.hpp           1D grid Hartree-Fock
  linalg.hpp         Matrix, GEMM, eigensolver, orthogonaliser
src/               implementation (basis_data.cpp is generated by tools/gen_basis_data.py)
apps/              qk-scf, qk-hf1d command-line programs
bench/             qk-bench
tests/             dependency-free test harness + 5 suites, test data
tools/             PySCF cross-check, NumPy 1D reference, basis-data generator
examples/          XYZ geometries, library_usage.cpp
.github/workflows/ CI: Ubuntu GCC/Clang/sanitizers, Windows MSVC, macOS
```

## Limitations and what I'd like to add next

Only Cartesian functions are supported for now. So 6-31G\* results match Gaussian's default (6D) and PySCF with `cart=True`, but correlation-consistent bases (cc-pVXZ), which load fine, are normally used with spherical harmonics, and their energies will differ slightly from the published values.

The SCF is closed-shell RHF only. UHF/ROHF, analytic gradients and density fitting would be the natural next steps.

Screening happens at the primitive-pair and shell-quartet levels, not yet at the primitive level. A Head-Gordon–Pople or Rys-quadrature kernel for high angular momentum would probably close most of the in-core gap with libcint (see [Performance](#performance)).

By the way, I developed on Linux with GCC and Clang. The Windows and macOS builds are set up in CI, but I never built them myself during development.

## References

1. L. E. McMurchie and E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978) — Hermite-Gaussian integral scheme.
2. T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure Theory* (Wiley, 2000), ch. 9.
3. A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry* (Dover, 1996) — H₂ and HeH⁺ reference values.
4. P. Pulay, *Chem. Phys. Lett.* **73**, 393 (1980); *J. Comput. Chem.* **3**, 556 (1982) — DIIS.
5. M. Häser and R. Ahlrichs, *J. Comput. Chem.* **10**, 104 (1989) — direct SCF and Schwarz screening.
6. D. T. Colbert and W. H. Miller, *J. Chem. Phys.* **96**, 1982 (1992) — sinc discrete variable representation.
7. J. H. Eberly, Q. Su, J. Javanainen, *Phys. Rev. Lett.* **62**, 881 (1989) — soft-Coulomb 1D atoms.
8. T. D. Crawford, *Programming Projects* (github.com/CrawfordGroup/ProgrammingProjects) — water STO-3G reference data.
9. Q. Sun et al., *J. Chem. Phys.* **153**, 024109 (2020) — PySCF, used for validation.

## License

MIT, see [LICENSE](LICENSE).
