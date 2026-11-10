#!/usr/bin/env python3
"""Independent NumPy/SciPy implementation of 1D soft-Coulomb RHF (used to
generate the reference energies in tests/test_hf1d.cpp)."""
import numpy as np
from scipy.linalg import eigh


def rhf1d(nuclei, nelec, a=1.0, lam=1.0, omega=0.0, xmin=-20.0, xmax=20.0, n=201):
    x = np.linspace(xmin, xmax, n)
    dx = x[1] - x[0]
    i = np.arange(n)
    d = i[:, None] - i[None, :]
    with np.errstate(divide="ignore"):
        T = np.where(d == 0, np.pi**2 / 3, 2.0 * (-1.0) ** np.abs(d) / np.where(d == 0, 1, d) ** 2)
    T /= 2 * dx**2
    v = 0.5 * omega**2 * x**2 - sum(Z / np.sqrt((x - X) ** 2 + a * a) for Z, X in nuclei)
    h = T + np.diag(v)
    W = lam / np.sqrt((x[:, None] - x[None, :]) ** 2 + a * a)
    enuc = sum(nuclei[A][0] * nuclei[B][0] / np.sqrt((nuclei[A][1] - nuclei[B][1]) ** 2 + a * a)
               for A in range(len(nuclei)) for B in range(A))
    nocc = nelec // 2
    _, C = eigh(h)
    D = C[:, :nocc] @ C[:, :nocc].T
    e_old = 0.0
    for it in range(500):
        F = h + np.diag(2 * W @ np.diag(D)) - D * W
        e = np.sum(D * (h + F)) + enuc
        if abs(e - e_old) < 1e-13:
            break
        e_old = e
        _, C = eigh(F)
        Dn = C[:, :nocc] @ C[:, :nocc].T
        D = 0.5 * D + 0.5 * Dn if it < 3 else Dn
    return e


if __name__ == "__main__":
    print("He  (Z=2, a=1)          %.12f" % rhf1d([(2, 0.0)], 2))
    print("H2  (R=1.6, a=1)        %.12f" % rhf1d([(1, -0.8), (1, 0.8)], 2))
    print("Be  (Z=4, 4e, a=1)      %.12f" % rhf1d([(4, 0.0)], 4, xmin=-25, xmax=25, n=301))
    print("He  converged grid      %.12f" % rhf1d([(2, 0.0)], 2, xmin=-30, xmax=30, n=601))
