#!/usr/bin/env python3
"""Cross-check qk integrals and energies against PySCF (libcint).

    pip install pyscf==2.6.2
    python tools/validate_pyscf.py build/qk-scf examples/h2o.xyz 6-31g*
    python tools/validate_pyscf.py build/qk-scf examples/h2o.xyz tests/data/spdf.g94 --basis-file

qk-scf writes its S, T, V and ERI arrays plus the exact shell list (basis.json)
with --dump; the same shells are fed to PySCF with cart=True, so both codes see
identical basis functions.  Because Cartesian d/f components are normalised
differently by the two codes, matrices are compared after scaling by
diag(S)^-1/2, which leaves every physical quantity unchanged.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from pyscf import gto, scf


def build_pyscf(meta):
    atoms, basis = [], {}
    for i, a in enumerate(meta["atoms"]):
        label = f"{gto.mole._symbol(a['Z'])}{i + 1}"
        atoms.append([label, a["xyz"]])
        basis[label] = []
    for sh in meta["shells"]:
        label = atoms[sh["atom"]][0]
        basis[label].append([sh["l"]] + [[e, c] for e, c in zip(sh["exps"], sh["coefs"])])
    mol = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=True, charge=meta["charge"], verbose=0)
    return mol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qk_scf")
    ap.add_argument("xyz")
    ap.add_argument("basis")
    ap.add_argument("--basis-file", action="store_true", help="BASIS is a .g94 file")
    ap.add_argument("--bohr", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [args.qk_scf, args.xyz, "--dump", tmp, "--quiet", "--conv", "1e-11"]
        cmd += ["--basis-file", args.basis] if args.basis_file else ["--basis", args.basis]
        if args.bohr:
            cmd.append("--bohr")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        m = re.search(r"RHF total energy\s+(-?\d+\.\d+)", proc.stdout)
        if m is None:
            sys.exit(f"qk-scf failed:\n{proc.stdout}{proc.stderr}")
        e_qk = float(m.group(1))
        meta = json.load(open(os.path.join(tmp, "basis.json")))
        S, T, V = (np.load(os.path.join(tmp, f"{k}.npy")) for k in "STV")
        eri_path = os.path.join(tmp, "ERI.npy")
        eri = np.load(eri_path) if os.path.exists(eri_path) else None

    mol = build_pyscf(meta)
    Sp = mol.intor("int1e_ovlp")
    if Sp.shape != S.shape:
        sys.exit(f"basis size mismatch: qk {S.shape} vs pyscf {Sp.shape}")
    dq, dp = 1 / np.sqrt(np.diag(S)), 1 / np.sqrt(np.diag(Sp))

    def n1(M, d):
        return M * d[:, None] * d[None, :]

    worst = 0.0
    rows = [("S", n1(S, dq), n1(Sp, dp)),
            ("T", n1(T, dq), n1(mol.intor("int1e_kin"), dp)),
            ("V", n1(V, dq), n1(mol.intor("int1e_nuc"), dp))]
    if eri is not None:
        Ep = mol.intor("int2e")
        nq = np.einsum("i,j,k,l->ijkl", dq, dq, dq, dq)
        np_ = np.einsum("i,j,k,l->ijkl", dp, dp, dp, dp)
        rows.append(("ERI", eri * nq, Ep * np_))
    print(f"{mol.nao} Cartesian functions, max l = {max(s['l'] for s in meta['shells'])}")
    for name, a, b in rows:
        err = np.max(np.abs(a - b))
        worst = max(worst, err)
        print(f"  max |{name:3s}(qk) - {name:3s}(pyscf)| = {err:.2e}")

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    e_ref = mf.kernel()
    print(f"  RHF energy  qk = {e_qk:.10f}   pyscf = {e_ref:.10f}   diff = {e_qk - e_ref:.2e}")
    ok = worst < 1e-10 and abs(e_qk - e_ref) < 1e-8
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
