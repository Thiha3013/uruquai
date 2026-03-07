"""Convert a QUBO matrix to Ising (h, J) coefficients and PennyLane Hamiltonians.

Substitution:  xᵢ = (1 − σᵢ) / 2,  σᵢ ∈ {−1, +1}

Ising coefficients
------------------
hᵢ  = −Q[i,i] / 2  −  (1/4) · Σⱼ≠ᵢ Q[min(i,j), max(i,j)]
Jᵢⱼ = Q[i,j] / 4       (i < j)

Energy function
---------------
E(σ) = Σᵢ hᵢ σᵢ + Σᵢ<ⱼ Jᵢⱼ σᵢ σⱼ

This matches the QUBO energy f(x) up to a constant offset (irrelevant for
optimisation).  The Ising and QUBO optima are the same bitstring.
"""
from __future__ import annotations

import numpy as np


def qubo_to_ising(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert upper-triangular QUBO matrix to Ising (h, J).

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        Upper-triangular QUBO matrix (lower triangle assumed zero).

    Returns
    -------
    h : np.ndarray, shape (n,)
        Local fields (bias terms).
    J : np.ndarray, shape (n, n)
        Coupling matrix (upper triangle, J[i,j] for i < j).
    """
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))

    for i in range(n):
        h[i] = -Q[i, i] / 2
        for j in range(n):
            if j != i:
                h[i] -= Q[min(i, j), max(i, j)] / 4

    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = Q[i, j] / 4

    return h, J


def build_pennylane_hamiltonian(h: np.ndarray, J: np.ndarray):
    """Build PennyLane cost and mixer Hamiltonians from Ising coefficients.

    Compatible with PennyLane ≥ 0.36 (uses ``qml.ops.LinearCombination``)
    and older releases (falls back to ``qml.Hamiltonian``).

    Parameters
    ----------
    h : np.ndarray, shape (n,)
        Ising local fields.
    J : np.ndarray, shape (n, n)
        Ising coupling matrix (upper triangle).

    Returns
    -------
    H_cost : PennyLane Hamiltonian
        Cost Hamiltonian — sum of Z and ZZ terms.
    H_mix : PennyLane Hamiltonian
        Mixer Hamiltonian — sum of X terms (standard transverse-field mixer).
    """
    import pennylane as qml

    n = len(h)
    coeffs: list[float] = []
    ops = []

    for i in range(n):
        if abs(h[i]) > 1e-12:
            coeffs.append(float(h[i]))
            ops.append(qml.PauliZ(i))

    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                coeffs.append(float(J[i, j]))
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    def _make(c, o):
        try:
            return qml.ops.LinearCombination(c, o)
        except AttributeError:
            return qml.Hamiltonian(c, o)

    H_cost = _make(coeffs, ops)
    H_mix  = _make([1.0] * n, [qml.PauliX(i) for i in range(n)])
    return H_cost, H_mix
