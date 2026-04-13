"""Decoded Quantum Interferometry (DQI) solver.

Reference: Jordan et al., "Optimization by Decoded Quantum Interferometry",
           Nature 2025 (arXiv:2408.08292). Google Quantum AI.

Problem class: max-XORSAT
--------------------------
Given binary constraint matrix B (m×n) and constraint vector v (length m),
maximise:

    f(x) = Σᵢ (-1)^(vᵢ + bᵢ·x mod 2)
         = (# satisfied clauses) − (# violated clauses)

Core idea
---------
Unlike QAOA which variational-optimises circuit parameters, DQI uses quantum
*interference*: it builds a superposition of all 2ⁿ bitstrings, then applies
a Quantum Fourier Transform (via Hadamard on the data register) so that
high-objective bitstrings interfere constructively and low-objective ones
destructively.  Measuring the final state samples good solutions with high
probability.

Connection to our QUBO
-----------------------
The Ising Hamiltonian H = Σᵢ hᵢ σᵢᶻ + Σᵢ<ⱼ Jᵢⱼ σᵢᶻ σⱼᶻ  maps to max-XORSAT:
  - Linear term hᵢ σᵢᶻ     → clause bᵢ = eᵢ,        vᵢ = 1 if hᵢ < 0
  - Quadratic Jᵢⱼ σᵢᶻ σⱼᶻ  → clause bᵢⱼ = eᵢ + eⱼ,  vᵢⱼ = 1 if Jᵢⱼ < 0

Note on clause weights: the paper handles weighted clauses by integer repetition
(floor(|w|/w_min) copies per clause). This implementation does NOT do that.
Instead it uses a top-K truncation: only the strongest max_clauses Ising terms
are kept as clauses. This is a sparse approximation of the full dense problem,
not an equivalent weighted reduction.

Scalability
-----------
- n qubits data + m qubits syndrome + ⌈log₂(ℓ+1)⌉ weight qubits total
- For a dense QUBO: full clause count = n + n(n-1)/2 → syndrome grows fast
- With top-K truncation (max_clauses=10): total qubits = n + 10 + 2
  e.g. n=6 → 18 qubits (not 29 as would be needed for full dense clauses)
- Shot count: 1024 samples from the final measurement

Seven-step circuit (per Jordan et al.)
---------------------------------------
1. StatePrep: embed optimal weight coefficients wₖ into W
2. Dicke state: prepare |Dₘᵏ⟩ on E, controlled on W = k
3. Uncompute W
4. Phase (-1)^(v·y) on E  (Z gates where vᵢ = 1)
5. Syndrome: compute Bᵀy into D via CNOTs
6. Decode syndrome → uncompute E via lookup table
7. Hadamard on D → measure D

Returns the same dict format as run_qaoa() for drop-in benchmark use.
"""
from __future__ import annotations

import time
import itertools

import numpy as np
import pennylane as qml

from .config import AppConfig

_N_SHOTS      = 1024   # measurement samples from the final state
_POLY_DEGREE  = 2      # ℓ: DQI polynomial degree (higher = better but larger circuit)
_MAX_CLAUSES  = 10     # top-K clause truncation (sparse approximation of dense QUBO)


# ── Ising → max-XORSAT conversion ────────────────────────────────────────────

def ising_to_xorsat(
    h: np.ndarray,
    J: np.ndarray,
    max_clauses: int = _MAX_CLAUSES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Ising (h, J) to a sparse unweighted max-XORSAT instance.

    Dense QUBOs produce O(n²) Ising terms. This function keeps only the
    top-max_clauses strongest terms as XOR clauses — a sparse approximation,
    not an equivalent reformulation. The paper's intended use case is sparse
    LDPC-structured problems; our sensor-placement QUBO is dense by construction
    (budget penalty creates couplings for all pairs i<j).

    Parameters
    ----------
    h           : (n,) Ising local fields
    J           : (n, n) Ising coupling matrix (upper triangular)
    max_clauses : int
        Keep only this many clauses (strongest weights first).
        Total circuit qubits = max_clauses + n + ceil(log2(ell+1)).
        Default _MAX_CLAUSES = 10 → 16 qubits for n=4, 18 qubits for n=6.

    Returns
    -------
    B            : (m, n) binary constraint matrix over GF(2)
    v            : (m,)  constraint vector
    raw_weights  : (m,)  absolute Ising weights for each clause (reference only)
    """
    n = len(h)
    all_rows: list[tuple[float, list[int], int]] = []   # (|weight|, row, v)

    for i in range(n):
        if abs(h[i]) > 1e-10:
            row = [0] * n;  row[i] = 1
            all_rows.append((abs(float(h[i])), row, 1 if h[i] < 0 else 0))

    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-10:
                row = [0] * n;  row[i] = 1;  row[j] = 1
                all_rows.append((abs(float(J[i, j])), row, 1 if J[i, j] < 0 else 0))

    if not all_rows:
        raise ValueError("No clauses generated — h and J are all zero.")

    # Keep strongest max_clauses terms
    all_rows.sort(key=lambda x: x[0], reverse=True)
    kept = all_rows[:max_clauses]

    raw_w = np.array([r[0] for r in kept], dtype=float)
    B     = np.array([r[1] for r in kept], dtype=int)
    v     = np.array([r[2] for r in kept], dtype=int)
    return B, v, raw_w


# ── DQI circuit helpers ───────────────────────────────────────────────────────

def _optimal_weights(m: int, ell: int) -> np.ndarray:
    """Principal eigenvector of the tridiagonal matrix with aₖ = √(k(m-k+1)).

    This gives the optimal weights wₖ for the degree-ℓ polynomial that
    maximises the DQI signal-to-noise ratio.
    """
    # Build symmetric tridiagonal: off-diagonals aₖ = √(k(m-k+1)), k=1..ell
    a = np.array([np.sqrt(k * (m - k + 1)) for k in range(1, ell + 1)])
    # Tridiagonal matrix of size (ell+1) × (ell+1)
    T = np.diag(a, -1) + np.diag(a, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(T)
    # Principal eigenvector = largest eigenvalue
    w = eigenvectors[:, np.argmax(eigenvalues)]
    return np.abs(w)   # take absolute values (sign is irrelevant for StatePrep)


def _generate_dicke_states(m: int, k: int) -> list[str]:
    """All m-bit strings with exactly k ones (Dicke state computational basis)."""
    result = []
    for positions in itertools.combinations(range(m), k):
        bits = ['0'] * m
        for p in positions:
            bits[p] = '1'
        result.append(''.join(bits))
    return result


def _build_syndrome_lut(B: np.ndarray) -> dict[str, str]:
    """Lookup table: syndrome string → minimum-weight error string.

    The syndrome lives in GF(2)^n (n = # data qubits = # columns of B).
    We enumerate all 2^n possible n-bit syndromes and for each find the
    minimum-Hamming-weight error pattern y (from GF(2)^m) that produces it.

    We search greedily by Hamming weight: try all weight-1 errors first,
    then weight-2, etc., stopping once all syndromes are covered.  This is
    O(m * 2^m_eff) in the worst case but m is capped at _MAX_CLAUSES=10 and
    the early stopping makes it fast in practice.
    """
    m, n = B.shape
    lut: dict[str, str] = {}
    # Zero error → zero syndrome
    lut['0' * n] = '0' * m

    # Search increasing Hamming weight
    for weight in range(1, m + 1):
        if len(lut) == (1 << n):
            break
        for positions in itertools.combinations(range(m), weight):
            y = np.zeros(m, dtype=int)
            for p in positions:
                y[p] = 1
            syndrome = (B.T @ y) % 2
            s_str = ''.join(map(str, syndrome))
            if s_str not in lut:
                lut[s_str] = ''.join(map(str, y))
        if len(lut) == (1 << n):
            break

    return lut


# ── Main DQI circuit ──────────────────────────────────────────────────────────

def _build_dqi_circuit(
    B: np.ndarray,
    v: np.ndarray,
    ell: int,
    n_shots: int,
) -> tuple[callable, int]:
    """Build and return the DQI QNode + total qubit count.

    Register layout (all wires are integers):
      W : weight register   wires [0 .. n_w-1]
      E : error register    wires [n_w .. n_w+m-1]
      D : data register     wires [n_w+m .. n_w+m+n-1]
    """
    m, n    = B.shape
    n_w     = int(np.ceil(np.log2(ell + 1))) if ell > 0 else 1
    total_q = n_w + m + n

    w_wires = list(range(0,       n_w))
    e_wires = list(range(n_w,     n_w + m))
    d_wires = list(range(n_w + m, n_w + m + n))

    w_opt_raw = _optimal_weights(m, ell)
    # Pad to length 2^n_w so StatePrep gets a power-of-2 sized vector
    w_padded  = np.zeros(1 << n_w)
    w_padded[:len(w_opt_raw)] = w_opt_raw
    w_opt = w_padded / np.linalg.norm(w_padded)
    lut     = _build_syndrome_lut(B)

    dev = qml.device("default.qubit", wires=total_q, shots=n_shots)

    @qml.qnode(dev)
    def circuit():
        # ── Step 1: Embed optimal weights into W ──────────────────────────
        qml.StatePrep(w_opt, wires=w_wires)

        # ── Step 2: Prepare Dicke state on E, controlled on W = k ─────────
        # For each k=1..ell, we want: when W=|k⟩, initialise E to |D_m^k⟩.
        # Strategy: X-flip W so that |k⟩ becomes |11...1⟩ (all-ones control),
        # apply controlled-StatePrep, then X-flip back.
        for k in range(1, ell + 1):
            k_bits = format(k, f'0{n_w}b')

            dicke_strings = _generate_dicke_states(m, k)
            amplitudes    = np.zeros(1 << m)
            for ds in dicke_strings:
                amplitudes[int(ds, 2)] = 1.0
            amplitudes /= np.linalg.norm(amplitudes)

            # X-flip bits that are 0 in k_bits so controls fire on |1⟩^⊗n_w
            flip_idx = [i for i, b in enumerate(k_bits) if b == '0']
            for fi in flip_idx:
                qml.PauliX(wires=w_wires[fi])

            qml.ctrl(qml.StatePrep, control=w_wires)(amplitudes, wires=e_wires)

            for fi in flip_idx:
                qml.PauliX(wires=w_wires[fi])

        # ── Step 3: Uncompute W ───────────────────────────────────────────
        qml.adjoint(qml.StatePrep)(w_opt, wires=w_wires)

        # ── Step 4: Phase (-1)^(v·y) on E — Z on positions where vᵢ = 1 ──
        for i, vi in enumerate(v):
            if vi == 1:
                qml.PauliZ(wires=e_wires[i])

        # ── Step 5: Syndrome computation s = Bᵀy — CNOT(e_i → d_j) ───────
        # B is (m×n): B[i,j] = 1 means column j participates in clause i
        # s_j = Σᵢ B[i,j] * y_i  mod 2
        for j in range(n):
            for i in range(m):
                if B[i, j] == 1:
                    qml.CNOT(wires=[e_wires[i], d_wires[j]])

        # ── Step 6: Decode syndrome → uncompute E via LUT ─────────────────
        # The LUT has up to 2^n syndrome entries. For each entry, one or more
        # error bits in E are flipped via MultiControlledX — so total gate count
        # depends on the decoded error pattern, not simply 2^n gates flat.
        # PennyLane decomposes each MultiControlledX into elementary gates,
        # which was the main bottleneck observed at n=6 during development.
        for s_str, y_str in lut.items():
            s_bits = [int(b) for b in s_str]
            y_bits = [int(b) for b in y_str]
            if sum(y_bits) == 0:
                continue
            flip_wires = [e_wires[i] for i in range(m) if y_bits[i] == 1]
            # Flip control qubits where syndrome bit = 0 (so control fires on 1)
            zero_ctrl = [d_wires[j] for j, b in enumerate(s_bits) if b == 0]
            for zw in zero_ctrl:
                qml.PauliX(wires=zw)
            for fw in flip_wires:
                qml.MultiControlledX(wires=d_wires + [fw])
            for zw in zero_ctrl:
                qml.PauliX(wires=zw)

        # ── Step 7: Hadamard on D, measure ───────────────────────────────
        for dw in d_wires:
            qml.Hadamard(wires=dw)

        return qml.counts(wires=d_wires)

    return circuit, total_q


# ── Best feasible solution decoder ───────────────────────────────────────────

def _best_feasible(counts: dict, cfg: AppConfig, n: int) -> dict:
    """Scan DQI measurement counts, return best feasible solution."""
    c_arr = cfg.costs
    w_arr = cfg.weights
    B_val = float(cfg.budget_10k)

    best = {"obj": -np.inf, "x": None, "cost": None, "prob": 0.0}
    total = sum(counts.values())

    for bitstr, count in counts.items():
        # PennyLane may return int keys or string keys depending on version
        if isinstance(bitstr, int):
            bits = [(bitstr >> (n - 1 - i)) & 1 for i in range(n)]
        else:
            bits = [int(b) for b in bitstr]
        x    = np.array(bits, dtype=float)
        cost = float(c_arr @ x)
        if cost <= B_val:
            obj = float(w_arr @ x)
            if obj > best["obj"]:
                best = {"obj": obj, "x": x.copy(), "cost": cost,
                        "prob": count / total}
    return best


# ── Public entry point ────────────────────────────────────────────────────────

def run_dqi(
    cfg: AppConfig,
    h: np.ndarray,
    J: np.ndarray,
    ell: int = _POLY_DEGREE,
    n_shots: int = _N_SHOTS,
    max_clauses: int | None = _MAX_CLAUSES,
) -> dict:
    """Run DQI and return results in the same dict format as run_qaoa().

    Parameters
    ----------
    cfg : AppConfig
    h   : (n,) Ising local fields (already scaled)
    J   : (n, n) Ising coupling matrix (upper triangular, already scaled)
    ell : int
        Polynomial degree (default 2).  Higher = better signal, larger circuit.
    n_shots : int
        Measurement samples (default 1024).
    max_clauses : int or None
        Keep only the top-K strongest Ising terms as XOR clauses.
        None → adaptive: min(2*n, 20), scaling with problem size.
        Total circuit qubits = max_clauses + n + ceil(log2(ell+1)).

    Returns
    -------
    dict with keys matching run_qaoa(): best_x, best_obj, best_cost, best_prob,
    time_ms, timing, convergence, grad_norm_history, cost_history, probs,
    opt_params, final_expval.
    """
    n = len(h)

    # Step A: convert Ising → max-XORSAT (sparse top-K truncation)
    t_build_0 = time.perf_counter()
    B, v, raw_weights = ising_to_xorsat(h, J, max_clauses=max_clauses)
    m = B.shape[0]

    n_w     = int(np.ceil(np.log2(ell + 1))) if ell > 0 else 1
    total_q = n_w + m + n
    print(f"  DQI  n={n}  clauses={m}/{n + n*(n-1)//2}(full)  "
          f"total_qubits={total_q}  ell={ell}  shots={n_shots}")

    # Step B: build circuit (LUT construction + gate decomposition)
    circuit, _ = _build_dqi_circuit(B, v, ell, n_shots)
    t_build_ms = (time.perf_counter() - t_build_0) * 1000

    # Step C: execute circuit
    t_exec_0 = time.perf_counter()
    counts = circuit()
    t_exec_ms = (time.perf_counter() - t_exec_0) * 1000

    # Step D: decode best feasible solution
    t_decode_0 = time.perf_counter()
    best = _best_feasible(counts, cfg, n)
    t_decode_ms = (time.perf_counter() - t_decode_0) * 1000

    t_ms = t_build_ms + t_exec_ms + t_decode_ms
    print(f"  DQI done  build={t_build_ms/1000:.1f}s  exec={t_exec_ms/1000:.1f}s  "
          f"decode={t_decode_ms/1000:.3f}s  "
          f"obj={best['obj']:.4f}  best_prob={best['prob']:.3f}")

    return {
        "best_x":            best["x"],
        "best_obj":          best["obj"],
        "best_cost":         best["cost"],
        "best_prob":         best["prob"],
        "time_ms":           t_ms,
        "cost_history":      [],
        "grad_norm_history": [],
        "probs":             None,
        "opt_params":        None,
        "final_expval":      float("nan"),
        "timing": {
            "circuit_init_ms": t_build_ms,
            "opt_loop_ms":     t_exec_ms,
            "decode_ms":       t_decode_ms,
            "step_avg_ms":     0.0,
        },
        "convergence": {
            "iter":    1,
            "time_ms": t_ms,
            "n_steps": 1,
            "reached": True,
        },
        "meta": {
            "m_clauses":   m,
            "total_qubits": total_q,
            "ell":         ell,
            "n_shots":     n_shots,
        },
    }
