"""Build a QUBO matrix for the sensor-placement knapsack problem.

Formulation (maximisation, no slack bits)
------------------------------------------
Maximise  Σᵢ wᵢ xᵢ   subject to  Σᵢ cᵢ xᵢ ≤ B,  xᵢ ∈ {0,1}

Written as an unconstrained minimisation (QAOA minimises):

    min f(x) = −Σᵢ wᵢ xᵢ  +  λ · (Σᵢ cᵢ xᵢ − B)²
             + Σᵢ<ⱼ ρᵢⱼ · xᵢ xⱼ     ← overlap penalty (optional)

Expanding the penalty and collecting terms (using xᵢ² = xᵢ):

    Q[i,i] = −wᵢ  +  λ · cᵢ · (cᵢ − 2B)        (diagonal)
    Q[i,j] =          2λ · cᵢ · cⱼ  +  ρᵢⱼ       (upper triangle, i < j)

λ choice
--------
λ must exceed Σwᵢ so that any budget violation costs more than the maximum
possible objective gain.  The default ``effective_lambda`` in AppConfig sets
λ = 1 + Σwᵢ, which satisfies this for normalised weights ∈ [0,1].

Overlap / redundancy model
--------------------------
If two sensors have overlapping detection footprints, deploying both gives
diminishing returns.  ``compute_overlap(cfg)`` computes a pairwise redundancy
matrix ρ (upper-triangle) from geographic distance and sensor detection radii.

    overlap_ij = max(0,  1 − dist(i,j) / (r_i + r_j))
    ρᵢⱼ        = overlap_ij · min(wᵢ, wⱼ)

Adding ρᵢⱼ to Q[i,j] discourages co-deploying sensors whose footprints
significantly overlap — only pay the cost if the marginal coverage is worth it.
"""
from __future__ import annotations

import math
import numpy as np

from .config import AppConfig


# ── Haversine distance ─────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ── Overlap model ──────────────────────────────────────────────────────────────

def compute_overlap(cfg: AppConfig) -> np.ndarray:
    """Compute the pairwise redundancy penalty matrix ρ (upper triangle).

    ρᵢⱼ > 0 means sensors i and j have overlapping coverage footprints;
    deploying both produces less combined value than the sum of their
    individual weights.

    Parameters
    ----------
    cfg : AppConfig

    Returns
    -------
    rho : np.ndarray, shape (n, n)
        Upper-triangular redundancy matrix.  rho[i,j] = 0 for i ≥ j.
    """
    n   = cfg.n_sites
    w   = cfg.weights
    rho = np.zeros((n, n))

    for i in range(n):
        si = cfg.sites[i]
        ri = si.detection_radius_km()
        for j in range(i + 1, n):
            sj = cfg.sites[j]
            rj = sj.detection_radius_km()
            dist = _haversine_km(si.lat, si.lon, sj.lat, sj.lon)
            overlap_frac = max(0.0, 1.0 - dist / (ri + rj))
            rho[i, j] = overlap_frac * min(w[i], w[j])

    return rho


# ── QUBO builder ───────────────────────────────────────────────────────────────

def build_qubo(cfg: AppConfig, overlap: bool = False) -> tuple[np.ndarray, dict]:
    """Return the QUBO matrix Q and a metadata dict.

    Parameters
    ----------
    cfg : AppConfig
        Populated configuration with sites, budget, and lambda.
    overlap : bool
        If True, add the pairwise redundancy penalty from ``compute_overlap``.
        Discourages deploying sensors with overlapping detection footprints.

    Returns
    -------
    Q : np.ndarray, shape (n, n)
        Upper-triangular QUBO matrix (Q[i,j] = 0 for i > j).
    meta : dict
        Summary statistics: n_qubits, lambda, budget, q_min, q_max, overlap.
    """
    n   = cfg.n_sites
    w   = cfg.weights
    c   = cfg.costs
    B   = float(cfg.budget_10k)
    lam = cfg.effective_lambda

    Q = np.zeros((n, n))

    # Objective term: −wᵢ on sensor diagonals
    for i in range(n):
        Q[i, i] -= w[i]

    # Penalty diagonal: λ·cᵢ·(cᵢ − 2B)
    for i in range(n):
        Q[i, i] += lam * c[i] * (c[i] - 2 * B)

    # Penalty off-diagonal: 2λ·cᵢ·cⱼ
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2 * lam * c[i] * c[j]

    # Optional: add pairwise redundancy penalties
    if overlap:
        rho = compute_overlap(cfg)
        Q += rho   # rho is already upper-triangular

    meta = {
        "n_qubits": n,
        "lambda":   lam,
        "budget":   B,
        "overlap":  overlap,
        "q_min":    float(Q.min()),
        "q_max":    float(Q.max()),
    }
    return Q, meta


# ── Brute-force reference ──────────────────────────────────────────────────────

def brute_force(cfg: AppConfig, Q: np.ndarray) -> dict:
    """Enumerate all 2^n bitstrings and return the best feasible solution.

    Returns
    -------
    dict with keys: obj (float), x (np.ndarray), cost (float).
    """
    n = cfg.n_sites
    c = cfg.costs
    w = cfg.weights
    B = float(cfg.budget_10k)

    best: dict = {"obj": -np.inf, "x": None, "cost": None}

    for idx in range(2 ** n):
        x    = np.array([int(b) for b in format(idx, f"0{n}b")], dtype=float)
        cost = float(c @ x)
        if cost <= B:
            obj = float(w @ x)
            if obj > best["obj"]:
                best = {"obj": obj, "x": x.copy(), "cost": cost}

    return best
