"""Build QUBO matrices for the sensor-placement knapsack problem.

Formulation (maximisation, no slack bits)
------------------------------------------
Maximise  Σᵢ wᵢ xᵢ   subject to  Σᵢ cᵢ xᵢ ≤ B,  xᵢ ∈ {0,1}

Written as an unconstrained minimisation (QAOA minimises):

    min f(x) = −Σᵢ wᵢ xᵢ  +  λ · (Σᵢ cᵢ xᵢ − B)²
             + Σᵢ<ⱼ ρᵢⱼ · xᵢ xⱼ     ← overlap penalty (optional)

Expanding the penalty and collecting terms (using xᵢ² = xᵢ):

    Q[i,i] = −wᵢ  +  λ · cᵢ · (cᵢ − 2B)        (diagonal)
    Q[i,j] =          2λ · cᵢ · cⱼ  +  ρᵢⱼ       (upper triangle, i < j)

Two λ variants
--------------
build_qubo()  — default λ = 1 + Σwᵢ.
    Conservative: guarantees feasibility is always cheaper than any budget
    violation.  Used by gradient-based solvers (PennyLane, RQAOA, DQI) where
    a stiff penalty landscape is fine because gradients still navigate it.

build_qubo_sampling()  — calibrated λ = (Σwᵢ / min_c²) × 1.3  via
    calibrate_lambda_for_sampling().
    Minimum valid λ that still enforces feasibility, with a 30% safety margin.
    For typical instances this is ~14x smaller than the default, keeping the
    objective signal visible to sampling-based solvers (Pulser, Pasqal) that
    cannot distinguish solution quality when the penalty dominates.

Overlap / redundancy model
--------------------------
If two sensors have overlapping detection footprints, deploying both gives
diminishing returns.  ``compute_overlap(cfg)`` computes a pairwise redundancy
matrix ρ (upper-triangle) from geographic distance and sensor detection radii.

    overlap_ij = max(0,  1 − dist(i,j) / (r_i + r_j))
    ρᵢⱼ        = overlap_ij · min(wᵢ, wⱼ)

Adding ρᵢⱼ to Q[i,j] discourages co-deploying sensors whose footprints
significantly overlap.
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
    """Build the shared QUBO matrix for all QUBO-based solvers.

    Objective:
        min  -sum_i w_i x_i
             + lam * (sum_i c_i x_i - B)^2
             + eps * sum_i x_i
             + optional overlap penalty

    The tiny eps * sum_i x_i term is a shared cardinality bias:
    when two solutions are otherwise very close, it slightly prefers
    using fewer selected sites.
    """
    n = cfg.n_sites                                   # number of candidate sites
    w = cfg.weights                                   # risk/utility weights
    c = cfg.costs                                     # site costs
    B = float(cfg.budget_10k)                         # budget in $10K units
    lam = cfg.effective_lambda                        # soft-budget penalty strength

    eps = 1e-2                                        # tiny shared cardinality tie-breaker

    Q = np.zeros((n, n), dtype=float)                 # upper-triangular QUBO matrix

    for i in range(n):                                # fill diagonal terms
        Q[i, i] -= w[i]                               # objective reward: maximize weight -> minimize -weight
        Q[i, i] += eps                                # tiny penalty per selected site (cardinality bias)
        Q[i, i] += lam * c[i] * (c[i] - 2 * B)       # diagonal part of budget penalty expansion

    for i in range(n):                                # fill off-diagonal coupling terms
        for j in range(i + 1, n):
            Q[i, j] += 2 * lam * c[i] * c[j]         # off-diagonal part of budget penalty expansion

    if overlap:                                       # optionally add overlap penalties
        rho = compute_overlap(cfg)                    # get overlap penalty matrix
        Q += rho                                      # add it into the shared QUBO

    meta = {                                          # metadata for reporting/debugging
        "n_qubits": n,
        "lambda": lam,
        "budget": B,
        "overlap": overlap,
        "q_min": float(Q.min()),
        "q_max": float(Q.max()),
        "tie_break_mode": "cardinality",
        "tie_break_eps": eps,
    }
    return Q, meta                                    # return QUBO and metadata


# ── Sampling-solver QUBO ───────────────────────────────────────────────────────

def calibrate_lambda_for_sampling(cfg: AppConfig, margin: float = 0.30) -> float:
    """Minimum valid lambda for sampling-based neutral-atom solvers.

    The conservative default lambda = 1 + sum(w) makes the QUBO ~10-15x stiffer
    than needed, burying the objective signal under the penalty.  For Pasqal and
    Pulser (sampling-based) the minimum-valid lambda is used instead:

        lam_min = sum(w) / min(c)^2

    This guarantees that the penalty for exceeding the budget by the smallest
    possible amount (one minimum-cost site) still exceeds the maximum achievable
    objective gain.  A safety margin of (1 + margin) is added on top.
    """
    c = cfg.costs
    w = cfg.weights
    min_c = float(c.min())
    sum_w = float(w.sum())
    lam_min = sum_w / (min_c ** 2) if min_c > 0 else 1.0
    return lam_min * (1.0 + margin)


def build_qubo_sampling(cfg: AppConfig, margin: float = 0.30,
                        overlap: bool = False) -> tuple[np.ndarray, dict]:
    """Build QUBO with calibrated minimum lambda for sampling-based solvers.

    Same (Q, meta) tuple as build_qubo() but uses the smallest lambda that still
    enforces feasibility.  This avoids burying the objective signal under a
    penalty that is 10-15x larger than necessary.

    Use this instead of build_qubo() for Pasqal and Pulser.
    """
    n   = cfg.n_sites
    w   = cfg.weights
    c   = cfg.costs
    B   = float(cfg.budget_10k)
    lam = calibrate_lambda_for_sampling(cfg, margin=margin)
    eps = 1e-2

    Q = np.zeros((n, n), dtype=float)
    for i in range(n):
        Q[i, i] -= w[i]
        Q[i, i] += eps
        Q[i, i] += lam * c[i] * (c[i] - 2 * B)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2 * lam * c[i] * c[j]

    if overlap:
        rho = compute_overlap(cfg)
        Q += rho

    meta = {
        "n_qubits":      n,
        "lambda":        lam,
        "lambda_default": 1.0 + float(w.sum()),
        "lambda_source": "calibrated_sampling",
        "budget":        B,
        "overlap":       overlap,
        "q_min":         float(Q.min()),
        "q_max":         float(Q.max()),
        "tie_break_mode": "cardinality",
        "tie_break_eps": eps,
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
