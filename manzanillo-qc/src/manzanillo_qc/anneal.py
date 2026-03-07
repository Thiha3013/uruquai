"""Simulated Annealing baseline using D-Wave's local sampler (no cloud account needed).

solve_sa() wraps SimulatedAnnealingSampler from dwave-samplers, which runs
entirely on CPU.  It operates directly on the QUBO matrix — no Ising conversion
needed — and returns the full sampleset so the caller can pick the best feasible
solution.

Install:
    pip install dimod dwave-samplers
"""
from __future__ import annotations

import time

import numpy as np


def solve_sa(
    Q: np.ndarray,
    offset: float = 0.0,
    num_reads: int = 5000,
    sweeps: int = 2000,
    seed: int = 123,
) -> tuple[object, float]:
    """Run Simulated Annealing on the QUBO and return (sampleset, elapsed_ms).

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        Upper-triangular QUBO matrix from build_qubo().
    offset : float
        Constant energy offset (λ·B² term).  Does not affect which solution
        is optimal; included so reported energies are correct.
    num_reads : int
        Number of independent SA runs (more = better chance of finding global min).
    sweeps : int
        Number of sweeps per run (more = slower but better mixing).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    sampleset : dimod.SampleSet
        All num_reads solutions sorted by energy (lowest first).
    elapsed_ms : float
        Wall-clock time for the SA call.
    """
    import dimod
    from dwave.samplers import SimulatedAnnealingSampler

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset=offset)
    sampler = SimulatedAnnealingSampler()

    t0 = time.perf_counter()
    sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=sweeps, seed=seed)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return sampleset, elapsed_ms


def best_feasible(sampleset, cfg) -> dict:
    """Extract the best feasible solution from a SA sampleset.

    Iterates samples in ascending energy order and returns the first one that
    satisfies the budget constraint Σcᵢxᵢ ≤ B.

    Parameters
    ----------
    sampleset : dimod.SampleSet
    cfg : AppConfig

    Returns
    -------
    dict with keys: x, obj, cost.  Returns all-zeros if no feasible sample found.
    """
    n = cfg.n_sites
    c = cfg.costs
    w = cfg.weights
    B = float(cfg.budget_10k)

    for sample, _ in sampleset.data(["sample", "energy"], sorted_by="energy"):
        x = np.array([sample[i] for i in range(n)], dtype=float)
        if float(c @ x) <= B:
            return {"x": x, "obj": float(w @ x), "cost": float(c @ x)}

    # No feasible sample found — return all-zeros
    x = np.zeros(n, dtype=float)
    return {"x": x, "obj": 0.0, "cost": 0.0}
