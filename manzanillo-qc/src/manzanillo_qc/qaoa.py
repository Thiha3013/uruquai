"""QAOA optimiser using PennyLane's AdamOptimizer.

Circuit structure (p layers)
-----------------------------
|+⟩^⊗n  →  [cost_layer(γₗ) · mixer_layer(βₗ)]_{l=1..p}  →  ⟨H_cost⟩

Optimisation
------------
Gradient-based Adam (``qml.AdamOptimizer``) minimises ⟨H_cost⟩ over
2p parameters (p gammas + p betas).  Starts from a random initialisation
in [0, π/2] with ``np.random.seed(42)`` for reproducibility.

Solution extraction
-------------------
After optimisation, ``qml.probs`` returns the full 2ⁿ distribution.
We scan the top-500 highest-probability bitstrings and return the
best *feasible* one (Σcᵢxᵢ ≤ B).
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from .config import AppConfig
from .ising import build_pennylane_hamiltonian
from .backends import get_device


def run_qaoa(cfg: AppConfig, h: np.ndarray, J: np.ndarray,
             direct_gates: bool = False) -> dict:
    """Run QAOA and return results.

    Parameters
    ----------
    cfg : AppConfig
        Run configuration (p_layers, n_steps, stepsize, backend, …).
    h : np.ndarray, shape (n,)
        Ising local fields from :func:`~manzanillo_qc.ising.qubo_to_ising`.
    J : np.ndarray, shape (n, n)
        Ising coupling matrix (upper triangle).

    Returns
    -------
    dict with keys:
        opt_params    – optimised [γ₁,…,γₚ, β₁,…,βₚ]
        cost_history  – ⟨H_cost⟩ at each Adam step
        probs         – full 2ⁿ probability distribution
        best_x        – best feasible sensor-selection vector (0/1)
        best_obj      – risk coverage Σwᵢxᵢ for best_x
        best_cost     – CAPEX Σcᵢxᵢ for best_x
        best_prob     – circuit probability of best_x
        final_expval  – ⟨H_cost⟩ at last Adam step
    """
    n = len(h)
    H_cost, H_mix = build_pennylane_hamiltonian(h, J)
    dev = get_device(cfg.backend, n)
    p   = cfg.p_layers

    # ── Circuit definitions ────────────────────────────────────────────────────
    if direct_gates:
        # Explicit RZ / IsingZZ / RX gates — faster with lightning.qubit + adjoint
        _couplings = [(i, j, J[i, j]) for i in range(n)
                      for j in range(i + 1, n) if abs(J[i, j]) > 1e-10]
        diff_method = "adjoint" if "lightning" in cfg.backend else "best"

        @qml.qnode(dev, diff_method=diff_method)
        def circuit_cost(params):
            gammas = params[:p]
            betas  = params[p:]
            for i in range(n):
                qml.Hadamard(wires=i)
            for l in range(p):
                for i in range(n):
                    qml.RZ(2.0 * gammas[l] * h[i], wires=i)
                for i, j, Jij in _couplings:
                    qml.IsingZZ(2.0 * gammas[l] * Jij, wires=[i, j])
                for i in range(n):
                    qml.RX(2.0 * betas[l], wires=i)
            return qml.expval(H_cost)

        @qml.qnode(dev)
        def circuit_probs(params):
            gammas = params[:p]
            betas  = params[p:]
            for i in range(n):
                qml.Hadamard(wires=i)
            for l in range(p):
                for i in range(n):
                    qml.RZ(2.0 * gammas[l] * h[i], wires=i)
                for i, j, Jij in _couplings:
                    qml.IsingZZ(2.0 * gammas[l] * Jij, wires=[i, j])
                for i in range(n):
                    qml.RX(2.0 * betas[l], wires=i)
            return qml.probs(wires=range(n))
    else:
        # PennyLane QAOA template layers
        @qml.qnode(dev)
        def circuit_cost(params):
            gammas = params[:p]
            betas  = params[p:]
            for i in range(n):
                qml.Hadamard(wires=i)
            for l in range(p):
                qml.qaoa.cost_layer(gammas[l], H_cost)
                qml.qaoa.mixer_layer(betas[l], H_mix)
            return qml.expval(H_cost)

        @qml.qnode(dev)
        def circuit_probs(params):
            gammas = params[:p]
            betas  = params[p:]
            for i in range(n):
                qml.Hadamard(wires=i)
            for l in range(p):
                qml.qaoa.cost_layer(gammas[l], H_cost)
                qml.qaoa.mixer_layer(betas[l], H_mix)
            return qml.probs(wires=range(n))

    # ── Optimisation ───────────────────────────────────────────────────────────
    np.random.seed(42)
    params = pnp.array(np.random.uniform(0, np.pi / 2, 2 * p), requires_grad=True)

    cost_history: list[float] = []
    optimizer = getattr(cfg, "optimizer", "adam")

    if optimizer == "neldermead":
        # Gradient-free Nelder-Mead via scipy — good for low-dim (2p params).
        # No diff_method overhead; each evaluation is one forward pass.
        from scipy.optimize import minimize as scipy_minimize

        def _objective(p_arr: np.ndarray) -> float:
            return float(circuit_cost(np.array(p_arr)))

        nm_res = scipy_minimize(
            _objective, np.array(params), method="Nelder-Mead",
            options={"maxiter": cfg.n_steps * 20, "xatol": 1e-5, "fatol": 1e-5},
        )
        params = pnp.array(nm_res.x, requires_grad=True)
        cost_history = [nm_res.fun]
        print(f"  Nelder-Mead finished: cost = {nm_res.fun:.4f}  "
              f"({nm_res.nit} iterations, success={nm_res.success})")
    else:
        opt = qml.AdamOptimizer(stepsize=cfg.stepsize)
        for step in range(cfg.n_steps):
            params, cost_val = opt.step_and_cost(circuit_cost, params)
            cost_history.append(float(cost_val))
            if (step + 1) % 50 == 0:
                print(f"  step {step+1:4d}  cost = {cost_val:.4f}")

    # ── Extract best feasible solution ─────────────────────────────────────────
    probs  = circuit_probs(params)
    c_arr  = cfg.costs
    w_arr  = cfg.weights
    B      = float(cfg.budget_10k)
    # For small n (≤10), 2^n ≤ 1024 so scanning 200 states is nearly exhaustive.
    # For large n (e.g. n=16, 2^n=65536), 200 is ~0.3% — a genuine test of
    # whether QAOA concentrates probability on good solutions.
    n_scan = min(50, len(probs))

    best: dict = {"obj": -np.inf, "x": None, "cost": None, "prob": 0.0}
    for idx in np.argsort(probs)[::-1][:n_scan]:
        x   = np.array([int(b) for b in format(idx, f"0{n}b")], dtype=float)
        tot = float(c_arr @ x)
        if tot <= B:
            obj = float(w_arr @ x)
            if obj > best["obj"]:
                best = {"obj": obj, "x": x.copy(), "cost": tot, "prob": float(probs[idx])}

    return {
        "opt_params":   params,
        "cost_history": cost_history,
        "probs":        probs,
        "best_x":       best["x"],
        "best_obj":     best["obj"],
        "best_cost":    best["cost"],
        "best_prob":    best["prob"],
        "final_expval": cost_history[-1] if cost_history else float("nan"),
    }
