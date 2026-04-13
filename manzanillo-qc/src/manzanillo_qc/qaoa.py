"""QAOA optimiser using PennyLane's AdamOptimizer.

Circuit structure (p layers)
-----------------------------
|+⟩^⊗n  →  [cost_layer(γₗ) · mixer_layer(βₗ)]_{l=1..p}  →  ⟨H_cost⟩

Optimisation
------------
Gradient-based Adam (``qml.AdamOptimizer``) minimises ⟨H_cost⟩ over
2p parameters (p gammas + p betas).  Starts from a random initialisation
in [0, π/2] controlled by ``seed`` (default 42) for reproducibility.

INTERP warm-start
-----------------
Pass ``warm_params`` (the ``opt_params`` from a p-1 layer run) to seed
the p-layer optimisation from a high-quality starting point.  The p-1
params are extended to length 2p by appending a zero for the new gamma
and beta — this ensures the initial p-layer circuit reduces exactly to
the previous (p-1)-layer circuit, preserving solution quality from the
start.  This is the standard technique to avoid barren plateaus as p
grows.

Barren plateau detection
------------------------
The gradient norm ||∇params|| is computed every ``_log_every`` steps
(same interval as the cost printout).  If it stays below
``_BARREN_PLATEAU_TOL`` for ``_BARREN_PLATEAU_WINDOW`` consecutive log
intervals a warning is printed.  ``grad_norm_history`` is returned so
the caller can inspect the full trajectory.

Solution extraction
-------------------
After optimisation, ``qml.probs`` returns the full 2ⁿ distribution.
We scan all bitstrings and return the best *feasible* one (Σcᵢxᵢ ≤ B).

Convergence criterion
---------------------
During Adam optimisation, convergence is detected when the absolute
improvement in objective over the last ``_CONV_WINDOW`` steps falls
below ``_CONV_TOL``.  The iteration and wall-clock time at that point
are recorded in the returned ``convergence`` dict.
"""
from __future__ import annotations

import time as _time

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from .config import AppConfig
from .ising import build_pennylane_hamiltonian
from .backends import get_device

_CONV_WINDOW = 20
_CONV_TOL    = 1e-3

_BARREN_PLATEAU_TOL    = 1e-4   # ||∇|| below this signals a barren plateau
_BARREN_PLATEAU_WINDOW = 5      # how many consecutive log intervals before warning


def run_qaoa(cfg: AppConfig, h: np.ndarray, J: np.ndarray,
             direct_gates: bool = False, seed: int = 42,
             warm_params: np.ndarray | None = None) -> dict:
    """Run QAOA and return results.

    Parameters
    ----------
    cfg : AppConfig
        Run configuration (p_layers, n_steps, stepsize, backend, …).
    h : np.ndarray, shape (n,)
        Ising local fields from :func:`~manzanillo_qc.ising.qubo_to_ising`.
    J : np.ndarray, shape (n, n)
        Ising coupling matrix (upper triangle).
    direct_gates : bool
        If True, use explicit RZ/IsingZZ/RX gates instead of qml.qaoa
        template layers.  Enables ``diff_method="adjoint"`` on
        lightning.qubit for faster gradient computation.
    seed : int
        Random seed for parameter initialisation (default 42).
    warm_params : np.ndarray or None
        INTERP warm-start.  Pass ``opt_params`` from a (p-1)-layer run
        to initialise the p-layer circuit from a known-good point.
        Shape must be 2*(p-1); a zero is appended for the new gamma and
        beta so the circuit starts equivalent to the p-1 layer circuit.
        Shape mismatch falls back to random initialisation with a warning.

    Returns
    -------
    dict with keys:
        opt_params       – optimised [γ₁,…,γₚ, β₁,…,βₚ]
        cost_history     – ⟨H_cost⟩ at each Adam step
        grad_norm_history– list of (step, ||∇||) sampled every _log_every steps
        probs            – full 2ⁿ probability distribution
        best_x           – best feasible sensor-selection vector (0/1)
        best_obj         – risk coverage Σwᵢxᵢ for best_x
        best_cost        – CAPEX Σcᵢxᵢ for best_x
        best_prob        – circuit probability of best_x
        final_expval     – ⟨H_cost⟩ at last Adam step
        timing           – dict: circuit_init_ms, opt_loop_ms, decode_ms,
                           step_avg_ms
        convergence      – dict: iter, time_ms, n_steps, reached
    """
    n = len(h)

    # ── Setup: Hamiltonians + device + circuit definitions ─────────────────────
    t_setup_start = _time.perf_counter()
    H_cost, H_mix = build_pennylane_hamiltonian(h, J)
    dev = get_device(cfg.backend, n)
    p   = cfg.p_layers

    if direct_gates:
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

    t_circuit_init_ms = (_time.perf_counter() - t_setup_start) * 1000

    # ── Parameter initialisation (INTERP warm-start or random) ────────────────
    if warm_params is not None:
        wp = np.asarray(warm_params, dtype=float).flatten()
        if len(wp) == 2 * (p - 1):
            # Append zero for new gamma and beta: circuit starts == (p-1)-layer circuit
            prev_gammas = wp[:p - 1]
            prev_betas  = wp[p - 1:]
            init_arr    = np.concatenate([prev_gammas, [0.0], prev_betas, [0.0]])
            params = pnp.array(init_arr, requires_grad=True)
            print(f"  INTERP init: warm-started from p={p-1} params "
                  f"(||warm||={np.linalg.norm(wp):.4f})")
        else:
            print(f"  INTERP init: shape mismatch "
                  f"(expected {2*(p-1)}, got {len(wp)}) — falling back to random init")
            np.random.seed(seed)
            params = pnp.array(np.random.uniform(0, np.pi / 2, 2 * p), requires_grad=True)
    else:
        np.random.seed(seed)
        params = pnp.array(np.random.uniform(0, np.pi / 2, 2 * p), requires_grad=True)

    cost_history:      list[float]        = []
    grad_norm_history: list[tuple[int, float]] = []   # (step, ||∇||)
    optimizer_name = getattr(cfg, "optimizer", "adam")
    conv_iter:    int   | None = None
    conv_time_ms: float | None = None

    t_opt_start = _time.perf_counter()

    if optimizer_name == "neldermead":
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
        if nm_res.success:
            conv_iter    = nm_res.nit
            conv_time_ms = (_time.perf_counter() - t_opt_start) * 1000
    else:
        opt = qml.AdamOptimizer(stepsize=cfg.stepsize)
        # Print more frequently for large n: every 10 steps if n>=20, else every 50
        _log_every = 10 if n >= 20 else 50
        _low_grad_count    = 0
        _plateau_warned    = False
        for step in range(cfg.n_steps):
            params, cost_val = opt.step_and_cost(circuit_cost, params)
            cost_history.append(float(cost_val))
            if (step + 1) % _log_every == 0:
                # Gradient norm: one extra circuit evaluation per log interval.
                # Acceptable overhead (~10% for large n) vs the diagnostic value.
                grad       = qml.grad(circuit_cost)(params)
                grad_norm  = float(np.linalg.norm(grad))
                grad_norm_history.append((step + 1, grad_norm))

                elapsed_s  = _time.perf_counter() - t_opt_start
                step_avg_s = elapsed_s / (step + 1)
                remaining  = step_avg_s * (cfg.n_steps - step - 1)
                eta_str    = (f"{int(remaining//3600)}h{int((remaining%3600)//60)}m"
                              if remaining > 60 else f"{remaining:.0f}s")
                print(f"  step {step+1:4d}/{cfg.n_steps}"
                      f"  cost={cost_val:.4f}"
                      f"  ||∇||={grad_norm:.2e}"
                      f"  {step_avg_s*1000:.0f}ms/step"
                      f"  ETA {eta_str}")

                if grad_norm < _BARREN_PLATEAU_TOL:
                    _low_grad_count += 1
                    if _low_grad_count >= _BARREN_PLATEAU_WINDOW and not _plateau_warned:
                        print(f"  WARNING: barren plateau detected — ||∇|| < "
                              f"{_BARREN_PLATEAU_TOL:.0e} for "
                              f"{_BARREN_PLATEAU_WINDOW} consecutive log intervals. "
                              f"Use INTERP warm-start (pass opt_params from p-1 run).")
                        _plateau_warned = True
                else:
                    _low_grad_count = 0

            # convergence: improvement over last _CONV_WINDOW steps < _CONV_TOL
            if conv_iter is None and len(cost_history) >= _CONV_WINDOW + 1:
                improvement = cost_history[-(1 + _CONV_WINDOW)] - cost_history[-1]
                if abs(improvement) < _CONV_TOL:
                    conv_iter    = step + 1
                    conv_time_ms = (_time.perf_counter() - t_opt_start) * 1000

    t_opt_loop_ms = (_time.perf_counter() - t_opt_start) * 1000

    # ── Extract best feasible solution ─────────────────────────────────────────
    t_decode_start = _time.perf_counter()
    probs  = circuit_probs(params)
    c_arr  = cfg.costs
    w_arr  = cfg.weights
    B      = float(cfg.budget_10k)
    n_scan = len(probs)

    # Vectorised bit extraction: avoids Python-loop string formatting which is
    # O(n_scan * n) and very slow at large n (e.g. n=24 → 16M Python iterations).
    sorted_idx = np.argsort(probs)[::-1][:n_scan]
    bits_shift = np.arange(n - 1, -1, -1, dtype=np.int64)
    X          = ((sorted_idx[:, None] >> bits_shift[None, :]) & 1).astype(float)
    costs_vec  = X @ c_arr
    objs_vec   = X @ w_arr
    feas_idx   = np.where(costs_vec <= B)[0]

    best: dict = {"obj": -np.inf, "x": None, "cost": None, "prob": 0.0}
    if len(feas_idx) > 0:
        i = int(feas_idx[np.argmax(objs_vec[feas_idx])])
        best = {
            "obj":  float(objs_vec[i]),
            "x":    X[i].copy(),
            "cost": float(costs_vec[i]),
            "prob": float(probs[sorted_idx[i]]),
        }

    t_decode_ms = (_time.perf_counter() - t_decode_start) * 1000

    return {
        "opt_params":        params,
        "cost_history":      cost_history,
        "grad_norm_history": grad_norm_history,
        "probs":             probs,
        "best_x":            best["x"],
        "best_obj":     best["obj"],
        "best_cost":    best["cost"],
        "best_prob":    best["prob"],
        "final_expval": cost_history[-1] if cost_history else float("nan"),
        "timing": {
            "circuit_init_ms": t_circuit_init_ms,
            "opt_loop_ms":     t_opt_loop_ms,
            "decode_ms":       t_decode_ms,
            "step_avg_ms":     t_opt_loop_ms / max(cfg.n_steps, 1),
        },
        "convergence": {
            "iter":    conv_iter,
            "time_ms": conv_time_ms,
            "n_steps": cfg.n_steps,
            "reached": conv_iter is not None,
        },
    }
