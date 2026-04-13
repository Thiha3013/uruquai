"""QAOA using PennyLane's JAX interface + JIT compilation.

Instead of PennyLane's AdamOptimizer (Python-level loop), this uses:
  - interface="jax" on the QNode → JAX traces the circuit once
  - jax.jit() → XLA compiles the full update step (circuit + gradient + Adam)
  - optax.adam → JAX-native Adam optimizer

The JIT-compiled update step runs entirely in XLA after the first call,
eliminating Python overhead on every iteration.  On Apple Silicon the XLA
CPU backend uses NEON/SIMD vectorisation; if jax-metal is installed it
can offload to the Metal GPU.

Same h, J, cfg interface as run_qaoa() — returns the same dict so it
drops into the existing benchmark table unchanged.
"""
from __future__ import annotations

import time as _time

import numpy as np

from .config import AppConfig
from .ising import build_pennylane_hamiltonian
from .backends import get_device

_CONV_WINDOW = 20
_CONV_TOL    = 1e-3


def run_qaoa_jax(cfg: AppConfig, h: np.ndarray, J: np.ndarray,
                 seed: int = 42) -> dict:
    """Run QAOA with JAX JIT-compiled optimisation loop.

    Parameters match run_qaoa() exactly so results are directly comparable.
    """
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError:
        raise ImportError(
            "JAX and optax are required: pip install jax optax"
        )

    # Enable 64-bit floats (default JAX uses 32-bit which reduces precision)
    jax.config.update("jax_enable_x64", True)

    import pennylane as qml

    n = len(h)
    p = cfg.p_layers

    # ── Setup ─────────────────────────────────────────────────────────────────
    t_setup_start = _time.perf_counter()

    H_cost, H_mix = build_pennylane_hamiltonian(h, J)
    # default.qubit with JAX interface — lightning does not support JAX
    dev = qml.device("default.qubit", wires=n)

    _couplings = [(i, j, float(J[i, j])) for i in range(n)
                  for j in range(i + 1, n) if abs(J[i, j]) > 1e-10]
    h_jax = jnp.array(h, dtype=jnp.float64)

    @qml.qnode(dev, interface="jax", diff_method="backprop")
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

    @qml.qnode(dev, interface="jax", diff_method="backprop")
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

    # ── JIT-compiled update step ───────────────────────────────────────────────
    optimizer = optax.adam(cfg.stepsize)

    @jax.jit
    def update(params, opt_state):
        cost_val, grads = jax.value_and_grad(circuit_cost)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, cost_val

    t_circuit_init_ms = (_time.perf_counter() - t_setup_start) * 1000

    # ── Optimisation ───────────────────────────────────────────────────────────
    np.random.seed(seed)
    params = jnp.array(
        np.random.uniform(0, np.pi / 2, 2 * p), dtype=jnp.float64
    )
    opt_state = optimizer.init(params)

    cost_history: list[float] = []
    conv_iter:    int   | None = None
    conv_time_ms: float | None = None

    t_opt_start = _time.perf_counter()

    # Warm-up: first call triggers JIT compilation — not counted in step timing
    print("  [JAX] compiling...")
    _dummy_p, _dummy_s, _ = update(params, opt_state)
    _ = circuit_probs(_dummy_p)   # also pre-compile probs circuit
    print("  [JAX] compiled — running optimisation ...")
    t_opt_start = _time.perf_counter()   # restart clock after compile

    for step in range(cfg.n_steps):
        params, opt_state, cost_val = update(params, opt_state)
        cost_history.append(float(cost_val))
        if (step + 1) % 50 == 0:
            print(f"  step {step+1:4d}  cost = {float(cost_val):.4f}")
        if conv_iter is None and len(cost_history) >= _CONV_WINDOW + 1:
            improvement = cost_history[-(1 + _CONV_WINDOW)] - cost_history[-1]
            if abs(improvement) < _CONV_TOL:
                conv_iter    = step + 1
                conv_time_ms = (_time.perf_counter() - t_opt_start) * 1000

    t_opt_loop_ms = (_time.perf_counter() - t_opt_start) * 1000

    # ── Decode ────────────────────────────────────────────────────────────────
    t_decode_start = _time.perf_counter()
    probs  = np.array(circuit_probs(params))
    c_arr  = cfg.costs
    w_arr  = cfg.weights
    B      = float(cfg.budget_10k)
    n_scan = min(50, len(probs))

    best: dict = {"obj": -np.inf, "x": None, "cost": None, "prob": 0.0}
    for idx in np.argsort(probs)[::-1][:n_scan]:
        x   = np.array([int(b) for b in format(idx, f"0{n}b")], dtype=float)
        tot = float(c_arr @ x)
        if tot <= B:
            obj = float(w_arr @ x)
            if obj > best["obj"]:
                best = {"obj": obj, "x": x.copy(), "cost": tot,
                        "prob": float(probs[idx])}

    t_decode_ms = (_time.perf_counter() - t_decode_start) * 1000

    return {
        "opt_params":   np.array(params),
        "cost_history": cost_history,
        "probs":        probs,
        "best_x":       best["x"],
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
