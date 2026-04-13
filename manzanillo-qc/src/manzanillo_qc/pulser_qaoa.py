"""Analog QAOA via Pulser.

Accepts a live-data AppConfig + QUBO matrix — no YAML file needed.
Rebuilds the QUBO with a calibrated minimum λ, performs atom-layout
optimisation, runs a T-sweep to find the best evolution time, then reports
results in the same dict format as run_qaoa().

Backend selection
-----------------
N < _PULSER_MPS_N : QutipEmulator (statevector, fast at small n)
N >= _PULSER_MPS_N: MPSBackend / emu-mps (MPS, scales as n×χ² instead
                    of 2ⁿ — lower memory and faster for n ≥ 16+)

Lambda calibration
------------------
The shared build_qubo() uses λ = 1 + Σwᵢ ≈ 7.  This makes the penalty
dominant over the objective terms, giving a flat off-diagonal Q that the
Nelder-Mead layout optimizer cannot distinguish from noise.  This solver
rebuilds Q with calibrated λ = (Σwᵢ / min_c²) × 1.3 via build_qubo_sampling(),
which is the smallest λ that still enforces feasibility.

T-sweep
-------
n < 12 : [1, 4, 7] µs  (3 points)
n ≥ 12 : [1, 3, 5, 7, 10, 12] µs  (6 points)
Best T is selected by lowest gap%; ties broken by shortest T.

Greedy repair
-------------
Neutral-atom systems tend to over-select sites (strong negative detuning on
all atoms).  _decode_best() now applies greedy repair to over-budget bitstrings:
iteratively drops the site with the worst wᵢ/cᵢ ratio until feasible.  The
best coverage across all repaired candidates is returned.
"""
from __future__ import annotations

import time
import numpy as np

from .config import AppConfig
from .qubo import build_qubo_sampling, calibrate_lambda_for_sampling


# T-sweep values (ns) — 7 points up to 7 µs, balanced for 12-14 qubit runtime
_T_VALUES = [int(t) for t in 1000 * np.linspace(1, 7, 7)]

# Switch to MPSBackend above this qubit count (statevector becomes expensive)
_PULSER_MPS_N = 16
_MPS_MAX_BOND_DIM = 256   # χ=256: ~50MB for n=24, accurate for low-entanglement Rydberg


def run_pulser_qaoa(cfg: AppConfig, Q: np.ndarray,
                    n_samples: int = 1000,
                    t_values: list | None = None) -> dict:
    """Run analog QAOA on a neutral-atom emulator (Pulser / QutipEmulator).

    Parameters
    ----------
    cfg       : AppConfig
    Q         : np.ndarray  shape (N, N) QUBO matrix
    n_samples : int         shots per simulation

    Returns
    -------
    dict compatible with run_qaoa() — keys: best_x, best_obj, best_cost,
    best_prob, time_ms, timing, convergence
    """
    try:
        from pulser import Pulse, Sequence, Register
        from pulser.devices import DigitalAnalogDevice
        from pulser.waveforms import InterpolatedWaveform
        from scipy.optimize import minimize
        from scipy.spatial.distance import pdist, squareform
    except ImportError:
        raise ImportError(
            "pulser is required: pip install pulser pulser-simulation"
        )

    N = len(cfg.sites)
    use_mps = N >= _PULSER_MPS_N
    if use_mps:
        try:
            from emu_mps import MPSBackend, MPSConfig
        except ImportError:
            use_mps = False
    if not use_mps:
        try:
            from pulser_simulation import QutipEmulator
        except ImportError:
            raise ImportError("pulser-simulation is required: pip install pulser-simulation")
    costs   = cfg.costs
    weights = cfg.weights
    BUDGET  = float(cfg.budget_10k)

    # Rebuild QUBO with calibrated minimum lambda for sampling.
    # The shared build_qubo() uses lambda = 1 + sum(w) ~ 7; this makes the
    # penalty ~15x larger than necessary, washing out the objective signal.
    # Pulser's layout optimizer maps Q off-diagonal values to physical Rydberg
    # interactions; a stiff penalty-dominated Q gives a flat interaction landscape
    # where atom positions don't encode site quality differences.
    lam_default = 1.0 + float(weights.sum())
    lam_cal     = calibrate_lambda_for_sampling(cfg)
    Q_cal, _    = build_qubo_sampling(cfg)
    print(f"  Pulser lambda  default={lam_default:.3f}  calibrated={lam_cal:.3f}"
          f"  ratio={lam_default/lam_cal:.1f}x")
    # Use Q_cal for the layout optimization and simulation.
    Q = Q_cal

    # ── Reference optimal (brute-force ≤20 qubits, MILP above) ────────────
    _BF_CUTOFF = 20
    best_bf_obj, OPTIMAL_BS = -np.inf, None
    if N <= _BF_CUTOFF:
        for mask in range(1 << N):
            x = np.array([(mask >> i) & 1 for i in range(N)], dtype=float)
            if float(costs @ x) <= BUDGET:
                obj = float(weights @ x)
                if obj > best_bf_obj:
                    best_bf_obj = obj
                    OPTIMAL_BS  = "".join(str(int((mask >> i) & 1)) for i in range(N))
        OPTIMAL = best_bf_obj
    else:
        from .benchmarks import milp_solve
        _ml = milp_solve(cfg)
        OPTIMAL = _ml["obj"] if _ml["success"] else 0.0
        OPTIMAL_BS = None   # exact bitstring unavailable; _opt_frac will return 0

    # ── Normalise Q off-diagonal ───────────────────────────────────────────
    Q_off = Q.copy()
    np.fill_diagonal(Q_off, 0)
    q_max = Q_off.max()
    Q_norm = Q_off / q_max if q_max != 0 else Q_off

    # ── Helper functions ───────────────────────────────────────────────────
    def _qubo_cost(bs):
        z = np.array(list(bs), dtype=int)
        return float(z @ Q @ z)

    def _repair(x_raw):
        """Greedy repair: drop the lowest w/c site until budget is satisfied.

        The emulator over-selects because all atoms see strong negative detuning.
        Repair iteratively removes the site with the worst risk-per-cost ratio
        until the selection is feasible — standard neutral-atom post-processing.
        """
        x = x_raw.copy()
        while float(costs @ x) > BUDGET:
            active = np.where(x > 0)[0]
            if len(active) == 0:
                break
            ratios = weights[active] / costs[active]
            worst = active[int(np.argmin(ratios))]
            x[worst] = 0.0
        return x

    def _decode_best(counter):
        best = {"obj": -np.inf, "x": None, "capex": None, "count": 0, "prob": 0.0}
        total = sum(counter.values())
        for bs, cnt in counter.items():
            x = np.array(list(bs), dtype=float)
            if float(costs @ x) > BUDGET:
                x = _repair(x)
            tot = float(costs @ x)
            if tot <= BUDGET:
                obj = float(weights @ x)
                if obj > best["obj"]:
                    best = {"obj": obj, "x": x.copy(), "capex": tot,
                            "count": cnt, "prob": cnt / total}
        return best

    def _feasible_frac(counter):
        total = sum(counter.values())
        feas  = sum(cnt for bs, cnt in counter.items()
                    if float(costs @ np.array(list(bs), dtype=float)) <= BUDGET)
        return feas / total

    def _opt_frac(counter):
        total = sum(counter.values())
        return counter.get(OPTIMAL_BS, 0) / total

    def _avg_cost(counter):
        total = sum(counter.values())
        return sum(cnt * _qubo_cost(bs) for bs, cnt in counter.items()) / total

    # ── Atom layout optimisation ───────────────────────────────────────────
    def _eval_mapping(coords_flat, Q_target, shape):
        coords = np.reshape(coords_flat, shape)
        new_Q  = squareform(DigitalAnalogDevice.interaction_coeff / pdist(coords) ** 6)
        return np.linalg.norm(new_Q - Q_target)

    # Multiple random restarts: Nelder-Mead fails in high dimensions from a
    # single start. Try _N_LAYOUT_RESTARTS seeds and keep the best layout.
    _N_LAYOUT_RESTARTS = 5
    backend_label = f"MPS chi={_MPS_MAX_BOND_DIM}" if use_mps else "QutipEmulator"
    print(f"  Pulser layout  n={N}  restarts={_N_LAYOUT_RESTARTS}  backend={backend_label}")
    t_layout_start = time.perf_counter()
    best_res = None
    for seed in range(_N_LAYOUT_RESTARTS):
        np.random.seed(seed)
        x0  = np.random.random((N, 2)).flatten() * 10
        res = minimize(_eval_mapping, x0, args=(Q_norm, (N, 2)),
                       method="Nelder-Mead", tol=1e-6,
                       options={"maxiter": 200_000, "maxfev": None})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
        print(f"    restart {seed+1}/{_N_LAYOUT_RESTARTS}  layout_err={res.fun:.4f}"
              f"  best_so_far={best_res.fun:.4f}"
              f"  ({(time.perf_counter()-t_layout_start)*1000:.0f}ms)")
    coords = np.reshape(best_res.x, (N, 2))

    # Fit within DigitalAnalogDevice radial limit (50 µm)
    coords -= coords.mean(axis=0)
    max_r   = np.max(np.linalg.norm(coords, axis=1))
    if max_r > 48.0:
        coords *= 48.0 / max_r

    t_layout_ms = (time.perf_counter() - t_layout_start) * 1000

    qubits = {str(i): coords[i] for i in range(N)}
    reg    = Register(qubits)

    Omega   = min(np.median(Q_norm[Q_norm > 0].flatten()), 1.5)
    delta_0 = -5
    delta_f =  5

    def _make_seq(T_ns):
        pulse = Pulse(
            InterpolatedWaveform(T_ns, [1e-9, Omega, 1e-9]),
            InterpolatedWaveform(T_ns, [delta_0, 0, delta_f]),
            0,
        )
        seq = Sequence(reg, DigitalAnalogDevice)
        seq.declare_channel("ising", "rydberg_global")
        seq.add(pulse, "ising")
        return seq

    # ── T-sweep ────────────────────────────────────────────────────────────
    T_GRID = t_values if t_values is not None else _T_VALUES
    sweep = []
    t_sweep_start = time.perf_counter()
    print(f"  Pulser T-sweep  {len(T_GRID)} points  "
          f"T=[{T_GRID[0]},{T_GRID[-1]}]ns  optimal={OPTIMAL:.4f}")
    for t_idx, T_ns in enumerate(T_GRID):
        t0_t  = time.perf_counter()
        seq_t = _make_seq(T_ns)
        if use_mps:
            backend_t = MPSBackend(seq_t, config=MPSConfig(max_bond_dim=_MPS_MAX_BOND_DIM))
            res_t     = backend_t.run()
            counts_t  = res_t.final_bitstrings      # Counter, total=1000 by default
        else:
            sim_t    = QutipEmulator.from_sequence(seq_t)
            res_t    = sim_t.run()
            counts_t = res_t.sample_final_state(N_samples=n_samples)
        t_ms     = (time.perf_counter() - t0_t) * 1000
        best_t   = _decode_best(counts_t)
        gap = 100*(OPTIMAL - best_t["obj"])/OPTIMAL if best_t["obj"] > 0 else 100
        elapsed_total = (time.perf_counter() - t_sweep_start) * 1000
        remaining_pts  = len(T_GRID) - (t_idx + 1)
        eta_ms         = (elapsed_total / (t_idx + 1)) * remaining_pts
        eta_str        = (f"{int(eta_ms/60000)}m{int((eta_ms%60000)/1000)}s"
                          if eta_ms > 60000 else f"{eta_ms/1000:.0f}s")
        print(f"    T={T_ns:6d}ns  [{t_idx+1}/{len(T_GRID)}]"
              f"  obj={best_t['obj']:7.4f}"
              f"  gap={gap:5.1f}%"
              f"  feas={_feasible_frac(counts_t)*100:5.1f}%"
              f"  {t_ms/1000:.1f}s"
              f"  ETA {eta_str}")
        sweep.append({
            "T_ns":      T_ns,
            "t_ms":      t_ms,
            "counts":    counts_t,
            "best_obj":  best_t["obj"],
            "best_x":    best_t["x"],
            "best_capex":best_t["capex"],
            "best_count":best_t["count"],
            "best_prob": best_t["prob"],
            "gap_pct":   gap,
            "feas_frac": _feasible_frac(counts_t),
            "opt_frac":  _opt_frac(counts_t),
            "avg_cost":  _avg_cost(counts_t),
            "sensors":   [int(i) for i in np.where(best_t["x"])[0]] if best_t["x"] is not None else [],
        })
    t_sweep_ms = (time.perf_counter() - t_sweep_start) * 1000

    # Best T: lowest gap, then fastest
    zero_gap = [r for r in sweep if r["gap_pct"] == 0.0]
    best     = min(zero_gap, key=lambda r: r["T_ns"]) if zero_gap \
               else min(sweep, key=lambda r: r["gap_pct"])

    total_ms = t_layout_ms + t_sweep_ms

    return {
        "best_x":    best["best_x"],
        "best_obj":  best["best_obj"],
        "best_cost": best["best_capex"],
        "best_prob": best["best_prob"],
        "time_ms":   total_ms,
        "timing": {
            "circuit_init_ms": t_layout_ms,
            "opt_loop_ms":     t_sweep_ms,
            "decode_ms":       0.0,
            "step_avg_ms":     t_sweep_ms / len(T_GRID),
        },
        "convergence": {
            "iter":    best["T_ns"],
            "time_ms": best["t_ms"],
            "n_steps": len(T_GRID),
            "reached": best["gap_pct"] == 0.0,
        },
        "_sweep": sweep,   # full T-sweep data for detailed output
        "_optimal": OPTIMAL,
    }
