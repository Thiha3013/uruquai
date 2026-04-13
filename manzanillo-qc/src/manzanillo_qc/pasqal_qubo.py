"""Pasqal QUBO solver wrapper (qubo-solver package).

Uses Pasqal's qubo-solver library with a local neutral-atom emulator backend.
No cloud credentials required.

Backend selection
-----------------
N < _PASQAL_MPS_N : LocalEmulator (QutipBackendV2, statevector — fast at small n)
N >= _PASQAL_MPS_N: LocalEmulator (MPSBackend / emu-mps, MPS — scales as n×χ²)

The solver embeds the QUBO into a neutral-atom register, shapes an adiabatic
drive pulse, runs the quantum emulation, and samples 2000 bitstrings from the
final state.

Lambda calibration
------------------
The shared build_qubo() uses λ = 1 + Σwᵢ ≈ 7 for typical instances.  This
makes the penalty term ~14x larger than necessary, burying the objective signal
in the emulated energy landscape.  This solver rebuilds the QUBO with a
calibrated minimum λ = (Σwᵢ / min_c²) × 1.3 via build_qubo_sampling(), which
is the smallest λ that still guarantees feasibility while keeping objective
differences visible to the sampler.

Greedy repair
-------------
Neutral-atom systems tend to over-select sites (strong negative detuning on all
atoms).  After sampling, every over-budget bitstring is repaired by iteratively
dropping the site with the worst risk-per-cost (wᵢ/cᵢ) ratio until feasible.
The best coverage across all repaired candidates is returned.

Symmetry breaking
-----------------
When site costs are equal, (kc−B)² = ((k+1)c−B)² gives over-budget solutions
equal penalty to feasible ones.  A small cost-proportional term μ·cᵢ is added
to each diagonal to break this symmetry.

Returns the same dict format as run_qaoa() so it slots straight into
the benchmark table.

Install:
    pip install git+https://github.com/pasqal-io/qubo-solver
    pip install emu-mps   (already installed — needed for n >= _PASQAL_MPS_N)
"""
from __future__ import annotations

import time
import numpy as np

from .config import AppConfig
from .qubo import build_qubo_sampling, calibrate_lambda_for_sampling

_N_RUNS = 2000      # bitstring samples from the emulator (more = better repair diversity)
_PASQAL_MPS_N   = 16   # switch to MPSBackend above this qubit count
_MPS_MAX_BOND_DIM = 256


def run_pasqal_qubo(cfg: AppConfig, Q: np.ndarray) -> dict:
    """Solve a QUBO using Pasqal's qubosolver (local quantum emulator).

    Parameters
    ----------
    cfg : AppConfig
    Q   : np.ndarray  shape (N, N) — kept for API compatibility; internally
          replaced by build_qubo_sampling(cfg) with calibrated lambda.

    Returns
    -------
    dict with keys matching run_qaoa() output:
        best_x, best_obj, best_cost, best_prob, time_ms, timing, convergence
    """
    try:
        from qubosolver.solver import QuboSolver
        from qubosolver import QUBOInstance, SolverConfig
        from qoolqit.execution.backends import LocalEmulator
    except ImportError:
        raise ImportError(
            "qubosolver not installed: "
            "pip install git+https://github.com/pasqal-io/qubo-solver"
        )

    N = len(cfg.sites)
    use_mps = N >= _PASQAL_MPS_N
    if use_mps:
        try:
            from emu_mps import MPSBackend, MPSConfig
        except ImportError:
            use_mps = False

    c = cfg.costs
    B = float(cfg.budget_10k)
    w = cfg.weights
    # N already set above

    # Rebuild QUBO with calibrated minimum lambda for sampling.
    # The shared build_qubo() uses lambda = 1 + sum(w) ~ 7, making the penalty
    # ~15x larger than needed; this buries the objective signal and causes the
    # emulator to sample almost exclusively infeasible bitstrings.
    # calibrate_lambda_for_sampling() returns lam_min * 1.3 which is the smallest
    # lambda that still guarantees any budget violation costs more than the best
    # feasible objective.
    lam_default = 1.0 + float(w.sum())
    lam_cal     = calibrate_lambda_for_sampling(cfg)
    Q_cal, _    = build_qubo_sampling(cfg)
    print(f"  Pasqal lambda  default={lam_default:.3f}  calibrated={lam_cal:.3f}"
          f"  ratio={lam_default/lam_cal:.1f}x")

    # Q_cal is upper-triangular; Pasqal expects symmetric form.
    Q_sym = (Q_cal + Q_cal.T) / 2

    # Break the (kc−B)²=((k+1)c−B)² penalty symmetry.
    # When all site costs are equal (e.g. c=10, B=55), selecting k sites and
    # k+1 sites produce identical penalty values, so the QUBO minimum favours
    # the over-budget (k+1) solution since it has higher coverage reward.
    # Adding a tiny cost-proportional term μ·cᵢ to each diagonal makes
    # over-budget selections marginally more expensive than their symmetric
    # feasible counterparts, guiding the emulator toward feasible bitstrings.
    # μ is chosen as 1e-3 × (typical Q scale) / max_c so it is large enough
    # to break symmetry but negligible compared to objective differences.
    q_abs_max = max(float(np.abs(Q_sym).max()), 1.0)
    mu = 1e-3 * q_abs_max / max(float(c.max()), 1.0)
    Q_asym = Q_sym.copy()
    for i in range(len(c)):
        Q_asym[i, i] += mu * float(c[i])

    # Scale to fit device detuning range.
    q_scale = max(float(np.abs(Q_asym).max()), 1.0)
    Q_scaled = Q_asym / q_scale

    if use_mps:
        backend_label = f"MPS chi={_MPS_MAX_BOND_DIM}"
        mps_cfg = MPSConfig(max_bond_dim=_MPS_MAX_BOND_DIM)
        backend = LocalEmulator(backend_type=MPSBackend, emulation_config=mps_cfg, runs=_N_RUNS)
    else:
        backend_label = "QutipBackendV2"
        backend = LocalEmulator(runs=_N_RUNS)

    print(f"  Pasqal  n={N}  runs={_N_RUNS}  backend={backend_label}  "
          f"budget=${B*10:.0f}K  mu={mu:.4f}")

    t0 = time.perf_counter()
    instance = QUBOInstance(coefficients=Q_scaled.astype(float))
    config = SolverConfig(
        use_quantum=True,
        backend=backend,
        do_postprocessing=True,
    )
    import logging as _logging
    _ql_log = _logging.getLogger("qoolqit")
    _prev_level = _ql_log.level
    _ql_log.setLevel(_logging.ERROR)   # suppress "runs specified in both" warning
    solver = QuboSolver(instance=instance, config=config)
    solution = solver.solve()
    _ql_log.setLevel(_prev_level)
    t_ms = (time.perf_counter() - t0) * 1000

    # qubosolver.solve() may return a single bitstring of shape (n,)
    # instead of a batch of shape (k, n), especially after postprocessing.
    raw = solution.bitstrings
    if hasattr(raw, "detach"):  # torch tensor
        raw = raw.detach().cpu().numpy()
    elif hasattr(raw, "numpy"):  # tensor-like
        raw = raw.numpy()
    raw = np.asarray(raw)

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    def _repair(x_raw):
        """Greedy repair: drop the lowest w/c site until budget is satisfied.

        The emulator tends to over-select (all atoms see strong negative detuning).
        Rather than discarding over-budget bitstrings, we iteratively remove the
        site with the worst risk-per-cost ratio until the selection is feasible.
        This is standard post-processing for neutral-atom QUBO solvers.
        """
        x = x_raw.copy()
        while float(c @ x) > B:
            active = np.where(x > 0)[0]
            if len(active) == 0:
                break
            ratios = w[active] / c[active]
            worst = active[int(np.argmin(ratios))]
            x[worst] = 0.0
        return x

    obj = -np.inf
    bs = None
    tot = None
    n_repaired = 0

    for row in raw:
        x = np.asarray(row, dtype=float).reshape(-1)
        if x.shape[0] != c.shape[0]:
            continue
        if float(c @ x) > B:
            x = _repair(x)
            n_repaired += 1
        cost = float(c @ x)
        if cost <= B:
            candidate_obj = float(w @ x)
            if candidate_obj > obj:
                obj = candidate_obj
                bs = x.copy()
                tot = cost

    print(f"  Pasqal done  {t_ms/1000:.1f}s  "
          f"obj={obj:.4f}  cost=${(tot or 0)*10:.0f}K  "
          f"bitstrings={raw.shape[0] if raw.ndim == 2 else 1}  "
          f"repaired={n_repaired}")

    return {
        "best_x": bs,
        "best_obj": obj,
        "best_cost": tot,
        "best_prob": None,
        "time_ms": t_ms,
        "timing": {
            "circuit_init_ms": 0.0,
            "opt_loop_ms": t_ms,
            "decode_ms": 0.0,
            "step_avg_ms": 0.0,
        },
        "convergence": {
            "iter": None,
            "time_ms": None,
            "n_steps": _N_RUNS,
            "reached": True,
        },
    }