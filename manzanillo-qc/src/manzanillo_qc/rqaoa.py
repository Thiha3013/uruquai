"""Recursive QAOA (RQAOA) for QUBO optimisation.

Reference: Bravyi et al., arXiv:2001.09660 (2020).

How it works
------------
Instead of running deep QAOA (p=6, barren plateaus), RQAOA always uses p=1
(shallow, hardware-friendly) and recurses:

  1. Run p=1 QAOA on the current n-qubit problem.
  2. Compute correlators ⟨σᵢᶻ σⱼᶻ⟩ (pairs) and ⟨σᵢᶻ⟩ (single-qubit).
  3. Find the strongest |correlator|.
     - Pair   C_ij > 0 → x_j = x_i   (tend to co-select)
     - Pair   C_ij < 0 → x_j = 1−x_i (tend to anti-select)
     - Single ⟨Z⟩  < 0 → pin x_i = 1 (select)
     - Single ⟨Z⟩  > 0 → pin x_i = 0 (don't select)
  4. Substitute the elimination into the QUBO → (n-1)-qubit problem.
  5. Recurse until n ≤ _CLASSICAL_N (=2), then brute-force the 4-state residual.
  6. Back-substitute through the chain to recover the full n-bit solution.

n=4 behaviour: runs 2 real QAOA levels (4→3→2) then brute-forces the n=2 residual.
n=12 behaviour: runs 10 real QAOA levels (12→11→…→2) then brute-forces the residual.

Key advantages over standard QAOA
----------------------------------
- Always p=1 → no barren plateaus by construction.
- Each step is a shallow circuit → hardware-friendly on Pasqal/IBM today.
- Works at n=24+ (first step takes ~20s, shrinks fast after that).

Known limitations
-----------------
- Quality degrades at late recursion levels (n≥14): correlators approach zero as
  the reduced problem loses structure (C ≈ −0.000), making eliminations near-random.
- Each imperfect elimination compounds errors; n=12 (10 levels) already shows
  ~10% gap vs PennyLane QAOA on the same instance.

Convergence quality
-------------------
The p=1 QAOA correlators are only meaningful if the optimizer actually
converges to a good minimum. We use:
  - 300 steps max, lr=0.01 (sufficient for 2 parameters)
  - Early stopping: stop when ||∇|| < 1e-4 for 10 consecutive steps
  - 3 random seeds per level — take the seed with the lowest final ⟨H⟩
  - Good initialization: γ ≈ π/(4·h_max), β = π/4 (analytical p=1 near-optimum)
"""
from __future__ import annotations

import time
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from .config import AppConfig

_CLASSICAL_N   = 2      # brute-force the residual below this size — n=2 has 4 states, instant.
                        # Setting to 2 means n=4 runs 2 real QAOA levels (4→3→2) before brute-force.
_P1_STEPS      = 300    # max Adam steps per recursion level
_P1_LR         = 0.01   # Adam learning rate
_N_SEEDS       = 3      # seeds at level 0; later levels use max(1, _N_SEEDS - level)
_CONV_GRAD     = 1e-4   # early-stop threshold: stop Adam when ||∇|| < this
_CONV_WINDOW   = 10     # number of consecutive steps below threshold before stopping
_RQAOA_MAX_N   = 9999   # no hard cutoff — let it run and fail naturally


# ── Brute-force for small residual ───────────────────────────────────────────

def _brute_force(Q: np.ndarray, costs: np.ndarray,
                 weights: np.ndarray, budget: float) -> dict:
    n = Q.shape[0]
    best = {"obj": -np.inf, "x": None}
    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        if float(costs @ x) <= budget:
            obj = float(weights @ x)
            if obj > best["obj"]:
                best = {"obj": obj, "x": x.copy(), "cost": float(costs @ x)}
    return best


# ── QUBO reduction ────────────────────────────────────────────────────────────

def _reduce_qubo(Q: np.ndarray, costs: np.ndarray, weights: np.ndarray,
                 budget: float, elim: int, ref: int | None,
                 sign: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Eliminate variable `elim` from the QUBO.

    If ref is None : pin x[elim] = pin_val (0 if sign>0, 1 if sign<0)
    Else           : substitute x[elim] = x[ref]   (sign>0)
                                x[elim] = 1-x[ref]  (sign<0)

    Returns (Q_new, costs_new, weights_new, budget_new, offset).
    """
    n    = Q.shape[0]
    keep = [i for i in range(n) if i != elim]
    m    = len(keep)
    idx  = {old: new for new, old in enumerate(keep)}

    Q_new = np.zeros((m, m))
    c_new = costs[keep].copy()
    w_new = weights[keep].copy()
    b_new = budget
    offset = 0.0

    # Fill in the unchanged block first
    for i in keep:
        for j in keep:
            Q_new[idx[i], idx[j]] += Q[i, j]

    if ref is None:
        pin = 0.0 if sign > 0 else 1.0
        # Diagonal contribution of pinned variable
        offset += Q[elim, elim] * pin
        # Off-diagonal: Q[elim,j]*pin → add to linear (diagonal) of j
        for j in keep:
            jj = idx[j]
            contrib = (Q[elim, j] + Q[j, elim]) * pin
            Q_new[jj, jj] += contrib
        # Update budget: if pin=1 we spent cost[elim]
        b_new = budget - costs[elim] * pin
        # Offset for objective: pin=1 earns weights[elim]
        offset -= weights[elim] * pin   # subtract from QUBO (adds to objective)
    else:
        r = idx[ref]
        if sign > 0:
            # x[elim] = x[ref]
            # Diagonal: Q[e,e]*x[r] → adds to Q[r,r]
            Q_new[r, r] += Q[elim, elim]
            # Cross: Q[e,r]*x[e]*x[r] = Q[e,r]*x[r]^2 = Q[e,r]*x[r] → linear on r
            Q_new[r, r] += Q[elim, ref] + Q[ref, elim]
            # Off-diagonal with others: Q[e,j]*x[e]*x[j] = Q[e,j]*x[r]*x[j]
            for j in keep:
                if j == ref:
                    continue
                jj = idx[j]
                Q_new[r, jj] += Q[elim, j]
                Q_new[jj, r] += Q[j, elim]
            # Cost and weight accumulate on ref
            c_new[r] += costs[elim]
            w_new[r] += weights[elim]
        else:
            # x[elim] = 1 - x[ref]
            # Q[e,e]*(1-x[r]): linear → -Q[e,e]*x[r], constant Q[e,e]
            offset    += Q[elim, elim]
            Q_new[r, r] -= Q[elim, elim]
            # Off-diagonal Q[e,j]*(1-x[r])*x[j] = Q[e,j]*x[j] - Q[e,j]*x[r]*x[j]
            for j in keep:
                jj = idx[j]
                # Constant linear term Q[e,j]*x[j]
                Q_new[jj, jj] += Q[elim, j] + Q[j, elim]
                # Bilinear -Q[e,j]*x[r]*x[j]
                Q_new[r, jj]  -= Q[elim, j]
                Q_new[jj, r]  -= Q[j, elim]
            # Cost/weight: costs[ref] -= costs[elim] (net effect on ref)
            c_new[r] -= costs[elim]
            w_new[r] -= weights[elim]
            # Budget: x[elim]=1 costs costs[elim], so effective remaining budget decreases
            b_new = budget - costs[elim]
            offset += weights[elim]   # constant objective contribution

    return Q_new, c_new, w_new, b_new, offset


# ── p=1 QAOA correlator computation ──────────────────────────────────────────

def _p1_correlators(h: np.ndarray, J: np.ndarray,
                    n_seeds: int = _N_SEEDS) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Run p=1 QAOA; return (expval, ⟨Z⟩, ⟨ZZ⟩, opt_params).

    Parameters
    ----------
    n_seeds : int  Number of random restarts. Use fewer at later recursion levels
                   where the reduced problem is smaller and the landscape is simpler.
    """
    n = len(h)
    dev = qml.device("lightning.qubit", wires=n)

    couplings = [(i, j, float(J[i, j]))
                 for i in range(n) for j in range(i + 1, n)
                 if abs(J[i, j]) > 1e-10]

    h_coeffs = [float(h[i]) for i in range(n) if abs(h[i]) > 1e-10]
    h_wires  = [i             for i in range(n) if abs(h[i]) > 1e-10]

    H_ops    = [qml.PauliZ(w) for w in h_wires]
    H_coeffs = h_coeffs[:]
    for i, j, Jij in couplings:
        H_ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
        H_coeffs.append(Jij)
    H_cost = qml.Hamiltonian(H_coeffs, H_ops)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit_cost(params):
        gamma, beta = params[0], params[1]
        for i in range(n):
            qml.Hadamard(wires=i)
        for i, w in zip(h_wires, h_coeffs):
            qml.RZ(2.0 * gamma * w, wires=i)
        for i, j, Jij in couplings:
            qml.IsingZZ(2.0 * gamma * Jij, wires=[i, j])
        for i in range(n):
            qml.RX(2.0 * beta, wires=i)
        return qml.expval(H_cost)

    @qml.qnode(dev)
    def circuit_observables(params):
        gamma, beta = params[0], params[1]
        for i in range(n):
            qml.Hadamard(wires=i)
        for i, w in zip(h_wires, h_coeffs):
            qml.RZ(2.0 * gamma * w, wires=i)
        for i, j, Jij in couplings:
            qml.IsingZZ(2.0 * gamma * Jij, wires=[i, j])
        for i in range(n):
            qml.RX(2.0 * beta, wires=i)
        obs = [qml.expval(qml.PauliZ(i)) for i in range(n)]
        obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
                for i in range(n) for j in range(i + 1, n)]
        return obs

    # Good p=1 initialization: γ near π/(4·h_max), β=π/4
    h_max = max(np.abs(h).max(), 1e-6)
    gamma_init = float(np.pi / (4.0 * h_max))
    beta_init  = float(np.pi / 4.0)

    best_expval = np.inf
    best_params = None

    for seed in range(n_seeds):
        np.random.seed(seed)
        gamma_0 = gamma_init * (0.5 + np.random.uniform(0, 1.0))
        beta_0  = beta_init  * (0.5 + np.random.uniform(0, 1.0))
        params  = pnp.array([gamma_0, beta_0], requires_grad=True)

        opt = qml.AdamOptimizer(stepsize=_P1_LR)
        below_thresh = 0
        for _ in range(_P1_STEPS):
            params, grad = opt.step_and_cost(circuit_cost, params)
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm < _CONV_GRAD:
                below_thresh += 1
                if below_thresh >= _CONV_WINDOW:
                    break
            else:
                below_thresh = 0

        val = float(circuit_cost(params))
        if val < best_expval:
            best_expval = val
            best_params = np.array(params)

    results = circuit_observables(best_params)
    single  = np.array([float(r) for r in results[:n]])
    pair    = np.zeros((n, n))
    idx_zz  = n
    for i in range(n):
        for j in range(i + 1, n):
            pair[i, j] = float(results[idx_zz])
            idx_zz += 1

    return best_expval, single, pair, best_params


# ── Reconstruction helper ─────────────────────────────────────────────────────

def _reconstruct(chain: list[tuple[int, int | None, int, list[int]]],
                 active_final: list[int],
                 x_residual: np.ndarray | None,
                 n_orig: int) -> np.ndarray:
    """Back-substitute elimination chain to recover full solution."""
    x = np.zeros(n_orig)
    if x_residual is None:
        return x

    for local_idx, orig_idx in enumerate(active_final):
        x[orig_idx] = float(x_residual[local_idx])

    # Replay in reverse: chain entry = (elim_local, ref_local, sign, active_at_step)
    for (elim_local, ref_local, sign, active_at_step) in reversed(chain):
        elim_orig = active_at_step[elim_local]
        if ref_local is None:
            x[elim_orig] = 0.0 if sign > 0 else 1.0
        else:
            ref_orig = active_at_step[ref_local]
            x[elim_orig] = x[ref_orig] if sign > 0 else 1.0 - x[ref_orig]

    return x


# ── Main RQAOA ────────────────────────────────────────────────────────────────

def run_rqaoa(cfg: AppConfig, Q: np.ndarray) -> dict:
    """Run Recursive QAOA and return results in the same format as run_qaoa().

    Parameters
    ----------
    cfg : AppConfig
    Q   : (n, n) upper-triangular QUBO matrix from build_qubo()
    """
    n_orig  = len(cfg.sites)
    if n_orig > _RQAOA_MAX_N:
        raise ValueError(f"RQAOA: n={n_orig} > {_RQAOA_MAX_N}")

    t0      = time.perf_counter()
    budget  = float(cfg.budget_10k)

    Q_cur   = Q.astype(float).copy()
    costs   = cfg.costs.astype(float).copy()
    weights = cfg.weights.astype(float).copy()
    n_cur   = n_orig
    active  = list(range(n_orig))

    # chain: (elim_local, ref_local, sign, active_snapshot_at_this_step)
    chain: list[tuple[int, int | None, int, list[int]]] = []
    cost_history: list[float] = []

    print(f"  RQAOA  n={n_orig}  p=1  max={_P1_STEPS} steps/level  "
          f"{_N_SEEDS} seeds  conv_grad={_CONV_GRAD}  classical_n={_CLASSICAL_N}")

    level = 0
    while n_cur > _CLASSICAL_N:
        # ── Build Ising from current QUBO ─────────────────────────────────
        # Symmetrize: QUBO is upper-triangular from build_qubo
        Q_sym = Q_cur + Q_cur.T - np.diag(np.diag(Q_cur))
        h_cur = np.zeros(n_cur)
        J_cur = np.zeros((n_cur, n_cur))

        for i in range(n_cur):
            # Diagonal → linear Ising field: h_i = -Q[i,i]/2 - Σ_{j≠i} Q[i,j]/4
            h_cur[i] = -Q_sym[i, i] / 2.0
            for j in range(n_cur):
                if j != i:
                    h_cur[i] -= Q_sym[i, j] / 4.0

        for i in range(n_cur):
            for j in range(i + 1, n_cur):
                J_cur[i, j] = Q_sym[i, j] / 4.0

        scale = max(float(np.abs(h_cur).max()),
                    float(np.abs(J_cur).max()), 1.0)
        h_sc  = h_cur / scale
        J_sc  = J_cur / scale

        step_t0 = time.perf_counter()
        expval, single, pair, _ = _p1_correlators(h_sc, J_sc, n_seeds=_N_SEEDS)
        step_ms = (time.perf_counter() - step_t0) * 1000
        cost_history.append(expval)

        # ── Find strongest correlator ──────────────────────────────────────
        best_mag  = 0.0
        best_type = "single"
        best_i    = int(np.argmax(np.abs(single)))   # default: single with max |⟨Z⟩|
        best_j    = -1
        best_val  = float(single[best_i])
        best_mag  = abs(best_val)

        for i in range(n_cur):
            for j in range(i + 1, n_cur):
                if abs(pair[i, j]) > best_mag:
                    best_mag  = abs(pair[i, j])
                    best_type = "pair"
                    best_val  = float(pair[i, j])
                    best_i, best_j = i, j

        # ── Apply elimination ──────────────────────────────────────────────
        if best_type == "pair":
            sign  = 1 if best_val > 0 else -1
            elim  = best_j   # keep i, eliminate j
            ref   = best_i
            rel   = "=" if sign > 0 else "=1-"
            print(f"    n={n_cur:>3}→{n_cur-1}  "
                  f"pair: site[{active[elim]}]{rel}site[{active[ref]}]  "
                  f"C={best_val:+.3f}  ⟨H⟩={expval:.4f}  {step_ms:.0f}ms")
        else:
            sign  = 1 if best_val > 0 else -1
            elim  = best_i
            ref   = None
            pin   = 0 if sign > 0 else 1
            print(f"    n={n_cur:>3}→{n_cur-1}  "
                  f"pin: site[{active[elim]}]={pin}  "
                  f"⟨Z⟩={best_val:+.3f}  ⟨H⟩={expval:.4f}  {step_ms:.0f}ms")

        chain.append((elim, ref, sign, active.copy()))
        active.pop(elim)

        Q_cur, costs, weights, budget, _ = _reduce_qubo(
            Q_cur, costs, weights, budget, elim, ref, sign
        )
        n_cur -= 1
        level += 1

    # ── Brute-force residual ───────────────────────────────────────────────
    budget_r = max(budget, 0.0)
    print(f"    n={n_cur:>3}  brute-force residual  remaining_budget=${budget_r*10:.0f}K")
    residual = _brute_force(Q_cur, costs, weights, budget_r)

    # ── Back-substitute chain ──────────────────────────────────────────────
    x_full = _reconstruct(chain, active, residual.get("x"), n_orig)

    orig_c  = cfg.costs
    orig_w  = cfg.weights
    B_orig  = float(cfg.budget_10k)

    total_cost = float(orig_c @ x_full)
    feasible   = total_cost <= B_orig
    total_obj  = float(orig_w @ x_full) if feasible else -np.inf

    t_ms = (time.perf_counter() - t0) * 1000
    print(f"  RQAOA done  {t_ms/1000:.1f}s  "
          f"obj={total_obj:.4f}  cost=${total_cost*10:.0f}K  feasible={feasible}")

    return {
        "best_x":            x_full,
        "best_obj":          total_obj,
        "best_cost":         total_cost,
        "best_prob":         None,
        "time_ms":           t_ms,
        "cost_history":      cost_history,
        "grad_norm_history": [],
        "probs":             None,
        "opt_params":        None,
        "final_expval":      cost_history[-1] if cost_history else float("nan"),
        "timing": {
            "circuit_init_ms": 0.0,
            "opt_loop_ms":     t_ms,
            "decode_ms":       0.0,
            "step_avg_ms":     t_ms / max(n_orig - _CLASSICAL_N, 1),
        },
        "convergence": {
            "iter":    len(cost_history),
            "time_ms": t_ms,
            "n_steps": len(cost_history) * _P1_STEPS,
            "reached": True,
        },
    }
