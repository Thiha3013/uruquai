"""Runtime-vs-qubits scaling benchmark on live data.

Fetches USGS/FDSN data once, builds instances for n = [4, 6, 8, 10, 12]
qubits (top-n highest-risk live sites), then runs all solvers at each size
with consistent settings and prints a gap% comparison table.

Solvers benchmarked
-------------------
    PennyLane QAOA  — p=6, AdamOptimizer, 300 steps
    Pulser          — analog QAOA, Rydberg layout, T-sweep, greedy repair
    Pasqal          — qubo-solver, neutral-atom emulator, greedy repair
    RQAOA           — recursive p=1 QAOA, 3 seeds/level, brute-force at n≤2
    Greedy          — O(n log n) classical baseline
    MILP            — exact branch-and-bound classical baseline

Gap% reference: brute-force for n≤20, MILP for n>20.

Saved plots (inside --plots-dir, default plots/):
    sensor_map_combined.png    — all solvers on one geographic map
    sensor_maps/<solver>.png   — one map per solver
    runtime_combined.png       — all solvers, time vs n (log-y)
    quality_combined.png       — all solvers, gap% vs n

Usage
-----
    PYTHONPATH=src python3 -m manzanillo_qc.scaling
    PYTHONPATH=src python3 -m manzanillo_qc.scaling --no-pulser --no-pasqal
    PYTHONPATH=src python3 -m manzanillo_qc.scaling --p-layers 3 --steps 400
    PYTHONPATH=src python3 -m manzanillo_qc.scaling --output scaling.json --no-plots
    PYTHONPATH=src python3 -m manzanillo_qc.scaling --plots-dir my_plots/
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .config import AppConfig
from .instance import fetch_instance, fetch_stations
from .qubo import build_qubo, brute_force
from .ising import qubo_to_ising
from .qaoa import run_qaoa
from .benchmarks import greedy, milp_solve, plot_sensor_map

# ── Defaults ───────────────────────────────────────────────────────────────────
#QUBIT_COUNTS = [4, 6, 8, 10, 12, 14]
#QUBIT_COUNTS = [16]
#QUBIT_COUNTS = [24]   # standalone large-n stress test — swap in when ready
BUDGET_10K      = 80          # $800K — allows ~6-8 sensors, scales better with n

# QAOA — p=6 layers (one more than baseline p=5 for better expressibility).
# Empirical convergence: n=8 ~step 180, n=14 ~step 221. 300 steps gives a
# comfortable buffer without wasting time on already-converged runs.
DEFAULT_P       = 6
DEFAULT_STEPS   = 300

# Pulser — T-sweep is n-dependent: larger n needs wider time range to find good evolution time
PULSER_T_NS        = [1_000, 4_000, 7_000]                           # n≤10
PULSER_T_NS_LARGE  = [1_000, 3_000, 5_000, 7_000, 10_000, 12_000]   # n≥12
PULSER_T_LARGE_N   = 12   # threshold for switching to wide T-sweep
PULSER_SAMPLES     = 500

# ── Scalability cutoffs ────────────────────────────────────────────────────────
# Above these n values the corresponding solver is skipped (OOM / impractical).
# MILP is used as gap% reference whenever brute-force is skipped.
_BF_MAX_N     = 20   # brute-force: 2^20 = 1M states, ~1 s; 2^24 = 16M, ~30 s; 2^50 → ∞
_QAOA_MAX_N   = 9999
_PULSER_MAX_N = 9999
_PASQAL_MAX_N = 9999
_RQAOA_MAX_N  = 9999

DEFAULT_PLOTS_DIR = "plots"


# ── Per-n runner ───────────────────────────────────────────────────────────────

def _run_n(sites_sorted: list, n: int,
           p: int, steps: int,
           include_pulser: bool,
           include_pasqal: bool,
           include_classical: bool = True,
           include_rqaoa: bool = True) -> dict:
    """Run all enabled solvers for one qubit count. Returns a metrics dict."""

    cfg = AppConfig(
        sites=sites_sorted[:n],
        budget_10k=BUDGET_10K,
        p_layers=p,
        n_steps=steps,
        stepsize=0.01,
        backend="lightning.qubit",
    )
    Q, meta = build_qubo(cfg)

    # ── Reference optimal (brute-force for n≤_BF_MAX_N, MILP above) ──────
    # brute_force is O(2^n): feasible up to ~n=20 (~1M states).
    # For larger n we fall back to MILP (exact, milliseconds at any n).
    if n <= _BF_MAX_N:
        bf      = brute_force(cfg, Q)
        optimal = bf["obj"]
        ref_src = "brute-force"
    else:
        # Run MILP now just for the reference; it will run again below for metrics.
        try:
            _ref_ml = milp_solve(cfg)
            optimal = _ref_ml["obj"] if _ref_ml["success"] else 0.0
        except Exception:
            optimal = 0.0
        ref_src = "MILP"

    # ── PennyLane QAOA ────────────────────────────────────────────────────
    # Statevector requires 2^n complex128 amplitudes; skip above _QAOA_MAX_N.
    qaoa_row = None
    if n <= _QAOA_MAX_N:
        h, J  = qubo_to_ising(Q)
        scale = max(np.abs(h).max(), np.abs(J).max(), 1.0)
        h, J  = h / scale, J / scale

        t0     = time.perf_counter()
        res    = run_qaoa(cfg, h, J, direct_gates=True, seed=0)
        q_ms   = (time.perf_counter() - t0) * 1000
        q_obj  = res["best_obj"] if res["best_obj"] not in (None, -np.inf) else 0.0
        q_gap  = 100 * (optimal - q_obj) / optimal if optimal > 0 else 100.0
        conv   = res["convergence"]

        qaoa_row = {
            "time_ms":   q_ms,
            "coverage":  q_obj,
            "gap_pct":   q_gap,
            "converged": conv["reached"],
            "conv_step": conv["iter"] if conv["reached"] else None,
            "best_prob": res.get("best_prob"),
            "lambda":    meta["lambda"],
            "x":         res.get("best_x"),
        }
    else:
        print(f"  [QAOA skipped: n={n}]", end="", flush=True)
        qaoa_row = {"skipped": True, "time_ms": 0, "coverage": None,
                    "gap_pct": None, "converged": False, "conv_step": None,
                    "best_prob": None, "lambda": meta["lambda"], "x": None}

    # ── Pulser ────────────────────────────────────────────────────────────
    pulser_row = None
    if include_pulser:
        try:
            from .pulser_qaoa import run_pulser_qaoa
            t0  = time.perf_counter()
            t_ns = PULSER_T_NS_LARGE if n >= PULSER_T_LARGE_N else PULSER_T_NS
            pr  = run_pulser_qaoa(cfg, Q,
                                  n_samples=PULSER_SAMPLES,
                                  t_values=t_ns)
            p_ms  = (time.perf_counter() - t0) * 1000
            p_obj = pr["best_obj"] if pr["best_obj"] not in (None, -np.inf) else 0.0
            p_gap = 100 * (optimal - p_obj) / optimal if optimal > 0 else 100.0
            pulser_row = {
                "time_ms":   p_ms,
                "coverage":  p_obj,
                "gap_pct":   p_gap,
                "converged": pr["convergence"]["reached"],
                "conv_step": pr["convergence"]["iter"],
                "best_prob": pr.get("best_prob"),
            }
        except Exception as e:
            pulser_row = {"error": str(e),
                          "time_ms": None, "coverage": None,
                          "gap_pct": None, "converged": False,
                          "conv_step": None, "best_prob": None}

    # ── Pasqal QUBO ───────────────────────────────────────────────────────
    pasqal_row = None
    if include_pasqal:
        try:
            from .pasqal_qubo import run_pasqal_qubo
            pq  = run_pasqal_qubo(cfg, Q)
            pq_obj = pq["best_obj"] if pq["best_obj"] not in (None, -np.inf) else 0.0
            pq_gap = 100 * (optimal - pq_obj) / optimal if optimal > 0 else 100.0
            pasqal_row = {
                "time_ms":   pq["time_ms"],
                "coverage":  pq_obj,
                "gap_pct":   pq_gap,
                "converged": True,
                "conv_step": None,
                "best_prob": None,
            }
        except Exception as e:
            pasqal_row = {"error": str(e),
                          "time_ms": None, "coverage": None,
                          "gap_pct": None, "converged": False,
                          "conv_step": None, "best_prob": None}

    # ── Greedy ────────────────────────────────────────────────────────────
    greedy_row = None
    if include_classical:
        gr     = greedy(cfg)
        gr_obj = gr["obj"]
        gr_gap = 100 * (optimal - gr_obj) / optimal if optimal > 0 else 100.0
        greedy_row = {
            "time_ms":  gr["time_ms"],
            "coverage": gr_obj,
            "gap_pct":  gr_gap,
        }

    # ── MILP ──────────────────────────────────────────────────────────────
    milp_row = None
    if include_classical:
        try:
            ml     = milp_solve(cfg)
            ml_obj = ml["obj"]
            ml_gap = 100 * (optimal - ml_obj) / optimal if optimal > 0 else 100.0
            milp_row = {
                "time_ms":  ml["time_ms"],
                "coverage": ml_obj,
                "gap_pct":  ml_gap,
                "success":  ml["success"],
                "x":        ml["x"],
            }
        except Exception as e:
            milp_row = {"error": str(e),
                        "time_ms": None, "coverage": None,
                        "gap_pct": None, "success": False}

    # ── RQAOA ────────────────────────────────────────────────────────────
    rqaoa_row = None
    if include_rqaoa:
        from .rqaoa import _CLASSICAL_N as _RQAOA_CLASSICAL_N
        if n <= _RQAOA_CLASSICAL_N:
            rqaoa_row = None  # skip: no QAOA steps would run at this size
        else:
            try:
                from .rqaoa import run_rqaoa
                rq     = run_rqaoa(cfg, Q)
                rq_obj = rq["best_obj"] if rq["best_obj"] not in (None, -np.inf) else 0.0
                rq_gap = 100 * (optimal - rq_obj) / optimal if optimal > 0 else 100.0
                rqaoa_row = {
                    "time_ms":   rq["time_ms"],
                    "coverage":  rq_obj,
                    "gap_pct":   rq_gap,
                    "converged": True,
                    "conv_step": None,
                    "x":         rq.get("best_x"),
                }
            except Exception as e:
                rqaoa_row = {"error": str(e),
                             "time_ms": None, "coverage": None,
                             "gap_pct": None, "converged": False,
                             "conv_step": None}

    return {
        "n":       n,
        "optimal": optimal,
        "ref_src": ref_src,
        "cfg":     cfg,
        "qaoa":    qaoa_row,
        "pulser":  pulser_row,
        "pasqal":  pasqal_row,
        "rqaoa":   rqaoa_row,
        "greedy":  greedy_row,
        "milp":    milp_row,
    }


# ── n=24 monitored standalone run ─────────────────────────────────────────────

def run_n24_monitored(sites_sorted: list,
                      p: int = DEFAULT_P,
                      steps: int = DEFAULT_STEPS,
                      budget_10k: int = 120,   # $1.2M — lets solver pick 8-10 sites
                      log_path: str = "n24_monitor.log") -> dict:
    """Monitored standalone run for n=24 with full logging to file + stdout.

    Logs every QAOA step (with ETA), timestamps each solver phase, and writes
    a full traceback if anything crashes.  Results saved to n24_results.json.

    Comment out the call in main() to skip this and run the normal benchmark.
    """
    import logging
    import traceback as _tb
    import sys

    logger = logging.getLogger("n24")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if not logger.handlers:
        logger.addHandler(logging.FileHandler(log_path, mode="w"))
        logger.addHandler(logging.StreamHandler(sys.stdout))
    for h in logger.handlers:
        h.setFormatter(fmt)

    def _log(msg):
        logger.info(msg)

    _log("=" * 60)
    _log(f"n=24 monitored run  |  p={p}  steps={steps}  budget=${BUDGET_10K*10}K")
    _log(f"QAOA max_n={_QAOA_MAX_N}, BF max_n={_BF_MAX_N}")
    _log("=" * 60)

    result = None
    try:
        _log("Phase 1/3: building QUBO …")
        cfg = AppConfig(
            sites=sites_sorted[:24],
            budget_10k=budget_10k,
            p_layers=p,
            n_steps=steps,
            stepsize=0.01,
            backend="lightning.qubit",
        )
        Q, meta = build_qubo(cfg)
        _log(f"  budget=${budget_10k*10}K  λ={meta['lambda']:.3f}  "
             f"Q range [{meta['q_min']:.2f}, {meta['q_max']:.2f}]")

        _log("Phase 2/3: MILP reference (since n=24 > BF cutoff) …")
        t0 = time.perf_counter()
        ml = milp_solve(cfg)
        _log(f"  MILP done in {(time.perf_counter()-t0)*1000:.1f}ms  "
             f"obj={ml['obj']:.4f}  cost=${ml['cost']*10:.0f}K  success={ml['success']}")
        optimal = ml["obj"] if ml["success"] else 0.0

        _log("Phase 3/3: PennyLane QAOA n=24 (INTERP staged warm-start) …")
        _log(f"  Staged: p=1..{p}, {steps} steps/layer  "
             f"(~{p} × 20–60s/step at n=24 → expect {p}–{p*5} hours total)")
        t0 = time.perf_counter()
        h, J  = qubo_to_ising(Q)
        scale = max(np.abs(h).max(), np.abs(J).max(), 1.0)
        h, J  = h / scale, J / scale

        # INTERP staged warm-start: run p=1,2,...,p sequentially.
        # Each layer warm-starts from the previous optimal params → avoids
        # barren plateaus that cause gap% to widen as n×p grows.
        warm = None
        res  = None
        for layer in range(1, p + 1):
            _cfg_l = AppConfig(
                sites=sites_sorted[:24],
                budget_10k=budget_10k,
                p_layers=layer,
                n_steps=steps,
                stepsize=0.01,
                backend="lightning.qubit",
            )
            _log(f"  INTERP layer p={layer}/{p} …")
            res  = run_qaoa(_cfg_l, h, J, direct_gates=True, seed=0,
                            warm_params=warm)
            warm = res["opt_params"]
            _l_obj = res["best_obj"] if res["best_obj"] not in (None, -np.inf) else 0.0
            _l_gap = 100 * (optimal - _l_obj) / optimal if optimal > 0 else 100.0
            _gnorm = res["grad_norm_history"][-1][1] if res["grad_norm_history"] else float("nan")
            _log(f"    p={layer} → coverage={_l_obj:.4f}  gap={_l_gap:.1f}%  "
                 f"||∇||_last={_gnorm:.2e}")

        q_ms  = (time.perf_counter() - t0) * 1000
        q_obj = res["best_obj"] if res["best_obj"] not in (None, -np.inf) else 0.0
        q_gap = 100 * (optimal - q_obj) / optimal if optimal > 0 else 100.0
        conv  = res["convergence"]

        _log(f"  QAOA done in {q_ms/1000:.1f}s")
        _log(f"  coverage={q_obj:.4f}  gap={q_gap:.1f}%  "
             f"converged={conv['reached']}  conv_step={conv['iter']}")
        _log(f"  MILP coverage={optimal:.4f} (reference)")

        result = {
            "n": 24, "optimal": optimal, "ref_src": "MILP",
            "qaoa": {
                "time_ms": q_ms, "coverage": q_obj, "gap_pct": q_gap,
                "converged": conv["reached"], "conv_step": conv["iter"],
                "best_prob": res.get("best_prob"), "lambda": meta["lambda"],
                "x": res.get("best_x"),
                "grad_norm_history": res.get("grad_norm_history", []),
            },
            "milp": {
                "time_ms": ml["time_ms"], "coverage": ml["obj"],
                "gap_pct": 0.0, "success": ml["success"], "x": ml["x"],
            },
            "cfg": cfg,
        }

        import json
        out_path = "n24_results.json"
        with open(out_path, "w") as f:
            json.dump({
                "n": 24, "optimal": optimal, "ref_src": "MILP",
                "qaoa_gap_pct": q_gap, "qaoa_time_s": q_ms / 1000,
                "qaoa_converged": conv["reached"], "qaoa_conv_step": conv["iter"],
                "milp_coverage": optimal,
            }, f, indent=2, default=float)
        _log(f"Results saved → {out_path}")

    except Exception as e:
        _log(f"CRASHED: {e}")
        _log(_tb.format_exc())
        raise

    _log("=" * 60)
    _log("n=24 run complete.")
    return result


# ── Main benchmark ─────────────────────────────────────────────────────────────

def run_scaling(p: int = DEFAULT_P,
                steps: int = DEFAULT_STEPS,
                include_pulser: bool = True,
                include_pasqal: bool = True,
                include_rqaoa: bool = True,
                include_classical: bool = True,
                no_plots: bool = False,
                plots_dir: str = DEFAULT_PLOTS_DIR,
                output: str | None = None) -> dict:

    print("=" * 72)
    print("  Qubit Scaling Benchmark  |  live USGS/FDSN data")
    print("=" * 72)
    print(f"\n  Qubit counts  : {QUBIT_COUNTS}")
    print(f"  Budget        : ${BUDGET_10K * 10}K")
    print(f"  QAOA          : p={p}, {steps} steps, lightning.qubit + direct gates")
    print(f"                  (convergence: cost improvement < 1e-3 over 20 steps)")
    if include_pulser:
        print(f"  Pulser        : {len(PULSER_T_NS)} T-values "
              f"({PULSER_T_NS[0]//1000}–{PULSER_T_NS[-1]//1000} µs), "
              f"{PULSER_SAMPLES} samples/point")
    if include_pasqal:
        print(f"  Pasqal QUBO   : CPLEX exact solver (classical reference)")
    if include_classical:
        print(f"  Classical     : Greedy (value/cost ratio) + MILP (scipy exact)")
    print(f"  Gap%          : (brute-force optimal − solver coverage) / optimal × 100")
    print()

    # ── Fetch live data once ───────────────────────────────────────────────
    print("Fetching live data (USGS catalog + FDSN stations) …")
    cfg_max = fetch_instance(budget_10k=BUDGET_10K, n_sites=max(QUBIT_COUNTS))
    sites_sorted = sorted(cfg_max.sites,
                          key=lambda s: s.risk_weight, reverse=True)
    stations_df = fetch_stations()
    print(f"  {len(sites_sorted)} sites ready  "
          f"(top {max(QUBIT_COUNTS)} by seismic risk)")
    print(f"  {len(stations_df)} existing FDSN stations fetched for map overlay\n")

    # ── n=24 standalone monitored run (uncomment to run instead of benchmark) ─
    # run_n24_monitored(sites_sorted, p=p, steps=steps)
    # return

    # ── Benchmark loop ─────────────────────────────────────────────────────
    results: dict[int, dict] = {}

    for n in QUBIT_COUNTS:
        print(f"  n={n:>2}", end="", flush=True)
        row = _run_n(sites_sorted, n, p, steps, include_pulser, include_pasqal,
                     include_classical, include_rqaoa)
        results[n] = row

        q = row["qaoa"]
        if q and not q.get("skipped"):
            conv_tag = (f"conv@{q['conv_step']}" if q["converged"]
                        else f"NOT CONV ({steps} steps)")
            print(f"  |  QAOA {q['time_ms']/1000:>5.1f}s  "
                  f"gap={q['gap_pct']:>5.1f}%  {conv_tag}", end="", flush=True)
        else:
            print(f"  |  QAOA skipped (n>{_QAOA_MAX_N})", end="", flush=True)

        if row["pulser"] and not row["pulser"].get("skipped") and "error" not in row["pulser"]:
            pu = row["pulser"]
            print(f"  |  Pulser {pu['time_ms']/1000:>5.1f}s  "
                  f"gap={pu['gap_pct']:>5.1f}%", end="", flush=True)
        elif row["pulser"] and row["pulser"].get("skipped"):
            print(f"  |  Pulser skipped (n>{_PULSER_MAX_N})", end="", flush=True)

        if row["pasqal"] and not row["pasqal"].get("skipped") and "error" not in row["pasqal"]:
            pq = row["pasqal"]
            print(f"  |  Pasqal {pq['time_ms']/1000:>5.2f}s  "
                  f"gap={pq['gap_pct']:>5.1f}%", end="", flush=True)
        elif row["pasqal"] and row["pasqal"].get("skipped"):
            print(f"  |  Pasqal skipped (n>{_PASQAL_MAX_N})", end="", flush=True)

        if row.get("rqaoa") and not row["rqaoa"].get("skipped") and "error" not in row["rqaoa"]:
            rq = row["rqaoa"]
            print(f"  |  RQAOA {rq['time_ms']/1000:>5.1f}s  "
                  f"gap={rq['gap_pct']:>5.1f}%", end="", flush=True)
        elif row.get("rqaoa") and row["rqaoa"].get("skipped"):
            print(f"  |  RQAOA skipped (n>{_RQAOA_MAX_N})", end="", flush=True)

        if row["greedy"]:
            gr = row["greedy"]
            print(f"  |  Greedy {gr['time_ms']/1000:>6.4f}s  "
                  f"gap={gr['gap_pct']:>5.1f}%", end="", flush=True)

        if row["milp"] and "error" not in row["milp"]:
            ml = row["milp"]
            print(f"  |  MILP {ml['time_ms']/1000:>6.4f}s  "
                  f"gap={ml['gap_pct']:>5.1f}%", end="", flush=True)

        print()

    _print_table(results, include_pulser, include_pasqal, include_classical, steps)

    if not no_plots:
        _plot_all(results, plots_dir, include_classical, stations_df)

    if output:
        def _json_row(r):
            out = {}
            for k, v in r.items():
                if k == "cfg":
                    continue
                if isinstance(v, dict):
                    out[k] = {sk: (sv.tolist() if hasattr(sv, "tolist") else sv)
                              for sk, sv in v.items()}
                else:
                    out[k] = v
            return out
        payload = {str(n): _json_row(r) for n, r in results.items()}
        payload["_meta"] = {
            "qubit_counts": QUBIT_COUNTS, "budget_10k": BUDGET_10K,
            "qaoa_p": p, "qaoa_steps": steps,
            "pulser_t_ns": PULSER_T_NS, "pulser_samples": PULSER_SAMPLES,
        }
        with open(output, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        print(f"\nResults saved → {output}")

    return results


# ── Summary table ──────────────────────────────────────────────────────────────

def _print_table(results: dict, include_pulser: bool,
                 include_pasqal: bool, include_classical: bool,
                 steps: int) -> None:
    ns = list(results.keys())

    # ── Build solver list in display order ────────────────────────────────────
    solvers: list[tuple[str, str]] = [("qaoa", "PennyLane QAOA")]
    if include_pulser:
        solvers.append(("pulser", "Pulser (analog)"))
    if include_pasqal:
        solvers.append(("pasqal", "Pasqal QUBO"))
    solvers.append(("rqaoa", "RQAOA"))
    if include_classical:
        solvers.append(("greedy", "Greedy"))
        solvers.append(("milp",   "MILP (exact)"))

    sep  = "─" * (10 + 10 * len(solvers))
    hdr  = f"  {'n':>4}  {'Optimal':>8}"
    for _, label in solvers:
        hdr += f"  {label[:9]:>9}"

    # ── Table 1: Optimality Gap % ─────────────────────────────────────────────
    print(f"\n  SOLUTION QUALITY  (gap % vs brute-force optimal)\n{sep}")
    print(hdr)
    print(sep)
    for n in ns:
        r    = results[n]
        line = f"  {n:>4}  {r['optimal']:>8.4f}"
        for key, _ in solvers:
            row = r.get(key)
            if row and "error" not in row and row.get("gap_pct") is not None:
                g = row["gap_pct"]
                line += f"  {g:>8.1f}%"
            else:
                line += f"  {'   n/a':>9}"
        print(line)
    print(sep)

    # ── Table 2: Wall-clock Runtime ───────────────────────────────────────────
    print(f"\n  RUNTIME\n{sep}")
    print(hdr)
    print(sep)
    for n in ns:
        r    = results[n]
        line = f"  {n:>4}  {'':>8}"
        for key, _ in solvers:
            row = r.get(key)
            if row and "error" not in row and row.get("time_ms") is not None:
                ms = row["time_ms"]
                if ms >= 1000:
                    line += f"  {ms/1000:>7.2f}s "
                else:
                    line += f"  {ms:>6.1f}ms "
            else:
                line += f"  {'   n/a':>9}"
        print(line)
    print(sep)

    # ── Table 3: QAOA convergence detail ─────────────────────────────────────
    print(f"\n  QAOA CONVERGENCE DETAIL\n{'─'*52}")
    print(f"  {'n':>4}  {'Conv?':>6}  {'@step':>6}  {'P(opt)':>7}  {'Gap%':>6}")
    print("─" * 52)
    no_conv = []
    for n in ns:
        q = results[n]["qaoa"]
        if q and not q.get("skipped"):
            conv = "yes" if q["converged"] else " NO "
            step = str(q["conv_step"]) if q["conv_step"] is not None else "  n/a"
            prob = f"{q['best_prob']:.4f}" if q["best_prob"] is not None else "   n/a"
            gap  = f"{q['gap_pct']:>5.1f}%" if q["gap_pct"] is not None else "  n/a"
            if not q["converged"]:
                no_conv.append(n)
            print(f"  {n:>4}  {conv:>6}  {step:>6}  {prob:>7}  {gap:>6}")
        else:
            print(f"  {n:>4}  {'skipped':>6}  {'n/a':>6}  {'n/a':>7}  {'n/a':>6}")
    print("─" * 52)
    if no_conv:
        print(f"  WARNING: QAOA did not converge at n={no_conv} "
              f"within {steps} steps.")

    # ── QAOA runtime scaling factor ───────────────────────────────────────────
    q_ms = [results[n]["qaoa"]["time_ms"] for n in ns]
    print(f"\n  QAOA runtime ×-factor per +2 qubits (expect ~4×):")
    for i in range(1, len(ns)):
        ratio = q_ms[i] / q_ms[i-1] if q_ms[i-1] > 0 else float("nan")
        print(f"    n={ns[i-1]:>2} → {ns[i]:>2}: {ratio:.1f}×")


# ── Plotting ───────────────────────────────────────────────────────────────────

_SOLVER_STYLE = {
    "PennyLane QAOA":             {"color": "#2166ac", "marker": "o"},
    "Pulser (neutral-atom emul.)":{"color": "#d6604d", "marker": "s"},
    "Pasqal QUBO (CPLEX)":        {"color": "#4dac26", "marker": "^"},
    "RQAOA":                      {"color": "#762a83", "marker": "v"},
    "Greedy":                     {"color": "#f4a582", "marker": "D"},
    "MILP (exact)":               {"color": "#1a9641", "marker": "P"},
}


def _fmt_time(x: float, _) -> str:
    return f"{x:.0f}s" if x >= 1 else f"{x*1000:.0f}ms"


def _fit_exp(ns, vals):
    try:
        good = [(n, v) for n, v in zip(ns, vals)
                if v is not None and v > 0]
        if len(good) < 2:
            return None
        gns, gvs = zip(*good)
        a, b = np.polyfit(gns, np.log(gvs), 1)
        return a, b
    except Exception:
        return None


def _save(fig, plots_dir: str, name: str) -> None:
    path = os.path.join(plots_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {path}")


def _annotate_pts(ax, xvals, yvals, fmt_fn, color):
    for x, y in zip(xvals, yvals):
        if y is not None:
            ax.annotate(fmt_fn(y), xy=(x, y), xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", fontsize=8, color=color)


def _plot_all(results: dict, plots_dir: str,
              include_classical: bool = True,
              existing_stations=None) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  (matplotlib not available — skipping plots)")
        return

    ns = list(results.keys())

    # collect series
    series = {}

    q_t = [results[n]["qaoa"]["time_ms"] / 1000 for n in ns]
    q_g = [results[n]["qaoa"]["gap_pct"]         for n in ns]
    q_c = [results[n]["qaoa"]["conv_step"]
           if results[n]["qaoa"]["converged"] else None for n in ns]
    series["PennyLane QAOA"] = {"ns": ns, "time": q_t, "gap": q_g}

    pu_ns = [n for n in ns
             if results[n]["pulser"] and
             "error" not in results[n]["pulser"] and
             results[n]["pulser"]["time_ms"] is not None]
    if pu_ns:
        series["Pulser (neutral-atom emul.)"] = {
            "ns":   pu_ns,
            "time": [results[n]["pulser"]["time_ms"] / 1000 for n in pu_ns],
            "gap":  [results[n]["pulser"]["gap_pct"]         for n in pu_ns],
        }

    pq_ns = [n for n in ns
             if results[n]["pasqal"] and
             "error" not in results[n]["pasqal"] and
             results[n]["pasqal"]["time_ms"] is not None]
    if pq_ns:
        series["Pasqal QUBO (CPLEX)"] = {
            "ns":   pq_ns,
            "time": [results[n]["pasqal"]["time_ms"] / 1000 for n in pq_ns],
            "gap":  [results[n]["pasqal"]["gap_pct"]         for n in pq_ns],
        }

    rq_ns = [n for n in ns
             if results[n].get("rqaoa") and
             "error" not in results[n]["rqaoa"] and
             results[n]["rqaoa"].get("time_ms") is not None]
    if rq_ns:
        series["RQAOA"] = {
            "ns":   rq_ns,
            "time": [results[n]["rqaoa"]["time_ms"] / 1000 for n in rq_ns],
            "gap":  [results[n]["rqaoa"]["gap_pct"]         for n in rq_ns],
        }

    if include_classical:
        gr_ns = [n for n in ns if results[n].get("greedy")]
        if gr_ns:
            series["Greedy"] = {
                "ns":   gr_ns,
                "time": [results[n]["greedy"]["time_ms"] / 1000 for n in gr_ns],
                "gap":  [results[n]["greedy"]["gap_pct"]         for n in gr_ns],
            }
        ml_ns = [n for n in ns
                 if results[n].get("milp") and
                 "error" not in results[n]["milp"] and
                 results[n]["milp"]["time_ms"] is not None]
        if ml_ns:
            series["MILP (exact)"] = {
                "ns":   ml_ns,
                "time": [results[n]["milp"]["time_ms"] / 1000 for n in ml_ns],
                "gap":  [results[n]["milp"]["gap_pct"]          for n in ml_ns],
            }

    def _draw_solver(ax, sname, sdata, logy=True):
        st = _SOLVER_STYLE[sname]
        ykey = "time" if logy else "gap"
        ax.plot(sdata["ns"], sdata[ykey],
                st["marker"] + "-", color=st["color"],
                lw=2, ms=7, label=sname)
        if logy:
            fit = _fit_exp(sdata["ns"], sdata["time"])
            if fit:
                a, b = fit
                xf = np.linspace(sdata["ns"][0], sdata["ns"][-1], 100)
                ax.plot(xf, np.exp(b + a * xf), "--",
                        color=st["color"], lw=1, alpha=0.4,
                        label=f"{sname.split()[0]} fit: ×{np.exp(2*a):.1f}/+2q")

    # ── 1. Combined runtime (LINEAR) ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for sname, sdata in series.items():
        st = _SOLVER_STYLE[sname]
        ax.plot(
            sdata["ns"], sdata["time"],
            st["marker"] + "-",
            color=st["color"],
            lw=2, ms=7, label=sname
        )
        _annotate_pts(
            ax,
            sdata["ns"],
            sdata["time"],
            lambda t: f"{t:.1f}s" if t >= 1 else f"{t*1000:.0f}ms",
            st["color"]
        )

    ax.set_xticks(ns)
    ax.set_xlabel("Number of qubits / atoms", fontsize=12)
    ax.set_ylabel("Wall-clock runtime (s)", fontsize=12)
    ax.set_title(
        "Runtime Scaling: All Solvers vs Qubit Count\n"
        "(live USGS/FDSN data, Manzanillo seismic network)",
        fontsize=11
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_time))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, plots_dir, "runtime/combined.png")
    plt.close(fig)

    # ── 2–4. Individual runtime plots (LINEAR) ───────────────────────────
    for sname, sdata in series.items():
        st = _SOLVER_STYLE[sname]
        slug = sname.lower().split()[0]   # pennylane / pulser / pasqal / greedy / milp

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            sdata["ns"], sdata["time"],
            st["marker"] + "-",
            color=st["color"],
            lw=2.5, ms=8, label=sname
        )
        _annotate_pts(
            ax,
            sdata["ns"],
            sdata["time"],
            lambda t: f"{t:.1f}s" if t >= 1 else f"{t*1000:.0f}ms",
            st["color"]
        )

        ax.set_xticks(sdata["ns"])
        ax.set_xlabel("Number of qubits / atoms", fontsize=12)
        ax.set_ylabel("Wall-clock runtime (s)", fontsize=12)
        ax.set_title(f"{sname} — Runtime vs Qubit Count\n(live data)", fontsize=11)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_time))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, plots_dir, f"runtime/{slug}.png")
        plt.close(fig)

    # ── 5. Combined quality (gap%) ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for sname, sdata in series.items():
        st = _SOLVER_STYLE[sname]
        ax.plot(sdata["ns"], sdata["gap"],
                st["marker"] + "-", color=st["color"],
                lw=2, ms=7, label=sname)
        _annotate_pts(ax, sdata["ns"], sdata["gap"],
                      lambda g: f"{g:.1f}%", st["color"])
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5,
               label="Optimal (0% gap)")
    ax.set_xlabel("Number of qubits / atoms", fontsize=12)
    ax.set_ylabel("Optimality gap  (% below brute-force)", fontsize=12)
    ax.set_title("Solution Quality vs Qubit Count\n"
                 "(gap% vs brute-force optimal, live data)", fontsize=11)
    ax.set_xticks(ns)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, plots_dir, "quality/combined.png")
    plt.close(fig)

    # ── 6. PennyLane quality + convergence ────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax1.plot(ns, q_g, "o-", color="#2166ac", lw=2, ms=7)
    _annotate_pts(ax1, ns, q_g, lambda g: f"{g:.1f}%", "#2166ac")
    ax1.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax1.set_ylabel("Optimality gap (%)", fontsize=11)
    ax1.set_title("PennyLane QAOA — Quality & Convergence vs Qubit Count\n"
                  "(live data)", fontsize=11)
    ax1.grid(True, alpha=0.3)

    conv_vals = [c if c is not None else np.nan for c in q_c]
    no_conv_n = [n for n, c in zip(ns, q_c) if c is None]
    ax2.plot(ns, conv_vals, "^-", color="#4dac26", lw=2, ms=7,
             label="Convergence step")
    for nc in no_conv_n:
        ax2.axvline(nc, color="red", lw=1.5, ls=":", alpha=0.8)
    if no_conv_n:
        ax2.plot([], [], ":", color="red", lw=1.5,
                 label=f"Not converged (need more steps)")
    ax2.set_xlabel("Number of qubits", fontsize=11)
    ax2.set_ylabel("Step at convergence", fontsize=11)
    ax2.set_xticks(ns)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, plots_dir, "quality/pennylane.png")
    plt.close(fig)

    # ── 7. Pulser quality ─────────────────────────────────────────────────
    if pu_ns:
        fig, ax = plt.subplots(figsize=(6, 4))
        pu_g = [results[n]["pulser"]["gap_pct"] for n in pu_ns]
        st   = _SOLVER_STYLE["Pulser (neutral-atom emul.)"]
        ax.plot(pu_ns, pu_g, st["marker"] + "-",
                color=st["color"], lw=2.5, ms=8,
                label="Pulser (neutral-atom emulator)")
        _annotate_pts(ax, pu_ns, pu_g, lambda g: f"{g:.1f}%", st["color"])
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5,
                   label="Optimal (0% gap)")
        ax.set_xlabel("Number of atoms", fontsize=12)
        ax.set_ylabel("Optimality gap  (% below brute-force)", fontsize=12)
        ax.set_title("Pulser Analog QAOA — Solution Quality vs Atom Count\n"
                     f"(live data, {len(PULSER_T_NS)} T-values)", fontsize=11)
        ax.set_xticks(pu_ns)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, plots_dir, "quality/pulser.png")
        plt.close(fig)

    # ── 8. Sensor maps ────────────────────────────────────────────────────
    # Combined map (all available solvers) + individual per-solver maps
    _SOLVER_MAP = [
        ("milp",   "MILP (optimal)"),
        ("qaoa",   "PennyLane QAOA"),
        ("pulser", "Pulser"),
        ("pasqal", "Pasqal"),
        ("rqaoa",  "RQAOA"),
        ("greedy", "Greedy"),
    ]
    for n in ns:
        row = results[n]
        cfg = row.get("cfg")
        if cfg is None:
            continue

        solver_selections = {}
        for key, label in _SOLVER_MAP:
            r = row.get(key)
            if r and r.get("x") is not None:
                solver_selections[label] = {"x": r["x"]}

        if not solver_selections:
            continue

        maps_dir = os.path.join(plots_dir, "sensor_maps")
        os.makedirs(maps_dir, exist_ok=True)

        # Combined map — all solvers side by side
        plot_sensor_map(
            cfg=cfg,
            results=solver_selections,
            existing_stations=existing_stations,
            title_suffix=f" — n={n} candidates, budget=${BUDGET_10K*10}K",
            save_path=os.path.join(maps_dir, f"n{n:02d}_combined.png"),
        )
        _save_msg = f"sensor_maps/n{n:02d}_combined.png"
        print(f"Plot saved → {_save_msg}")

        # Individual per-solver maps in subfolders
        for label, sel in solver_selections.items():
            slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            sub_dir = os.path.join(maps_dir, slug)
            os.makedirs(sub_dir, exist_ok=True)
            plot_sensor_map(
                cfg=cfg,
                results={label: sel},
                existing_stations=existing_stations,
                title_suffix=f" — n={n} candidates, budget=${BUDGET_10K*10}K",
                save_path=os.path.join(sub_dir, f"n{n:02d}.png"),
            )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="manzanillo-qc-scaling",
        description="Qubit scaling benchmark on live data (n=4–14)",
    )
    parser.add_argument("--p-layers", type=int, default=DEFAULT_P, metavar="N",
                        help=f"QAOA circuit depth (default {DEFAULT_P})")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, metavar="N",
                        help=f"QAOA training steps (default {DEFAULT_STEPS})")
    parser.add_argument("--no-pulser", action="store_true",
                        help="Skip Pulser")
    parser.add_argument("--no-pasqal", action="store_true",
                        help="Skip Pasqal QUBO")
    parser.add_argument("--no-rqaoa", action="store_true",
                        help="Skip RQAOA")
    parser.add_argument("--no-classical", action="store_true",
                        help="Skip classical solvers (Greedy, MILP)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip matplotlib plots")
    parser.add_argument("--plots-dir", default=DEFAULT_PLOTS_DIR, metavar="DIR",
                        help=f"Directory for saved plots (default: {DEFAULT_PLOTS_DIR})")
    parser.add_argument("--output", "-o", metavar="JSON",
                        help="Save full results to JSON file")
    args = parser.parse_args()

    run_scaling(
        p=args.p_layers,
        steps=args.steps,
        include_pulser=not args.no_pulser,
        include_pasqal=not args.no_pasqal,
        include_rqaoa=not args.no_rqaoa,
        include_classical=not args.no_classical,
        no_plots=args.no_plots,
        plots_dir=args.plots_dir,
        output=args.output,
    )


if __name__ == "__main__":
    main()
