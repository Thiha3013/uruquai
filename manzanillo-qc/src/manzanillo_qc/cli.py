"""Command-line entry-point for manzanillo-qc.

All runs use live USGS/FDSN data fetched at runtime.

Usage
-----
    # Basic run (live data, $350K budget):
    PYTHONPATH=src python3 -m manzanillo_qc.cli --budget 35

    # With Pasqal QUBO + Pulser analog QAOA:
    PYTHONPATH=src python3 -m manzanillo_qc.cli --budget 35 --pasqal --pulser

    # Tune QAOA depth and steps:
    PYTHONPATH=src python3 -m manzanillo_qc.cli --budget 35 --p-layers 5 --steps 400

    # 2×2 backend comparison:
    PYTHONPATH=src python3 -m manzanillo_qc.cli --budget 35 --compare

    # Save plots and JSON results:
    PYTHONPATH=src python3 -m manzanillo_qc.cli --budget 35 --output results.json --save-plots plots/

    # Runtime-vs-qubits scaling benchmark (separate command):
    PYTHONPATH=src python3 -m manzanillo_qc.scaling
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .config import AppConfig
from .qubo import build_qubo, brute_force
from .ising import qubo_to_ising
from .qaoa import run_qaoa
from .qaoa_jax import run_qaoa_jax
from .benchmarks import (greedy, milp_solve, print_benchmark_table,
                         compute_auc_metrics, print_auc_table,
                         plot_convergence, plot_probs, plot_solver_comparison,
                         plot_sensor_map,
                         plot_runtime_bar, plot_timing_breakdown,
                         plot_seed_variance)


def _run(args: argparse.Namespace) -> None:
    # ── 1. Fetch live data ──────────────────────────────────────────────────────
    existing_stations = None
    from .instance import fetch_instance
    t0 = time.perf_counter()
    cfg = fetch_instance(budget_10k=args.budget)
    t_fetch = (time.perf_counter() - t0) * 1000

    # Always try to fetch existing FDSN stations for the map (silently skip if offline)
    try:
        from .instance import fetch_stations
        existing_stations = fetch_stations()
    except Exception:
        existing_stations = None

    # ── 2. Optional: replace risk_weight with scenario-based utility ───────────
    if args.utility:
        from .utility import build_utility_weights, DEFAULT_SCENARIOS, print_scenario_report
        print("\n── Hazard Scenario Utility ─────────────────────────────────────────")
        cfg = build_utility_weights(cfg, DEFAULT_SCENARIOS)
        print_scenario_report(cfg, DEFAULT_SCENARIOS)
        print()

    if args.backend:
        cfg = cfg.model_copy(update={"backend": args.backend})
    if args.optimizer:
        cfg = cfg.model_copy(update={"optimizer": args.optimizer})
    if args.steps:
        cfg = cfg.model_copy(update={"n_steps": args.steps})
    if args.p_layers:
        cfg = cfg.model_copy(update={"p_layers": args.p_layers})
    if args.lr:
        cfg = cfg.model_copy(update={"stepsize": args.lr})

    n_seeds = args.seeds

    print(f"\nLoaded {cfg.n_sites} site candidates  |  budget = ${cfg.budget_10k * 10}K")
    mode = f"lightning.qubit + direct gates" if not args.compare and not args.jax else cfg.backend
    print(f"Backend: {mode}  |  p = {cfg.p_layers}  |  steps = {cfg.n_steps}"
          + (f"  |  seeds = {n_seeds}" if n_seeds > 1 else ""))

    # ── 3. Build QUBO ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    Q, meta = build_qubo(cfg, overlap=args.overlap)
    t_qubo = (time.perf_counter() - t0) * 1000
    overlap_str = " + overlap penalty" if args.overlap else ""
    print(f"\nQUBO{overlap_str}: {meta['n_qubits']} qubits  |  "
          f"λ={meta['lambda']:.3f}  |  Q ∈ [{meta['q_min']:.2f}, {meta['q_max']:.2f}]")

    # ── 3b. Ising conversion ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    h, J = qubo_to_ising(Q)
    # Normalise h,J so QAOA rotation angles stay well-conditioned.
    # Optimal solution is unchanged by scaling; without this, large QUBO
    # values (e.g. live data with λ~8) cause RZ(2γh) to over-rotate badly.
    _ising_scale = max(np.abs(h).max(), np.abs(J).max(), 1.0)
    h = h / _ising_scale
    J = J / _ising_scale
    t_ising = (time.perf_counter() - t0) * 1000

    # ── 4. Classical benchmarks ────────────────────────────────────────────────
    print("\n── Classical Benchmarks ────────────────────────────────────────────")

    t0 = time.perf_counter()
    if cfg.n_sites <= 20:
        bf = brute_force(cfg, Q)
        bf["time_ms"] = (time.perf_counter() - t0) * 1000
    else:
        bf = {"obj": None, "x": None, "cost": None,
              "time_ms": 0, "skipped": True,
              "reason": f"n={cfg.n_sites} > 20: brute-force infeasible"}

    g = greedy(cfg)
    m = milp_solve(cfg)

    classical = {
        "Brute-force": bf,
        "Greedy":      g,
        "MILP":        m,
    }

    # ── 4b. Pasqal QUBO solver (optional) ──────────────────────────────────────
    if args.pasqal:
        from .pasqal_qubo import run_pasqal_qubo
        print("  Running Pasqal QUBO solver ...")
        pq = run_pasqal_qubo(cfg, Q)
        classical["Pasqal QUBO"] = {
            "x":         pq["best_x"],
            "obj":       pq["best_obj"],
            "cost":      pq["best_cost"],
            "time_ms":   pq["time_ms"],
            "best_prob": None,
        }

    # ── 4c. Simulated Annealing (optional) ─────────────────────────────────────
    if args.anneal:
        from .anneal import solve_sa, best_feasible as sa_best_feasible
        print(f"  Running SA ({args.reads} reads × {args.sweeps} sweeps) ...")
        offset = cfg.effective_lambda * float(cfg.budget_10k) ** 2
        sampleset, t_sa = solve_sa(
            Q, offset=offset,
            num_reads=args.reads, sweeps=args.sweeps, seed=args.seed,
        )
        sa_sol = sa_best_feasible(sampleset, cfg)
        sa_sol["time_ms"] = t_sa
        classical["Anneal-SA"] = sa_sol

    # ── 5. QAOA ────────────────────────────────────────────────────────────────
    print("\n── QAOA ────────────────────────────────────────────────────────────")

    qaoa_results = {}
    qaoa_raw     = {}
    timing_rows  = []   # list of dicts (richer than the old list-of-tuples)

    if args.compare:
        # 2×2 grid: backend × gate-style  (only runs with --compare flag)
        variants = [
            ("default.qubit",   False, "default  + templates",   False),
            ("default.qubit",   True,  "default  + direct gates",False),
            ("lightning.qubit", False, "lightning + templates",   False),
            ("lightning.qubit", True,  "lightning + direct gates",False),
            ("jax",             True,  "JAX + JIT + backprop",   True),
        ]
    elif args.jax:
        variants = [("jax", True, "JAX + JIT + backprop", True)]
    else:
        # Default: best single variant — lightning.qubit + direct gates + adjoint
        variants = [("lightning.qubit", True, f"QAOA p={cfg.p_layers}", False)]

    for backend_str, direct, label, use_jax in variants:
        cfg_v = cfg.model_copy(update={"backend": backend_str})
        seed_results = []

        for s in range(n_seeds):
            seed_tag = f"  seed={s}" if n_seeds > 1 else ""
            print(f"\n  [{label}]{seed_tag}  p={cfg_v.p_layers}, {cfg_v.n_steps} steps ...")
            t0 = time.perf_counter()
            if use_jax:
                result = run_qaoa_jax(cfg_v, h, J, seed=s)
            else:
                result = run_qaoa(cfg_v, h, J, direct_gates=direct, seed=s)
            result["time_ms"] = (time.perf_counter() - t0) * 1000
            seed_results.append(result)

        objs  = [r["best_obj"] for r in seed_results]
        times = [r["time_ms"]  for r in seed_results]

        # Best seed drives the benchmark table and plots
        best_result = max(seed_results, key=lambda r: r["best_obj"])

        qaoa_results[label] = {
            "x":         best_result["best_x"],
            "obj":       best_result["best_obj"],
            "cost":      best_result["best_cost"],
            "time_ms":   float(np.mean(times)),
            "best_prob": best_result["best_prob"],
        }
        qaoa_raw[label] = best_result

        timing_rows.append({
            "label":        label,
            "mean_ms":      float(np.mean(times)),
            "std_ms":       float(np.std(times)),
            "best_ms":      float(min(times)),
            "mean_obj":     float(np.mean(objs)),
            "std_obj":      float(np.std(objs)),
            "best_obj":     float(max(objs)),
            "breakdown":    best_result.get("timing", {}),
            "seed_results": seed_results,
        })

    # ── 5b. Pulser analog QAOA (optional) ──────────────────────────────────────
    if args.pulser:
        from .pulser_qaoa import run_pulser_qaoa
        print("\n── Pulser Analog QAOA ──────────────────────────────────────────────")
        print("  Running atom layout optimisation + T-sweep ...")
        pr = run_pulser_qaoa(cfg, Q)
        label_p = f"Pulser analog"
        qaoa_results[label_p] = {
            "x":         pr["best_x"],
            "obj":       pr["best_obj"],
            "cost":      pr["best_cost"],
            "time_ms":   pr["time_ms"],
            "best_prob": pr["best_prob"],
        }
        qaoa_raw[label_p] = pr
        timing_rows.append({
            "label":        label_p,
            "mean_ms":      pr["time_ms"],
            "std_ms":       0.0,
            "best_ms":      pr["time_ms"],
            "mean_obj":     pr["best_obj"] if pr["best_obj"] != -np.inf else 0.0,
            "std_obj":      0.0,
            "best_obj":     pr["best_obj"] if pr["best_obj"] != -np.inf else 0.0,
            "breakdown":    pr.get("timing", {}),
            "seed_results": [pr],
        })
        sweep = pr.get("_sweep", [])
        if sweep:
            print(f"\n  T-sweep results ({len(sweep)} points):")
            print(f"  {'T(ns)':>7}  {'Sim(ms)':>8}  {'Best cov':>9}  "
                  f"{'Gap%':>6}  {'Feasible%':>10}  Sensors")
            for r in sweep:
                marker = " <--" if r["T_ns"] == pr["convergence"]["iter"] else ""
                print(f"  {r['T_ns']:>7}  {r['t_ms']:>8.0f}  {r['best_obj']:>9.4f}  "
                      f"{r['gap_pct']:>5.1f}%  {r['feas_frac']*100:>9.1f}%  "
                      f"{r['sensors']}{marker}")

    # ── 6. Benchmark table ─────────────────────────────────────────────────────
    all_results = {**classical, **qaoa_results}

    if args.compare:
        print("\n  Note: same instance / same objective function / same budget.")
        print("  Solution quality is apples-to-apples across all variants.")
        print("  Runtime is NOT fully apples-to-apples: QAOA uses iterative")
        print("  training while classical solvers run once.")

    print_benchmark_table(all_results, cfg)

    # ── 7. Multi-seed statistics ───────────────────────────────────────────────
    if n_seeds > 1:
        print(f"\n── Multi-seed Statistics  (seeds 0 … {n_seeds - 1}) "
              "────────────────────────────────")
        for row in timing_rows:
            sr = row["seed_results"]
            conv_reached = [r["convergence"]["iter"] for r in sr
                            if r["convergence"]["reached"]]
            if conv_reached:
                conv_str = (f"{len(conv_reached)}/{n_seeds} converged, "
                            f"mean iter = {np.mean(conv_reached):.0f}")
            else:
                conv_str = f"not reached within {cfg.n_steps} steps (all {n_seeds} seeds)"
            print(f"\n  {row['label']}")
            print(f"    objective   :  mean={row['mean_obj']:.4f}  "
                  f"std={row['std_obj']:.4f}  best={row['best_obj']:.4f}")
            print(f"    runtime     :  mean={row['mean_ms']:.1f}ms  "
                  f"std={row['std_ms']:.1f}ms  best={row['best_ms']:.1f}ms")
            print(f"    convergence :  {conv_str}")

    # ── 8. Timing breakdown ────────────────────────────────────────────────────
    print("\n── Timing Breakdown ────────────────────────────────────────────────")
    if t_fetch:
        print(f"  Instance fetch (USGS + FDSN) : {t_fetch:>8.1f} ms")
    print(f"  QUBO build                   : {t_qubo:>8.1f} ms")
    print(f"  Ising conversion             : {t_ising:>8.1f} ms")

    if args.compare:
        baseline_ms = timing_rows[0]["mean_ms"]
        std_col = n_seeds > 1

        # Runtime comparison
        hdr = f"  {'Variant':<30}  {'Mean(ms)':>10}"
        if std_col:
            hdr += f"  {'Std(ms)':>8}"
        hdr += f"  {'Speedup':>8}"
        print(f"\n{hdr}")
        print(f"  {'─'*30}  {'─'*10}" + ("  " + "─"*8 if std_col else "") + "  " + "─"*8)
        for row in timing_rows:
            speedup = baseline_ms / row["mean_ms"] if row["mean_ms"] > 0 else 0
            line = f"  {row['label']:<30}  {row['mean_ms']:>10.1f}"
            if std_col:
                line += f"  {row['std_ms']:>8.1f}"
            line += f"  {speedup:>7.1f}×"
            print(line)

        # Per-variant phase breakdown
        print(f"\n  {'Variant':<30}  {'Init(ms)':>9}  {'OptLoop(ms)':>11}  "
              f"{'Decode(ms)':>10}  {'Step avg(ms)':>13}")
        print(f"  {'─'*30}  {'─'*9}  {'─'*11}  {'─'*10}  {'─'*13}")
        for row in timing_rows:
            bd = row.get("breakdown", {})
            print(f"  {row['label']:<30}  "
                  f"{bd.get('circuit_init_ms', 0):>9.1f}  "
                  f"{bd.get('opt_loop_ms', 0):>11.1f}  "
                  f"{bd.get('decode_ms', 0):>10.1f}  "
                  f"{bd.get('step_avg_ms', 0):>13.2f}")
    else:
        for row in timing_rows:
            bd = row.get("breakdown", {})
            conv = row["seed_results"][0]["convergence"]
            if conv["reached"]:
                conv_str = f"step {conv['iter']}  ({conv['time_ms']:.0f} ms)"
            else:
                conv_str = f"not reached in {conv['n_steps']} steps"
            print(f"  {row['label']:<30}: {row['mean_ms']:>8.1f} ms")
            print(f"    init={bd.get('circuit_init_ms',0):.1f}ms  "
                  f"opt={bd.get('opt_loop_ms',0):.1f}ms  "
                  f"decode={bd.get('decode_ms',0):.1f}ms  "
                  f"step_avg={bd.get('step_avg_ms',0):.2f}ms")
            print(f"    convergence: {conv_str}")

    # ── 9. Plots ───────────────────────────────────────────────────────────────
    save_dir = args.save_plots

    def _sp(name: str) -> str | None:
        if save_dir is None:
            return None
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, name)

    if not args.no_plots:
        suffix = " (utility weights)" if args.utility else ""
        suffix += " + overlap" if args.overlap else ""
        plot_convergence(qaoa_raw, title_suffix=suffix,
                         save_path=_sp("convergence.png"))
        last_result = list(qaoa_raw.values())[-1]
        plot_probs(last_result, cfg, title_suffix=suffix,
                   save_path=_sp("probs.png"))
        plot_solver_comparison(all_results, title_suffix=suffix,
                               save_path=_sp("solver_comparison.png"))
        plot_sensor_map(cfg, all_results, existing_stations=existing_stations,
                        title_suffix=suffix, save_path=_sp("sensor_map.png"))
        plot_runtime_bar(timing_rows, title_suffix=suffix,
                         save_path=_sp("runtime_bar.png"))
        plot_timing_breakdown(timing_rows, classical=classical,
                              title_suffix=suffix,
                              save_path=_sp("timing_breakdown.png"))
        if n_seeds > 1:
            plot_seed_variance(timing_rows, title_suffix=suffix,
                               save_path=_sp("seed_variance.png"))

        if save_dir:
            print(f"\nPlots saved → {save_dir}/")

    # ── 10. Optional JSON output ───────────────────────────────────────────────
    if args.output:
        def _ser(r: dict) -> dict:
            out = {k: v for k, v in r.items() if k != "x"}
            if r.get("x") is not None:
                out["sensors"] = list(np.where(r["x"])[0].tolist())
            return out

        # Build per-QAOA-variant seed data for JSON
        qaoa_seed_data: dict = {}
        for row in timing_rows:
            per_seed = []
            for sr in row["seed_results"]:
                sensors = ([int(i) for i in np.where(sr["best_x"])[0]]
                           if sr.get("best_x") is not None else [])
                per_seed.append({
                    "best_obj":     sr["best_obj"],
                    "best_cost":    sr["best_cost"],
                    "time_ms":      sr["time_ms"],
                    "sensors":      sensors,
                    "timing":       sr.get("timing", {}),
                    "convergence":  sr.get("convergence", {}),
                })
            qaoa_seed_data[row["label"]] = {
                "seeds":    per_seed,
                "mean_obj": row["mean_obj"],
                "std_obj":  row["std_obj"],
                "best_obj": row["best_obj"],
                "mean_ms":  row["mean_ms"],
                "std_ms":   row["std_ms"],
                "best_ms":  row["best_ms"],
            }

        payload = {name: _ser(r) for name, r in all_results.items()}
        payload["_qaoa_seeds"] = qaoa_seed_data
        payload["_meta"] = {
            "n_sites":    cfg.n_sites,
            "budget_10k": cfg.budget_10k,
            "overlap":    args.overlap,
            "utility":    args.utility,
            "n_seeds":    n_seeds,
            "t_qubo_ms":  t_qubo,
            "t_ising_ms": t_ising,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        print(f"\nResults saved → {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="manzanillo-qc",
        description="Quantum sensor placement for the Manzanillo seismic network",
    )
    parser.add_argument(
        "--budget", "-b", type=int, default=25, metavar="UNITS",
        help="Budget in $10K units (default 25 = $250K)",
    )
    parser.add_argument(
        "--overlap", action="store_true",
        help="Add geographic overlap/redundancy penalty to QUBO",
    )
    parser.add_argument(
        "--utility", action="store_true",
        help="Replace seismicity density weights with hazard-scenario utility scores",
    )
    parser.add_argument(
        "--output", "-o", metavar="JSON",
        help="Write full results to a JSON file",
    )
    parser.add_argument(
        "--backend", metavar="DEVICE",
        help="PennyLane backend to use, e.g. default.qubit or lightning.qubit (overrides config)",
    )
    parser.add_argument(
        "--optimizer", choices=["adam", "neldermead"],
        help="Optimizer for QAOA training: adam (default) or neldermead (gradient-free)",
    )
    parser.add_argument(
        "--steps", type=int, default=None, metavar="N",
        help="Override QAOA training steps (default: from config)",
    )
    parser.add_argument(
        "--p-layers", type=int, default=None, metavar="N",
        help="Override QAOA circuit depth p (default: from config)",
    )
    parser.add_argument(
        "--lr", type=float, default=None, metavar="F",
        help="Override Adam learning rate (default: from config)",
    )
    parser.add_argument(
        "--seeds", type=int, default=1, metavar="N",
        help="Number of independent random seeds per QAOA variant (default 1)",
    )
    parser.add_argument(
        "--pasqal", action="store_true",
        help="Also run Pasqal QUBO solver (qubosolver, classical SA+tabu backend)",
    )
    parser.add_argument(
        "--pulser", action="store_true",
        help="Also run Pulser analog QAOA (QutipEmulator, T-sweep, ~25s for 12-14 qubits)",
    )
    parser.add_argument(
        "--anneal", action="store_true",
        help="Also run Simulated Annealing baseline (requires dimod + dwave-samplers)",
    )
    parser.add_argument(
        "--reads", type=int, default=5000, metavar="N",
        help="SA: number of independent reads (default 5000)",
    )
    parser.add_argument(
        "--sweeps", type=int, default=2000, metavar="N",
        help="SA: sweeps per read (default 2000)",
    )
    parser.add_argument(
        "--seed", type=int, default=123, metavar="N",
        help="SA: random seed (default 123)",
    )
    parser.add_argument(
        "--direct-gates", action="store_true",
        help="Use explicit RZ/IsingZZ/RX gates instead of qml.qaoa templates",
    )
    parser.add_argument(
        "--jax", action="store_true",
        help="Use JAX JIT-compiled backprop instead of PennyLane optimiser",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run 2×2 grid (default/lightning × templates/direct-gates) "
             "and print side-by-side timing comparison",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib plots (useful for headless/CI runs)",
    )
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Directory to save PNG plots instead of displaying them",
    )
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
