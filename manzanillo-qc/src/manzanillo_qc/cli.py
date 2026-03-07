"""Command-line entry-point for manzanillo-qc.

Usage
-----
    # Basic run from YAML config:
    manzanillo-qc --config examples/config_small.yaml

    # With overlap/redundancy model:
    manzanillo-qc --config examples/config_small.yaml --overlap

    # Replace seismicity density with hazard-scenario utility weights:
    manzanillo-qc --config examples/config_small.yaml --utility

    # Fetch live USGS/FDSN data, custom budget:
    manzanillo-qc --budget 30

    # Save full results to JSON:
    manzanillo-qc --config examples/config_small.yaml --output results.json

    # Without installing the package:
    python -m manzanillo_qc.cli --config examples/config_small.yaml
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .config import AppConfig
from .qubo import build_qubo, brute_force
from .ising import qubo_to_ising
from .qaoa import run_qaoa
from .benchmarks import (greedy, milp_solve, print_benchmark_table,
                         compute_auc_metrics, print_auc_table,
                         plot_convergence, plot_probs, plot_solver_comparison,
                         plot_sensor_map)


def _run(args: argparse.Namespace) -> None:
    # ── 1. Load config ─────────────────────────────────────────────────────────
    existing_stations = None
    t_fetch = 0.0
    if args.config:
        cfg = AppConfig.from_yaml(args.config)
    else:
        from .instance import fetch_instance, fetch_stations
        t0 = time.perf_counter()
        cfg = fetch_instance(budget_10k=args.budget)
        existing_stations = fetch_stations()
        t_fetch = (time.perf_counter() - t0) * 1000

    # ── 2. Optional: replace risk_weight with scenario-based utility ───────────
    if args.utility:
        from .utility import build_utility_weights, DEFAULT_SCENARIOS, print_scenario_report
        print("\n── Hazard Scenario Utility ─────────────────────────────────────")
        cfg = build_utility_weights(cfg, DEFAULT_SCENARIOS)
        print_scenario_report(cfg, DEFAULT_SCENARIOS)
        print()

    if args.backend:
        cfg = cfg.model_copy(update={"backend": args.backend})
    if args.optimizer:
        cfg = cfg.model_copy(update={"optimizer": args.optimizer})
    if args.steps:
        cfg = cfg.model_copy(update={"n_steps": args.steps})

    print(f"\nLoaded {cfg.n_sites} site candidates  |  budget = ${cfg.budget_10k * 10}K")
    print(f"Backend: {cfg.backend}  |  p = {cfg.p_layers}  |  steps = {cfg.n_steps}")

    # ── 3. Build QUBO ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    Q, meta = build_qubo(cfg, overlap=args.overlap)
    t_qubo = (time.perf_counter() - t0) * 1000
    overlap_str = " + overlap penalty" if args.overlap else ""
    print(f"\nQUBO{overlap_str}: {meta['n_qubits']} qubits  |  "
          f"λ={meta['lambda']:.3f}  |  Q ∈ [{meta['q_min']:.2f}, {meta['q_max']:.2f}]")

    # ── 4. Classical benchmarks ────────────────────────────────────────────────
    print("\n── Classical Benchmarks ────────────────────────────────────────────")

    t0 = time.perf_counter()
    bf = brute_force(cfg, Q)
    bf["time_ms"] = (time.perf_counter() - t0) * 1000

    g = greedy(cfg)
    m = milp_solve(cfg)

    classical = {
        "Brute-force": bf,
        "Greedy":      g,
        "MILP":        m,
    }

    # ── 4b. Simulated Annealing (optional) ─────────────────────────────────────
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

    print_benchmark_table(classical, cfg)

    # ── 5. QAOA ────────────────────────────────────────────────────────────────
    print("\n── QAOA ────────────────────────────────────────────────────────────")
    h, J = qubo_to_ising(Q)
    qaoa_results = {}
    qaoa_raw     = {}
    timing_rows  = []

    if args.compare:
        # 2×2 grid: backend × gate-style — isolates each improvement separately
        variants = [
            ("default.qubit",   False, "default  + templates"),
            ("default.qubit",   True,  "default  + direct gates"),
            ("lightning.qubit", False, "lightning + templates"),
            ("lightning.qubit", True,  "lightning + direct gates"),
        ]
    else:
        variants = [(cfg.backend, args.direct_gates, f"QAOA p={cfg.p_layers}")]

    for backend_str, direct, label in variants:
        cfg_v = cfg.model_copy(update={"backend": backend_str})
        print(f"\n  [{label}]  p={cfg_v.p_layers}, {cfg_v.n_steps} steps ...")
        t0 = time.perf_counter()
        result = run_qaoa(cfg_v, h, J, direct_gates=direct)
        t_ms = (time.perf_counter() - t0) * 1000
        result["time_ms"] = t_ms
        qaoa_results[label] = {
            "x":         result["best_x"],
            "obj":       result["best_obj"],
            "cost":      result["best_cost"],
            "time_ms":   t_ms,
            "best_prob": result["best_prob"],
        }
        qaoa_raw[label] = result
        timing_rows.append((label, t_ms))

    all_results = {**classical, **qaoa_results}
    print_benchmark_table(all_results, cfg)

    # ── 7. Timing breakdown ────────────────────────────────────────────────────
    print("\n── Timing Breakdown ────────────────────────────────────────────────")
    if t_fetch:
        print(f"  Instance fetch (USGS + FDSN) : {t_fetch:>8.1f} ms")
    print(f"  QUBO build                   : {t_qubo:>8.1f} ms")
    if args.compare:
        print(f"\n  {'Variant':<30}  {'Time':>10}  {'Speedup vs baseline':>20}")
        print(f"  {'─'*30}  {'─'*10}  {'─'*20}")
        baseline_ms = timing_rows[0][1]
        for lbl, t_ms in timing_rows:
            speedup = baseline_ms / t_ms if t_ms > 0 else 0
            print(f"  {lbl:<30}  {t_ms:>8.1f}ms  {speedup:>17.1f}×")
    else:
        for lbl, t_ms in timing_rows:
            print(f"  QAOA {lbl:<30}: {t_ms:>8.1f} ms")

    # ── 8. Plots ───────────────────────────────────────────────────────────────
    if not args.no_plots:
        suffix = " (utility weights)" if args.utility else ""
        suffix += " + overlap" if args.overlap else ""
        plot_convergence(qaoa_raw, title_suffix=suffix)
        last_result = list(qaoa_raw.values())[-1]
        plot_probs(last_result, cfg, title_suffix=suffix)
        plot_solver_comparison(all_results, title_suffix=suffix)
        plot_sensor_map(cfg, all_results, existing_stations=existing_stations, title_suffix=suffix)

    # ── 9. Optional JSON output ────────────────────────────────────────────────
    if args.output:
        def _ser(r: dict) -> dict:
            out = {k: v for k, v in r.items() if k != "x"}
            if r.get("x") is not None:
                out["sensors"] = list(np.where(r["x"])[0].tolist())
            return out

        payload = {name: _ser(r) for name, r in all_results.items()}
        payload["_meta"] = {
            "n_sites":   cfg.n_sites,
            "budget_10k": cfg.budget_10k,
            "overlap":   args.overlap,
            "utility":   args.utility,
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
        "--config", "-c", metavar="YAML",
        help="Path to YAML config (default: fetch live USGS/FDSN data)",
    )
    parser.add_argument(
        "--budget", "-b", type=int, default=25, metavar="UNITS",
        help="Budget in $10K units (default 25 = $250K). Ignored if --config is set.",
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
        "--compare", action="store_true",
        help="Run both template (default.qubit) and direct-gates (lightning.qubit) "
             "variants and print a side-by-side timing comparison",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib plots (useful for headless/CI runs)",
    )
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
