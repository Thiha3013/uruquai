"""Classical benchmarks for the sensor-placement problem.

Provides two fast classical solvers so you can quantify the quality/time
trade-off against QAOA:

greedy(cfg)
    O(n log n).  Sort candidates by risk_weight / capex_10k (value per dollar),
    pick greedily until the budget is exhausted.  Runs in microseconds.
    Not always optimal (classic counterexample for 0-1 knapsack) but a strong
    practical baseline — often what a human planner would do intuitively.

milp_solve(cfg)
    Exact MILP via scipy.optimize.milp (available since scipy 1.7).
    Solves the LP relaxation with branch-and-bound.  Always finds the global
    optimum.  Runs in milliseconds for n ≤ 30.  This is the standard academic
    benchmark for proving quantum advantage.

Both functions return a dict with the same keys as qubo.brute_force:
    x    (np.ndarray)  binary selection vector
    obj  (float)       risk coverage Σwᵢxᵢ
    cost (float)       total CAPEX Σcᵢxᵢ  (in $10K units)

Usage
-----
    from manzanillo_qc.benchmarks import greedy, milp_solve

    g = greedy(cfg)
    print(f"Greedy: {g['obj']:.4f}, ${g['cost']*10:.0f}K, sensors={list(np.where(g['x'])[0])}")

    m = milp_solve(cfg)
    print(f"MILP  : {m['obj']:.4f}, ${m['cost']*10:.0f}K, sensors={list(np.where(m['x'])[0])}")
"""
from __future__ import annotations

import time

import numpy as np

from .config import AppConfig


# ── Greedy ─────────────────────────────────────────────────────────────────────

def greedy(cfg: AppConfig) -> dict:
    """Greedy knapsack heuristic: pick by value-per-dollar until budget runs out.

    Parameters
    ----------
    cfg : AppConfig

    Returns
    -------
    dict with keys: x, obj, cost, time_ms.
    """
    w = cfg.weights
    c = cfg.costs
    B = float(cfg.budget_10k)
    n = cfg.n_sites

    t0 = time.perf_counter()

    # Sort by value-per-unit-cost descending
    ratio = np.where(c > 0, w / c, 0.0)
    order = np.argsort(ratio)[::-1]

    x         = np.zeros(n, dtype=float)
    remaining = B
    for i in order:
        if c[i] <= remaining:
            x[i] = 1.0
            remaining -= c[i]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "x":       x,
        "obj":     float(w @ x),
        "cost":    float(c @ x),
        "time_ms": elapsed_ms,
    }


# ── MILP ───────────────────────────────────────────────────────────────────────

def milp_solve(cfg: AppConfig) -> dict:
    """Exact MILP solver for the binary knapsack using scipy.optimize.milp.

    Solves:
        min  −wᵀx          (i.e. maximise coverage)
        s.t. cᵀx ≤ B
             0 ≤ x ≤ 1,  x integer

    Requires scipy ≥ 1.7.  Always returns the global optimum.

    Parameters
    ----------
    cfg : AppConfig

    Returns
    -------
    dict with keys: x, obj, cost, time_ms, success.
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        raise ImportError("scipy >= 1.7 is required for milp_solve(). "
                          "Install with: pip install 'scipy>=1.7'")

    w = cfg.weights
    c = cfg.costs
    B = float(cfg.budget_10k)
    n = cfg.n_sites

    t0 = time.perf_counter()

    c_obj        = -w                               # minimise −w (maximise w)
    constraints  = LinearConstraint(c.reshape(1, -1), lb=-np.inf, ub=B)
    bounds       = Bounds(lb=np.zeros(n), ub=np.ones(n))
    integrality  = np.ones(n)                       # 1 = integer variable

    res = milp(c_obj, constraints=constraints, integrality=integrality, bounds=bounds)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if res.success:
        x = np.round(res.x).astype(float)
    else:
        x = np.zeros(n, dtype=float)

    return {
        "x":       x,
        "obj":     float(w @ x),
        "cost":    float(c @ x),
        "time_ms": elapsed_ms,
        "success": res.success,
    }


# ── AUC / PR-AUC metrics ───────────────────────────────────────────────────────

def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve and AUC.  Pure numpy, no sklearn needed.

    Returns (fpr, tpr, auc_value).
    """
    order  = np.argsort(scores)[::-1]
    y_sort = y_true[order]
    n_pos  = y_true.sum()
    n_neg  = len(y_true) - n_pos

    tpr = np.concatenate([[0.0], np.cumsum(y_sort)   / max(n_pos, 1), [1.0]])
    fpr = np.concatenate([[0.0], np.cumsum(1-y_sort) / max(n_neg, 1), [1.0]])
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def _pr_auc(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute Precision-Recall curve and area (Average Precision).

    Returns (recall, precision, ap_value).
    """
    order  = np.argsort(scores)[::-1]
    y_sort = y_true[order]
    n_pos  = y_true.sum()

    tp  = np.cumsum(y_sort)
    fp  = np.cumsum(1 - y_sort)
    rec = tp / max(n_pos, 1)
    pre = tp / (tp + fp)

    recall    = np.concatenate([[0.0], rec])
    precision = np.concatenate([[1.0], pre])
    ap = float(np.trapz(precision, recall))
    return recall, precision, ap


def solver_scores(cfg: AppConfig, qaoa_probs: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return a continuous ranking score per location for each solver.

    These scores are used to compute AUC/PR-AUC — they represent how
    confidently each method thinks a given location should be deployed.

    Scores
    ------
    QAOA         P(xᵢ=1) — marginal probability summed from full distribution
    Greedy       wᵢ / cᵢ  — value-per-dollar ratio (its internal ranking)
    Risk weight  wᵢ        — raw seismicity/utility score (MILP / brute-force baseline)
    Random       uniform   — random baseline for reference
    """
    w = cfg.weights
    c = cfg.costs
    n = cfg.n_sites

    scores: dict[str, np.ndarray] = {}

    # Risk weight — the underlying utility used by exact solvers
    scores["Risk weight"] = w.copy()

    # Greedy ranking score
    scores["Greedy"] = np.where(c > 0, w / c, 0.0)

    # QAOA marginal probability P(xᵢ = 1)
    if qaoa_probs is not None:
        marginal = np.zeros(n)
        for idx in range(len(qaoa_probs)):
            bits = format(idx, f"0{n}b")
            for i, b in enumerate(bits):
                if b == "1":
                    marginal[i] += qaoa_probs[idx]
        scores["QAOA"] = marginal

    # Random baseline
    rng = np.random.default_rng(0)
    scores["Random"] = rng.uniform(0, 1, size=n)

    return scores


def compute_auc_metrics(
    cfg: AppConfig,
    optimal_x: np.ndarray,
    qaoa_probs: np.ndarray | None = None,
) -> dict[str, dict]:
    """Compute ROC-AUC and PR-AUC for each solver against the optimal solution.

    The optimal solution (brute-force / MILP) defines ground-truth labels:
        y_true[i] = 1  if sensor i is in the optimal set, else 0

    Each solver provides a continuous score per location (see solver_scores()).
    AUC and PR-AUC measure how well that ranking separates optimal from
    suboptimal locations — independent of the deployment threshold.

    Parameters
    ----------
    cfg : AppConfig
    optimal_x : np.ndarray
        Binary vector from brute_force() or milp_solve() — the ground truth.
    qaoa_probs : np.ndarray, optional
        Full 2^n probability array from run_qaoa()["probs"].

    Returns
    -------
    dict  solver_name → {"roc_auc": float, "pr_auc": float,
                          "fpr": array, "tpr": array,
                          "recall": array, "precision": array}
    """
    y_true  = optimal_x.astype(float)
    scores  = solver_scores(cfg, qaoa_probs)
    metrics = {}

    for name, sc in scores.items():
        fpr, tpr, roc = _roc_auc(y_true, sc)
        rec, pre, ap  = _pr_auc(y_true, sc)
        metrics[name] = {
            "roc_auc":   roc,
            "pr_auc":    ap,
            "fpr":       fpr,
            "tpr":       tpr,
            "recall":    rec,
            "precision": pre,
        }

    return metrics


def print_auc_table(metrics: dict[str, dict]) -> None:
    """Print a compact AUC/PR-AUC comparison table."""
    print(f"\n{'Solver':<18}  {'ROC-AUC':>9}  {'PR-AUC':>9}  Note")
    print("─" * 65)
    for name, m in metrics.items():
        note = ""
        if name == "Random":
            note = "← baseline (chance level)"
        elif m["roc_auc"] == 1.0:
            note = "← perfect ranking"
        print(f"{name:<18}  {m['roc_auc']:>9.4f}  {m['pr_auc']:>9.4f}  {note}")
    print("\nInterpretation:")
    print("  ROC-AUC  1.0 = perfect, 0.5 = random. Measures ranking of optimal sensors.")
    print("  PR-AUC   higher = better precision on top-ranked locations.")


def plot_roc_pr(metrics: dict[str, dict], title_suffix: str = "",
                save_path: str | None = None) -> None:
    """Plot ROC and PR curves for all solvers side by side."""
    import matplotlib.pyplot as plt

    colours = {"Risk weight": "#27ae60", "Greedy": "#e67e22",
               "QAOA": "#9b59b6", "Random": "#95a5a6"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ROC
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC=0.50)")
    for name, m in metrics.items():
        if name == "Random":
            continue
        c = colours.get(name, "steelblue")
        ax.plot(m["fpr"], m["tpr"], lw=1.8, color=c,
                label=f"{name}  (AUC={m['roc_auc']:.3f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve{title_suffix}")
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    # PR
    ax2 = axes[1]
    n_pos   = sum(1 for m in metrics.values() if "precision" in m)
    for name, m in metrics.items():
        if name == "Random":
            continue
        c = colours.get(name, "steelblue")
        ax2.plot(m["recall"], m["precision"], lw=1.8, color=c,
                 label=f"{name}  (AP={m['pr_auc']:.3f})")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall Curve{title_suffix}")
    ax2.legend(fontsize=8); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ── Comparison table ───────────────────────────────────────────────────────────

def print_benchmark_table(results: dict[str, dict], cfg: AppConfig) -> None:
    """Pretty-print a comparison table across multiple solvers.

    Parameters
    ----------
    results : dict
        Mapping of solver name → result dict (x, obj, cost, time_ms).
    cfg : AppConfig
        Used to compute the optimality gap relative to the best known solution.
    """
    best_obj = max(r["obj"] for r in results.values())

    header = f"\n{'Solver':<18}  {'Coverage':>9}  {'CAPEX':>8}  {'Gap%':>6}  {'Time':>8}  {'P(best)':>8}  Sensors"
    print(header)
    print("─" * 90)
    for name, r in results.items():
        obj_val  = r["obj"] if r["obj"] not in (None, -np.inf, np.inf) else 0.0
        gap_pct  = 100 * (best_obj - obj_val) / max(best_obj, 1e-9)
        sensors  = [int(i) for i in np.where(r["x"])[0]] if r.get("x") is not None else []
        t_str    = f"{r.get('time_ms', 0):.1f} ms"
        prob_str = f"{r['best_prob']:.4f}" if r.get("best_prob") is not None else "  n/a  "
        cost_str = f"${r['cost']*10:>6.0f}K" if r.get("cost") is not None else "     n/a "
        obj_str  = f"{obj_val:>9.4f}" if obj_val != 0.0 or r["obj"] == 0.0 else "    no feasible sol"
        print(f"{name:<18}  {obj_str}  "
              f"{cost_str}  {gap_pct:>5.1f}%  {t_str:>8}  {prob_str:>8}  {sensors}")


# ── Plots ───────────────────────────────────────────────────────────────────────

def plot_convergence(qaoa_results: dict | dict, title_suffix: str = "",
                     save_path: str | None = None) -> None:
    """Plot QAOA objective value per training step and per wall-clock time.

    Produces a 2-panel figure:
      Left  – objective vs training step  (same curve for all variants because
               the math is identical; confirms numerical equivalence)
      Right – objective vs wall-clock time  (diverges because each variant has
               a different step duration; this is the meaningful comparison)

    The right panel uses step_avg_ms from the timing dict to reconstruct the
    cumulative wall-clock axis without requiring per-step timestamps.

    Parameters
    ----------
    qaoa_results : dict
        Either a single run_qaoa() result dict, or a mapping of
        label → run_qaoa() result dict (e.g. {"default + templates": res, …}).
    """
    import matplotlib.pyplot as plt

    colours = ["#9b59b6", "#e67e22", "#27ae60", "#3498db"]

    # Normalise to {label: result} dict
    if "cost_history" in qaoa_results:
        named = {"QAOA": qaoa_results}
    else:
        named = qaoa_results

    has_timing = all(
        r.get("timing", {}).get("step_avg_ms") is not None
        for r in named.values()
    )
    multi = len(named) > 1

    if multi and has_timing:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    else:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = None

    for (label, res), colour in zip(named.items(), colours):
        history = res.get("cost_history", [])
        if not history:
            continue
        steps = list(range(len(history)))
        ax1.plot(steps, history, lw=1.5, color=colour, label=label)

        if ax2 is not None:
            step_ms = res["timing"]["step_avg_ms"]
            wallclock = [i * step_ms for i in steps]
            ax2.plot(wallclock, history, lw=1.5, color=colour, label=label)

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Objective value  (lower = better)")
    ax1.set_title(f"Convergence vs step{title_suffix}")
    ax1.legend(fontsize=8)

    if ax2 is not None:
        ax2.set_xlabel("Wall-clock time (ms)")
        ax2.set_ylabel("Objective value  (lower = better)")
        ax2.set_title(f"Convergence vs wall-clock time{title_suffix}")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_probs(qaoa_result: dict, cfg: AppConfig, top_k: int = 10,
               title_suffix: str = "", save_path: str | None = None) -> None:
    """Bar chart of the top-k highest-probability bitstrings from QAOA."""
    import matplotlib.pyplot as plt

    probs = qaoa_result.get("probs")
    if probs is None:
        print("No probs in result — skipping probability plot.")
        return

    n = cfg.n_sites
    top = np.argsort(probs)[::-1][:top_k]
    labels = [format(i, f"0{n}b") for i in top]
    colours = ["#e74c3c" if i == top[0] else "#9b59b6" for i in top]

    plt.figure(figsize=(9, 3))
    plt.bar(labels, probs[top], color=colours)
    plt.xticks(rotation=45, fontsize=8)
    plt.ylabel("Probability")
    plt.title(f"Top-{top_k} bitstrings{title_suffix}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_solver_comparison(results: dict[str, dict], title_suffix: str = "",
                           save_path: str | None = None) -> None:
    """Bar chart comparing coverage and CAPEX across solvers."""
    import matplotlib.pyplot as plt

    names    = list(results.keys())
    coverage = [r["obj"]        for r in results.values()]
    capex    = [(r["cost"] * 10 if r.get("cost") is not None else 0) for r in results.values()]

    x   = np.arange(len(names))
    w   = 0.35
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.bar(x - w / 2, coverage, w, label="Coverage Σwᵢxᵢ", color="#27ae60")
    ax1.set_ylabel("Coverage Σwᵢxᵢ", color="#27ae60")
    ax1.tick_params(axis="y", labelcolor="#27ae60")

    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, capex, w, label="CAPEX ($K)", color="#3498db", alpha=0.7)
    ax2.set_ylabel("CAPEX ($K)", color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_title(f"Solver comparison{title_suffix}")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_sensor_map(cfg: AppConfig, results: dict[str, dict],
                    existing_stations=None, title_suffix: str = "",
                    save_path: str | None = None) -> None:
    """Scatter map of candidate sites with each solver's selection highlighted."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        _has_cartopy = True
    except ImportError:
        _has_cartopy = False

    lats = np.array([s.lat for s in cfg.sites])
    lons = np.array([s.lon for s in cfg.sites])

    pad = 0.4
    lon_min, lon_max = lons.min() - pad, lons.max() + pad
    lat_min, lat_max = lats.min() - pad, lats.max() + pad

    solver_colours = {
        "Brute-force": "#e74c3c",
        "Greedy":      "#e67e22",
        "MILP":        "#27ae60",
    }
    for k in results:
        if "QAOA" in k:
            solver_colours[k] = "#9b59b6"

    n_solvers = len(results)
    fig_w = max(5 * n_solvers, 8)

    if _has_cartopy:
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            1, n_solvers,
            figsize=(fig_w, 5),
            subplot_kw={"projection": proj},
        )
        if n_solvers == 1:
            axes = [axes]
        for ax in axes:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
            ax.add_feature(cfeature.OCEAN.with_scale("10m"),
                           facecolor="#cde8f5", zorder=0)
            ax.add_feature(cfeature.LAND.with_scale("10m"),
                           facecolor="#eae6df", zorder=0)
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                           linewidth=0.8, edgecolor="#444444", zorder=1)
            ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                           linewidth=0.5, edgecolor="#777777",
                           linestyle="--", zorder=1)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                              color="gray", alpha=0.4, linestyle="--",
                              xlocs=mticker.MaxNLocator(3),
                              ylocs=mticker.MaxNLocator(3))
            gl.top_labels   = False
            gl.right_labels = False
        transform = ccrs.PlateCarree()
    else:
        fig, axes = plt.subplots(1, n_solvers, figsize=(fig_w, 5),
                                 sharex=True, sharey=True)
        if n_solvers == 1:
            axes = [axes]
        transform = None

    def _scatter(ax, x, y, **kw):
        if transform is not None:
            ax.scatter(x, y, transform=transform, **kw)
        else:
            ax.scatter(x, y, **kw)

    for ax, (name, r) in zip(axes, results.items()):
        # Existing FDSN stations
        if existing_stations is not None and len(existing_stations):
            _scatter(ax, existing_stations["lon"].values,
                     existing_stations["lat"].values,
                     marker="+", c="#2c3e50", s=30, linewidths=0.8,
                     zorder=2, label="Existing stations")
        # Candidate sites — unselected
        _scatter(ax, lons, lats, c="#bdc3c7", s=50, zorder=3,
                 label="Candidates")
        # Selected sites
        sel_idx = []
        if r.get("x") is not None:
            sel_idx = list(np.where(r["x"])[0])
            _scatter(ax, lons[sel_idx], lats[sel_idx],
                     c=solver_colours.get(name, "steelblue"),
                     s=120, edgecolors="k", linewidths=0.6,
                     zorder=4, label="Selected")
        # Labels only for selected sites — avoids clutter
        for i in sel_idx:
            s = cfg.sites[i]
            txt = f"Loc-{i+1}"
            if transform is not None:
                ax.text(s.lon + 0.05, s.lat + 0.05, txt,
                        fontsize=7, fontweight="bold",
                        transform=transform, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.7))
            else:
                ax.annotate(txt, (s.lon, s.lat), fontsize=7,
                            fontweight="bold",
                            xytext=(4, 4), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="none", alpha=0.7))
        ax.set_title(name, fontsize=10, pad=4)

    axes[0].legend(fontsize=8, loc="lower left",
                   framealpha=0.85, edgecolor="#cccccc")
    fig.suptitle(f"Sensor placement{title_suffix}", fontsize=12, y=1.01)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ── New benchmarking plots ──────────────────────────────────────────────────────

def plot_runtime_bar(timing_rows: list[dict], title_suffix: str = "",
                     save_path: str | None = None) -> None:
    """Bar chart of mean runtime per QAOA variant with optional std error bars.

    Parameters
    ----------
    timing_rows : list of dicts
        Each dict must have keys: label, mean_ms, std_ms.
    """
    import matplotlib.pyplot as plt

    labels   = [r["label"]   for r in timing_rows]
    means    = [r["mean_ms"] for r in timing_rows]
    stds     = [r["std_ms"]  for r in timing_rows]
    colours  = ["#9b59b6", "#e67e22", "#27ae60", "#3498db"]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, means, color=colours[:len(labels)], width=0.5)
    has_variance = any(s > 0 for s in stds)
    if has_variance:
        ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                    capsize=4, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title(f"QAOA variant runtime{title_suffix}")
    # Annotate bars with speedup relative to first bar
    if means[0] > 0:
        for i, (bar, m) in enumerate(zip(bars, means)):
            spd = means[0] / m if m > 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2, m + max(stds[i], 0) + 5,
                    f"{spd:.1f}×", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_timing_breakdown(timing_rows: list[dict],
                          classical: dict[str, dict] | None = None,
                          title_suffix: str = "",
                          save_path: str | None = None) -> None:
    """Stacked bar chart: timing phases per solver.

    Classical solvers get a single "Solve" bar (their total time_ms).
    QAOA variants get three stacked segments: Circuit init / Opt loop / Decode.
    Y-axis is log scale so classical (< 1 ms) and QAOA (> 1000 ms) are both
    visible.  A second panel shows the QAOA phase breakdown zoomed in so the
    small circuit-init and decode segments are readable.

    Parameters
    ----------
    timing_rows : list of dicts
        QAOA variants — each dict must have keys: label, breakdown.
    classical : dict, optional
        Classical solver results from the benchmark table, e.g.
        {"Brute-force": {"time_ms": 0.6}, "Greedy": …, "MILP": …}.
    """
    import matplotlib.pyplot as plt

    # ── Build combined bar data ────────────────────────────────────────────────
    all_labels   = []
    solve_vals   = []   # classical solve OR QAOA opt_loop
    init_vals    = []   # QAOA circuit_init  (N/A for classical → 0)
    decode_vals  = []   # QAOA decode        (N/A for classical → 0)
    is_classical = []   # bool flag per bar

    if classical:
        for name, r in classical.items():
            all_labels.append(name)
            solve_vals.append(r.get("time_ms", 0.0))
            init_vals.append(0.0)
            decode_vals.append(0.0)
            is_classical.append(True)

    for row in timing_rows:
        bd = row.get("breakdown", {})
        all_labels.append(row["label"])
        solve_vals.append(bd.get("opt_loop_ms", 0.0))
        init_vals.append(bd.get("circuit_init_ms", 0.0))
        decode_vals.append(bd.get("decode_ms", 0.0))
        is_classical.append(False)

    n = len(all_labels)
    x = np.arange(n)
    w = 0.5

    solve_c  = "#27ae60"   # green  — classical solve
    qaoa_c   = "#9b59b6"   # purple — QAOA opt loop
    init_c   = "#3498db"   # blue   — circuit init
    decode_c = "#e67e22"   # orange — decode

    bar_solve_c = [solve_c if cl else qaoa_c for cl in is_classical]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left panel: all solvers, log scale ────────────────────────────────────
    totals = np.array(solve_vals) + np.array(init_vals) + np.array(decode_vals)

    # Draw each bar individually so classical gets green, QAOA gets stacked
    for i in range(n):
        if is_classical[i]:
            ax1.bar(i, solve_vals[i], width=w, color=solve_c)
        else:
            ax1.bar(i, init_vals[i],  width=w, color=init_c,   bottom=0)
            ax1.bar(i, solve_vals[i], width=w, color=qaoa_c,
                    bottom=init_vals[i])
            ax1.bar(i, decode_vals[i], width=w, color=decode_c,
                    bottom=init_vals[i] + solve_vals[i])

    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("Time (ms)  [log scale]")
    ax1.set_title(f"All solvers — log scale{title_suffix}")

    # Custom legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(color=solve_c,  label="Classical: solve"),
        Patch(color=init_c,   label="QAOA: circuit init"),
        Patch(color=qaoa_c,   label="QAOA: opt loop"),
        Patch(color=decode_c, label="QAOA: decode"),
    ]
    ax1.legend(handles=legend_items, fontsize=7)

    # Annotate N/A on classical bars for missing phases
    for i, cl in enumerate(is_classical):
        if cl:
            ax1.text(i, totals[i] * 1.5, "init: N/A\ndecode: N/A",
                     ha="center", va="bottom", fontsize=6, color="#7f8c8d")

    # ── Right panel: QAOA only, linear scale, zoomed on small phases ──────────
    qaoa_indices = [i for i, cl in enumerate(is_classical) if not cl]
    qaoa_labels  = [all_labels[i] for i in qaoa_indices]
    qaoa_x       = np.arange(len(qaoa_labels))

    for j, i in enumerate(qaoa_indices):
        ax2.bar(j, init_vals[i],   width=w, color=init_c,   bottom=0,
                label="Circuit init" if j == 0 else "")
        ax2.bar(j, solve_vals[i],  width=w, color=qaoa_c,
                bottom=init_vals[i],
                label="Opt loop" if j == 0 else "")
        ax2.bar(j, decode_vals[i], width=w, color=decode_c,
                bottom=init_vals[i] + solve_vals[i],
                label="Decode" if j == 0 else "")

    ax2.set_xticks(qaoa_x)
    ax2.set_xticklabels(qaoa_labels, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("Time (ms)  [linear scale]")
    ax2.set_title(f"QAOA variants — phase breakdown{title_suffix}")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_seed_variance(timing_rows: list[dict], title_suffix: str = "",
                       save_path: str | None = None) -> None:
    """2-panel error-bar plot: objective and runtime mean ± std across seeds.

    Parameters
    ----------
    timing_rows : list of dicts
        Each dict must have keys: label, mean_obj, std_obj, mean_ms, std_ms.
    """
    import matplotlib.pyplot as plt

    labels   = [r["label"]    for r in timing_rows]
    mean_obj = [r["mean_obj"] for r in timing_rows]
    std_obj  = [r["std_obj"]  for r in timing_rows]
    mean_ms  = [r["mean_ms"]  for r in timing_rows]
    std_ms   = [r["std_ms"]   for r in timing_rows]
    x = np.arange(len(labels))

    colours = ["#9b59b6", "#e67e22", "#27ae60", "#3498db"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for i, (mo, so, c) in enumerate(zip(mean_obj, std_obj, colours)):
        ax1.errorbar(i, mo, yerr=so, fmt="o", color=c, capsize=5, markersize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Best objective (coverage)")
    ax1.set_title(f"Objective mean ± std across seeds{title_suffix}")

    for i, (mm, sm, c) in enumerate(zip(mean_ms, std_ms, colours)):
        ax2.errorbar(i, mm, yerr=sm, fmt="o", color=c, capsize=5, markersize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Runtime (ms)")
    ax2.set_title(f"Runtime mean ± std across seeds{title_suffix}")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
