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


def plot_roc_pr(metrics: dict[str, dict], title_suffix: str = "") -> None:
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
        gap_pct  = 100 * (best_obj - r["obj"]) / max(best_obj, 1e-9)
        sensors  = [int(i) for i in np.where(r["x"])[0]] if r.get("x") is not None else []
        t_str    = f"{r.get('time_ms', 0):.1f} ms"
        prob_str = f"{r['best_prob']:.4f}" if r.get("best_prob") is not None else "  n/a  "
        print(f"{name:<18}  {r['obj']:>9.4f}  "
              f"${r['cost']*10:>6.0f}K  {gap_pct:>5.1f}%  {t_str:>8}  {prob_str:>8}  {sensors}")


# ── Plots ───────────────────────────────────────────────────────────────────────

def plot_convergence(qaoa_results: dict | dict, title_suffix: str = "") -> None:
    """Plot QAOA objective value per training step for one or more runs.

    Lower objective = better: the optimizer is minimising the cost Hamiltonian
    expectation, which is equivalent to maximising sensor coverage.

    Parameters
    ----------
    qaoa_results : dict
        Either a single run_qaoa() result dict, or a mapping of
        label → run_qaoa() result dict (e.g. {"p=2": res2, "p=3": res3}).
    """
    import matplotlib.pyplot as plt

    colours = ["#9b59b6", "#e67e22", "#27ae60", "#3498db"]

    # Normalise to {label: result} dict
    if "cost_history" in qaoa_results:
        named = {"QAOA": qaoa_results}
    else:
        named = qaoa_results

    plt.figure(figsize=(7, 3))
    for (label, res), colour in zip(named.items(), colours):
        history = res.get("cost_history", [])
        if history:
            plt.plot(history, lw=1.5, color=colour, label=label)
    plt.xlabel("Training step")
    plt.ylabel("Objective value  (lower = better)")
    plt.title(f"QAOA optimisation progress{title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_probs(qaoa_result: dict, cfg: AppConfig, top_k: int = 10,
               title_suffix: str = "") -> None:
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
    plt.show()


def plot_solver_comparison(results: dict[str, dict], title_suffix: str = "") -> None:
    """Bar chart comparing coverage and CAPEX across solvers."""
    import matplotlib.pyplot as plt

    names    = list(results.keys())
    coverage = [r["obj"]        for r in results.values()]
    capex    = [r["cost"] * 10  for r in results.values()]   # convert to $K

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
    plt.show()


def plot_sensor_map(cfg: AppConfig, results: dict[str, dict],
                    existing_stations=None, title_suffix: str = "") -> None:
    """Scatter map of candidate sites with each solver's selection highlighted.

    Parameters
    ----------
    existing_stations : pd.DataFrame, optional
        DataFrame with 'lat' and 'lon' columns for existing FDSN stations.
        If provided, drawn as small black crosses on every panel.
    """
    import matplotlib.pyplot as plt

    lats = np.array([s.lat for s in cfg.sites])
    lons = np.array([s.lon for s in cfg.sites])

    solver_colours = {
        "Brute-force": "#e74c3c",
        "Greedy":      "#e67e22",
        "MILP":        "#27ae60",
    }
    for k in results:
        if "QAOA" in k:
            solver_colours[k] = "#9b59b6"

    n_solvers = len(results)
    fig, axes = plt.subplots(1, n_solvers, figsize=(4 * n_solvers, 4),
                             sharex=True, sharey=True)
    if n_solvers == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        # Existing FDSN stations
        if existing_stations is not None and len(existing_stations):
            ax.scatter(existing_stations["lon"], existing_stations["lat"],
                       marker="+", c="#2c3e50", s=40, linewidths=0.8,
                       zorder=1, label="Existing stations")
        # Candidate sites
        ax.scatter(lons, lats, c="#bdc3c7", s=60, zorder=2, label="Candidates")
        # Selected sites
        if r.get("x") is not None:
            sel = np.where(r["x"])[0]
            ax.scatter(lons[sel], lats[sel],
                       c=solver_colours.get(name, "steelblue"),
                       s=150, zorder=3, edgecolors="k", linewidths=0.5,
                       label="Selected")
        for i, s in enumerate(cfg.sites):
            ax.annotate(s.name, (s.lon, s.lat),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Lon")
    axes[0].set_ylabel("Lat")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Sensor placement{title_suffix}", fontsize=11)
    fig.tight_layout()
    plt.show()
