# manzanillo-qc

Quantum sensor placement optimisation for the Manzanillo seismic network (v0.2.0).

Given a set of candidate deployment locations with seismicity risk weights and
CAPEX estimates, the package solves a binary knapsack problem:

> **Maximise** Σ wᵢ xᵢ   subject to   Σ cᵢ xᵢ ≤ B,  xᵢ ∈ {0,1}

using QAOA (PennyLane) and benchmarks it against exact and heuristic classical
solvers.  Results include solver comparisons and AUC / PR-AUC ranking metrics.

---

## Package layout

```
manzanillo-qc/
├── pyproject.toml
├── requirements.txt
├── examples/
│   └── config_small.yaml      8 Manzanillo candidate sites, budget=$250K
├── photonic/                  CV-QAOA experiment (separate venv, see below)
│   ├── cv_qaoa.py
│   ├── requirements.txt
│   └── README.md
└── src/manzanillo_qc/
    ├── config.py       Pydantic data models (SiteCandidate, AppConfig) + YAML loader
    ├── instance.py     Live data pipeline: USGS catalog + FDSN stations → AppConfig
    ├── qubo.py         QUBO matrix builder + overlap/redundancy model
    ├── ising.py        QUBO → Ising (h, J) + PennyLane Hamiltonian factory
    ├── qaoa.py         QAOA circuit + AdamOptimizer loop
    ├── backends.py     PennyLane device factory
    ├── utility.py      Hazard-scenario utility builder
    ├── benchmarks.py   Greedy heuristic + exact MILP + AUC/PR-AUC metrics
    └── cli.py          Command-line entry-point
```

---

## Quick start

```bash
cd manzanillo-qc

# Run with the bundled 8-site example (no network calls):
PYTHONPATH=src python3 -m manzanillo_qc.cli --config examples/config_small.yaml

# With geographic overlap/redundancy penalty:
PYTHONPATH=src python3 -m manzanillo_qc.cli --config examples/config_small.yaml --overlap

# Replace seismicity density with hazard-scenario utility weights:
PYTHONPATH=src python3 -m manzanillo_qc.cli --config examples/config_small.yaml --utility

# Fetch live USGS/FDSN data, custom budget:
PYTHONPATH=src python3 -m manzanillo_qc.cli --budget 30

# Save all results to JSON:
PYTHONPATH=src python3 -m manzanillo_qc.cli --config examples/config_small.yaml --output results.json
```

If the package is installed (`pip install -e .`) omit `PYTHONPATH=src` and use
`manzanillo-qc` directly.

---

## Problem context

The study region covers **18–20°N, 100–105°W** (Manzanillo / Colima, Mexico),
one of the most seismically active segments of the Middle America Trench.

### Seismicity risk weights (default)

A 20-year USGS catalog (2005–2025, M ≥ 2.0) is energy-weighted per 0.5° cell:

```
risk_score_i = Σ_{events in cell i}  10^(1.5 · M)
```

Normalised to [0, 1], this becomes the risk weight wᵢ.

### Sensor tiers

CAPEX and sensor type are assigned by risk tier:

| Sensor type  | CAPEX  | Risk threshold       | Detection radius |
|--------------|--------|----------------------|-----------------|
| Broadband    | $100 K | risk ≥ 0.60          | 150 km           |
| Short-period | $70 K  | 0.30 ≤ risk < 0.60   | 75 km            |
| MEMS node    | $40 K  | risk < 0.30          | 30 km            |

---

## Modules

### `config.py` — data models

```python
from manzanillo_qc.config import AppConfig, SiteCandidate

cfg = AppConfig.from_yaml("examples/config_small.yaml")
print(cfg.n_sites, cfg.effective_lambda)

site = cfg.sites[0]
print(site.detection_radius_km())    # physical detection footprint
print(site.detection_threshold_g())  # minimum PGA the sensor can detect
```

**`SiteCandidate`** fields: `name`, `lat`, `lon`, `risk_weight` ∈ [0,1],
`capex_10k`, `sensor_type`, `greenfield`.

**`AppConfig`** key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `budget_10k` | 25 | Total budget in $10 K units ($250 K) |
| `penalty_lambda` | None | QUBO penalty λ; None → auto = 1 + Σwᵢ |
| `p_layers` | 2 | QAOA circuit depth |
| `n_steps` | 200 | Adam optimiser iterations |
| `stepsize` | 0.01 | Adam learning rate |
| `backend` | `"default.qubit"` | PennyLane device name |

---

### `qubo.py` — QUBO matrix

```python
from manzanillo_qc.qubo import build_qubo, brute_force

Q, meta = build_qubo(cfg)                  # basic QUBO
Q, meta = build_qubo(cfg, overlap=True)    # with redundancy penalty

bf = brute_force(cfg, Q)
print(bf["obj"], bf["cost"], bf["x"])
```

**Minimisation form** (no slack bits, n qubits only):

```
min f(x) = −Σᵢ wᵢ xᵢ  +  λ · (Σᵢ cᵢ xᵢ − B)²  +  Σᵢ<ⱼ ρᵢⱼ xᵢxⱼ

Q[i,i] = −wᵢ + λ · cᵢ · (cᵢ − 2B)
Q[i,j] =  2λ · cᵢ · cⱼ  +  ρᵢⱼ          (i < j, overlap optional)
```

**Overlap / redundancy model** (`--overlap`):

```
overlap_ij = max(0, 1 − dist(i,j) / (r_i + r_j))
ρᵢⱼ        = overlap_ij · min(wᵢ, wⱼ)
```

Adds a positive off-diagonal term that discourages co-deploying sensors whose
detection footprints significantly overlap.

`meta` keys: `n_qubits`, `lambda`, `budget`, `q_min`, `q_max`.

---

### `ising.py` — Ising conversion

```python
from manzanillo_qc.ising import qubo_to_ising, build_pennylane_hamiltonian

h, J          = qubo_to_ising(Q)
H_cost, H_mix = build_pennylane_hamiltonian(h, J)
```

Substitution xᵢ = (1 − σᵢ)/2 maps the QUBO to Ising fields h and couplings J.
`build_pennylane_hamiltonian` is version-safe: uses `LinearCombination`
(PennyLane ≥ 0.36) with fallback to `qml.Hamiltonian`.

---

### `qaoa.py` — QAOA circuit

```python
from manzanillo_qc.qaoa import run_qaoa

result = run_qaoa(cfg, h, J)
print(result["best_obj"], result["best_cost"])
print(result["best_x"])     # binary sensor selection vector
print(result["probs"])      # full 2^n probability distribution
```

**Circuit** (p layers, n qubits):

```
|+⟩^⊗n  →  [cost_layer(γₗ) · mixer_layer(βₗ)]_{l=1..p}  →  measure probs
```

**Optimiser**: `qml.AdamOptimizer`, minimises ⟨H_cost⟩ over 2p parameters.
Starts from random angles in [0, π/2], seed 42.

**Solution extraction**: scans the top-500 highest-probability bitstrings,
returns the best feasible one (Σcᵢxᵢ ≤ B).

Return dict keys: `best_x`, `best_obj`, `best_cost`, `best_prob`, `probs`,
`cost_history`, `opt_params`, `final_expval`.

---

### `utility.py` — hazard scenario utility

```python
from manzanillo_qc.utility import build_utility_weights, DEFAULT_SCENARIOS, print_scenario_report

cfg_u = build_utility_weights(cfg, DEFAULT_SCENARIOS)  # returns new AppConfig
print_scenario_report(cfg_u, DEFAULT_SCENARIOS)
```

Replaces seismicity density weights with **scenario-derived utility scores**.
Four realistic Manzanillo fault scenarios are defined:

| Scenario | Type | Mag | Prob |
|----------|------|-----|------|
| S1 | Thrust interface (1995-like event) | M7.5 | 0.40 |
| S2 | Inland crustal near Colima volcano | M5.5 | 0.25 |
| S3 | Deep intraslab | M6.0 | 0.20 |
| S4 | Northwest coastal segment | M6.5 | 0.15 |

PGA is estimated via:

```
log10(PGA_g) = 0.5·M − log10(R_km) − 0.003·R_km − 2.5
```

Utility of sensor i = Σ_s  p_s · I(PGA_i_s > threshold_i)

Weights are normalised to [0, 1] before returning.  With `--utility` the
optimal sensor set typically spreads geographically to cover different fault
systems rather than clustering in the highest-seismicity zone.

---

### `benchmarks.py` — classical solvers + AUC metrics

```python
from manzanillo_qc.benchmarks import greedy, milp_solve, compute_auc_metrics, print_auc_table

g = greedy(cfg)
m = milp_solve(cfg)
# Both return: {x, obj, cost, time_ms}

auc = compute_auc_metrics(cfg, bf["x"], qaoa_probs=result["probs"])
print_auc_table(auc)
```

#### Classical solvers

**`greedy(cfg)`** — O(n log n) heuristic.  Sorts by wᵢ/cᵢ (value per dollar),
picks greedily until budget exhausted.  Not guaranteed optimal but a strong
practical baseline.

**`milp_solve(cfg)`** — Exact branch-and-bound via `scipy.optimize.milp`.
Always finds the global optimum.  Requires scipy ≥ 1.7.

#### AUC / PR-AUC metrics

Ground truth labels = brute-force optimal binary vector.
Each solver provides a continuous ranking score per location:

| Solver | Score used |
|--------|-----------|
| Risk weight | wᵢ (raw seismicity score) |
| Greedy | wᵢ / cᵢ (its internal ranking ratio) |
| QAOA | P(xᵢ=1) — marginal probability from full distribution |
| Random | Uniform random (baseline) |

AUC = 1.0 means the solver's ranking perfectly separates optimal from
sub-optimal locations.  AUC = 0.5 is chance level.

---

### `cli.py` — command line

```
manzanillo-qc --config examples/config_small.yaml
manzanillo-qc --config examples/config_small.yaml --overlap
manzanillo-qc --config examples/config_small.yaml --utility
manzanillo-qc --budget 30
manzanillo-qc --config examples/config_small.yaml --output results.json
```

Flags:

| Flag | Description |
|------|-------------|
| `--config YAML` | Path to YAML config (default: fetch live USGS/FDSN data) |
| `--budget N` | Budget in $10K units (default 25). Ignored if --config set. |
| `--overlap` | Add geographic overlap/redundancy penalty to QUBO |
| `--utility` | Replace seismicity density weights with hazard-scenario utilities |
| `--output JSON` | Write full results to a JSON file |

CLI output sections:
1. Config summary (sites, budget, backend, p, steps)
2. Classical benchmarks table (Brute-force, Greedy, MILP)
3. QAOA result added to comparison table
4. AUC / PR-AUC metrics table

---

## Results from `examples/config_small.yaml`

8 candidate sites, budget = $250 K, p = 2, Adam 200 steps:

```
Solver               Coverage     CAPEX    Gap%      Time  Sensors
────────────────────────────────────────────────────────────────────────────────
Brute-force            1.0201  $   220K    0.0%    0.6 ms  [0, 1, 2, 3]
Greedy                 1.0201  $   220K    0.0%    0.0 ms  [0, 1, 2, 3]
MILP                   1.0201  $   220K    0.0%    1.8 ms  [0, 1, 2, 3]
QAOA (p=2)             1.0201  $   220K    0.0%  1614.8 ms  [0, 1, 2, 3]

Solver                ROC-AUC     PR-AUC  Note
─────────────────────────────────────────────────────────────────────────────
Risk weight            1.0000     1.0000  ← perfect ranking
Greedy                 1.0000     1.0000  ← perfect ranking
QAOA                   0.2500     0.5280
Random                 0.0625     0.3092  ← baseline (chance level)
```

The QAOA ROC-AUC of 0.25 on this small instance reflects a very flat
probability distribution (~0.001 per bitstring), so marginal probabilities
do not sharply discriminate sites.  With `--utility` the distribution
sharpens and QAOA achieves AUC = 1.0.

---

## Photonic experiments

A continuous-variable (CV) QAOA experiment using Strawberry Fields lives in
`photonic/`.  It must run in a separate virtual environment because
`strawberryfields >= 0.23` requires `numpy < 2` and conflicts with
`pennylane >= 0.36`.

```bash
cd photonic
python3 -m venv .venv-photonic
source .venv-photonic/bin/activate
pip install -r requirements.txt
python cv_qaoa.py --config ../examples/config_small.yaml --n-shots 100 --n-steps 30
```

If `strawberryfields` is not installed, `cv_qaoa.py` falls back to a classical
Monte Carlo sampler automatically.

See `photonic/README.md` for the CV encoding details.

---

## Full pipeline in Python

```python
from manzanillo_qc.config import AppConfig
from manzanillo_qc.qubo import build_qubo, brute_force
from manzanillo_qc.ising import qubo_to_ising
from manzanillo_qc.qaoa import run_qaoa
from manzanillo_qc.benchmarks import greedy, milp_solve, compute_auc_metrics, print_auc_table

cfg = AppConfig.from_yaml("examples/config_small.yaml")

Q, meta = build_qubo(cfg, overlap=False)
bf      = brute_force(cfg, Q)

g = greedy(cfg)
m = milp_solve(cfg)

h, J   = qubo_to_ising(Q)
result = run_qaoa(cfg, h, J)

auc = compute_auc_metrics(cfg, bf["x"], qaoa_probs=result["probs"])
print_auc_table(auc)
```

---

## Installation

```bash
# Editable install (recommended for development):
cd manzanillo-qc
pip install -e .

# Or add to sys.path without installing (used in quantum_exp.ipynb):
import sys; sys.path.insert(0, "manzanillo-qc/src")
```

---

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| pennylane | ≥ 0.36 | QAOA circuit + optimiser |
| numpy | ≥ 1.24 | Matrix arithmetic |
| pandas | ≥ 2.0 | Data pipeline |
| pydantic | ≥ 2.0 | Config validation |
| pyyaml | ≥ 6.0 | YAML config loader |
| scipy | ≥ 1.7 | MILP benchmark (`milp_solve`) |
| requests | ≥ 2.28 | USGS / FDSN API calls |
| matplotlib | ≥ 3.9 | Optional — `plot_roc_pr()` |

---

## Relationship to `quantum_exp.ipynb`

`quantum_exp.ipynb` is the analysis notebook.  Sections §1–§3 fetch data and
build the candidate list inline; §4–§5 import from this package:

```
quantum_exp.ipynb §4  →  manzanillo_qc.qubo.build_qubo()
quantum_exp.ipynb §5  →  manzanillo_qc.ising + manzanillo_qc.qaoa.run_qaoa()
```

The inline notebook version used 13 variables (8 sensor + 5 slack bits,
COBYLA optimiser).  This package uses 8 variables (soft penalty, Adam
optimiser), matching the project specification.
