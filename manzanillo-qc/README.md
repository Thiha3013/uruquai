# manzanillo-qc

Quantum sensor placement optimisation for the Manzanillo seismic network.

Given candidate deployment locations with seismicity risk weights and CAPEX
estimates, the package solves a binary knapsack problem:

> **Maximise** Σ wᵢ xᵢ   subject to   Σ cᵢ xᵢ ≤ B,  xᵢ ∈ {0,1}

across five solvers (PennyLane QAOA, Pulser analog QAOA, Pasqal qubo-solver,
RQAOA, DQI) and two classical baselines (Greedy, MILP), with a scalability
benchmark that finds where each solver breaks.

---

## Package layout

```
manzanillo-qc/
├── pyproject.toml
├── requirements.txt
├── results.json                  last benchmark run output
├── plots/                        generated figures (sensor maps, runtime, quality)
├── run_dqi.py                    standalone DQI scaling benchmark
└── src/manzanillo_qc/
    ├── config.py       Pydantic data models (SiteCandidate, AppConfig) + YAML loader
    ├── instance.py     Live data pipeline: USGS catalog + FDSN stations → AppConfig
    ├── qubo.py         QUBO matrix builder + brute-force reference
    ├── ising.py        QUBO → Ising (h, J) + PennyLane Hamiltonian factory
    ├── qaoa.py         PennyLane QAOA (AdamOptimizer, INTERP warm-start, grad norm)
    ├── pulser_qaoa.py  Pulser analog QAOA (Rydberg layout + T-sweep + greedy repair)
    ├── pasqal_qubo.py  Pasqal qubo-solver (neutral-atom emulator + greedy repair)
    ├── rqaoa.py        Recursive QAOA (p=1 per level, multi-seed, brute-force at n≤2)
    ├── dqi.py          Decoded Quantum Interferometry (Jordan et al., Nature 2025)
    ├── anneal.py       D-Wave simulated annealing (CPU, no cloud needed)
    ├── backends.py     PennyLane device factory
    ├── utility.py      Hazard-scenario utility builder (replaces seismicity density weights)
    ├── benchmarks.py   Greedy + MILP classical solvers + sensor map plots (cartopy)
    ├── scaling.py      Scalability benchmark runner (n=4–12, all solvers)
    └── cli.py          Command-line entry-point (single-run mode)
```

---

## Quick start

### Scalability benchmark (main use case)

Fetches live USGS/FDSN data, runs all solvers across n=4,6,8,10,12 qubits,
prints a comparison table, and saves plots:

```bash
cd manzanillo-qc
PYTHONPATH=src python3 -m manzanillo_qc.scaling
```

Common flags:

```bash
# Skip slow solvers during development
PYTHONPATH=src python3 -m manzanillo_qc.scaling --no-pulser --no-pasqal

# Save results to JSON
PYTHONPATH=src python3 -m manzanillo_qc.scaling --output results.json

# Custom QAOA depth and steps
PYTHONPATH=src python3 -m manzanillo_qc.scaling --p-layers 4 --steps 200

# Skip plots
PYTHONPATH=src python3 -m manzanillo_qc.scaling --no-plots
```

Run in background (survives terminal close):

```bash
nohup PYTHONPATH=src python3 -m manzanillo_qc.scaling > solver.log 2>&1 &
tail -f solver.log
```

### DQI standalone benchmark

```bash
cd manzanillo-qc
PYTHONPATH=src python3 run_dqi.py
```

### Single-run CLI

```bash
# Live USGS/FDSN data, custom budget:
PYTHONPATH=src python3 -m manzanillo_qc.cli --budget 80

# Save results to JSON:
PYTHONPATH=src python3 -m manzanillo_qc.cli --budget 80 --output results.json
```

---

## Study region

- **Region**: 18.3–20.8°N, 105.7–102.7°W (Manzanillo / Colima, Mexico)
- **Catalog**: USGS M≥2.0 events 2005–2025
- **Grid**: 10×10, offshore cells removed via cartopy 1:10m land mask
- **Candidates**: top-risk + low-risk greenfield cells (no existing FDSN station in same grid cell)
- **Budget default**: $800K (BUDGET_10K=80)

---

## Problem formulation

### QUBO (no slack bits)

```
min f(x) = −Σᵢ wᵢ xᵢ  +  λ · (Σᵢ cᵢ xᵢ − B)²

Q[i,i] = −wᵢ + λ · cᵢ · (cᵢ − 2B)
Q[i,j] =  2λ · cᵢ · cⱼ          (i < j)
```

**Default λ** (gradient solvers — PennyLane, RQAOA, DQI): `λ = 1 + Σwᵢ`.
Guarantees feasibility is always cheaper than any budget violation.

**Calibrated λ** (sampling solvers — Pasqal, Pulser): `λ = (Σwᵢ / min_c²) × 1.3`
via `build_qubo_sampling()`. Uses the minimum valid λ to keep the objective signal
visible relative to the penalty; avoids burying solution quality differences.

n qubits only — no slack bits.

---

## Solvers

### PennyLane QAOA (`qaoa.py`)

- Circuit: p=6 layers of cost + mixer, `lightning.qubit` with direct Ising gates + adjoint diff
- Optimiser: `qml.AdamOptimizer`, 300 steps, lr=0.01
- INTERP warm-start: p=1→2→…→p chain, each layer seeds the next (prevents barren plateaus)
- Gradient norm `||∇||` tracked every 10 steps; barren plateau warning if sustained < 1e-4
- Scalability limit: n≤28 (statevector RAM: 2^28 × 16 bytes ≈ 4 GB)

```python
from manzanillo_qc.qaoa import run_qaoa
result = run_qaoa(cfg, h, J, direct_gates=True)
```

### Pulser analog QAOA (`pulser_qaoa.py`)

- Embeds QUBO into a Rydberg atom register via Nelder-Mead layout optimisation (5 restarts)
- Uses calibrated λ (`build_qubo_sampling`) so objective signal is not buried under penalty
- T-sweep: n<12: [1,4,7]µs (3 points); n≥12: [1,3,5,7,10,12]µs (6 points)
- Greedy repair post-processing: over-budget bitstrings are repaired by dropping the
  worst risk-per-cost site until feasible (standard neutral-atom post-processing)
- Backend: QutipEmulator (n<16) or MPSBackend / emu-mps (n≥16, χ=256)
- 500 shots per T point

```python
from manzanillo_qc.pulser_qaoa import run_pulser_qaoa
result = run_pulser_qaoa(cfg, Q)
```

### Pasqal qubo-solver (`pasqal_qubo.py`)

- Uses Pasqal's `qubo-solver` library with local neutral-atom emulator
- Uses calibrated λ (`build_qubo_sampling`) instead of passed-in Q
- Symmetry-breaking diagonal term `μ·cᵢ` to prevent equal-penalty degenerate solutions
- Greedy repair post-processing on all 2000 sampled bitstrings before picking best feasible
- Backend: QutipBackendV2 (n<16) or MPSBackend / emu-mps (n≥16, χ=256)

```python
from manzanillo_qc.pasqal_qubo import run_pasqal_qubo
result = run_pasqal_qubo(cfg, Q)
```

### RQAOA (`rqaoa.py`)

- Always uses p=1 QAOA (no barren plateaus by construction)
- At each level: runs p=1 QAOA, finds the strongest correlator ⟨σᵢᶻσⱼᶻ⟩ or ⟨σᵢᶻ⟩,
  eliminates one variable, substitutes into the QUBO → recurse
- Brute-forces the residual at n≤2 (4 states, instant)
- n=4 runs 2 real QAOA levels (4→3→2), n=12 runs 10 levels
- 3 random seeds per level with early stopping (grad norm < 1e-4 for 10 consecutive steps)
- Quality degrades at late recursion levels (n=14+): correlators approach zero as the
  reduced problem loses structure

```python
from manzanillo_qc.rqaoa import run_rqaoa
result = run_rqaoa(cfg, Q)
```

### DQI (`dqi.py`)

- Decoded Quantum Interferometry (Jordan et al., Nature 2025)
- Encodes the QUBO as a random satisfiability instance, uses quantum interference
  to decode the optimal assignment
- _MAX_CLAUSES=10 (top-K clause truncation for sparse approximation)
- Runtime bottleneck: LUT / MultiControlledX decomposition scales ~5-6x per +2 qubits
- Use `run_dqi.py` for standalone scaling benchmark (n=4–12)

```python
from manzanillo_qc.dqi import run_dqi
from manzanillo_qc.ising import qubo_to_ising
h, J = qubo_to_ising(Q)
result = run_dqi(cfg, h, J)
```

### Classical baselines (`benchmarks.py`)

- **Greedy**: O(n log n), sorts by wᵢ/cᵢ, strong practical baseline
- **MILP**: exact branch-and-bound via `scipy.optimize.milp`, used as gap% reference for n>20

---

## Greedy repair (Pulser + Pasqal)

Neutral-atom solvers tend to over-select sites because all atoms experience
strong negative detuning. Both solvers apply greedy repair to each sampled bitstring:

1. While total cost > budget: drop the selected site with the lowest wᵢ/cᵢ ratio
2. Keep the repaired bitstring as a feasible candidate
3. Return the best coverage across all repaired candidates

This is standard post-processing for neutral-atom QUBO solvers and avoids
discarding near-optimal solutions that are only slightly over budget.

---

## MPS backend (emu-mps)

For n≥16, Pulser and Pasqal automatically switch from statevector (QutipBackendV2,
memory scales as 2^n) to MPS (MPSBackend, memory scales as n×χ²):

| n  | Statevector | MPS χ=256 |
|----|-------------|-----------|
| 16 | 16 MB       | 3 MB      |
| 24 | 268 MB      | 50 MB     |
| 32 | 4 GB        | 200 MB    |

`emu-mps` is already installed. No configuration needed — the switch is automatic.

---

## Scalability cutoffs

| Solver          | Hard skip above | Reason                           |
|-----------------|-----------------|----------------------------------|
| Brute-force ref | n=20            | O(2^n) enumeration               |
| PennyLane QAOA  | none (n≤9999)   | Statevector RAM ~4 GB at n=28    |
| Pulser          | none (n≤9999)   | Layout optimisation in 2n dims   |
| Pasqal          | none (n≤9999)   | Emulator runtime                 |
| RQAOA           | none (n≤9999)   | Degrades at late recursion levels|

Above n=20 the gap% reference switches from brute-force to MILP (exact, milliseconds).
No solver is replaced with a classical fallback when it fails — it is skipped.

---

## Scalability benchmark defaults

| Parameter          | Value                              | Description                     |
|--------------------|------------------------------------|---------------------------------|
| `QUBIT_COUNTS`     | [4, 6, 8, 10, 12]                  | Default sweep                   |
| `BUDGET_10K`       | 80                                 | $800K — ~40% selection ratio    |
| `DEFAULT_P`        | 6                                  | QAOA circuit depth               |
| `DEFAULT_STEPS`    | 300                                | Adam iterations per layer        |
| `PULSER_T_NS`      | [1000, 4000, 7000]                 | T-sweep for n<12 (µs)           |
| `PULSER_T_NS_LARGE`| [1000, 3000, 5000, 7000, 10000, 12000] | T-sweep for n≥12 (µs)      |
| `PULSER_SAMPLES`   | 500                                | Shots per T point                |

---

## Installation

```bash
cd manzanillo-qc
pip install -e .
```

Or without installing (used in notebooks):

```python
import sys; sys.path.insert(0, "manzanillo-qc/src")
```

### Key dependencies

| Package | Role |
|---------|------|
| pennylane ≥ 0.36 | QAOA circuit + optimiser |
| pennylane-lightning | Fast statevector + adjoint diff |
| pulser, pulser-simulation | Pulser analog QAOA |
| emu-mps ≥ 2.6 | MPS backend for n≥16 |
| qubo-solver (Pasqal) | Pasqal neutral-atom emulator |
| scipy ≥ 1.7 | MILP benchmark |
| cartopy | Coastline/land features on sensor maps + land mask |
| shapely | Point-in-polygon land filter for candidate selection |
| requests | USGS / FDSN API calls |
| matplotlib | Sensor maps + plots |

Install qubo-solver:

```bash
pip install git+https://github.com/pasqal-io/qubo-solver
```

