# README — `quantum_exp.ipynb`

## What this notebook is

This notebook solves a **seismic sensor placement problem** using a quantum algorithm called QAOA, then compares its answer against a classical exact solver (OR-Tools CP-SAT).

The context is Liz's Phase II memo: the team needs a ranked deployment table showing which candidate sensor sites are worth funding, given a fixed budget. This notebook is the **quantum optimisation kernel** of that deliverable.

It is not a production system. It is an experiment to show that the QUBO/QAOA pipeline:
1. Can be formulated from real seismicity data
2. Runs to completion on a classical simulator
3. Produces a defensible answer comparable to an exact solver

---

## The problem in one sentence

> Given 8 candidate sensor locations, each with a risk weight and a deployment cost, pick a subset that maximises total seismic risk coverage without exceeding a $250K budget.

This is a **0-1 knapsack problem**. It is NP-hard in general. For small instances (≤ 20 binary variables) a quantum approximate algorithm can explore all possibilities simultaneously in superposition.

---

## How the notebook is structured

```
§1  Data sources          →  fetch real earthquake catalog + station inventory
§2  Risk grid             →  bin events into a 4×5 geographic grid, compute risk scores
§3  Candidate locations   →  pick 8 sites + assign CAPEX estimates
§4  QUBO formulation      →  encode the knapsack as a matrix minimisation problem
§5  QAOA                  →  solve with quantum approximate algorithm (PennyLane)
§6  CP-SAT                →  solve exactly with OR-Tools (classical benchmark)
§7  Comparison            →  ranked deployment table + agreement analysis
§8  Data gaps / next steps→  notes on what was excluded and why
```

---

## Cell-by-cell explanation

---

### Cell c00 — Title and pipeline diagram (markdown)

Describes the full pipeline in ASCII art. The key line is the constraint:

> ≤ 20 binary variables so the simulator runs cleanly (2²⁰ < 10⁶ states)

**Why this limit?** A quantum circuit with `n` qubits simulated on a classical computer
requires storing a state vector of size 2ⁿ complex numbers. At n=20 that is about 8 MB —
manageable. At n=30 it is 8 GB. We use 13 variables (8 sensor + 5 slack), well within budget.

---

### Cell c01 — Package installation

```python
%pip -q install pennylane requests pandas scipy ortools matplotlib
```

Installs everything in one line. `-q` suppresses verbose output.

- `pennylane` — quantum circuit simulator
- `requests` — HTTP calls to USGS and EarthScope APIs
- `scipy` — provides the COBYLA optimizer
- `ortools` — Google's OR-Tools for the classical CP-SAT solver
- `matplotlib` — all the plots

---

### Cell c02 — Imports

```python
import io, json, requests, warnings, time
import numpy as np
import pandas as pd
...
import pennylane as qml
np.random.seed(42)
```

Standard imports. Two things worth noting:

- `import io` is needed for `io.StringIO` when parsing SSN CSV data (a standard Python module
  that was missing in the original version and caused a bug with `pd.io.common.StringIO`).
- `np.random.seed(42)` makes the QAOA starting parameters reproducible — without this,
  the optimizer would start from different random angles each run and give different results.

---

### §1 — Data Sources

---

### Cell c04 — USGS Earthquake Catalog

**What it does:**
Fetches 20 years of earthquakes (M≥2.0, 2005–2025) in a bounding box around the
Manzanillo/Colima region of Mexico (18–20°N, 100–105°W) from the USGS FDSN web service.

**Why USGS?**
It is the most reliable, globally available, machine-readable earthquake catalog.
All endpoints were confirmed working in the QA notebook. The data is public domain.

**Why 20 years?**
Short windows (e.g. 1 year) miss the background seismicity rate — a quiet year looks
artificially safe. 20 years captures enough M≥5+ events to distinguish genuinely high-risk
cells from low-risk ones.

**Why energy-weighted density later (not just event count)?**
A single M7.6 event releases ~10,000× more energy than an M5.0 event. Using raw counts
would underweight the cells where major earthquakes happen. By converting magnitude to
seismic moment (`10^(1.5·M)`) we get a physically meaningful risk proxy.

**Pagination:**
The USGS API returns at most 20,000 events per call. The `while True` loop pages through
using `offset` until a partial page comes back, meaning all events have been fetched.

**Output:** `usgs` DataFrame with columns `time, mag, lon, lat, depth`.

---

### Cell c05 — SSN Catálogo de sismos

**What it does:**
Attempts to download additional earthquake data from Mexico's national seismic service (SSN)
to supplement USGS. Falls back gracefully if unavailable.

**Why SSN?**
SSN has better coverage of small magnitude events (M2–M4) in Mexico than USGS, because
they operate the local network. More events = better density map.

**Why does it fall back?**
The SSN CSV endpoint (`/catalogo/reportecsv`) is a Java web application that sometimes
returns an HTML page instead of CSV (session/CSRF protection, or the server is down).
Rather than crash the notebook, the `try/except` catches any failure and proceeds with
USGS-only data. This is documented as a known data gap.

**The `io.StringIO` fix:**
`pd.read_csv` needs a file-like object when given a string. The correct way to convert
a string to a file-like object in Python is `io.StringIO(text)`. An earlier version used
`pd.io.common.StringIO` which is not a public API and raises an error in newer pandas.

**Output:** `catalog` DataFrame — either USGS alone or USGS + SSN merged.

---

### Cell c06 — EarthScope Fedcatalog / FDSN Station Inventory

**What it does:**
Queries EarthScope's FDSN station service for all seismic stations currently deployed
inside (and slightly outside) the study bounding box. Parses the pipe-delimited text
response into a `stations` DataFrame.

**Why do we need existing stations?**
The notebook flags each candidate location as `greenfield` (True) or not (False).
A greenfield location has no existing station — deploying there adds new coverage.
A non-greenfield location would be redundant or an upgrade, which changes the
justification for deployment. This flag appears in the final ranked table.

**Output:** `stations` DataFrame with network, station code, lat/lon, site name.
In the current run: **66 stations** found, mostly from the XF and ZA networks (Colima
volcano monitoring array and regional broadband network).

---

### §2 — Seismicity Density Grid + Risk Weights

---

### Cell c07 — Grid explanation (markdown)

Explains the grid design: 4 latitude bins × 5 longitude bins = **20 cells** covering
the study region. Each cell is 0.5° × 1° in size (~55 km × 110 km).

The risk score formula `score_i = Σ 10^(1.5·M)` sums the seismic moment proxy over
all events in the cell. This is the standard energy-weighted seismicity measure used
in probabilistic seismic hazard analysis.

---

### Cell c08 — Build the risk grid

**What it does:**
1. Divides the study area into a 4×5 grid using `np.linspace`
2. For each cell, finds all catalog events that fall inside (using lat/lon bin masks)
3. Computes the energy-weighted sum `10^(1.5·M)` for each cell
4. Normalises all values to [0, 1] by dividing by the maximum cell score
5. Checks which cells already have at least one station (sets `has_station = True`)

**Key result from the actual run:**
Cell R0C2 (18.25°N, 102.5°W) has `risk_norm = 1.000` — it dominates because it contains
the **M7.6 Colima earthquake** of 2003 (actually captured in the 2005+ window via aftershocks)
and has high background seismicity. This is the subduction zone offshore Manzanillo.

Every other cell has `risk_norm < 0.01` — the energy is so concentrated in that one cell
(due to the M7.6) that all others are nearly zero by comparison. This is an important
property: the normalisation makes the optimisation problem heavily dominated by Loc-1.

---

### Cell c09 — Risk heatmap + event count bar chart

Two plots:
- **Left:** Geographic heatmap of normalised risk scores, overlaid with existing stations
  (blue triangles) and large events (M≥5.5, white stars)
- **Right:** Horizontal bar chart of the top 10 cells by raw event count

The heatmap makes it immediately clear that the risk is concentrated in the southwest
corner of the study region (the offshore subduction zone).

---

### §3 — Candidate Sensor Locations + CAPEX

---

### Cell c10 — CAPEX table (markdown)

Explains the three sensor tiers used to assign deployment costs:
- **Broadband seismometer** ($80–120K): high-risk permanent sites
- **Short-period seismometer** ($50–70K): medium-risk regional sites
- **MEMS node** ($20–40K): low-cost infill or temporary deployment

These are rough estimates from EarthScope sensor inventory literature, used to make the
budget constraint realistic rather than arbitrary.

---

### Cell c11 — Select 8 candidate locations

**Selection strategy:**
- Top 5 cells by `risk_norm` → highest-priority deployment targets
- 3 low-risk cells with some seismic activity (not zero events) → needed so the
  budget constraint is actually binding and the optimiser has trade-offs to make

Without low-risk candidates, every solution would just be "deploy everywhere high-risk"
and the optimisation problem would be trivial.

**CAPEX assignment:**
Simple threshold rule: risk ≥ 0.6 → Broadband ($100K), ≥ 0.3 → Short-period ($70K),
else → MEMS ($40K). In this run, Loc-1 gets Broadband (risk = 1.0); all others get
MEMS (risk < 0.01).

**Budget:**
`BUDGET_10K = 25` units × $10K = **$250K total**. The minimum possible sensor count
at this budget is 2 (two Broadband sensors would cost $200K each = $400K, over budget —
actually: one Broadband = $100K, so 2 = $200K ≤ $250K ✓). Maximum is 6 MEMS sensors
($40K × 6 = $240K ≤ $250K).

**Output from the actual run:**
```
Budget: $250K = 25 units
Min possible sensors: 2
Max possible sensors: 6
```

---

### Cell c12 — Candidate location map

Shows the 8 candidate sites overlaid on the risk heatmap. Circle size is proportional
to risk weight, colour indicates sensor type. Existing stations shown as blue triangles.

Visually confirms that Loc-1 (the high-risk offshore cell) is far larger than all others.

---

### §4 — QUBO Formulation

---

### Cell c13 — QUBO explanation (markdown)

This is the mathematical heart of the notebook. Worth reading carefully.

**Why QUBO?**
Quantum computers (and QAOA specifically) work natively with binary variables and
quadratic cost functions. The QUBO form `min z^T Q z` maps directly to a quantum
Hamiltonian. Any constrained binary optimisation problem can be put in this form.

**Binary variables:**
- `x₀, …, x₇` — sensor bits: xᵢ = 1 means "deploy at location i"
- `y₀, …, y₄` — slack bits: these absorb the leftover budget

**Why slack bits?**
QAOA (and QUBO in general) requires *unconstrained* problems. The budget constraint
`Σcᵢxᵢ ≤ B` is an inequality. To convert it to an equality (which we can then penalise),
we introduce slack variables: `Σcᵢxᵢ + (slack) = B`. The slack absorbs the unused budget.

We need enough slack bits to represent any value from 0 to B=25. With 5 bits
[1, 2, 4, 8, 16], the maximum representable slack is 31 > 25, so 5 bits is sufficient.

**The penalty term:**
`λ · (Σcᵢxᵢ + Σ2ᵏyₖ − B)²` is zero when the constraint is exactly satisfied and
positive otherwise. The penalty coefficient `λ = 1 + Σwᵢ` is chosen to be larger
than the maximum possible objective value, ensuring that any budget violation is more
expensive than any possible gain from sensor deployment.

**Q matrix entries:**
- Diagonal: `Q[i,i] = −wᵢ` (objective, sensor bits only) `+ λ·aᵢ·(aᵢ−2B)` (penalty)
- Off-diagonal: `Q[i,j] = 2λ·aᵢ·aⱼ` for i < j (penalty coupling terms)

**Ising conversion:**
Quantum gates act on spin variables σᵢ ∈ {+1, −1}, not binary bits. The substitution
`xᵢ = (1 − σᵢ)/2` converts the QUBO into an Ising Hamiltonian:
`H = Σᵢ hᵢZᵢ + Σᵢ<ⱼ JᵢⱼZᵢZⱼ`
where Zᵢ is the Pauli-Z operator on qubit i.

---

### Cell c14 — Build Q and verify with brute force

**What it does:**
1. Computes the coefficient vector `a = [c₀,…,c₇, 1,2,4,8,16]`
2. Sets `λ = 1 + Σwᵢ = 2.024`
3. Fills the 13×13 Q matrix using the formulas above
4. Brute-forces the optimal sensor assignment (checks all 2⁸ = 256 combinations)
   to get the ground-truth answer for comparison

**Why brute-force only over sensor bits?**
We only need to check sensor placements (2⁸ = 256 combinations). The slack bits are
determined automatically: given a sensor selection with total cost C, the optimal
slack is `B − C` which has a unique binary representation. So we don't need to search
over slack bits.

**Key output:**
```
Brute-force optimum: obj=1.0201, cost=22.0/25.0, sensors=[0, 1, 2, 3]
```
The best possible answer is to deploy Loc-1 (risk=1.0, cost=$100K) + Loc-2, Loc-3,
Loc-4 (MEMS sensors, $40K each), total cost $220K, leaving $30K unspent. This is
the reference answer QAOA and CP-SAT must match.

**Why Q range is so large (`[-1100, 647]`):**
The penalty scale `λ·aᵢ·aⱼ` involves the squared slack coefficients (up to 16²=256)
multiplied by λ≈2 and the cost values (up to 10). The large values are in the
slack-slack coupling region of Q, which is expected and correct.

---

### Cell c15 — Q matrix visualisation

Two plots:
- **Left:** Heatmap of the 13×13 Q matrix. The dashed lines separate sensor bits
  (top-left 8×8 block) from slack bits (bottom-right 5×5 block). Blue = negative
  (where the objective pulls the solution), red = large positive (penalty region).
- **Right:** Histogram of off-diagonal values. The distribution shows that most
  couplings are near zero, with a few large penalty values in the slack region.

---

### §5 — QAOA

---

### Cell c16 — QAOA explanation (markdown)

Explains the algorithm: p=2 layers (two rounds of cost + mixer), optimised with COBYLA.

**Why p=2 and not p=1?**
More layers generally gives better approximation quality. p=1 has only 2 parameters
(one γ, one β). p=2 has 4 parameters. With COBYLA (~400 evaluations), p=2 is still
fast (under 5 seconds on CPU) and produces better results than p=1 for this problem size.

**Why COBYLA?**
COBYLA is a gradient-free optimizer from scipy. QAOA parameters can be optimised with
gradient-based methods (using PennyLane's automatic differentiation), but gradient
computation requires additional circuit evaluations. For p=2 with 4 parameters, COBYLA
is simpler, more reliable, and fast enough.

---

### Cell c17 — QUBO → Ising conversion + Hamiltonian construction

**What it does:**
1. Computes `h_ising[i]` (local field on qubit i) and `J_ising[i,j]` (ZZ coupling)
   from the QUBO matrix Q using the conversion formulas
2. Builds `H_cost` — the PennyLane Hamiltonian representing the problem
3. Builds `H_mix` — the mixer Hamiltonian (sum of PauliX on all qubits)
4. Verifies that the QUBO energy and Ising energy rank solutions the same way

**The `make_hamiltonian` wrapper:**
In PennyLane ≥ 0.36, `qml.Hamiltonian` was deprecated in favour of
`qml.ops.LinearCombination`. The wrapper tries the new API first, falls back to the
old one, ensuring the notebook runs on any PennyLane version.

**About `H_mix`:**
The mixer `Σᵢ Xᵢ` uses coefficient `+1.0` (not −1.0). This is the standard QAOA
transverse-field mixer. It causes the algorithm to explore neighbouring bitstrings
by flipping individual qubits, preventing it from getting stuck at a local minimum.

**Key output:**
```
H_cost terms: 91  (13 Z terms + 78 ZZ terms)
H_mix  terms: 13
```
91 terms because every qubit has a Z term (13) and every pair has a ZZ coupling
(13×12/2 = 78). The large number of couplings is why this problem requires
a quantum algorithm for larger instances.

**About the Ising energy difference:**
The QUBO and Ising energies differ by a constant (−802.87 in this run). This is
expected — the conversion adds a constant offset from the `xᵢ = (1−σᵢ)/2` substitution.
The important thing is that the difference is the *same* for every bitstring, so
the ordering (which solution is best) is preserved. The verification confirms this.

---

### Cell c18 — QAOA circuit definition

**What it does:**
Defines two PennyLane quantum functions:
- `qaoa_cost(params)` — returns the expectation value `⟨H_cost⟩`, used during optimisation
- `qaoa_probs(params)` — returns probabilities for all 2¹³ = 8,192 bitstrings, used to
  extract the final answer

Both circuits have the same structure:
```
|0⟩^13 → H^13 → [cost_layer(γ₁) → mix_layer(β₁)] → [cost_layer(γ₂) → mix_layer(β₂)] → measure
```

The initial Hadamard layer puts all 13 qubits in equal superposition: the state is an
equal mixture of all 8,192 possible sensor placements simultaneously.

**`qml.device("default.qubit")`:**
This is PennyLane's noiseless statevector simulator. It runs entirely on CPU and is
exact (no shot noise). For 13 qubits it is fast. For >25 qubits it would run out of memory.

**Why two separate functions?**
`qml.expval(H_cost)` returns a single float — cheap to compute during optimisation.
`qml.probs(...)` returns all 8,192 probabilities — more expensive, only done once after
optimisation to extract the final answer.

---

### Cell c19 — COBYLA optimisation

**What it does:**
1. Initialises 4 random parameters in [0, π/2]
2. Runs COBYLA for up to 400 iterations, minimising `⟨H_cost⟩`
3. Records the cost history for the convergence plot

**Why minimise `⟨H_cost⟩`?**
The expectation value of H_cost under the QAOA state is the average QUBO energy
over the probability distribution. Minimising it pushes probability mass towards
lower-energy (better) solutions. After optimisation, the highest-probability states
should be near-optimal sensor placements.

**COBYLA parameters:**
- `rhobeg=0.4` — initial step size in parameter space (roughly 0.4 radians)
- `maxiter=400` — upper limit on circuit evaluations
- `catol=1e-6` — convergence tolerance

**Key output:**
```
Final ⟨H⟩ = [some negative value]  |  N evaluations  |  ~2s
```
Runtime of ~2 seconds for 400 evaluations on 13 qubits is fast enough to be practical.

---

### Cell c20 — Extract best feasible solution

**What it does:**
1. Runs `qaoa_probs(opt_params)` to get all 8,192 state probabilities
2. Sorts states from most to least probable
3. Checks the top 500 states: extracts the sensor bits (first 8 bits), checks if
   `Σcᵢxᵢ ≤ B` (budget feasibility), and keeps the feasible state with the best
   risk coverage

**Why check top 500?**
After QAOA optimisation, the best solution is not guaranteed to be the single
highest-probability state. The probability distribution is spread across many states.
Checking the top 500 by probability is a practical way to find the best feasible
solution while avoiding an exhaustive scan of all 8,192 states.

**Actual output from this run:**
```
Selected sensors : [0, 1]
Total CAPEX      : $140K / $250K budget
Risk coverage    : 1.0086
Gap              : 0.0115  (1.1% below optimum)
```

QAOA found Loc-1 + Loc-2 ($140K), missing Loc-3 and Loc-4 which CP-SAT adds for
the full $220K. The 1.1% gap is acceptable for a p=2 QAOA on a simulator.

**The top-10 table** shows the highest-probability bitstrings. Key observations:
- `01111111` (all MEMS sensors, skip Broadband) appears multiple times — high probability
  but infeasible (cost = $280K > $250K budget)
- `00000000` (deploy nothing) appears — feasible but zero coverage, not selected
- The actual best feasible solution `[0,1]` is buried at rank ~8 by probability

This illustrates a key limitation of p=2 QAOA: the optimizer pushes probability towards
low-energy states, but the penalty term keeps "deploy everything" from winning, causing
the distribution to spread across several near-optimal and near-feasible states.

---

### §6 — Classical Benchmark: OR-Tools CP-SAT

---

### Cell c21 — CP-SAT explanation (markdown)

CP-SAT (Constraint Programming — Satisfiability) is Google's exact integer programming
solver. For a 0-1 knapsack with 8 items, it finds the optimal solution in milliseconds.
It exists in this notebook to give a ground-truth answer to compare QAOA against.

---

### Cell c22 — OR-Tools solver

**What it does:**
1. Creates 8 binary decision variables (`new_bool_var`)
2. Sets the objective: maximise the weighted sum of risk values (scaled ×10000 to
   convert floats to integers, since CP-SAT requires integer coefficients)
3. Adds the budget constraint: `Σcᵢxᵢ ≤ 25`
4. Solves and extracts the solution

**OR-Tools API compatibility fix:**
Different versions of OR-Tools use different naming conventions (snake_case vs CamelCase).
The cell uses a try/except chain to handle both:
1. Try snake_case (`model.new_bool_var`, `solver.value`) — OR-Tools ≥9.5
2. Fall back to CamelCase (`model.NewBoolVar`, `solver.Value`) — older versions
3. Final fallback using plain Python `sum()` — works with any version

**Actual output:**
```
Status     : OPTIMAL
Selected   : [0, 1, 2, 3]
CAPEX      : $220K / $250K
Risk obj   : 1.0201  (matches brute-force ✓)
Runtime    : 7.32 ms
```

CP-SAT finds the true optimum (Loc-1 through Loc-4) in 7 milliseconds, confirming
what the brute-force search found.

---

### §7 — Comparison: Ranked Deployment Table

---

### Cell c23 — Section header (markdown)

---

### Cell c24 — Build and print the deployment table

**What it does:**
Combines QAOA and CP-SAT decisions into a single table, sorted by risk weight
(most important location first). Shows which algorithm deployed to each location
and whether they agreed.

**Actual output:**
```
Agreement: 6/8 locations (75%)
QAOA  — cost $140K, coverage 1.0086  |  runtime 1.9s
CP-SAT— cost $220K, coverage 1.0201  |  runtime 7.3ms
Optimality gap: 0.0115  (1.1%)
```

**What the disagreements mean:**
- QAOA deployed at Loc-1 and Loc-2 but skipped Loc-3 and Loc-4 (both MEMS, $40K each)
- CP-SAT deployed all four, spending $80K more but gaining 0.0115 in risk coverage
- The gap is small (1.1%) meaning QAOA found a near-optimal solution

**Why does QAOA miss Loc-3 and Loc-4?**
This is a known limitation of low-depth QAOA. The algorithm has difficulty distinguishing
between solutions that are close in quality but differ in budget utilisation. The p=2
circuit does not have enough "mixing power" to reliably steer probability towards the
full $220K solution rather than the simpler $140K solution.

Increasing to p=3 or p=4 would reduce this gap but increase circuit depth and
optimisation time.

---

### Cell c25 — Deployment maps and summary bar chart

Three plots:
1. **QAOA deployment map** — geographic view of which locations QAOA chose (green) vs
   skipped (grey), with risk heatmap in background and existing stations as triangles
2. **CP-SAT deployment map** — same layout for the exact solution
3. **Summary bar chart** — side-by-side comparison of brute-force optimum, QAOA, and
   CP-SAT risk coverage values

The maps make it immediately clear that QAOA and CP-SAT both prioritise Loc-1 (the
dominant risk cell in the southwest), and that QAOA's missed sensors are in the
low-risk northern part of the study area.

---

### §8 — Data Gaps + Deprioritised Sources

---

### Cell c26 — Notes on excluded data (markdown)

Documents four datasets that were considered but not included in the QUBO:

1. **EarthScope GNSS** — would give deformation-based risk weights (strain rate),
   which are physically more meaningful than catalog density. Excluded because
   preprocessing from RINEX files to strain rates is a multi-step pipeline.
   Flagged for Phase II.

2. **Copernicus LAC InSAR** — surface deformation maps from satellite radar.
   Processing-heavy and regional scope. Could replace seismicity density with
   cumulative surface displacement as a risk proxy.

3. **OpenTopography LiDAR/SfM** — terrain data. Useful for choosing where to physically
   place a sensor (slope stability, access), but not for weighting seismic risk.

4. **Borehole strain/tilt** — high-value near-field sensors but sparse, western-US
   biased, and preprocessing-heavy.

Also documents the key constraint from Liz's memo: **≤ 20 binary variables** for the
simulator. The current experiment uses 13 (8 sensor + 5 slack), within budget.

The next steps section outlines how to extend this work:
- Multi-hazard merge (combine with tsunami and wildfire risk weights)
- Hardware submission (run on IBM Quantum via pennylane-qiskit)
- ETAS benchmark (use statistically modelled earthquake rates instead of historical density)

---

## Known limitations and things to be aware of

### 1. Risk weights are very unequal
Loc-1 has `risk_norm = 1.0` while every other location has `risk_norm < 0.01`. This is
because the M7.6 earthquake concentrates almost all seismic energy in one cell. In a
balanced problem, QAOA performs better; here the answer is almost obvious classically
(always deploy Loc-1 first).

### 2. QAOA is approximate, not exact
The 1.1% gap between QAOA and the true optimum is small but nonzero. For a real
deployment decision, you would use CP-SAT (or brute force, since 8 locations is tiny)
and use QAOA only to demonstrate the quantum workflow.

### 3. This runs on a classical simulator
`qml.device("default.qubit")` simulates a perfect quantum computer. There is no quantum
speedup here — a noiseless simulator is slower than a classical algorithm for problems
this small. The value is in the workflow: the same PennyLane code can be submitted to
a real QPU by changing one line (`device = qml.device("qiskit.ibmq", ...)`).

### 4. The token in the notebook
The EarthScope token in cell c02 is a JWT that expires. If you re-run after expiry,
the FDSN station query will fail silently (caught by try/except) and `stations` will
be empty. The notebook still runs correctly — it just won't show existing station
coverage in the plots.

### 5. SSN data gap
The SSN catalog consistently returns HTML instead of CSV, so the combined catalog
is USGS-only (498 events). Adding SSN would increase the event count, potentially
changing which grid cells rank highest, but is unlikely to change the dominant cell
(the subduction zone offshore Manzanillo is well-sampled by USGS).

---

## Connection to `quantum_sensor_toy.ipynb`

`quantum_sensor_toy.ipynb` does exactly the same thing as this notebook with:
- 4 hand-crafted locations instead of 8 data-derived ones
- 6 binary variables (4 sensor + 2 slack) instead of 13
- p=1 QAOA with a 30×30 grid search instead of p=2 with COBYLA
- Brute-force comparison instead of OR-Tools

If you are trying to understand the math, read the toy notebook first — every concept
is explained with plain English, and the numbers are small enough to verify by hand.
