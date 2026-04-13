# Quantum Sensor Placement — Manzanillo Seismic Network

Internship project exploring quantum optimisation for seismic sensor network
design in the Manzanillo / Colima region of Mexico.

Two things live here:
- **manzanillo-qc** — quantum/classical solver benchmark package
- **data catalog** — QA notebook validating 35 geophysical datasets

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Thiha3013/uruquai.git
cd uruquai
```

### 2. Install dependencies

Python 3.10+ is required.

```bash
cd manzanillo-qc
pip install -r requirements.txt
```

To also install the package itself (optional — not needed for running via PYTHONPATH):

```bash
pip install -e .
```

### 3. API keys (optional — only needed for the data catalog notebook)

Create a file called `keys.py` in the root directory:

```python
# keys.py — do NOT commit this file
EARTHSCOPE_TOKEN = "your_token_here"   # from login.earthscope.org
OPENTOPO_API_KEY = "your_key_here"     # from portal.opentopography.org
```

If you skip this step, the data catalog notebook will still run — datasets
that require authentication will be marked `FILE_AUTH / MANUAL` instead of
validated.

The manzanillo-qc benchmark does **not** need any API keys (it uses the public
USGS and IRIS FDSN services).

---

## Running the quantum benchmark

All commands below should be run from inside `manzanillo-qc/`:

```bash
cd manzanillo-qc
```

**Scalability benchmark** — runs all solvers across n = 4, 6, 8, 10, 12, 14
qubits on live USGS/FDSN data and saves plots to `scaling_plots/`:

```bash
PYTHONPATH=src python3 -m manzanillo_qc.scaling
```

**Single full benchmark** at a fixed qubit count and budget:

```bash
PYTHONPATH=src python3 -m manzanillo_qc.cli --budget 35 --p-layers 5 --steps 400 --pulser
```

**Faster test run** (skip Pulser and Pasqal):

```bash
PYTHONPATH=src python3 -m manzanillo_qc.scaling --no-pulser --no-pasqal
```

**Run in background** (useful for long scaling runs):

```bash
nohup PYTHONPATH=src python3 -m manzanillo_qc.scaling > solver.log 2>&1 &
tail -f solver.log   # follow progress
```

Plots are saved to `manzanillo-qc/scaling_plots/`. Pass `--no-plots` to skip them.

See [manzanillo-qc/README.md](manzanillo-qc/README.md) for full documentation
on solvers, flags, and problem formulation.

---

## Data catalog

Open `data_catalog_QA_notebook.ipynb` in Jupyter and run all cells.

`data_catalog_v1.1.csv` is the underlying dataset index (35 entries) that the
notebook reads and validates.

---

## Project structure

```
QA_notebook/
├── README.md                        this file
├── keys.py                          API keys — not committed
├── data_catalog_v1.1.csv            dataset index (35 entries)
├── data_catalog_QA_notebook.ipynb   QA validation notebook
└── manzanillo-qc/                   quantum solver benchmark package
    ├── README.md                    full package documentation
    ├── run_dqi.py                   standalone DQI benchmark
    └── src/manzanillo_qc/           source code
```
