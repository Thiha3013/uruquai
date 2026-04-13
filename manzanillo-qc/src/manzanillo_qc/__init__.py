"""manzanillo_qc — Quantum sensor placement for the Manzanillo seismic region.

Package layout
--------------
config.py      – Pydantic data models (SiteCandidate, AppConfig) + YAML loader
instance.py    – Live data pipeline: USGS catalog + FDSN stations → SiteCandidates
               – Land mask via cartopy 1:10m; excludes offshore and existing-station cells
qubo.py        – QUBO matrix builder + build_qubo_sampling() for sampling solvers
ising.py       – QUBO → Ising (h, J) conversion + PennyLane Hamiltonian factory
qaoa.py        – PennyLane QAOA (AdamOptimizer, INTERP warm-start, grad norm tracking)
pulser_qaoa.py – Pulser analog QAOA (Rydberg layout + T-sweep + greedy repair)
pasqal_qubo.py – Pasqal qubo-solver (neutral-atom emulator, calibrated λ, greedy repair)
rqaoa.py       – Recursive QAOA (p=1 per level, 3 seeds, brute-force at n≤2)
dqi.py         – Decoded Quantum Interferometry (Jordan et al., Nature 2025)
anneal.py      – D-Wave simulated annealing (CPU, no cloud credentials)
backends.py    – PennyLane device factory
utility.py     – Hazard-scenario utility builder (replaces seismicity density weights)
benchmarks.py  – Greedy + MILP classical baselines + sensor map plots (cartopy)
scaling.py     – Scalability benchmark (live data, n=4–12, all solvers)
cli.py         – Command-line entry-point (single-run mode)
"""
__version__ = "0.3.0"
__all__ = ["config", "instance", "qubo", "ising", "qaoa", "pulser_qaoa", "pasqal_qubo",
           "rqaoa", "dqi", "anneal", "backends", "cli", "utility", "benchmarks", "scaling"]
