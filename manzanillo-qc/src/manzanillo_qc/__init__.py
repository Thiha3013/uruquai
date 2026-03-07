"""manzanillo_qc — Quantum sensor placement for the Manzanillo seismic region.

Package layout
--------------
config.py     – Pydantic data models (SiteCandidate, AppConfig) + YAML loader
instance.py   – Live data pipeline: USGS catalog + FDSN stations → SiteCandidates
qubo.py       – QUBO matrix builder + overlap/redundancy model
ising.py      – QUBO → Ising (h, J) conversion + PennyLane Hamiltonian factory
qaoa.py       – QAOA circuit + AdamOptimizer loop
backends.py   – PennyLane device factory
utility.py    – Hazard-scenario utility builder (replaces seismicity density weights)
benchmarks.py – Classical benchmarks: greedy heuristic + exact MILP (scipy)
cli.py        – Command-line entry-point
"""
__version__ = "0.2.0"
__all__ = ["config", "instance", "qubo", "ising", "qaoa", "backends", "cli",
           "utility", "benchmarks"]
