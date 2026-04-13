"""Standalone DQI scaling benchmark."""
from manzanillo_qc.instance import fetch_instance
from manzanillo_qc.qubo import build_qubo
from manzanillo_qc.ising import qubo_to_ising
from manzanillo_qc.dqi import run_dqi
import numpy as np

QUBIT_COUNTS = [4, 6, 8, 10, 12]
BUDGET_10K   = 80

print(f"{'n':>4}  {'clauses':>8}  {'qubits':>7}  {'time':>8}  {'obj':>8}  {'gap':>6}")
print("-" * 55)

for n in QUBIT_COUNTS:
    cfg     = fetch_instance(n_sites=n, budget_10k=BUDGET_10K)
    Q, _    = build_qubo(cfg)
    h, J    = qubo_to_ising(Q)

    # brute-force optimal for gap%
    from manzanillo_qc.qubo import brute_force
    opt = brute_force(cfg, Q)
    optimal = opt["obj"] if opt["obj"] not in (None, -np.inf) else 0.0

    result  = run_dqi(cfg, h, J)
    obj     = result["best_obj"] if result["best_obj"] not in (None, -np.inf) else 0.0
    gap     = 100 * (optimal - obj) / optimal if optimal > 0 else 100.0
    t_s     = result["time_ms"] / 1000
    m       = result["meta"]["m_clauses"]
    tq      = result["meta"]["total_qubits"]

    print(f"{n:>4}  {m:>8}  {tq:>7}  {t_s:>7.1f}s  {obj:>8.4f}  {gap:>5.1f}%")
