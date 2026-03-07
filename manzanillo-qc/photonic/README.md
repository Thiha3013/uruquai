# Photonic Experiments

Continuous-variable (CV) QAOA for sensor placement using Strawberry Fields.

## Why isolated?

`strawberryfields >= 0.23` requires `numpy < 2` and conflicts with
`pennylane >= 0.36` (used in the main package).  Running both in the same
environment causes import errors.  Keep them separate.

## Setup

```bash
python -m venv .venv-photonic
source .venv-photonic/bin/activate        # Windows: .venv-photonic\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# From the photonic/ directory:
python cv_qaoa.py --config ../examples/config_small.yaml --n-shots 100 --n-steps 30
```

If `strawberryfields` is not installed, `cv_qaoa.py` falls back to a
classical Monte Carlo sampler automatically — useful for testing the
interface without the photonic dependencies.

## How it works

| Step | Qubit QAOA | CV-QAOA (photonic) |
|------|------------|-------------------|
| Encode problem | Ising Hamiltonian H_cost | Displacement α_i = h_i / scale |
| Encode couplings | ZZ terms in H_cost | Two-mode squeezing r_ij ∝ J_ij |
| Variational layer | R_x(β) mixer | Single-mode rotation R(φ_i) |
| Measurement | Pauli-Z expectation | Fock measurement, threshold → binary |
| Optimiser | Adam on circuit params | Adam on rotation angles φ |

## Status

**Experimental.**  The CV encoding is approximate for large coupling values.
Real photonic hardware would use time-domain multiplexing (TDM) loops.
This code is a proof-of-concept for the methodology — not production-ready.
