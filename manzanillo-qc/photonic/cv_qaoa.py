"""Continuous-Variable QAOA for sensor placement — Photonic experiment.

STATUS: Experimental / legacy-plugin dependent.
Run in a SEPARATE virtual environment (see requirements.txt).

Why photonic?
-------------
Qubit-based QAOA uses discrete (spin-1/2) quantum systems.  Photonic
quantum computers use *continuous-variable* (CV) systems — modes of light
described by amplitude and phase quadratures (x̂, p̂).  In principle, CV
hardware can run at room temperature and scale to many modes via time-domain
multiplexing.

How this works
--------------
The binary sensor-placement QUBO is mapped to a CV system as follows:

1.  Encode problem biases as displacement amplitudes α_i = h_i / scale
    (h_i are the Ising local fields from qubo_to_ising).

2.  Encode pairwise couplings as two-mode squeezing between modes i and j
    with coupling strength r_ij ∝ J_ij.

3.  Apply a variational layer of single-mode rotations (φ_i) — the CV
    analogue of the mixer layer in qubit QAOA.

4.  Measure each mode in the Fock basis.  Threshold the measurement:
        n_i ≥ threshold → sensor deployed (xᵢ = 1)
        n_i <  threshold → skip         (xᵢ = 0)

5.  Optimise rotation angles φ using Adam to minimise the expected
    QUBO energy evaluated on the thresholded measurements (Monte Carlo).

Limitations
-----------
- Threshold measurement introduces stochasticity; multiple shots are needed.
- Two-mode squeezing is approximate for large J_ij values.
- This is a proof-of-concept; real photonic hardware would use TDM loops.
- Requires strawberryfields >= 0.23 which conflicts with pennylane >= 0.36.

Usage
-----
    # From photonic/ directory, with the photonic venv activated:
    python cv_qaoa.py --n-shots 200 --n-steps 50

    # Or import as a module:
    from cv_qaoa import run_cv_qaoa
    result = run_cv_qaoa(h, J, n_shots=200, n_steps=50)
"""
from __future__ import annotations

import argparse
import sys
import numpy as np

try:
    import strawberryfields as sf
    from strawberryfields.ops import Dgate, S2gate, Rgate, MeasureFock
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False


# ── Fallback: classical Monte Carlo simulation ─────────────────────────────────

def _classical_fallback(
    h: np.ndarray,
    J: np.ndarray,
    n_shots: int = 200,
    seed: int = 42,
) -> dict:
    """Classical Monte Carlo fallback when strawberryfields is not installed.

    Samples random binary strings, weights by Boltzmann factor exp(-E/T),
    and returns the lowest-energy feasible sample found.  Not quantum —
    purely illustrative of the structure.
    """
    rng = np.random.default_rng(seed)
    n   = len(h)
    T   = 1.0      # temperature (lower = greedier)

    best = {"x": None, "energy": np.inf}
    for _ in range(n_shots):
        sigma = rng.choice([-1.0, 1.0], size=n)
        E     = float(h @ sigma + sum(J[i, j] * sigma[i] * sigma[j]
                                      for i in range(n) for j in range(i + 1, n)))
        if E < best["energy"]:
            best = {"x": ((1 - sigma) / 2).astype(float), "energy": E}

    return best


# ── CV-QAOA circuit ────────────────────────────────────────────────────────────

def _pga_from_ising(
    h: np.ndarray,
    J: np.ndarray,
    phi: np.ndarray,
    scale: float,
    threshold: int,
    n_shots: int,
) -> np.ndarray:
    """Run one CV circuit sample and return binary placement vector.

    Each mode i gets:
      - Displacement α_i = h_i / scale
      - Two-mode squeezing with all j: r_ij = J_ij / scale (clipped)
      - Rotation by phi_i
      - Fock measurement, threshold → binary
    """
    n = len(h)
    prog = sf.Program(n)
    eng  = sf.Engine("fock", backend_options={"cutoff_dim": 6})

    with prog.context as q:
        # Displacement — encode local Ising fields
        for i in range(n):
            Dgate(float(h[i] / scale)) | q[i]

        # Two-mode squeezing — encode pairwise couplings
        for i in range(n):
            for j in range(i + 1, n):
                r = float(np.clip(abs(J[i, j]) / scale, 0, 1.0))
                if r > 1e-4:
                    S2gate(r) | (q[i], q[j])

        # Rotation mixer — variational layer
        for i in range(n):
            Rgate(float(phi[i])) | q[i]

        # Measure in Fock basis
        MeasureFock() | q

    result = eng.run(prog)
    samples = result.samples[0]    # shape (n,) integer photon numbers
    return (np.array(samples) >= threshold).astype(float)


def run_cv_qaoa(
    h: np.ndarray,
    J: np.ndarray,
    n_shots: int = 200,
    n_steps: int = 50,
    lr: float = 0.05,
    scale: float = 5.0,
    threshold: int = 1,
    seed: int = 42,
) -> dict:
    """Run CV-QAOA and return the best binary placement vector found.

    Parameters
    ----------
    h : np.ndarray
        Ising local fields (from ising.qubo_to_ising).
    J : np.ndarray
        Ising couplings (upper triangle).
    n_shots : int
        Circuit samples per optimisation step (Monte Carlo expectation).
    n_steps : int
        Optimisation iterations.
    lr : float
        Adam learning rate for rotation angles.
    scale : float
        Normalisation factor for displacements (h_i / scale).
    threshold : int
        Fock measurement threshold: n_photons >= threshold → deploy.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: x, energy, phi (optimised angles), method.
    """
    if not _SF_AVAILABLE:
        print("WARNING: strawberryfields not installed.  "
              "Running classical Monte Carlo fallback.\n"
              "Install with: pip install strawberryfields>=0.23", file=sys.stderr)
        res = _classical_fallback(h, J, n_shots=n_shots, seed=seed)
        res["method"] = "classical_mc_fallback"
        return res

    rng = np.random.default_rng(seed)
    n   = len(h)
    phi = rng.uniform(0, 2 * np.pi, size=n)  # variational angles

    # Simple Adam state
    m  = np.zeros(n)
    v  = np.zeros(n)
    b1, b2, eps = 0.9, 0.999, 1e-8

    best = {"x": None, "energy": np.inf, "phi": phi.copy()}

    def _ising_energy(x: np.ndarray) -> float:
        sigma = 1 - 2 * x
        return float(h @ sigma + sum(J[i, j] * sigma[i] * sigma[j]
                                     for i in range(n) for j in range(i + 1, n)))

    def _expected_energy(phi_: np.ndarray) -> float:
        """Monte Carlo estimate of expected Ising energy over n_shots samples."""
        energies = []
        for _ in range(n_shots):
            x = _pga_from_ising(h, J, phi_, scale, threshold, n_shots=1)
            energies.append(_ising_energy(x))
        return float(np.mean(energies))

    def _finite_diff_grad(phi_: np.ndarray, delta: float = 0.05) -> np.ndarray:
        grad = np.zeros(n)
        e0   = _expected_energy(phi_)
        for i in range(n):
            phi_p    = phi_.copy(); phi_p[i] += delta
            grad[i]  = (_expected_energy(phi_p) - e0) / delta
        return grad

    print(f"CV-QAOA: {n} modes, {n_shots} shots/step, {n_steps} steps")
    for step in range(1, n_steps + 1):
        grad = _finite_diff_grad(phi)

        # Adam update
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        m_hat = m / (1 - b1 ** step)
        v_hat = v / (1 - b2 ** step)
        phi  -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Track best sample
        for _ in range(n_shots):
            x = _pga_from_ising(h, J, phi, scale, threshold, n_shots=1)
            e = _ising_energy(x)
            if e < best["energy"]:
                best = {"x": x.copy(), "energy": e, "phi": phi.copy()}

        if step % 10 == 0:
            print(f"  step {step:3d}  best energy = {best['energy']:.4f}")

    best["method"] = "cv_qaoa_sf"
    return best


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CV-QAOA photonic sensor placement experiment",
    )
    parser.add_argument("--config", "-c", metavar="YAML",
                        default="../examples/config_small.yaml",
                        help="Path to AppConfig YAML (default: ../examples/config_small.yaml)")
    parser.add_argument("--n-shots", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    # Add package to path
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

    from manzanillo_qc.config import AppConfig
    from manzanillo_qc.qubo import build_qubo
    from manzanillo_qc.ising import qubo_to_ising

    cfg   = AppConfig.from_yaml(args.config)
    Q, _  = build_qubo(cfg)
    h, J  = qubo_to_ising(Q)

    result = run_cv_qaoa(h, J, n_shots=args.n_shots, n_steps=args.n_steps, lr=args.lr)

    print(f"\n=== CV-QAOA result ===")
    print(f"Method  : {result['method']}")
    print(f"Energy  : {result['energy']:.4f}")
    if result.get("x") is not None:
        import numpy as np
        w = cfg.weights
        c = cfg.costs
        x = result["x"]
        print(f"Sensors : {list(np.where(x)[0])}")
        print(f"Coverage: {float(w @ x):.4f}")
        print(f"CAPEX   : ${float(c @ x) * 10:.0f}K / ${cfg.budget_10k * 10}K")


if __name__ == "__main__":
    main()
