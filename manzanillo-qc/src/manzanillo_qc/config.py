"""Pydantic data models for the sensor-placement problem.

SiteCandidate
    One candidate deployment location with risk weight and CAPEX.

AppConfig
    Full run configuration: sites + QAOA hyper-parameters + budget.
    Can be loaded from a YAML file via ``AppConfig.from_yaml(path)``.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import yaml
from pydantic import BaseModel, Field


_DETECTION_RADII_KM: dict[str, float] = {
    "Broadband":    150.0,   # can detect distant M3+ events
    "Short-period":  75.0,
    "MEMS":          30.0,
}

_DETECTION_THRESHOLDS_G: dict[str, float] = {
    "Broadband":    0.001,   # very sensitive
    "Short-period": 0.005,
    "MEMS":         0.020,
}


class SiteCandidate(BaseModel):
    name: str
    lat: float
    lon: float
    risk_weight: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalised energy-weighted seismicity risk ∈ [0, 1]",
    )
    capex_10k: int = Field(
        ..., gt=0,
        description="Deployment cost in $10 K units (e.g. 10 → $100 K broadband)",
    )
    sensor_type: str = "MEMS"
    greenfield: bool = True

    def detection_radius_km(self) -> float:
        """Detection radius in km based on sensor type."""
        return _DETECTION_RADII_KM.get(self.sensor_type, 30.0)

    def detection_threshold_g(self) -> float:
        """Minimum detectable PGA in g based on sensor type."""
        return _DETECTION_THRESHOLDS_G.get(self.sensor_type, 0.020)


class AppConfig(BaseModel):
    budget_10k: int = Field(25, description="Total budget in $10 K units ($250 K default)")
    penalty_lambda: Optional[float] = Field(
        None,
        description="QUBO penalty strength λ; None → auto = 1 + Σwᵢ",
    )
    p_layers: int = Field(2, description="QAOA circuit depth p")
    n_steps: int = Field(100, description="Adam optimiser iterations")
    stepsize: float = Field(0.01, description="Adam learning rate")
    backend: str = Field("lightning.qubit", description="PennyLane device string")
    optimizer: str = Field("adam", description="Optimizer: 'adam' or 'neldermead'")
    sites: List[SiteCandidate] = Field(default_factory=list)

    # ── YAML loader ────────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    # ── Convenience properties ─────────────────────────────────────────────────
    @property
    def n_sites(self) -> int:
        return len(self.sites)

    @property
    def weights(self) -> np.ndarray:
        return np.array([s.risk_weight for s in self.sites], dtype=float)

    @property
    def costs(self) -> np.ndarray:
        return np.array([s.capex_10k for s in self.sites], dtype=float)

    @property
    def effective_lambda(self) -> float:
        """Return penalty strength; auto-computes if not set explicitly."""
        if self.penalty_lambda is not None:
            return self.penalty_lambda
        return 1.0 + float(self.weights.sum())
