"""Hazard-scenario utility builder.

Instead of using raw energy-weighted seismicity density as risk weights, this
module derives sensor utility from a set of explicit hazard scenarios.

Each scenario describes a realistic earthquake event (source location,
magnitude, importance weight).  For every candidate sensor we compute whether
it can detect that event using a simple ground-motion attenuation model, then
aggregate across scenarios to produce a utility score per site.

Attenuation model (simplified Boore-Atkinson style)
----------------------------------------------------
    log10(PGA_g) = 0.5·M − log10(R_km) − 0.003·R_km − 2.5

Detection
---------
A sensor detects a scenario event if PGA_at_sensor > detection_threshold:
    Broadband    threshold = 0.001 g  (very sensitive)
    Short-period threshold = 0.005 g
    MEMS         threshold = 0.020 g

Utility
-------
    utility_i = Σ_s  p_s · I(PGA(sensor_i, scenario_s) > threshold_i)

Normalised to [0, 1] across all candidates before being written back into
the AppConfig as new risk_weight values.

DEFAULT_SCENARIOS
-----------------
Four scenarios chosen to represent the main fault systems near Manzanillo:

    S1  Thrust interface M7.5   18.0°N, 103.5°W   p=0.40
        Replicates the 1995 Manzanillo event on the subduction interface.

    S2  Inland crustal M5.5     19.5°N, 103.7°W   p=0.25
        Colima volcanic region, shallow crustal seismicity.

    S3  Deep intraslab M6.0     18.5°N, 102.5°W   p=0.20
        Benioff zone event, deeper focal depth (depth_km=60).

    S4  Coastal M6.5            19.0°N, 104.8°W   p=0.15
        Northwest segment of the subduction margin.

Usage
-----
    from manzanillo_qc.utility import build_utility_weights, DEFAULT_SCENARIOS

    cfg_original = AppConfig.from_yaml("examples/config_small.yaml")
    cfg_utility  = build_utility_weights(cfg_original)

    # cfg_utility.sites[i].risk_weight now reflects scenario-based utility
    # The rest of the pipeline (build_qubo, run_qaoa) works unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from .config import AppConfig, SiteCandidate


# ── Hazard scenario dataclass ──────────────────────────────────────────────────

@dataclass
class HazardScenario:
    name:     str
    lat:      float          # source latitude
    lon:      float          # source longitude
    mag:      float          # moment magnitude
    weight:   float          # importance weight (should sum to 1 across scenarios)
    depth_km: float = 15.0   # focal depth (affects PGA via effective distance)


DEFAULT_SCENARIOS: List[HazardScenario] = [
    HazardScenario(
        name="Thrust interface M7.5",
        lat=18.0, lon=-103.5, mag=7.5, weight=0.40, depth_km=20.0,
    ),
    HazardScenario(
        name="Inland crustal M5.5",
        lat=19.5, lon=-103.7, mag=5.5, weight=0.25, depth_km=10.0,
    ),
    HazardScenario(
        name="Deep intraslab M6.0",
        lat=18.5, lon=-102.5, mag=6.0, weight=0.20, depth_km=60.0,
    ),
    HazardScenario(
        name="Coastal M6.5 NW",
        lat=19.0, lon=-104.8, mag=6.5, weight=0.15, depth_km=15.0,
    ),
]


# ── Attenuation + detection ────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _pga_g(mag: float, epi_km: float, depth_km: float) -> float:
    """Estimate PGA in g using simplified attenuation.

    Uses hypocentral distance (combines epicentral + depth).
    Formula: log10(PGA) = 0.5·M − log10(R) − 0.003·R − 2.5
    Clipped at a minimum of 1 km to avoid log(0).
    """
    R = math.sqrt(epi_km ** 2 + depth_km ** 2)
    R = max(R, 1.0)
    log_pga = 0.5 * mag - math.log10(R) - 0.003 * R - 2.5
    return 10 ** log_pga


def _detects(site: SiteCandidate, scenario: HazardScenario) -> bool:
    """Return True if the sensor at `site` can detect `scenario`."""
    epi_km = _haversine_km(site.lat, site.lon, scenario.lat, scenario.lon)
    pga    = _pga_g(scenario.mag, epi_km, scenario.depth_km)
    return pga >= site.detection_threshold_g()


# ── Public API ─────────────────────────────────────────────────────────────────

def scenario_utility(
    site: SiteCandidate,
    scenarios: List[HazardScenario] = DEFAULT_SCENARIOS,
) -> float:
    """Compute raw utility for a single site across all scenarios.

    Returns a value in [0, 1] where 1 means the sensor detects every scenario
    weighted by its importance.
    """
    return sum(s.weight for s in scenarios if _detects(site, s))


def build_utility_weights(
    cfg: AppConfig,
    scenarios: List[HazardScenario] = DEFAULT_SCENARIOS,
) -> AppConfig:
    """Recompute risk_weight for every site using hazard-scenario utility.

    Replaces the raw energy-weighted seismicity density (risk_weight) with
    a scenario-derived utility score normalised to [0, 1].  All other fields
    in AppConfig and each SiteCandidate are left unchanged.

    Parameters
    ----------
    cfg : AppConfig
        Original configuration (risk_weights will be replaced).
    scenarios : list of HazardScenario
        Hazard scenarios to evaluate.  Defaults to DEFAULT_SCENARIOS.

    Returns
    -------
    AppConfig
        New config with updated risk_weight fields.
    """
    raw = np.array([scenario_utility(s, scenarios) for s in cfg.sites])
    max_u = raw.max()
    normalised = (raw / max_u).tolist() if max_u > 0 else raw.tolist()

    new_sites = [
        s.model_copy(update={"risk_weight": float(u)})
        for s, u in zip(cfg.sites, normalised)
    ]
    return cfg.model_copy(update={"sites": new_sites})


def print_scenario_report(
    cfg: AppConfig,
    scenarios: List[HazardScenario] = DEFAULT_SCENARIOS,
) -> None:
    """Print a table showing which sensors detect which scenarios."""
    header = f"{'Site':<30}" + "".join(f"  {s.name[:12]:>12}" for s in scenarios) + "   Utility"
    print(header)
    print("-" * len(header))
    for site in cfg.sites:
        detections = [_detects(site, s) for s in scenarios]
        raw_u = scenario_utility(site, scenarios)
        row = f"{site.name:<30}" + "".join(f"  {'✓':>12}" if d else f"  {'—':>12}" for d in detections)
        row += f"   {raw_u:.3f}"
        print(row)
