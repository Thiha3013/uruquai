"""Live data pipeline: USGS catalog + FDSN station inventory → AppConfig.

Replicates the data-fetching logic from quantum_exp.ipynb §1–§3 as
importable functions, so the notebook can call ``fetch_instance()`` instead
of embedding the raw API code inline.

Quick start
-----------
>>> from manzanillo_qc.instance import fetch_instance
>>> cfg = fetch_instance()          # ~5 s, hits USGS + IRIS live
>>> print(cfg.n_sites, cfg.sites[0])
"""
from __future__ import annotations

import requests
import numpy as np
import pandas as pd

from .config import AppConfig, SiteCandidate

# ── Study region ───────────────────────────────────────────────────────────────
MINLAT, MAXLAT =  18.0,  20.0
MINLON, MAXLON = -105.0, -100.0
START,  END    = "2005-01-01", "2025-01-01"
MINMAG         =  2.0
API_LIMIT      = 19_000

USGS_BASE    = "https://earthquake.usgs.gov/fdsnws/event/1/query"
FDSN_STATION = "https://service.iris.edu/fdsnws/station/1/query"

N_LAT, N_LON = 4, 5
N_TOP_RISK   = 5
N_LOW_RISK   = 3
DEFAULT_BUDGET = 25   # $250 K


# ── Internal helpers ───────────────────────────────────────────────────────────

def _fetch_usgs() -> pd.DataFrame:
    """Page through the USGS FDSN catalog and return a lat/lon/mag DataFrame."""
    frames, offset = [], 1
    while True:
        r = requests.get(USGS_BASE, params=dict(
            format="geojson",
            minlatitude=MINLAT, maxlatitude=MAXLAT,
            minlongitude=MINLON, maxlongitude=MAXLON,
            starttime=START, endtime=END,
            minmagnitude=MINMAG,
            orderby="time-asc", limit=API_LIMIT, offset=offset,
        ), timeout=60, headers={"User-Agent": "manzanillo-qc/1.0"})
        r.raise_for_status()
        obj = r.json()
        rows = [
            {"lat": f["geometry"]["coordinates"][1],
             "lon": f["geometry"]["coordinates"][0],
             "mag": f["properties"].get("mag")}
            for f in obj.get("features", [])
        ]
        chunk = pd.DataFrame(rows).dropna(subset=["mag"])
        frames.append(chunk)
        if len(chunk) < API_LIMIT:
            break
        offset += API_LIMIT
    return pd.concat(frames, ignore_index=True)


def _fetch_stations() -> pd.DataFrame:
    """Query IRIS FDSN station service; returns empty DataFrame on failure."""
    try:
        r = requests.get(FDSN_STATION, params=dict(
            minlatitude=MINLAT - 1, maxlatitude=MAXLAT + 1,
            minlongitude=MINLON - 1, maxlongitude=MAXLON + 1,
            level="station", format="text",
        ), timeout=30, headers={"User-Agent": "manzanillo-qc/1.0"})
        if r.status_code != 200:
            return pd.DataFrame()
        rows = []
        for ln in r.text.splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) >= 4:
                try:
                    rows.append({"lat": float(parts[2]), "lon": float(parts[3])})
                except ValueError:
                    pass
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df[(df.lat.between(MINLAT, MAXLAT)) & (df.lon.between(MINLON, MAXLON))]
    except Exception:
        return pd.DataFrame()


def _build_risk_grid(
    catalog: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    """Bin catalog events into a N_LAT×N_LON grid; compute energy-weighted risk."""
    lat_edges = np.linspace(MINLAT, MAXLAT, N_LAT + 1)
    lon_edges = np.linspace(MINLON, MAXLON, N_LON + 1)
    lat_centres = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centres = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    rows = []
    for i, lat_c in enumerate(lat_centres):
        for j, lon_c in enumerate(lon_centres):
            mask = (
                (catalog.lat >= lat_edges[i]) & (catalog.lat < lat_edges[i + 1]) &
                (catalog.lon >= lon_edges[j]) & (catalog.lon < lon_edges[j + 1])
            )
            ev     = catalog[mask]
            energy = float(ev.mag.apply(lambda m: 10 ** (1.5 * m)).sum()) if len(ev) else 0.0
            rows.append({
                "lat_c": lat_c, "lon_c": lon_c,
                "lat_bin": i, "lon_bin": j,
                "n_events": len(ev), "energy": energy,
            })

    grid = pd.DataFrame(rows)
    max_energy = grid["energy"].max()
    grid["risk_norm"] = (grid["energy"] / max_energy).round(4) if max_energy > 0 else 0.0

    # Flag cells that already have an existing station
    grid["has_station"] = False
    if len(stations):
        for _, st in stations.iterrows():
            ii = int(np.searchsorted(lat_edges[1:], st.lat))
            jj = int(np.searchsorted(lon_edges[1:], st.lon))
            if 0 <= ii < N_LAT and 0 <= jj < N_LON:
                grid.loc[(grid.lat_bin == ii) & (grid.lon_bin == jj), "has_station"] = True

    return grid


def _select_candidates(grid: pd.DataFrame) -> pd.DataFrame:
    """Pick N_TOP_RISK high-risk + N_LOW_RISK low-risk (but active) cells.

    Cells that already have an existing FDSN station are excluded — no point
    proposing a new sensor right on top of one that already exists.
    """
    available = grid[~grid["has_station"]]   # only greenfield cells
    top  = available.nlargest(N_TOP_RISK, "risk_norm").copy()
    rest = available[~available.index.isin(top.index)]
    low  = rest.nsmallest(N_LOW_RISK, "risk_norm").nlargest(N_LOW_RISK, "n_events").copy()
    cands = pd.concat([top, low], ignore_index=True)

    def _capex(row) -> int:
        if row.risk_norm >= 0.6:   return 10   # broadband  ~$100 K
        elif row.risk_norm >= 0.3: return  7   # short-period ~$70 K
        else:                      return  4   # MEMS ~$40 K

    cands["capex_10k"]   = cands.apply(_capex, axis=1)
    cands["sensor_type"] = cands["capex_10k"].map({10: "Broadband", 7: "Short-period", 4: "MEMS"})
    cands["greenfield"]  = ~cands["has_station"]
    cands["name"] = [
        f"Loc-{i+1} ({r.lat_c:.2f}\u00b0N,{r.lon_c:.2f}\u00b0W)"
        for i, (_, r) in enumerate(cands.iterrows())
    ]
    return cands.reset_index(drop=True)


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_stations() -> pd.DataFrame:
    """Return existing FDSN station positions (lat, lon) in the study area."""
    return _fetch_stations()


def fetch_instance(budget_10k: int = DEFAULT_BUDGET, **kwargs) -> AppConfig:
    """Fetch live data and return a fully populated AppConfig.

    Parameters
    ----------
    budget_10k : int
        Total deployment budget in $10 K units (default 25 → $250 K).
    **kwargs
        Passed through to AppConfig (e.g. ``p_layers=3``, ``n_steps=300``).

    Returns
    -------
    AppConfig
        Ready to pass to :func:`~manzanillo_qc.qubo.build_qubo`.
    """
    print("Fetching USGS earthquake catalog …")
    catalog = _fetch_usgs()
    print(f"  {len(catalog):,} events M\u2265{MINMAG} (2005\u20132025)")

    print("Fetching FDSN station inventory …")
    stations = _fetch_stations()
    print(f"  {len(stations)} existing stations in study area")

    grid  = _build_risk_grid(catalog, stations)
    cands = _select_candidates(grid)

    sites = [
        SiteCandidate(
            name=str(row["name"]),
            lat=float(row.lat_c),
            lon=float(row.lon_c),
            risk_weight=float(row.risk_norm),
            capex_10k=int(row.capex_10k),
            sensor_type=str(row.sensor_type),
            greenfield=bool(row.greenfield),
        )
        for _, row in cands.iterrows()
    ]

    return AppConfig(budget_10k=budget_10k, sites=sites, **kwargs)
