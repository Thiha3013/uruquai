"""Live data pipeline: USGS catalog + FDSN station inventory → AppConfig.

Study region
------------
18.3–20.8°N, 105.7–102.7°W (Manzanillo / Colima, Mexico).
USGS M≥2.0 events 2005–2025.

Pipeline
--------
1. Fetch USGS earthquake catalog (paginated FDSN/GeoJSON).
2. Fetch existing FDSN stations in the study area (IRIS).
3. Bin catalog into a 10×10 grid; compute log-energy-weighted risk per cell.
4. Filter offshore cells using cartopy 1:10m land mask (shapely point-in-polygon).
5. Exclude cells that already contain an existing FDSN station (same grid cell).
6. Select top-risk + low-risk greenfield cells as candidate sites.
7. Assign sensor type (Broadband/Short-period/MEMS) and CAPEX by risk tier.
8. Return an AppConfig ready to pass to build_qubo().

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
MINLAT, MAXLAT =  18.3,  20.8   # Manzanillo/Colima — shifted slightly north
MINLON, MAXLON = -105.7, -102.7  # shifted slightly east for more land coverage
START,  END    = "2005-01-01", "2025-01-01"
MINMAG         =  2.0
API_LIMIT      = 19_000

USGS_BASE    = "https://earthquake.usgs.gov/fdsnws/event/1/query"
FDSN_STATION = "https://service.iris.edu/fdsnws/station/1/query"

N_LAT, N_LON = 10, 10  # finer grid: 100 cells vs 64, more candidate diversity
N_TOP_RISK   = 8
N_LOW_RISK   = 4
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
    # Log-normalise: raw energy spans ~10 orders of magnitude (M2→M7.5),
    # so one large event crushes all smaller cells in linear space.
    # log10(energy+1) compresses the range to ~3–11, giving meaningful
    # risk spread across cells — standard practice in seismic hazard.
    grid["log_energy"] = np.log10(grid["energy"] + 1)
    max_log = grid["log_energy"].max()
    grid["risk_norm"] = (grid["log_energy"] / max_log).round(4) if max_log > 0 else 0.0

    # Flag cells that already have an existing station
    grid["has_station"] = False
    if len(stations):
        for _, st in stations.iterrows():
            ii = int(np.searchsorted(lat_edges[1:], st.lat))
            jj = int(np.searchsorted(lon_edges[1:], st.lon))
            if 0 <= ii < N_LAT and 0 <= jj < N_LON:
                grid.loc[(grid.lat_bin == ii) & (grid.lon_bin == jj), "has_station"] = True

    return grid


def _is_on_land(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Return boolean array: True if point is on land (uses cartopy 1:10m)."""
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import Point
        import shapely.ops
        land_shp = shpreader.natural_earth(resolution="10m", category="physical", name="land")
        land_geom = shapely.ops.unary_union(list(shpreader.Reader(land_shp).geometries()))
        return np.array([land_geom.contains(Point(lon, lat)) for lat, lon in zip(lats, lons)])
    except Exception:
        return np.ones(len(lats), dtype=bool)


def _select_candidates(grid: pd.DataFrame,
                       n_top: int = N_TOP_RISK,
                       n_low: int = N_LOW_RISK) -> pd.DataFrame:
    """Pick n_top high-risk + n_low low-risk (but active) cells.

    Cells that already have an existing FDSN station are excluded — no point
    proposing a new sensor right on top of one that already exists.
    Offshore cells are excluded via land mask.
    """
    # Filter out offshore cells
    on_land = _is_on_land(grid["lat_c"].values, grid["lon_c"].values)
    grid = grid[on_land].copy()

    available = grid[~grid["has_station"]]   # only greenfield cells
    if n_top + n_low > len(available):
        import warnings
        warnings.warn(
            f"Requested {n_top + n_low} candidates but only {len(available)} "
            f"greenfield cells available in the {N_LAT}×{N_LON} grid. "
            f"Increase N_LAT/N_LON or expand the study region.",
            RuntimeWarning, stacklevel=3,
        )
    n_top = min(n_top, len(available))
    top   = available.nlargest(n_top, "risk_norm").copy()
    rest  = available[~available.index.isin(top.index)]
    n_low = min(n_low, len(rest))
    if n_low > 0:
        low   = rest.nsmallest(n_low, "risk_norm").nlargest(n_low, "n_events").copy()
        cands = pd.concat([top, low], ignore_index=True)
    else:
        cands = top.copy()

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


def fetch_instance(budget_10k: int = DEFAULT_BUDGET,
                   n_sites: int = 12,
                   **kwargs) -> AppConfig:
    """Fetch live data and return a fully populated AppConfig.

    Parameters
    ----------
    budget_10k : int
        Total deployment budget in $10 K units (default 25 → $250 K).
    n_sites : int
        Number of candidate sites to generate (default 12).
        Roughly 2/3 high-risk + 1/3 low-risk cells are selected.
        Range 4–14 is well-supported by the 6×6 live grid.
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

    n_low = max(1, n_sites // 3)
    n_top = n_sites - n_low
    grid  = _build_risk_grid(catalog, stations)
    cands = _select_candidates(grid, n_top=n_top, n_low=n_low)

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
