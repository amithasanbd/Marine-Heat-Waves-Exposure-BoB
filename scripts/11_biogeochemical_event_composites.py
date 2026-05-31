#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11 Biogeochemical Event Composites

Refined for GitHub + Zenodo release supporting the manuscript:
"An Emerging Marine Heatwave Exposure Regime in the Bay of Bengal:
Stratification-Mediated Persistence and Compound Heat-Oxygen Risk".

Notes
-----
- This file was converted from the author's original Google Colab-exported notebook.
- Colab-only shell commands, Google Drive mount calls, and personal Drive paths were
  removed or replaced with repository-relative path variables.
- Configure data locations through environment variables or by editing DATA_ROOT below.
- Large raw NOAA/Copernicus files should not be committed to GitHub; archive processed
  and figure-ready data in Zenodo as described in docs/source_data_access.md.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "scripts" in Path(__file__).parts else Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MHW_DATA_ROOT", PROJECT_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("MHW_OUTPUT_ROOT", PROJECT_ROOT / "outputs"))
for _p in (DATA_ROOT, OUTPUT_ROOT):
    _p.mkdir(parents=True, exist_ok=True)



# %% [markdown]
# #**Biochemical Time Series Multiaxis Analysis**
# 
# Mean MHW intensity covaries with NO₃⁻, PO₄³⁻, Si and DO across the  Bay of Bengal seasonally


# %% Cell 2
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.ticker import MaxNLocator, MultipleLocator


# Paths
mhw_path = str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)/BoB_MHW_Seasonal_1995_2025.xlsx")

nut_base  =  Path (str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)/Biochemical Dataset (1995-2025)"))

nut_files = [
    "nutrients(1995-2000).nc",
    "nutrients(2001-2005).nc",
    "nutrients(2006-2010).nc",
    "nutrients(2011-2015).nc",
    "nutrients(2016-2020).nc",
    "nutrients(2021-2025).nc",
]

# Seasons / mapping

seasons_plot = ["Pre-monsoon", "Monsoon", "Post-monsoon", "Winter"]

season_to_mhw = {
    "Pre-monsoon": "Pre-Monsoon",
    "Monsoon": "Monsoon",
    "Post-monsoon": "Post-Monsoon",
    "Winter": "Winter",
}

season_months = {
    "Pre-monsoon": [3, 4, 5],
    "Monsoon": [6, 7, 8, 9],
    "Post-monsoon": [10, 11],
    "Winter": [1, 2, 12],
}

years_full = np.arange(1995, 2026)

# Regions

regions = {
    "Northern BoB": {"lat_min": 18, "lat_max": 23, "lon_min": 84, "lon_max": 94},
    "Central BoB":  {"lat_min": 12, "lat_max": 18, "lon_min": 80, "lon_max": 100},
    "Southern BoB": {"lat_min": 5,  "lat_max": 12, "lon_min": 80, "lon_max": 100},
}

region_order = ["Northern BoB", "Central BoB", "Southern BoB"]  # now COLUMNS
season_order = ["Pre-monsoon", "Monsoon", "Post-monsoon", "Winter"]  # now ROWS

def region_mask_df(df, rname):
    r = regions[rname]
    if rname == "Northern BoB":
        return (
            (df["lat"] >= r["lat_min"]) & (df["lat"] <= r["lat_max"]) &
            (df["lon"] >= r["lon_min"]) & (df["lon"] <= r["lon_max"])
        )
    else:
        return (
            (df["lat"] >= r["lat_min"]) & (df["lat"] < r["lat_max"]) &
            (df["lon"] >= r["lon_min"]) & (df["lon"] <= r["lon_max"])
        )


# Helpers for NetCDF

def parse_years_from_filename(fname: str):
    m = re.search(r"\((\d{4})-(\d{4})\)", fname)
    if not m:
        raise ValueError(f"Cannot parse years from filename: {fname}")
    return int(m.group(1)), int(m.group(2))

def detect_lat_lon_names(ds):
    lat_candidates = ["latitude", "lat", "y"]
    lon_candidates = ["longitude", "lon", "x"]
    lat_name = next((c for c in lat_candidates if c in ds.coords or c in ds.dims), None)
    lon_name = next((c for c in lon_candidates if c in ds.coords or c in ds.dims), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Could not detect lat/lon names. Coords: {list(ds.coords)}, dims: {list(ds.dims)}")
    return lat_name, lon_name

def ensure_datetime_time(ds, start_year: int):
    if "time" not in ds.coords and "time" not in ds.dims:
        raise ValueError("Dataset has no 'time' dimension/coord.")
    t = ds["time"]
    if np.issubdtype(t.dtype, np.datetime64):
        return ds

    try:
        ds2 = xr.decode_cf(ds)
        if np.issubdtype(ds2["time"].dtype, np.datetime64):
            return ds2
    except Exception:
        pass

    n = int(ds.sizes["time"])
    new_time = pd.date_range(f"{start_year}-01-01", periods=n, freq="MS")
    return ds.assign_coords(time=new_time)

def subset_region_xr(ds, rname, lat_name, lon_name):
    r = regions[rname]
    if rname == "Northern BoB":
        lat_sel = slice(r["lat_min"], r["lat_max"])
    else:
        lat_sel = slice(r["lat_min"], r["lat_max"] - 1e-6)
    lon_sel = slice(r["lon_min"], r["lon_max"])
    return ds.sel({lat_name: lat_sel, lon_name: lon_sel})

def set_axis_limits_nice(ax, y, nbins=5):
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if not np.any(finite):
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
        return

    vmin = float(np.nanmin(y[finite]))
    vmax = float(np.nanmax(y[finite]))

    if vmax > vmin:
        pad = 0.05 * (vmax - vmin)
        ax.set_ylim(max(0.0, vmin - pad), vmax + pad)
    else:
        ax.set_ylim(max(0.0, vmin * 0.9), vmax * 1.1 if vmax != 0 else 1.0)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))


# 1) MHW intensity

df_mhw = pd.read_excel(mhw_path, sheet_name="MHW")
df_mhw = df_mhw[["lon", "lat", "year", "Season", "intensity_mean"]].dropna()
df_mhw["year"] = df_mhw["year"].astype(int)

intensity = {r: {} for r in regions.keys()}
for rname in regions.keys():
    dfr = df_mhw[region_mask_df(df_mhw, rname)].copy()
    for s in seasons_plot:
        smhw = season_to_mhw[s]
        dfs = dfr[dfr["Season"] == smhw]
        s_year = dfs.groupby("year")["intensity_mean"].mean()
        intensity[rname][s] = s_year.reindex(years_full)


# 2) Nutrients (ACTUAL), surface only

nut_vars = ["no3", "po4", "si", "o2"]

actual_series = {v: {r: {} for r in regions.keys()} for v in nut_vars}
for v in nut_vars:
    for rname in regions.keys():
        for s in seasons_plot:
            actual_series[v][rname][s] = pd.Series(index=years_full, dtype=float)

for fname in nut_files:
    fpath = nut_base / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Not found: {fpath}")

    y0, _ = parse_years_from_filename(fname)

    ds = xr.open_dataset(fpath, decode_times=True)
    ds = ensure_datetime_time(ds, start_year=y0)
    lat_name, lon_name = detect_lat_lon_names(ds)

    ds_surf = ds.isel(depth=0) if "depth" in ds.dims else ds

    for rname in regions.keys():
        dsr = subset_region_xr(ds_surf, rname, lat_name, lon_name)

        for s in seasons_plot:
            months = season_months[s]
            dsm = dsr.where(dsr["time"].dt.month.isin(months), drop=True)

            for v in nut_vars:
                if v not in dsm.data_vars:
                    continue

                yearly = (
                    dsm[v]
                    .groupby("time.year")
                    .mean(dim=["time", lat_name, lon_name], skipna=True)
                )

                yrs = yearly["year"].values.astype(int)
                vals = yearly.values.astype(float)

                s_master = actual_series[v][rname][s]
                s_master.loc[yrs] = vals
                actual_series[v][rname][s] = s_master

    ds.close()


# 3) Plot layout: ROWS = seasons (4), COLS = regions (3)

nrows = len(season_order)   # 4
ncols = len(region_order)   # 3

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 14), sharex=True)

line_colors = {"no3": "red", "po4": "green", "si": "orange", "o2": "purple"}
right_labels = {
    "no3": "no3 conc. (mmol/m³)",
    "po4": "po4 conc. (mmol/m³)",
    "si":  "si conc. (mmol/m³)",
    "o2":  "DO conc. (mmol/m³)",
}

# 4 right axes spacing (only on LAST COLUMN)
right_outward = {"no3": 0, "po4": 55, "si": 110, "o2": 165}

legend_handles = None
legend_labels = None

for row_idx, season in enumerate(season_order):
    for col_idx, rname in enumerate(region_order):
        ax = axes[row_idx, col_idx]

        # Bars: MHW intensity
        yvals = intensity[rname][season]
        valid = ~yvals.isna()
        bars = ax.bar(years_full[valid], yvals[valid], color="steelblue", label="Average Mean MHW Intensity")

        ax.set_ylim(0.5, 2.0)
        ax.set_yticks(np.arange(0.5, 2.01, 0.3))
        ax.set_xlim(1994.5, 2025.5)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.xaxis.set_major_locator(MultipleLocator(5))

        # Show LEFT y ticks ONLY first column
        if col_idx != 0:
            ax.tick_params(axis="y", left=False, labelleft=False)

        # Right axes (4 variables)
        ax_no3 = ax.twinx()
        ax_po4 = ax.twinx()
        ax_si  = ax.twinx()
        ax_o2  = ax.twinx()

        for a in (ax_po4, ax_si, ax_o2):
            a.set_frame_on(True)
            a.patch.set_visible(False)

        ax_no3.spines["right"].set_position(("outward", right_outward["no3"]))
        ax_po4.spines["right"].set_position(("outward", right_outward["po4"]))
        ax_si.spines["right"].set_position(("outward", right_outward["si"]))
        ax_o2.spines["right"].set_position(("outward", right_outward["o2"]))

        s_no3 = actual_series["no3"][rname][season].reindex(years_full).values
        s_po4 = actual_series["po4"][rname][season].reindex(years_full).values
        s_si  = actual_series["si"][rname][season].reindex(years_full).values
        s_o2  = actual_series["o2"][rname][season].reindex(years_full).values

        set_axis_limits_nice(ax_no3, s_no3, nbins=5)
        set_axis_limits_nice(ax_po4, s_po4, nbins=5)
        set_axis_limits_nice(ax_si,  s_si,  nbins=5)
        set_axis_limits_nice(ax_o2,  s_o2,  nbins=5)

        ln_no3, = ax_no3.plot(years_full, s_no3, color=line_colors["no3"], marker="o", linewidth=1.4, markersize=3.0, label="NO3")
        ln_po4, = ax_po4.plot(years_full, s_po4, color=line_colors["po4"], marker="o", linewidth=1.4, markersize=3.0, label="PO4")
        ln_si,  = ax_si.plot (years_full, s_si,  color=line_colors["si"],  marker="o", linewidth=1.4, markersize=3.0, label="Si")
        ln_o2,  = ax_o2.plot (years_full, s_o2,  color=line_colors["o2"],  marker="o", linewidth=1.4, markersize=3.0, label="DO")

        # Show RIGHT y ticks/labels ONLY last column
        if col_idx == ncols - 1:
            for a in (ax_no3, ax_po4, ax_si, ax_o2):
                a.tick_params(axis="y", right=True, labelright=True, labelsize=8, pad=2)

            ax_no3.set_ylabel(right_labels["no3"], fontsize=9, fontweight="bold")
            ax_po4.set_ylabel(right_labels["po4"], fontsize=9, fontweight="bold")
            ax_si.set_ylabel(right_labels["si"],  fontsize=9, fontweight="bold")
            ax_o2.set_ylabel(right_labels["o2"],  fontsize=9, fontweight="bold")

            # (Optional fine control) move PO4 label slightly left:
            # ax_po4.yaxis.set_label_coords(1.06, 0.5)

        else:
            for a in (ax_no3, ax_po4, ax_si, ax_o2):
                a.tick_params(axis="y", right=False, labelright=False)
                a.spines["right"].set_visible(False)
                a.set_ylabel("")

        # Column titles = Regions (TOP row only)
        if row_idx == 0:
            ax.set_title(rname, fontsize=12, fontweight="bold")

        # Row labels = Seasons (LEFT side only)
        if col_idx == 0:
            ax.text(
                -0.35, 0.5, season,
                fontsize=12, fontweight="bold",
                ha="right", va="center",
                rotation=90, transform=ax.transAxes
            )

            if row_idx == 1:
                ax.text(
                    -0.25, 0.05, "Average Mean MHW Intensity (°C)",
                    fontsize=12, fontweight="bold",
                    rotation=90, va="center", ha="center",
                    transform=ax.transAxes
                )

        # x labels only bottom row
        if row_idx == nrows - 1:
            ax.set_xlabel("Year", fontsize=10, fontweight="bold")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            ax.tick_params(axis="x", labelbottom=False)

        # Legend once
        if legend_handles is None:
            legend_handles = [bars, ln_no3, ln_po4, ln_si, ln_o2]
            legend_labels = ["Average Mean MHW Intensity", "NO3", "PO4", "Si", "DO"]

# Global legend
fig.legend(legend_handles, legend_labels, loc="upper center", ncol=5, fontsize=10, frameon=False)

# More space between panels (so lines are clearer)
plt.subplots_adjust(
    left=0.10, right=0.80, top=0.92, bottom=0.08,
    wspace=0.18,  # increase if still too close
    hspace=0.18
)


# Save

out_png = str(OUTPUT_ROOT / "MHW_Intensity_Nutrients_DO_1995_2025_1080dpi.png")
plt.savefig(out_png, dpi=1080, bbox_inches="tight", facecolor="white")
plt.show()
print("Saved PNG:", out_png)


# %% [markdown]
# # **MHWs Event-based ecosystem composites for NO₃⁻, PO₄³⁻, Si and DO anomalies.**


# %% Cell 4
# ============================================================
# BAY OF BENGAL BIOGEOCHEMICAL EVENT-COMPOSITE ANALYSIS
# Q1-style Colab workflow
#
# PURPOSE
#   Convert Section 3.6 from descriptive co-variability
#   to event-based inference:
#   MHW vs non-MHW composites for NO3, PO4, Si, DO
#
# OUTPUTS
#   1) Tidy merged dataset
#   2) Composite statistics table
#   3) Publication-ready effect-size figure
#   4) Excel workbook with all sheets
#
# EXPECTED INPUTS
#   - Seasonal MHW Excel workbook (1995–2025)
#   - Biogeochemical NetCDF files (1995–2025)
#
# AUTHOR NOTE
#   Designed for Google Colab with ~12 GB RAM
# ============================================================

# -------------------------
# 1) Install dependencies
# -------------------------
# Colab shell command removed for repository reproducibility: !pip -q install xarray netCDF4 cftime scipy pandas numpy matplotlib openpyxl xlsxwriter dask bottleneck

# -------------------------
# 2) Imports
# -------------------------
import os
import re
import gc
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -------------------------
# 3) Mount Google Drive
# -------------------------
# Google Colab Drive import removed; use config/paths_template.yaml and local data folders.
# Google Drive mount removed; configure DATA_ROOT in this script or via environment variable MHW_DATA_ROOT.

# -------------------------
# 4) USER PATHS
# -------------------------
MHW_XLSX = str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)/BoB_MHW_Seasonal_1995_2025.xlsx")
BIO_DIR  = str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)/Biochemical Dataset (1995-2025)")

OUTDIR = Path(str(DATA_ROOT / "BoB_biogeochem_event_composites_netcdf"))
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_MERGED_CSV  = OUTDIR / "BoB_biogeochem_event_input_merged.csv"
OUT_STATS_CSV   = OUTDIR / "BoB_biogeochem_event_composite_statistics.csv"
OUT_XLSX        = OUTDIR / "BoB_biogeochem_event_composite_statistics.xlsx"
OUT_FIG_PNG     = OUTDIR / "Figure12_event_based_biogeochem_effectsizes_1200dpi.png"
OUT_FIG_PDF     = OUTDIR / "Figure12_event_based_biogeochem_effectsizes.pdf"

# -------------------------
# 5) STUDY SETTINGS
# -------------------------
START_YEAR = 1995
END_YEAR   = 2025

LAT_MIN, LAT_MAX = 5.0, 22.0
LON_MIN, LON_MAX = 80.0, 100.0

# Sub-basin limits
SUBBASIN_BOUNDS = {
    "Northern": (16.0, 22.0),
    "Central":  (10.0, 16.0),
    "Southern": (5.0, 10.0),
}

SEASON_MONTHS = {
    "Winter":       [12, 1, 2],
    "Pre-monsoon":  [3, 4, 5],
    "Monsoon":      [6, 7, 8, 9],
    "Post-monsoon": [10, 11],
}

SEASON_ORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
SUBBASIN_ORDER = ["Northern", "Central", "Southern"]

BIO_VARS = {
    "no3": {
        "aliases": ["no3", "nitrate"],
        "pretty": "NO₃⁻",
        "unit": "mmol m$^{-3}$",
        "color": "#d73027",
    },
    "po4": {
        "aliases": ["po4", "phosphate"],
        "pretty": "PO₄³⁻",
        "unit": "mmol m$^{-3}$",
        "color": "#1a9850",
    },
    "si": {
        "aliases": ["si", "silicate"],
        "pretty": "Si",
        "unit": "mmol m$^{-3}$",
        "color": "#fdae61",
    },
    "do": {
        "aliases": ["o2", "oxygen", "dissolved_oxygen", "dissolved oxygen", "do"],
        "pretty": "DO",
        "unit": "mmol m$^{-3}$",
        "color": "#7b3294",
    },
}

TIME_CANDIDATES = ["time", "TIME", "t"]
LAT_CANDIDATES  = ["lat", "latitude", "LAT", "nav_lat", "y"]
LON_CANDIDATES  = ["lon", "longitude", "LON", "nav_lon", "x"]
DEPTH_CANDIDATES = ["depth", "deptht", "lev", "olevel", "z"]

SURFACE_MAX_DEPTH = 10.0      # use near-surface layer for ecological response
CHUNKS = {"time": 90}
N_BOOT = 5000
RANDOM_SEED = 42
MIN_N = 4

# -------------------------
# 6) Plot style
# -------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
})


# %% Cell 5
# -------------------------
# 7) Helper functions
# -------------------------
import gc
from dataclasses import dataclass

def normtxt(x):
    x = str(x).strip().lower()
    x = x.replace("–", "-").replace("—", "-").replace("_", " ")
    x = re.sub(r"\s+", " ", x)
    return x

def find_name(ds, candidates, search_data_vars=True):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
        if search_data_vars and name in ds.data_vars:
            return name
    return None

def ensure_lon_0_360(ds_or_da, lon_name):
    lon = ds_or_da[lon_name].values
    if np.nanmin(lon) < 0:
        ds_or_da = ds_or_da.assign_coords({lon_name: lon % 360}).sortby(lon_name)
    return ds_or_da

def ensure_lat_ascending(da, lat_name):
    vals = da[lat_name].values
    if vals[0] > vals[-1]:
        da = da.sortby(lat_name)
    return da

def subset_domain(da, lat_name, lon_name, lat_min, lat_max, lon_min, lon_max):
    da = ensure_lat_ascending(da, lat_name)
    return da.sel({lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})

def area_weighted_mean(da, lat_name, lon_name):
    weights = np.cos(np.deg2rad(da[lat_name]))
    weights = weights / weights.mean()
    return da.weighted(weights).mean(dim=[lat_name, lon_name], skipna=True)

def month_to_season(month):
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season
    return None

def bootstrap_mean_diff(a, b, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < MIN_N or len(b) < MIN_N:
        return np.nan, np.nan, np.nan

    obs = np.mean(a) - np.mean(b)
    boots = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        boots[i] = np.mean(aa) - np.mean(bb)

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, lo, hi

def welch_p(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < MIN_N or len(b) < MIN_N:
        return np.nan
    return stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue

def hedges_g(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    n1, n2 = len(a), len(b)
    if n1 < MIN_N or n2 < MIN_N:
        return np.nan

    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if not np.isfinite(sp) or sp == 0:
        return np.nan

    d = (np.mean(a) - np.mean(b)) / sp
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return d * correction

def detect_subbasin(x):
    t = normtxt(x)
    if "north" in t:
        return "Northern"
    if "central" in t or "middle" in t:
        return "Central"
    if "south" in t:
        return "Southern"
    return None

def detect_season(x):
    t = normtxt(x)
    if "pre-monsoon" in t or "pre monsoon" in t or "premonsoon" in t:
        return "Pre-monsoon"
    if "post-monsoon" in t or "post monsoon" in t or "postmonsoon" in t:
        return "Post-monsoon"
    if "winter" in t:
        return "Winter"
    if "monsoon" in t and "pre" not in t and "post" not in t:
        return "Monsoon"
    return None

def save_figure(fig, png_path, pdf_path):
    fig.savefig(png_path, dpi=1200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

# -------------------------
# 8) MHW Event Catalogue Parsing Functions (adapted from NBpyfH45QAvu)
# -------------------------
@dataclass
class RegionMonthExposure:
    region: str
    year_month: pd.Timestamp
    season: str
    season_year: int
    mhw_grid_days: int
    total_grid_days: int
    exposure_fraction: float
    exposed_grid_cells: int
    mean_intensity_exposed: float

def event_gridcell_region(df: pd.DataFrame, region_name: str) -> pd.Series:
    r = SUBBASIN_BOUNDS[region_name]
    # Adjusting for lat_max handling based on notebook's region_mask_df (Northern BoB has inclusive upper bound)
    if region_name == "Northern": # Using simplified names as defined in xszR4EBJUZEQ
        lat_ok = (df["lat"] >= r[0]) & (df["lat"] <= r[1])
    else:
        lat_ok = (df["lat"] >= r[0]) & (df["lat"] < r[1])
    lon_ok = (df["lon"] >= LON_MIN) & (df["lon"] <= LON_MAX)
    return lat_ok & lon_ok

def load_mhw_events_raw(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="MHW")
    required = ["lon", "lat", "date_start", "date_end", "intensity_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in MHW workbook (MHW sheet): {missing}")

    df = df.copy()
    df["date_start"] = pd.to_datetime(df["date_start"])
    df["date_end"] = pd.to_datetime(df["date_end"])
    if "Season" in df.columns:
        df["Season"] = df["Season"].astype(str)
    return df

def build_mhw_flags_from_events(df_events: pd.DataFrame) -> pd.DataFrame:
    records = []

    # Use SUBBASIN_ORDER as defined in xszR4EBJUZEQ (e.g., "Northern", "Central", "Southern")
    for region in SUBBASIN_ORDER:
        dfr = df_events.loc[event_gridcell_region(df_events, region)].copy()
        if dfr.empty:
            continue

        unique_cells = dfr[["lon", "lat"]].drop_duplicates()
        n_cells = len(unique_cells)

        parts = []
        if not dfr.empty:
            for row in dfr.itertuples(index=False):
                days = pd.date_range(row.date_start, row.date_end, freq="D")
                tmp = pd.DataFrame({
                    "date": days,
                    "lon": row.lon,
                    "lat": row.lat,
                    "intensity_mean": row.intensity_mean,
                })
                parts.append(tmp)

        if not parts:
            continue

        daily = pd.concat(parts, ignore_index=True)
        daily["year_month"] = daily["date"].values.astype("datetime64[M]")

        agg = (
            daily.groupby(["year_month", "lon", "lat"], as_index=False)
            .agg(mhw_days=("date", "count"), intensity_mean=("intensity_mean", "mean"))
        )

        for ym, g in agg.groupby("year_month"):
            month_days = pd.Period(str(pd.Timestamp(ym).strftime("%Y-%m"))).days_in_month
            mhw_grid_days = int(g["mhw_days"].sum())
            total_grid_days = int(n_cells * month_days)
            exposure_fraction = mhw_grid_days / total_grid_days if total_grid_days > 0 else 0.0
            exposed_grid_cells = g[["lon", "lat"]].drop_duplicates().shape[0]

            weights = g["mhw_days"].to_numpy(dtype=float)
            mean_intensity_exposed = np.average(g["intensity_mean"], weights=weights) if weights.sum() > 0 else np.nan
            ts = pd.Timestamp(ym)

            records.append(RegionMonthExposure(
                region=region,
                year_month=ts,
                season=month_to_season(ts.month),
                season_year=ts.year + 1 if ts.month == 12 else ts.year,
                mhw_grid_days=mhw_grid_days,
                total_grid_days=total_grid_days,
                exposure_fraction=exposure_fraction,
                exposed_grid_cells=int(exposed_grid_cells),
                mean_intensity_exposed=float(mean_intensity_exposed),
            ))

    out = pd.DataFrame([r.__dict__ for r in records])
    out = out.rename(columns={'region': 'subbasin'})
    out["is_mhw"] = (
        (out["exposure_fraction"] >= EXPOSURE_THRESHOLD) &
        (out["exposed_grid_cells"] >= MIN_EXPOSED_GRID_CELLS)
    ).astype(int)
    return out[['year_month', 'subbasin', 'season', 'is_mhw']].assign(year=lambda x: x['year_month'].dt.year)

# -------------------------
# Replace problematic load_mhw_flags call
# -------------------------
print("Reading seasonal MHW workbook...")
raw_mhw_events = load_mhw_events_raw(MHW_XLSX)
mhw_flags = build_mhw_flags_from_events(raw_mhw_events)
print(mhw_flags.head())
print("Rows:", len(mhw_flags))
print(mhw_flags.groupby(["subbasin", "season", "is_mhw"]).size())

# -------------------------
# 9) Find and classify biogeochemical NetCDF files
# -------------------------
bio_dir = Path(BIO_DIR)
nc_files = sorted(list(bio_dir.glob("*.nc")) + list(bio_dir.glob("*.nc4")) + list(bio_dir.glob("**/*.nc")))
if len(nc_files) == 0:
    raise FileNotFoundError(f"No NetCDF files found in {BIO_DIR}")

def detect_bio_var_from_filename(path):
    t = normtxt(path.name)
    for key, meta in BIO_VARS.items():
        for alias in meta["aliases"]:
            if alias in t:
                return key
    return None

var_file_map = {}
for f in nc_files:
    v = detect_bio_var_from_filename(f)
    if v is not None:
        var_file_map.setdefault(v, []).append(str(f))

print("\nDetected files by variable:")
for k, v in var_file_map.items():
    print(k, "->", len(v), "files")

missing_vars = [k for k in BIO_VARS.keys() if k not in var_file_map]
if missing_vars:
    print("\n[WARN] No NetCDF files auto-detected for:", missing_vars)
    print("You may need to rename files or edit detect_bio_var_from_filename().")


# %% Cell 6
# -------------------------
# 10) Read each biogeochemical NetCDF and compute sub-basin seasonal means
# -------------------------
def load_one_variable_to_tidy(var_key, files):
    print(f"\nProcessing variable: {var_key}")
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        chunks=CHUNKS,
        coords="minimal",
        data_vars="minimal",
        compat="override",
        parallel=False
    )

    time_name  = find_name(ds, TIME_CANDIDATES)
    lat_name   = find_name(ds, LAT_CANDIDATES)
    lon_name   = find_name(ds, LON_CANDIDATES)
    depth_name = find_name(ds, DEPTH_CANDIDATES)

    if time_name is None or lat_name is None or lon_name is None:
        raise ValueError(f"{var_key}: Could not detect time/lat/lon names")

    # detect actual data variable
    data_var = None
    for dv in ds.data_vars:
        dvn = normtxt(dv)
        if any(alias in dvn for alias in BIO_VARS[var_key]["aliases"]):
            data_var = dv
            break

    if data_var is None:
        # fallback: use first data var
        data_var = list(ds.data_vars)[0]
        print(f"[WARN] {var_key}: using fallback data variable -> {data_var}")

    print(f"Using data variable: {data_var}")

    da = ds[data_var]
    da = ensure_lon_0_360(da, lon_name)
    da = subset_domain(da, lat_name, lon_name, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    da = da.sel({time_name: slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")})

    # keep near-surface values
    if depth_name is not None and depth_name in da.dims:
        depth_vals = np.asarray(da[depth_name].values, dtype=float)
        surface_mask = depth_vals <= SURFACE_MAX_DEPTH
        if np.any(surface_mask):
            da = da.sel({depth_name: depth_vals[surface_mask]})
            da = da.mean(dim=depth_name, skipna=True)
        else:
            da = da.isel({depth_name: 0})

    # squeeze singleton extra dims
    extra_dims = [d for d in da.dims if d not in [time_name, lat_name, lon_name]]
    for d in extra_dims:
        if da.sizes[d] == 1:
            da = da.isel({d: 0})
        else:
            raise ValueError(f"{var_key}: unexpected extra non-singleton dimension {d}")

    parts = []

    for basin, (lat0, lat1) in SUBBASIN_BOUNDS.items():
        sub = subset_domain(da, lat_name, lon_name, lat0, lat1, LON_MIN, LON_MAX)
        ts = area_weighted_mean(sub, lat_name, lon_name).compute()

        time_values = pd.to_datetime(ts[time_name].values)
        vals = np.asarray(ts.values, dtype=float)

        tmp = pd.DataFrame({"date": time_values, "value": vals})
        tmp["year"] = tmp["date"].dt.year
        tmp["month"] = tmp["date"].dt.month
        tmp["season"] = tmp["month"].map(month_to_season)
        tmp["subbasin"] = basin
        tmp["variable"] = var_key

        tmp = tmp[(tmp["year"] >= START_YEAR) & (tmp["year"] <= END_YEAR)]
        tmp = tmp.dropna(subset=["season", "value"])

        # seasonal mean by year
        seasonal = (
            tmp.groupby(["year", "subbasin", "season", "variable"], as_index=False)["value"]
            .mean()
        )
        parts.append(seasonal)

        del sub, ts, tmp, seasonal
        gc.collect()

    del ds, da
    gc.collect()

    return pd.concat(parts, ignore_index=True)

bio_parts = []
for var_key in BIO_VARS.keys():
    try:
        # Pass all nc_files to load_one_variable_to_tidy for each variable
        bio_parts.append(load_one_variable_to_tidy(var_key, nc_files))
    except Exception as e:
        print(f"[WARN] Failed to process {var_key}: {e}")

if len(bio_parts) == 0:
    raise ValueError("No biogeochemical variables could be processed from NetCDF files.")

bio_df = pd.concat(bio_parts, ignore_index=True)
bio_df = bio_df.sort_values(["variable", "subbasin", "season", "year"]).reset_index(drop=True)

print("\nBiogeochemical seasonal dataset:")
print(bio_df.head())
print("Rows:", len(bio_df))

# -------------------------
# 11) Merge MHW flags with biogeochemical dataset
# -------------------------
merged = bio_df.merge(
    mhw_flags,
    on=["year", "subbasin", "season"],
    how="inner"
).copy()

merged = merged.dropna(subset=["value", "is_mhw"])
merged["is_mhw"] = merged["is_mhw"].astype(int)

print("\nMerged dataset preview:")
print(merged.head())
print("Rows:", len(merged))
print(merged.groupby(["variable", "is_mhw"]).size().unstack(fill_value=0))

merged.to_csv(OUT_MERGED_CSV, index=False)

# -------------------------
# 12) Composite statistics
# -------------------------
rows = []

for basin in SUBBASIN_ORDER:
    for season in SEASON_ORDER:
        for var_key in BIO_VARS.keys():
            sub = merged[
                (merged["subbasin"] == basin) &
                (merged["season"] == season) &
                (merged["variable"] == var_key)
            ].copy()

            mhw_vals = sub.loc[sub["is_mhw"] == 1, "value"].astype(float).values
            non_vals = sub.loc[sub["is_mhw"] == 0, "value"].astype(float).values

            mhw_mean = np.nanmean(mhw_vals) if len(mhw_vals) else np.nan
            non_mean = np.nanmean(non_vals) if len(non_vals) else np.nan

            delta, ci_low, ci_high = bootstrap_mean_diff(
                mhw_vals, non_vals, n_boot=N_BOOT, seed=RANDOM_SEED
            )
            pval = welch_p(mhw_vals, non_vals)
            g = hedges_g(mhw_vals, non_vals)

            rows.append({
                "subbasin": basin,
                "season": season,
                "variable": var_key,
                "variable_pretty": BIO_VARS[var_key]["pretty"],
                "unit": BIO_VARS[var_key]["unit"],
                "n_mhw": len(mhw_vals),
                "n_nonmhw": len(non_vals),
                "mhw_mean": mhw_mean,
                "nonmhw_mean": non_mean,
                "delta_mhw_minus_nonmhw": delta,
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "p_value_welch": pval,
                "hedges_g": g
            })

stats_df = pd.DataFrame(rows)
stats_df["significant_0_05"] = stats_df["p_value_welch"] < 0.05
stats_df["direction"] = np.where(
    stats_df["delta_mhw_minus_nonmhw"] > 0, "Higher during MHW",
    np.where(stats_df["delta_mhw_minus_nonmhw"] < 0, "Lower during MHW", "No change")
)

stats_df.to_csv(OUT_STATS_CSV, index=False)


# %% Cell 7
# -------------------------
# 14) Q1-style figure: 4-panel effect-size composite
# FINAL FIXED VERSION
# -------------------------
from matplotlib.lines import Line2D

fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.8), constrained_layout=False)
axes = axes.flatten()

# IMPORTANT:
# These names must exactly match stats_df["subbasin"]
SUBBASIN_ORDER = ["Northern", "Central", "Southern"]
SEASON_ORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]

basin_colors = {
    "Northern": "#d73027",
    "Central":  "#4575b4",
    "Southern": "#1a9850"
}

plot_order = ["no3", "po4", "si", "do"]

# These are ONLY vertical plotting offsets so the 3 basin points
# do not overlap at the same seasonal y-position.
offsets = {
    "Northern": -0.18,
    "Central":   0.00,
    "Southern":  0.18
}

season_to_y = {season: i for i, season in enumerate(SEASON_ORDER)}

panel_titles = {
    "no3": "NO$_3^-$",
    "po4": "PO$_4^{3-}$",
    "si": "Si",
    "do": "DO"
}

# Figure-level legend handles
legend_handles = [
    Line2D([0], [0], marker="o", color=basin_colors["Northern"],
           markerfacecolor=basin_colors["Northern"], markeredgecolor=basin_colors["Northern"],
           linewidth=1.4, markersize=6, label="Northern"),
    Line2D([0], [0], marker="o", color=basin_colors["Central"],
           markerfacecolor=basin_colors["Central"], markeredgecolor=basin_colors["Central"],
           linewidth=1.4, markersize=6, label="Central"),
    Line2D([0], [0], marker="o", color=basin_colors["Southern"],
           markerfacecolor=basin_colors["Southern"], markeredgecolor=basin_colors["Southern"],
           linewidth=1.4, markersize=6, label="Southern"),
    Line2D([0], [0], marker="o", color="black",
           markerfacecolor="none", markeredgecolor="black",
           linewidth=0, markersize=8, label="p < 0.05")
]

for ax, var_key in zip(axes, plot_order):
    sub = stats_df[stats_df["variable"] == var_key].copy()

    # enforce plotting order
    sub["season"] = pd.Categorical(sub["season"], categories=SEASON_ORDER, ordered=True)
    sub["subbasin"] = pd.Categorical(sub["subbasin"], categories=SUBBASIN_ORDER, ordered=True)
    sub = sub.sort_values(["season", "subbasin"]).reset_index(drop=True)

    panel_has_points = False

    for basin in SUBBASIN_ORDER:
        sb = sub[sub["subbasin"] == basin].copy()
        if sb.empty:
            continue

        # Map each season to base y and add a small basin-specific offset
        ys = sb["season"].map(season_to_y).astype(float).values + offsets[basin]

        x = sb["delta_mhw_minus_nonmhw"].astype(float).values
        lo = sb["ci_low_95"].astype(float).values
        hi = sb["ci_high_95"].astype(float).values

        valid = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(ys)
        if valid.sum() == 0:
            continue

        panel_has_points = True

        x = x[valid]
        ys = ys[valid]
        lo = lo[valid]
        hi = hi[valid]

        xerr_left = x - lo
        xerr_right = hi - x

        ax.errorbar(
            x, ys,
            xerr=[xerr_left, xerr_right],
            fmt="o",
            capsize=3.2,
            markersize=5.6,
            color=basin_colors[basin],
            ecolor=basin_colors[basin],
            elinewidth=1.35,
            linewidth=1.35,
            zorder=3
        )

        # ring significant results
        sig = sb.loc[valid, "significant_0_05"].fillna(False).values
        if np.any(sig):
            ax.scatter(
                x[sig], ys[sig],
                s=90,
                facecolors="none",
                edgecolors="black",
                linewidths=1.15,
                zorder=5
            )

    # zero reference line
    ax.axvline(0, linestyle="--", linewidth=1.0, color="0.45", zorder=1)

    # y-axis formatting
    ax.set_yticks(np.arange(len(SEASON_ORDER)))
    ax.set_yticklabels(SEASON_ORDER)
    ax.set_ylim(-0.4, len(SEASON_ORDER) - 1 + 0.4)

    # titles and labels
    ax.set_title(f"({chr(97 + plot_order.index(var_key))}) {panel_titles[var_key]}", pad=6)
    ax.set_xlabel(f"Effect size (MHW − non-MHW), {BIO_VARS[var_key]['unit']}")

    # style
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if not panel_has_points:
        ax.text(
            0.5, 0.5,
            "No valid estimates",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=13, fontweight="bold"
        )

# Main figure title
fig.suptitle(
    "Event-based ecosystem composites for Bay of Bengal marine heatwaves (1995–2025)",
    fontsize=18, fontweight="bold", y=0.975
)

# Centered legend BELOW title, not inside panel (a)
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.905),
    ncol=4,
    frameon=False,
    columnspacing=1.8,
    handletextpad=0.6,
    borderaxespad=0.0
)

# Layout tuned for title + legend + panels
fig.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.08,
    top=0.89,
    wspace=0.20,
    hspace=0.22
)

save_figure(fig, OUT_FIG_PNG, OUT_FIG_PDF)
plt.show()
plt.close(fig)

print(f"[OK] Saved figure PNG: {OUT_FIG_PNG}")
print(f"[OK] Saved figure PDF: {OUT_FIG_PDF}")


# %% Cell 8
# -------------------------
# 15) Console summary for manuscript insertion
# -------------------------
print("\n================ SIGNIFICANT EVENT-BASED RESPONSES ================\n")

sig_df = stats_df[stats_df["significant_0_05"] == True].copy()

if sig_df.empty:
    print("No basin-season combinations reached p < 0.05 under current data/method settings.")
else:
    for basin in SUBBASIN_ORDER:
        for season in SEASON_ORDER:
            tmp = sig_df[(sig_df["subbasin"] == basin) & (sig_df["season"] == season)]
            if tmp.empty:
                continue
            print(f"{basin} | {season}")
            for _, r in tmp.iterrows():
                print(
                    f"  {r['variable_pretty']}: Δ = {r['delta_mhw_minus_nonmhw']:.4f} "
                    f"({r['unit']}), 95% CI [{r['ci_low_95']:.4f}, {r['ci_high_95']:.4f}], "
                    f"p = {r['p_value_welch']:.4g}, Hedges g = {r['hedges_g']:.3f}"
                )
            print()

print("Done.")
