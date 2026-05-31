#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08 Bvf N2 Calculation

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
# # **Hovmöller diagrams showing vertical distributions of Brunt–Väisälä frequency (N²)**
# 
# **Multi Observation Global Ocean 3D Temperature Salinity Height Geostrophic Current and MLD Dataset:**
# https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012/description
# 
# 
# 
# ---
# 
# **Prepared By:** Md. Zuhaib Kabir, Md. Amit Hasan,
# **Publish Date:** 16-03-2026


# %% [markdown]
# **YEAR: 1995-2009**


# %% Cell 3
# Install
# Colab shell command removed for repository reproducibility: !pip -q install gsw xarray netCDF4 cftime dask pandas openpyxl

import os
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from datetime import datetime


YEARS = list(range(1995, 2010))   # 1995–2009
NROWS, NCOLS = 5, 3
#OUT_DIR= Input BVF NetCDF Dataset
#XLSX_PATH= MHW EVENT_SUMMARY_XLSX Dataset
OUT_DIR = str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)/BVF 3D Dataset (1995-2024)")
XLSX_PATH = str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)/MHWs Category Datasets & Outputs/MHW SST Indices Data (1995-2025)/MHWs event summary.xlsx")

DEPTH_CALC_MAX = 1000
DEPTH_PLOT_MAX = 200
DZ_PLOT = 1

MLD_METHOD = "temp"     # "temp" or "dens"
REF_DEPTH = 10.0
DELTA_T = 0.2
DELTA_SIGMA0 = 0.03
SMOOTH_DAYS = 7

# Layout control (reduce gaps)
TOP = 0.945
RIGHT = 0.90
WSPACE = 0.10
HSPACE = 0.36

# Colorbar position/size
CBAR_RIGHT = 0.92
CBAR_WIDTH = 0.02
CBAR_BOTTOM = 0.20
CBAR_HEIGHT = 0.60

#  Output
OUT_PNG = str(OUTPUT_ROOT / "BVF_MLD_MHW_1995_2009_1200dpi.png")
OUT_PDF = str(OUTPUT_ROOT / "BVF_MLD_MHW_1995_2009.pdf")



def find_dim_name(ds, candidates):
    for c in candidates:
        if c in ds.dims or c in ds.coords:
            return c
    return None


def interp_depth_time(Z_time_depth, depth_src, depth_tgt):
    depth_src = np.asarray(depth_src)
    depth_tgt = np.asarray(depth_tgt)

    if np.any(np.diff(depth_src) <= 0):
        order = np.argsort(depth_src)
        depth_src = depth_src[order]
        Z_time_depth = Z_time_depth[:, order]

    out = np.empty((Z_time_depth.shape[0], depth_tgt.size), dtype=float)
    for t in range(Z_time_depth.shape[0]):
        z = Z_time_depth[t, :]
        z = np.where(np.isfinite(z), z, 0.0)
        out[t, :] = np.interp(depth_tgt, depth_src, z, left=z[0], right=z[-1])
    return out


def compute_mld_temp(depth, T, ref_depth=10.0, deltaT=0.2):
    depth = np.asarray(depth)
    ref_idx = int(np.argmin(np.abs(depth - ref_depth)))

    Tref = T[:, ref_idx][:, None]
    dT = (Tref - T)
    dT_search = dT[:, ref_idx:]
    depth_search = depth[ref_idx:]

    mask = dT_search >= deltaT
    any_true = mask.any(axis=1)
    idx = mask.argmax(axis=1)

    mld = np.full((T.shape[0],), np.nan, dtype=float)
    for t in range(T.shape[0]):
        if not any_true[t]:
            continue
        k = idx[t]
        if k == 0:
            mld[t] = depth_search[0]
            continue

        d0, d1 = depth_search[k-1], depth_search[k]
        y0, y1 = dT_search[t, k-1], dT_search[t, k]
        if np.isfinite(y0) and np.isfinite(y1) and (y1 - y0) != 0:
            mld[t] = d0 + (deltaT - y0) * (d1 - d0) / (y1 - y0)
        else:
            mld[t] = depth_search[k]
    return mld


def compute_mld_density(depth, SA, CT, ref_depth=10.0, delta_sigma0=0.03):
    depth = np.asarray(depth)
    ref_idx = int(np.argmin(np.abs(depth - ref_depth)))

    sigma0 = gsw.sigma0(SA, CT)
    ref = sigma0[:, ref_idx][:, None]
    dSig = sigma0 - ref
    dSig_search = dSig[:, ref_idx:]
    depth_search = depth[ref_idx:]

    mask = dSig_search >= delta_sigma0
    any_true = mask.any(axis=1)
    idx = mask.argmax(axis=1)

    mld = np.full((SA.shape[0],), np.nan, dtype=float)
    for t in range(SA.shape[0]):
        if not any_true[t]:
            continue
        k = idx[t]
        if k == 0:
            mld[t] = depth_search[0]
            continue

        d0, d1 = depth_search[k-1], depth_search[k]
        y0, y1 = dSig_search[t, k-1], dSig_search[t, k]
        if np.isfinite(y0) and np.isfinite(y1) and (y1 - y0) != 0:
            mld[t] = d0 + (delta_sigma0 - y0) * (d1 - d0) / (y1 - y0)
        else:
            mld[t] = depth_search[k]
    return mld


# Excel (year-wise sheets)
xls = pd.ExcelFile(XLSX_PATH)
sheet_names = set(xls.sheet_names)

def get_events_for_year_sheet(year):
    sname = str(year)
    if sname not in sheet_names:
        return []

    df = pd.read_excel(xls, sheet_name=sname)
    if "start_date" not in df.columns:
        return []

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    else:
        df["end_date"] = pd.NaT

    if "duration_days" in df.columns:
        df["duration_days"] = pd.to_numeric(df["duration_days"], errors="coerce")
    else:
        df["duration_days"] = np.nan

    df = df.dropna(subset=["start_date"]).copy()

    missing_end = df["end_date"].isna() & df["duration_days"].notna()
    df.loc[missing_end, "end_date"] = df.loc[missing_end, "start_date"] + pd.to_timedelta(
        df.loc[missing_end, "duration_days"].round().astype(int), unit="D"
    )

    df = df.dropna(subset=["start_date", "end_date"]).sort_values("start_date")
    df = df[df["start_date"].dt.year == year]

    return [(r["start_date"].to_pydatetime(), r["end_date"].to_pydatetime()) for _, r in df.iterrows()]


def clip_event_to_year(start, end, year):
    year_start = datetime(year, 1, 1)
    year_end = datetime(year, 12, 31, 23, 59, 59)
    s = max(start, year_start)
    e = min(end, year_end)
    if e <= s:
        return None
    return s, e


def compute_year_fields(year):
    nc_path = os.path.join(
        OUT_DIR,
        f"cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_to-so_80-100E_5-25N_0-2000m_{year}.nc"
    )
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"Missing file for {year}:\n{nc_path}")

    ds = xr.open_dataset(nc_path, chunks={"time": 30})

    time_name  = find_dim_name(ds, ["time", "TIME"])
    depth_name = find_dim_name(ds, ["depth", "DEPTH", "deptht", "lev", "z"])
    lat_name   = find_dim_name(ds, ["latitude", "lat", "LATITUDE", "nav_lat"])
    lon_name   = find_dim_name(ds, ["longitude", "lon", "LONGITUDE", "nav_lon"])

    ds = ds.sel({depth_name: slice(0, DEPTH_CALC_MAX)})

    temp = ds["to"]
    sal  = ds["so"]

    if (lat_name is not None) and (lon_name is not None):
        temp = temp.mean(dim=[lat_name, lon_name], skipna=True)
        sal  = sal.mean(dim=[lat_name, lon_name], skipna=True)

    temp, sal = xr.align(temp, sal)
    temp = temp.compute()
    sal  = sal.compute()

    depth = temp[depth_name].values.astype(float)
    time  = temp[time_name].values

    T = temp.values.astype(float)
    SP = sal.values.astype(float)

    if np.nanmean(SP) < 1.0:
        SP *= 1000.0

    if (lat_name is not None) and (lon_name is not None):
        lat0 = float(ds[lat_name].mean().compute().values)
        lon0 = float(ds[lon_name].mean().compute().values)
    else:
        lat0, lon0 = 15.0, 90.0

    p1d = gsw.p_from_z(-depth, lat0)
    p1d = np.maximum.accumulate(p1d)
    P = np.repeat(p1d[np.newaxis, :], T.shape[0], axis=0)

    SA = gsw.SA_from_SP(SP, P, lon0*np.ones_like(P), lat0*np.ones_like(P))
    CT = gsw.CT_from_t(SA, T, P)

    N2_T, p_mid_T = gsw.Nsquared(SA.T, CT.T, P.T, lat=lat0)
    N2 = N2_T.T
    p_mid = p_mid_T.T

    depth_mid = (-gsw.z_from_p(p_mid, lat0)).astype(float)
    depth_mid_1d = depth_mid[0, :]

    N2 = np.where(np.isfinite(N2), N2, 0.0)
    Zmid = N2 * 1e4

    depth_grid = np.arange(0, DEPTH_PLOT_MAX + DZ_PLOT, DZ_PLOT)
    Zgrid = interp_depth_time(Zmid, depth_mid_1d, depth_grid)
    Zgrid = np.clip(Zgrid, 0, 10)

    if MLD_METHOD.lower() == "temp":
        mld = compute_mld_temp(depth, T, ref_depth=REF_DEPTH, deltaT=DELTA_T)
    else:
        mld = compute_mld_density(depth, SA, CT, ref_depth=REF_DEPTH, delta_sigma0=DELTA_SIGMA0)

    mld = pd.Series(mld).rolling(SMOOTH_DAYS, center=True, min_periods=1).mean().to_numpy()

    return time, depth_grid, Zgrid, mld



# Plot

# EXACT 1-step scale: 0,1,2,...,10
levels = np.arange(0, 11, 1)

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(22, 18), sharey=True)
axes = axes.ravel()
mappable = None

for idx, year in enumerate(YEARS):
    ax = axes[idx]

    time, depth_grid, Zgrid, mld = compute_year_fields(year)
    events = get_events_for_year_sheet(year)

    cf = ax.contourf(time, depth_grid, Zgrid.T, levels=levels, cmap="RdYlBu_r", extend="max")
    if mappable is None:
        mappable = cf

    ax.invert_yaxis()
    ax.set_ylim(DEPTH_PLOT_MAX, 0)
    ax.set_yticks(np.arange(0, DEPTH_PLOT_MAX + 1, 20))

    ax.plot(time, mld, "k", lw=1.7)

    base_y = 0.82 * DEPTH_PLOT_MAX
    step_y = 0.06 * DEPTH_PLOT_MAX

    for j, (start, end) in enumerate(events):
        clipped = clip_event_to_year(start, end, year)
        if clipped is None:
            continue
        s, e = clipped

        ax.axvline(s, color="k", linestyle="--", lw=0.9)
        ax.axvline(e, color="k", linestyle="--", lw=0.9)

        y = base_y + j * step_y
        ax.annotate("", xy=(e, y), xytext=(s, y),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=1.2, shrinkA=0, shrinkB=0))

    ax.set_title(f"Year: {year}", fontsize=11, pad=5)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))  # Jan, Feb, ...
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    ax.set_xlabel("Date")
    if idx % NCOLS == 0:
        ax.set_ylabel("Depth (m)")

for k in range(len(YEARS), len(axes)):
    fig.delaxes(axes[k])

legend_handles = [
    Patch(facecolor="lightgray", edgecolor="black", label="BVF"),
    Line2D([0], [0], color="black", lw=1.7, label="MLD"),
    Line2D([0], [0], color="black", lw=1.0, linestyle="--", label="MHW event"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.98), columnspacing=1.2, handletextpad=0.6)

fig.subplots_adjust(top=TOP, right=RIGHT, wspace=WSPACE, hspace=HSPACE)

cax = fig.add_axes([CBAR_RIGHT, CBAR_BOTTOM, CBAR_WIDTH, CBAR_HEIGHT])
cbar = fig.colorbar(mappable, cax=cax, ticks=np.arange(0, 11, 1))
cbar.set_label(r"Brunt–Väisälä frequency ($s^{-2}$) × $10^{-4}$")


# SAVE (1200 dpi) to /content so it shows in left Files panel

fig.savefig(OUT_PNG, dpi=1200, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")

print("Saved PNG:", OUT_PNG)
print("Saved PDF:", OUT_PDF)

plt.show()


# %% [markdown]
# **YEAR: 2010-2024**


# %% Cell 5
# Colab shell command removed for repository reproducibility: !pip -q install gsw xarray netCDF4 cftime dask pandas openpyxl

import os
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from datetime import datetime


YEARS = list(range(2010, 2025))   # 1995–2009
NROWS, NCOLS = 5, 3

#OUT_DIR= Input BVF NetCDF Dataset
#XLSX_PATH= MHW EVENT_SUMMARY_XLSX Dataset
OUT_DIR = str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)/BVF 3D Dataset (1995-2024)")
XLSX_PATH = str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)/MHWs Category Datasets & Outputs/MHW SST Indices Data (1995-2025)/MHWs event summary.xlsx")

DEPTH_CALC_MAX = 1000
DEPTH_PLOT_MAX = 200
DZ_PLOT = 1

MLD_METHOD = "temp"     # "temp" or "dens"
REF_DEPTH = 10.0
DELTA_T = 0.2
DELTA_SIGMA0 = 0.03
SMOOTH_DAYS = 7

# Layout control
TOP = 0.945
RIGHT = 0.90
WSPACE = 0.10
HSPACE = 0.36

# Colorbar position/size
CBAR_RIGHT = 0.92
CBAR_WIDTH = 0.02
CBAR_BOTTOM = 0.20
CBAR_HEIGHT = 0.60

# Output
OUT_PNG = str(OUTPUT_ROOT / "BVF_MLD_MHW_2010_2024_1200dpi.png")
OUT_PDF = str(OUTPUT_ROOT / "BVF_MLD_MHW_2010_2024.pdf")



def find_dim_name(ds, candidates):
    for c in candidates:
        if c in ds.dims or c in ds.coords:
            return c
    return None


def interp_depth_time(Z_time_depth, depth_src, depth_tgt):
    depth_src = np.asarray(depth_src)
    depth_tgt = np.asarray(depth_tgt)

    if np.any(np.diff(depth_src) <= 0):
        order = np.argsort(depth_src)
        depth_src = depth_src[order]
        Z_time_depth = Z_time_depth[:, order]

    out = np.empty((Z_time_depth.shape[0], depth_tgt.size), dtype=float)
    for t in range(Z_time_depth.shape[0]):
        z = Z_time_depth[t, :]
        z = np.where(np.isfinite(z), z, 0.0)
        out[t, :] = np.interp(depth_tgt, depth_src, z, left=z[0], right=z[-1])
    return out


def compute_mld_temp(depth, T, ref_depth=10.0, deltaT=0.2):
    depth = np.asarray(depth)
    ref_idx = int(np.argmin(np.abs(depth - ref_depth)))

    Tref = T[:, ref_idx][:, None]
    dT = (Tref - T)
    dT_search = dT[:, ref_idx:]
    depth_search = depth[ref_idx:]

    mask = dT_search >= deltaT
    any_true = mask.any(axis=1)
    idx = mask.argmax(axis=1)

    mld = np.full((T.shape[0],), np.nan, dtype=float)
    for t in range(T.shape[0]):
        if not any_true[t]:
            continue
        k = idx[t]
        if k == 0:
            mld[t] = depth_search[0]
            continue

        d0, d1 = depth_search[k-1], depth_search[k]
        y0, y1 = dT_search[t, k-1], dT_search[t, k]
        if np.isfinite(y0) and np.isfinite(y1) and (y1 - y0) != 0:
            mld[t] = d0 + (deltaT - y0) * (d1 - d0) / (y1 - y0)
        else:
            mld[t] = depth_search[k]
    return mld


def compute_mld_density(depth, SA, CT, ref_depth=10.0, delta_sigma0=0.03):
    depth = np.asarray(depth)
    ref_idx = int(np.argmin(np.abs(depth - ref_depth)))

    sigma0 = gsw.sigma0(SA, CT)
    ref = sigma0[:, ref_idx][:, None]
    dSig = sigma0 - ref
    dSig_search = dSig[:, ref_idx:]
    depth_search = depth[ref_idx:]

    mask = dSig_search >= delta_sigma0
    any_true = mask.any(axis=1)
    idx = mask.argmax(axis=1)

    mld = np.full((SA.shape[0],), np.nan, dtype=float)
    for t in range(SA.shape[0]):
        if not any_true[t]:
            continue
        k = idx[t]
        if k == 0:
            mld[t] = depth_search[0]
            continue

        d0, d1 = depth_search[k-1], depth_search[k]
        y0, y1 = dSig_search[t, k-1], dSig_search[t, k]
        if np.isfinite(y0) and np.isfinite(y1) and (y1 - y0) != 0:
            mld[t] = d0 + (delta_sigma0 - y0) * (d1 - d0) / (y1 - y0)
        else:
            mld[t] = depth_search[k]
    return mld


# Excel
xls = pd.ExcelFile(XLSX_PATH)
sheet_names = set(xls.sheet_names)

def get_events_for_year_sheet(year):
    sname = str(year)
    if sname not in sheet_names:
        return []

    df = pd.read_excel(xls, sheet_name=sname)
    if "start_date" not in df.columns:
        return []

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    else:
        df["end_date"] = pd.NaT

    if "duration_days" in df.columns:
        df["duration_days"] = pd.to_numeric(df["duration_days"], errors="coerce")
    else:
        df["duration_days"] = np.nan

    df = df.dropna(subset=["start_date"]).copy()

    missing_end = df["end_date"].isna() & df["duration_days"].notna()
    df.loc[missing_end, "end_date"] = df.loc[missing_end, "start_date"] + pd.to_timedelta(
        df.loc[missing_end, "duration_days"].round().astype(int), unit="D"
    )

    df = df.dropna(subset=["start_date", "end_date"]).sort_values("start_date")
    df = df[df["start_date"].dt.year == year]

    return [(r["start_date"].to_pydatetime(), r["end_date"].to_pydatetime()) for _, r in df.iterrows()]


def clip_event_to_year(start, end, year):
    year_start = datetime(year, 1, 1)
    year_end = datetime(year, 12, 31, 23, 59, 59)
    s = max(start, year_start)
    e = min(end, year_end)
    if e <= s:
        return None
    return s, e


def compute_year_fields(year):
    nc_path = os.path.join(
        OUT_DIR,
        f"cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_to-so_80-100E_5-25N_0-2000m_{year}.nc"
    )
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"Missing file for {year}:\n{nc_path}")

    ds = xr.open_dataset(nc_path, chunks={"time": 30})

    time_name  = find_dim_name(ds, ["time", "TIME"])
    depth_name = find_dim_name(ds, ["depth", "DEPTH", "deptht", "lev", "z"])
    lat_name   = find_dim_name(ds, ["latitude", "lat", "LATITUDE", "nav_lat"])
    lon_name   = find_dim_name(ds, ["longitude", "lon", "LONGITUDE", "nav_lon"])

    ds = ds.sel({depth_name: slice(0, DEPTH_CALC_MAX)})

    temp = ds["to"]
    sal  = ds["so"]

    if (lat_name is not None) and (lon_name is not None):
        temp = temp.mean(dim=[lat_name, lon_name], skipna=True)
        sal  = sal.mean(dim=[lat_name, lon_name], skipna=True)

    temp, sal = xr.align(temp, sal)
    temp = temp.compute()
    sal  = sal.compute()

    depth = temp[depth_name].values.astype(float)
    time  = temp[time_name].values

    T = temp.values.astype(float)
    SP = sal.values.astype(float)

    if np.nanmean(SP) < 1.0:
        SP *= 1000.0

    if (lat_name is not None) and (lon_name is not None):
        lat0 = float(ds[lat_name].mean().compute().values)
        lon0 = float(ds[lon_name].mean().compute().values)
    else:
        lat0, lon0 = 15.0, 90.0

    p1d = gsw.p_from_z(-depth, lat0)
    p1d = np.maximum.accumulate(p1d)
    P = np.repeat(p1d[np.newaxis, :], T.shape[0], axis=0)

    SA = gsw.SA_from_SP(SP, P, lon0*np.ones_like(P), lat0*np.ones_like(P))
    CT = gsw.CT_from_t(SA, T, P)

    N2_T, p_mid_T = gsw.Nsquared(SA.T, CT.T, P.T, lat=lat0)
    N2 = N2_T.T
    p_mid = p_mid_T.T

    depth_mid = (-gsw.z_from_p(p_mid, lat0)).astype(float)
    depth_mid_1d = depth_mid[0, :]

    N2 = np.where(np.isfinite(N2), N2, 0.0)
    Zmid = N2 * 1e4

    depth_grid = np.arange(0, DEPTH_PLOT_MAX + DZ_PLOT, DZ_PLOT)
    Zgrid = interp_depth_time(Zmid, depth_mid_1d, depth_grid)
    Zgrid = np.clip(Zgrid, 0, 10)

    if MLD_METHOD.lower() == "temp":
        mld = compute_mld_temp(depth, T, ref_depth=REF_DEPTH, deltaT=DELTA_T)
    else:
        mld = compute_mld_density(depth, SA, CT, ref_depth=REF_DEPTH, delta_sigma0=DELTA_SIGMA0)

    mld = pd.Series(mld).rolling(SMOOTH_DAYS, center=True, min_periods=1).mean().to_numpy()

    return time, depth_grid, Zgrid, mld


# Plot

# EXACT 1-step scale: 0,1,2,...,10
levels = np.arange(0, 11, 1)

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(22, 18), sharey=True)
axes = axes.ravel()
mappable = None

for idx, year in enumerate(YEARS):
    ax = axes[idx]

    time, depth_grid, Zgrid, mld = compute_year_fields(year)
    events = get_events_for_year_sheet(year)

    cf = ax.contourf(time, depth_grid, Zgrid.T, levels=levels, cmap="RdYlBu_r", extend="max")
    if mappable is None:
        mappable = cf

    ax.invert_yaxis()
    ax.set_ylim(DEPTH_PLOT_MAX, 0)
    ax.set_yticks(np.arange(0, DEPTH_PLOT_MAX + 1, 20))

    ax.plot(time, mld, "k", lw=1.7)

    base_y = 0.82 * DEPTH_PLOT_MAX
    step_y = 0.06 * DEPTH_PLOT_MAX

    for j, (start, end) in enumerate(events):
        clipped = clip_event_to_year(start, end, year)
        if clipped is None:
            continue
        s, e = clipped

        ax.axvline(s, color="k", linestyle="--", lw=0.9)
        ax.axvline(e, color="k", linestyle="--", lw=0.9)

        y = base_y + j * step_y
        ax.annotate("", xy=(e, y), xytext=(s, y),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=1.2, shrinkA=0, shrinkB=0))

    ax.set_title(f"Year: {year}", fontsize=11, pad=5)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))  # Jan, Feb, ...
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    ax.set_xlabel("Date")
    if idx % NCOLS == 0:
        ax.set_ylabel("Depth (m)")

for k in range(len(YEARS), len(axes)):
    fig.delaxes(axes[k])

legend_handles = [
    Patch(facecolor="lightgray", edgecolor="black", label="BVF"),
    Line2D([0], [0], color="black", lw=1.7, label="MLD"),
    Line2D([0], [0], color="black", lw=1.0, linestyle="--", label="MHW event"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.98), columnspacing=1.2, handletextpad=0.6)

fig.subplots_adjust(top=TOP, right=RIGHT, wspace=WSPACE, hspace=HSPACE)

cax = fig.add_axes([CBAR_RIGHT, CBAR_BOTTOM, CBAR_WIDTH, CBAR_HEIGHT])
cbar = fig.colorbar(mappable, cax=cax, ticks=np.arange(0, 11, 1))
cbar.set_label(r"Brunt–Väisälä frequency ($s^{-2}$) × $10^{-4}$")


# SAVE (1200 dpi) to /content so it shows in left Files panel

fig.savefig(OUT_PNG, dpi=1200, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")

print("Saved PNG:", OUT_PNG)
print("Saved PDF:", OUT_PDF)

plt.show()
