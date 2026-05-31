#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10 Upper Ocean Persistence Diagnostics

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



# %% Cell 1
# ============================================================
# BAY OF BENGAL UPPER-OCEAN PERSISTENCE DIAGNOSTICS
# FINAL COLAB SCRIPT
#
# Outputs:
#   1) Reproduced 3-panel Figure 11
#   2) Enhanced 4-panel persistence mechanism figure
#
# Panels:
#   (a) Shallow N² anomaly during high-exposure years
#   (b) Daily MHW occurrence scaling with shallow N²
#   (c) Daily MLD contrast between MHW and non-MHW days
#   (d) Vertical N² composite profile (MHW vs non-MHW)
#
# Designed for Google Colab (~12 GB RAM)
# ============================================================

# -------------------------
# 1) Install dependencies
# -------------------------
# Colab shell command removed for repository reproducibility: !pip -q install xarray netCDF4 cftime scipy pandas numpy matplotlib gsw openpyxl dask bottleneck statsmodels

# -------------------------
# 2) Imports
# -------------------------
import os
import glob
import gc
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import gsw

warnings.filterwarnings("ignore")

# -------------------------
# 3) Mount Google Drive
# -------------------------
# Google Colab Drive import removed; use config/paths_template.yaml and local data folders.
# Google Drive mount removed; configure DATA_ROOT in this script or via environment variable MHW_DATA_ROOT.

# -------------------------
# 4) USER SETTINGS
# -------------------------
DATA_DIR = str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)/BVF 3D Dataset (1995-2024)")
FILE_PATTERN = "*.nc"

MHW_MASK_CSV = str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)/BoB_MHW_daily_mask_1995_2024.csv")

OUTDIR = str(DATA_ROOT / "BoB_upper_ocean_persistence_outputs_FINAL_FIG11")
os.makedirs(OUTDIR, exist_ok=True)

OUT_DAILY_CSV       = os.path.join(OUTDIR, "BoB_upper_ocean_daily_metrics.csv")
OUT_ANNUAL_CSV      = os.path.join(OUTDIR, "BoB_upper_ocean_annual_metrics.csv")
OUT_SUMMARY_CSV     = os.path.join(OUTDIR, "BoB_upper_ocean_summary_metrics.csv")
OUT_PANELB_CSV      = os.path.join(OUTDIR, "BoB_upper_ocean_panelB_input.csv")
OUT_PROFILE_CSV     = os.path.join(OUTDIR, "BoB_vertical_N2_composite_profiles.csv")

OUT_DAILY_XLSX      = os.path.join(OUTDIR, "BoB_upper_ocean_daily_metrics.xlsx")
OUT_ANNUAL_XLSX     = os.path.join(OUTDIR, "BoB_upper_ocean_annual_metrics.xlsx")
OUT_SUMMARY_XLSX    = os.path.join(OUTDIR, "BoB_upper_ocean_summary_metrics.xlsx")
OUT_PANELB_XLSX     = os.path.join(OUTDIR, "BoB_upper_ocean_panelB_input.xlsx")
OUT_PROFILE_XLSX    = os.path.join(OUTDIR, "BoB_vertical_N2_composite_profiles.xlsx")
OUT_WORKBOOK_XLSX   = os.path.join(OUTDIR, "BoB_upper_ocean_master_workbook.xlsx")

OUT_FIG3_PNG        = os.path.join(OUTDIR, "Figure11_reproduced_3panel_1200dpi.png")
OUT_FIG4_PNG        = os.path.join(OUTDIR, "Figure11_enhanced_4panel_1200dpi.png")

LAT_MIN, LAT_MAX = 5.0, 22.0
LON_MIN, LON_MAX = 80.0, 100.0

SHALLOW_TOP = 0
SHALLOW_BOTTOM = 50
DEEP_TOP = 50
DEEP_BOTTOM = 200
PROFILE_MAX_DEPTH = 200

START_DATE = "1995-01-01"
END_DATE   = "2024-12-31"

HIGH_EXPOSURE_QUANTILE = 0.75
N_BOOT = 3000
RANDOM_SEED = 42
DPI = 1200
CHUNKS = {"time": 120}

TIME_CANDIDATES   = ["time", "TIME", "t"]
LAT_CANDIDATES    = ["lat", "latitude", "LAT", "nav_lat", "y"]
LON_CANDIDATES    = ["lon", "longitude", "LON", "nav_lon", "x"]
DEPTH_CANDIDATES  = ["depth", "deptht", "lev", "olevel", "z", "level"]

TEMP_CANDIDATES   = ["to", "thetao", "temperature", "temp", "votemper"]
SALT_CANDIDATES   = ["so", "salinity", "vosaline", "salt"]
MLD_CANDIDATES    = ["mlotst", "mld", "mixed_layer_depth", "mixed_layer_depth_0.03", "MLD"]

# -------------------------
# 5) Plot style
# -------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 15,
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
    "savefig.edgecolor": "white"
})

# -------------------------
# 6) Helper functions
# -------------------------
def find_name(ds, candidates, search_data_vars=True):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
        if search_data_vars and name in ds.data_vars:
            return name
    return None

def ensure_lon_0_360(ds, lon_name):
    lon = ds[lon_name].values
    if np.nanmin(lon) < 0:
        ds = ds.assign_coords({lon_name: lon % 360}).sortby(lon_name)
    return ds

def ensure_lat_ascending(da, lat_name):
    vals = da[lat_name].values
    if vals[0] > vals[-1]:
        da = da.sortby(lat_name)
    return da

def subset_bob(da, lat_name, lon_name):
    da = ensure_lat_ascending(da, lat_name)
    return da.sel({lat_name: slice(LAT_MIN, LAT_MAX), lon_name: slice(LON_MIN, LON_MAX)})

def area_weighted_mean(da, lat_name, lon_name):
    weights = np.cos(np.deg2rad(da[lat_name]))
    weights = weights / weights.mean()
    return da.weighted(weights).mean(dim=[lat_name, lon_name], skipna=True)

def season_from_month(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Pre-monsoon"
    elif m in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-monsoon"

def clean_mld_values(arr, min_valid=0.0, max_valid=300.0):
    arr = np.asarray(arr, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr[(arr < min_valid) | (arr > max_valid)] = np.nan
    return arr

def suspicious_constant_series(arr, tol_std=0.25):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 10:
        return True
    return np.nanstd(arr) < tol_std

def sci_fmt(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.3e}"

def savefig_png(fig, png_path, dpi=DPI):
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        transparent=False
    )

def bootstrap_mean_diff(a, b, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 10 or len(b) < 10:
        return np.nan, np.nan, np.nan, np.nan

    obs = np.mean(a) - np.mean(b)
    boots = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        boots[i] = np.mean(aa) - np.mean(bb)

    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    _, pval = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return obs, ci_low, ci_high, pval

def regression_with_ci(x, y, alpha=0.05):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 8:
        return {
            "n": len(x), "slope": np.nan, "intercept": np.nan,
            "r2": np.nan, "p": np.nan,
            "slope_ci_low": np.nan, "slope_ci_high": np.nan
        }

    res = stats.linregress(x, y)
    df = len(x) - 2
    tcrit = stats.t.ppf(1 - alpha/2, df)

    return {
        "n": len(x),
        "slope": res.slope,
        "intercept": res.intercept,
        "r2": res.rvalue**2,
        "p": res.pvalue,
        "slope_ci_low": res.slope - tcrit * res.stderr,
        "slope_ci_high": res.slope + tcrit * res.stderr
    }

def interpolated_crossing_depth(depth, profile, target, ref_depth=10.0, crossing="greater"):
    depth = np.asarray(depth, dtype=float)
    profile = np.asarray(profile, dtype=float)

    valid = np.isfinite(depth) & np.isfinite(profile)
    if valid.sum() < 3:
        return np.nan

    depth = depth[valid]
    profile = profile[valid]

    if np.any(np.diff(depth) < 0):
        idx = np.argsort(depth)
        depth = depth[idx]
        profile = profile[idx]

    ref_idx = np.argmin(np.abs(depth - ref_depth))

    for k in range(ref_idx + 1, len(depth)):
        y0, y1 = profile[k - 1], profile[k]
        d0, d1 = depth[k - 1], depth[k]

        if crossing == "greater":
            crossed = (y0 < target <= y1) or (y0 >= target and k - 1 == ref_idx)
        else:
            crossed = (y0 > target >= y1) or (y0 <= target and k - 1 == ref_idx)

        if crossed:
            if np.isclose(y1, y0):
                return float(d1)
            frac = (target - y0) / (y1 - y0)
            frac = np.clip(frac, 0, 1)
            return float(d0 + frac * (d1 - d0))
    return np.nan

def compute_mld_from_density_profile(depth, rho_profile, threshold=0.03, ref_depth=10.0):
    valid = np.isfinite(depth) & np.isfinite(rho_profile)
    if valid.sum() < 3:
        return np.nan
    d = np.asarray(depth, dtype=float)[valid]
    r = np.asarray(rho_profile, dtype=float)[valid]
    if np.any(np.diff(d) < 0):
        idx = np.argsort(d)
        d = d[idx]
        r = r[idx]
    ref_idx = np.argmin(np.abs(d - ref_depth))
    target = r[ref_idx] + threshold
    return interpolated_crossing_depth(d, r, target, ref_depth=ref_depth, crossing="greater")

def compute_ild_from_temperature_profile(depth, temp_profile, threshold=0.2, ref_depth=10.0):
    valid = np.isfinite(depth) & np.isfinite(temp_profile)
    if valid.sum() < 3:
        return np.nan
    d = np.asarray(depth, dtype=float)[valid]
    t = np.asarray(temp_profile, dtype=float)[valid]
    if np.any(np.diff(d) < 0):
        idx = np.argsort(d)
        d = d[idx]
        t = t[idx]
    ref_idx = np.argmin(np.abs(d - ref_depth))
    target = t[ref_idx] - threshold
    return interpolated_crossing_depth(d, t, target, ref_depth=ref_depth, crossing="less")

def compute_mld_proxy_from_n2(depth_mid, n2_profile, max_depth=100.0):
    d = np.asarray(depth_mid, dtype=float)
    n = np.asarray(n2_profile, dtype=float)
    mask = (d >= 0) & (d <= max_depth) & np.isfinite(n)
    if mask.sum() < 2:
        return np.nan
    return float(d[mask][np.nanargmax(n[mask])])

def build_full_daily_mhw_series(mhw_raw, start_date, end_date):
    full_dates = pd.date_range(start_date, end_date, freq="D")
    daily = pd.DataFrame({"date": full_dates})

    src = mhw_raw.copy()
    src["date"] = pd.to_datetime(src["date"])

    if "is_mhw" not in src.columns:
        if "event_id" in src.columns:
            src["is_mhw"] = src["event_id"].notna().astype(int)
        else:
            raise KeyError("Need 'is_mhw' or 'event_id' in MHW file.")

    src_daily = src.groupby("date", as_index=False)["is_mhw"].max()
    daily = daily.merge(src_daily, on="date", how="left")

    daily["is_mhw"] = daily["is_mhw"].fillna(0).astype(int)
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["season"] = daily["month"].apply(season_from_month)
    return daily

def season_zscore(series, season):
    out = pd.Series(np.nan, index=series.index, dtype=float)
    for s in ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]:
        mask = season == s
        vals = series.loc[mask].astype(float)
        mu = vals.mean()
        sd = vals.std(ddof=0)
        if np.isfinite(sd) and sd > 0:
            out.loc[mask] = (vals - mu) / sd
    return out

def prepare_panel_b_glm_df(daily_df):
    d = daily_df.copy()
    d = d[np.isfinite(d["N2_shallow"]) & np.isfinite(d["is_mhw"])].copy()
    d["x"] = season_zscore(d["N2_shallow"], d["season"])
    d = d[np.isfinite(d["x"])].copy()
    d["y"] = d["is_mhw"].astype(int)
    return d

def fit_panel_b_glm(panel_b_df):
    if len(panel_b_df) < 100 or panel_b_df["y"].std(ddof=0) == 0:
        return None
    try:
        exog = sm.add_constant(panel_b_df["x"].values)
        endog = panel_b_df["y"].values
        model = sm.GLM(endog, exog, family=sm.families.Binomial())
        return model.fit()
    except Exception:
        return None

def bootstrap_profile_ci(arr2d, n_boot=1000, seed=42):
    """
    arr2d: (n_samples, n_depth)
    returns mean, low, high
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr2d, dtype=float)
    arr = arr[np.isfinite(arr).any(axis=1)]
    if arr.shape[0] < 10:
        return None, None, None

    mean_prof = np.nanmean(arr, axis=0)
    boots = np.empty((n_boot, arr.shape[1]), dtype=float)

    for i in range(n_boot):
        idx = rng.choice(np.arange(arr.shape[0]), size=arr.shape[0], replace=True)
        boots[i, :] = np.nanmean(arr[idx, :], axis=0)

    low = np.nanpercentile(boots, 2.5, axis=0)
    high = np.nanpercentile(boots, 97.5, axis=0)
    return mean_prof, low, high


# %% Cell 2
# -------------------------
# 7) Load MHW daily mask and rebuild full daily calendar
# -------------------------
mhw_raw = pd.read_csv(MHW_MASK_CSV)

date_col = next((c for c in ["date", "time", "start_date"] if c in mhw_raw.columns), None)
if date_col is None:
    raise KeyError("No date column found in MHW CSV")

mhw_raw = mhw_raw.rename(columns={date_col: "date"})
mhw_raw["date"] = pd.to_datetime(mhw_raw["date"])

if "is_mhw" not in mhw_raw.columns:
    if "event_id" in mhw_raw.columns:
        mhw_raw["is_mhw"] = mhw_raw["event_id"].notna().astype(int)
    else:
        raise KeyError("Need either 'is_mhw' or 'event_id' in the MHW CSV")

mhw_raw = mhw_raw[(mhw_raw["date"] >= START_DATE) & (mhw_raw["date"] <= END_DATE)].copy()
mhw_daily_full = build_full_daily_mhw_series(mhw_raw, START_DATE, END_DATE)

annual_mhw_days = (
    mhw_daily_full.groupby("year", as_index=False)["is_mhw"]
    .sum()
    .rename(columns={"is_mhw": "mhw_days"})
)

print("Full daily MHW calendar created.")
print("Daily length:", len(mhw_daily_full))

# -------------------------
# 8) Open CMEMS dataset
# -------------------------
files = sorted(glob.glob(os.path.join(DATA_DIR, FILE_PATTERN)))
if len(files) == 0:
    raise FileNotFoundError(f"No .nc files found in {DATA_DIR}")

print(f"Found {len(files)} netCDF files")

# --- MODIFICATION START ---
valid_files = []
problematic_files = []
for f in files:
    try:
        # Attempt to open each file individually
        with xr.open_dataset(f, decode_times=False) as temp_ds:
            # Check if it's a valid netCDF (e.g., has dimensions)
            if temp_ds.dims:
                valid_files.append(f)
            else:
                print(f"Skipping empty or malformed file: {f}")
                problematic_files.append(f)
    except Exception as e:
        print(f"Error opening file {f}: {e}")
        problematic_files.append(f)

if not valid_files:
    raise RuntimeError("No valid netCDF files could be opened.")

print(f"Found {len(valid_files)} valid netCDF files and skipped {len(problematic_files)} problematic files.")

ds = xr.open_mfdataset(
    valid_files,
    combine="by_coords",
    chunks=CHUNKS,
    compat="override",
    coords="minimal",
    data_vars="minimal",
    parallel=False
)
# --- MODIFICATION END ---

time_name  = find_name(ds, TIME_CANDIDATES)
lat_name   = find_name(ds, LAT_CANDIDATES)
lon_name   = find_name(ds, LON_CANDIDATES)
depth_name = find_name(ds, DEPTH_CANDIDATES)
temp_name  = find_name(ds, TEMP_CANDIDATES)
salt_name  = find_name(ds, SALT_CANDIDATES)
mld_name   = find_name(ds, MLD_CANDIDATES)

needed = {
    "time": time_name, "lat": lat_name, "lon": lon_name,
    "depth": depth_name, "temp": temp_name, "salt": salt_name
}
missing = [k for k, v in needed.items() if v is None]
if missing:
    raise ValueError(f"Missing required dataset variables: {missing}")

print("Variables selected:")
print(" time :", time_name)
print(" lat  :", lat_name)
print(" lon  :", lon_name)
print(" depth:", depth_name)
print(" temp :", temp_name)
print(" salt :", salt_name)
print(" mld  :", mld_name if mld_name is not None else "not found -> compute from fallback")

ds = ensure_lon_0_360(ds, lon_name)

temp = subset_bob(ds[temp_name], lat_name, lon_name).sel({time_name: slice(START_DATE, END_DATE)})
salt = subset_bob(ds[salt_name], lat_name, lon_name).sel({time_name: slice(START_DATE, END_DATE)})

for da_name, da in [("temp", temp), ("salt", salt)]:
    extra_dims = [d for d in da.dims if d not in [time_name, depth_name, lat_name, lon_name]]
    for d in extra_dims:
        if da.sizes[d] == 1:
            da = da.isel({d: 0})
        else:
            raise ValueError(f"{da_name} has unexpected non-singleton extra dim: {d}")
    if da_name == "temp":
        temp = da
    else:
        salt = da

try:
    temp_test = float(temp.isel({time_name: slice(0, 2)}).mean().compute())
    if temp_test > 100:
        temp = temp - 273.15
        print("Converted temperature from Kelvin to Celsius")
except Exception:
    pass

if mld_name is not None:
    mld = subset_bob(ds[mld_name], lat_name, lon_name).sel({time_name: slice(START_DATE, END_DATE)})
    extra_dims = [d for d in mld.dims if d not in [time_name, lat_name, lon_name]]
    for d in extra_dims:
        if mld.sizes[d] == 1:
            mld = mld.isel({d: 0})
        else:
            raise ValueError(f"MLD has unexpected non-singleton extra dim: {d}")
else:
    mld = None


# %% Cell 3
# -------------------------
# 9) Daily basin-mean profiles
# -------------------------
print("\nComputing DAILY basin-mean T/S profiles...")
temp_prof = area_weighted_mean(temp, lat_name, lon_name).compute()
salt_prof = area_weighted_mean(salt, lat_name, lon_name).compute()

if mld is not None:
    print("Computing DAILY basin-mean MLD from CMEMS MLD...")
    mld_ts = area_weighted_mean(mld, lat_name, lon_name).compute()
else:
    mld_ts = None

del ds, temp, salt
if mld is not None:
    del mld
gc.collect()

temp_prof = temp_prof.rename({time_name: "date", depth_name: "depth"})
salt_prof = salt_prof.rename({time_name: "date", depth_name: "depth"})
if mld_ts is not None:
    mld_ts = mld_ts.rename({time_name: "date"})

times = pd.to_datetime(temp_prof["date"].values)
depth = np.asarray(temp_prof["depth"].values, dtype=float)

# -------------------------
# 10) Compute density, N², robust daily MLD
# -------------------------
print("Computing TEOS-10 density profiles, N², and robust daily MLD...")

if np.any(np.diff(depth) < 0):
    sort_idx = np.argsort(depth)
    depth = depth[sort_idx]
    temp_prof = temp_prof.isel(depth=sort_idx)
    salt_prof = salt_prof.isel(depth=sort_idx)

lat_ref = float((LAT_MIN + LAT_MAX) / 2.0)
lon_ref = float((LON_MIN + LON_MAX) / 2.0)

pressure_1d = gsw.p_from_z(-depth, lat_ref)

T_np = np.asarray(temp_prof.values, dtype=float)
S_np = np.asarray(salt_prof.values, dtype=float)
P = np.broadcast_to(pressure_1d[None, :], T_np.shape)

SA = gsw.SA_from_SP(S_np, P, lon_ref, lat_ref)
CT = gsw.CT_from_t(SA, T_np, P)
rho = gsw.sigma0(SA, CT) + 1000.0

N2, p_mid = gsw.Nsquared(SA, CT, P, lat=lat_ref, axis=1)
depth_mid = 0.5 * (depth[:-1] + depth[1:])

if N2.shape[1] != len(depth_mid):
    raise ValueError(f"N2 depth dimension ({N2.shape[1]}) != depth_mid length ({len(depth_mid)})")

shallow_mask = (depth_mid >= SHALLOW_TOP) & (depth_mid <= SHALLOW_BOTTOM)
deep_mask = (depth_mid >= DEEP_TOP) & (depth_mid <= DEEP_BOTTOM)
profile_mask = depth_mid <= PROFILE_MAX_DEPTH

if shallow_mask.sum() == 0:
    raise ValueError("No N² points found in shallow layer.")
if deep_mask.sum() == 0:
    raise ValueError("No N² points found in deep layer.")

N2_shallow = np.nanmean(N2[:, shallow_mask], axis=1)
N2_deep = np.nanmean(N2[:, deep_mask], axis=1)
N2_profile_0_200 = N2[:, profile_mask]
depth_mid_0_200 = depth_mid[profile_mask]

if mld_ts is not None:
    MLD_primary = clean_mld_values(np.asarray(mld_ts.values, dtype=float))
else:
    MLD_primary = np.full(rho.shape[0], np.nan, dtype=float)

MLD_density = clean_mld_values(np.array([
    compute_mld_from_density_profile(depth, rho[i, :], threshold=0.03, ref_depth=10.0)
    for i in range(rho.shape[0])
], dtype=float))

MLD_temp = clean_mld_values(np.array([
    compute_ild_from_temperature_profile(depth, T_np[i, :], threshold=0.2, ref_depth=10.0)
    for i in range(T_np.shape[0])
], dtype=float))

MLD_proxy = clean_mld_values(np.array([
    compute_mld_proxy_from_n2(depth_mid, N2[i, :], max_depth=100.0)
    for i in range(N2.shape[0])
], dtype=float))

mld_candidates = {
    "CMEMS_MLD": MLD_primary.copy(),
    "Density_MLD": MLD_density.copy(),
    "Temperature_ILD": MLD_temp.copy(),
    "N2_proxy_depth": MLD_proxy.copy()
}

candidate_scores = []
for name, arr in mld_candidates.items():
    finite_frac = np.mean(np.isfinite(arr))
    stdv = np.nanstd(arr)
    suspicious = suspicious_constant_series(arr)
    score = finite_frac * max(stdv, 0) * (0 if suspicious else 1)
    candidate_scores.append((name, finite_frac, stdv, suspicious, score))

candidate_df = pd.DataFrame(
    candidate_scores,
    columns=["name", "finite_fraction", "std_m", "suspicious_constant", "score"]
)
print(candidate_df.to_string(index=False))

best_name = candidate_df.sort_values(
    ["score", "finite_fraction", "std_m"], ascending=False
).iloc[0]["name"]

MLD_vals = mld_candidates[best_name].copy()
ranked_names = candidate_df.sort_values(
    ["score", "finite_fraction", "std_m"], ascending=False
)["name"].tolist()

for nm in ranked_names:
    arr = mld_candidates[nm]
    fill_mask = ~np.isfinite(MLD_vals) & np.isfinite(arr)
    MLD_vals[fill_mask] = arr[fill_mask]

MLD_vals = clean_mld_values(MLD_vals)

print(f"Selected primary MLD field: {best_name}")
print(f"Final MLD finite fraction : {np.mean(np.isfinite(MLD_vals)):.3f}")
print(f"Final MLD std (m)         : {np.nanstd(MLD_vals):.3f}")

# -------------------------
# 11) Build final daily dataframe
# -------------------------
daily_df = pd.DataFrame({
    "date": pd.to_datetime(times),
    "year": pd.to_datetime(times).year,
    "month": pd.to_datetime(times).month,
    "season": [season_from_month(m) for m in pd.to_datetime(times).month],
    "N2_shallow": N2_shallow,
    "N2_deep": N2_deep,
    "MLD": MLD_vals,
    "MLD_source_selected": best_name
})

daily_df = daily_df.merge(
    mhw_daily_full[["date", "is_mhw"]],
    on="date",
    how="left"
)

daily_df["is_mhw"] = daily_df["is_mhw"].fillna(0).astype(int)

print("\nDaily diagnostics:")
print("Daily rows:", len(daily_df))
print("MHW count :", int(daily_df["is_mhw"].sum()))
print("N2_shallow std:", float(daily_df["N2_shallow"].dropna().std(ddof=0)))
print("MLD std      :", float(daily_df["MLD"].dropna().std(ddof=0)))

# -------------------------
# 12) Annual metrics
# -------------------------
annual_df = (
    daily_df.groupby("year", as_index=False)
    .agg(
        N2_shallow=("N2_shallow", "mean"),
        N2_deep=("N2_deep", "mean"),
        MLD=("MLD", "mean"),
        MHW_days_from_daily=("is_mhw", "sum")
    )
)

annual_df = annual_df.merge(annual_mhw_days, on="year", how="inner")


# %% Cell 4
# -------------------------
# 13) Panel A metric
# -------------------------
threshold = annual_df["mhw_days"].quantile(HIGH_EXPOSURE_QUANTILE)
annual_df["high_exposure"] = annual_df["mhw_days"] >= threshold

high_vals = annual_df.loc[annual_df["high_exposure"], "N2_shallow"].values
other_vals = annual_df.loc[~annual_df["high_exposure"], "N2_shallow"].values

delta_n2, delta_n2_ci_low, delta_n2_ci_high, delta_n2_p = bootstrap_mean_diff(
    high_vals, other_vals, n_boot=N_BOOT, seed=RANDOM_SEED
)

# -------------------------
# 14) Panel B metric
# -------------------------
panel_b_df = prepare_panel_b_glm_df(daily_df)
glm_result = fit_panel_b_glm(panel_b_df)
panel_b_ok = glm_result is not None

if panel_b_ok:
    slope_b = glm_result.params[1]
    ci_b = glm_result.conf_int()
    slope_low, slope_high = ci_b[1, 0], ci_b[1, 1]
    p_b = glm_result.pvalues[1]

    try:
        pseudo_r2 = 1 - (glm_result.deviance / glm_result.null_deviance) if glm_result.null_deviance != 0 else np.nan
    except Exception:
        pseudo_r2 = np.nan

    try:
        rho_b, rho_p = stats.spearmanr(panel_b_df["x"].values, panel_b_df["y"].values, nan_policy="omit")
    except Exception:
        rho_b, rho_p = np.nan, np.nan
else:
    slope_b = slope_low = slope_high = p_b = pseudo_r2 = rho_b = rho_p = np.nan

reg_annual = regression_with_ci(
    annual_df["N2_shallow"].values,
    annual_df["mhw_days"].values
)

# -------------------------
# 15) Panel C metric
# -------------------------
mhw_mld = daily_df.loc[(daily_df["is_mhw"] == 1) & np.isfinite(daily_df["MLD"]), "MLD"].values
non_mhw_mld = daily_df.loc[(daily_df["is_mhw"] == 0) & np.isfinite(daily_df["MLD"]), "MLD"].values

panel_c_ok = (
    len(mhw_mld) >= 20 and
    len(non_mhw_mld) >= 20 and
    np.nanstd(mhw_mld) > 0 and
    np.nanstd(non_mhw_mld) > 0
)

if panel_c_ok:
    delta_mld, delta_mld_ci_low, delta_mld_ci_high, delta_mld_p = bootstrap_mean_diff(
        mhw_mld, non_mhw_mld, n_boot=N_BOOT, seed=RANDOM_SEED
    )
else:
    delta_mld, delta_mld_ci_low, delta_mld_ci_high, delta_mld_p = np.nan, np.nan, np.nan, np.nan

# -------------------------
# 16) Panel D metric: vertical N² composite profile
# -------------------------
mhw_idx = daily_df["is_mhw"].values == 1
non_idx = daily_df["is_mhw"].values == 0

N2_mhw_profiles = N2_profile_0_200[mhw_idx, :]
N2_non_profiles = N2_profile_0_200[non_idx, :]

mhw_mean_prof, mhw_low_prof, mhw_high_prof = bootstrap_profile_ci(
    N2_mhw_profiles, n_boot=1000, seed=RANDOM_SEED
)
non_mean_prof, non_low_prof, non_high_prof = bootstrap_profile_ci(
    N2_non_profiles, n_boot=1000, seed=RANDOM_SEED + 1
)

profile_df = pd.DataFrame({
    "depth_mid_m": depth_mid_0_200,
    "N2_MHW_mean": mhw_mean_prof,
    "N2_MHW_ci_low": mhw_low_prof,
    "N2_MHW_ci_high": mhw_high_prof,
    "N2_nonMHW_mean": non_mean_prof,
    "N2_nonMHW_ci_low": non_low_prof,
    "N2_nonMHW_ci_high": non_high_prof
})

# -------------------------
# 17) Save tables
# -------------------------
summary = pd.DataFrame([
    {
        "metric": "SHALLOW N2 ANOMALY DURING HIGH-EXPOSURE YEARS",
        "estimate": delta_n2,
        "ci_low": delta_n2_ci_low,
        "ci_high": delta_n2_ci_high,
        "p_value": delta_n2_p,
        "units": "s^-2"
    },
    {
        "metric": "DAILY MHW OCCURRENCE SCALING (binomial GLM)",
        "estimate": slope_b,
        "ci_low": slope_low,
        "ci_high": slope_high,
        "p_value": p_b,
        "units": "log-odds per 1-SD N2"
    },
    {
        "metric": "DAILY MHW OCCURRENCE SCALING pseudo-R2",
        "estimate": pseudo_r2,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
        "units": "unitless"
    },
    {
        "metric": "N2-MHW DAYS REGRESSION SLOPE (annual secondary)",
        "estimate": reg_annual["slope"],
        "ci_low": reg_annual["slope_ci_low"],
        "ci_high": reg_annual["slope_ci_high"],
        "p_value": reg_annual["p"],
        "units": "days per s^-2"
    },
    {
        "metric": "N2-MHW DAYS REGRESSION R2 (annual secondary)",
        "estimate": reg_annual["r2"],
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
        "units": "unitless"
    },
    {
        "metric": "MLD DIFFERENCE BETWEEN MHW AND NON-MHW DAILY WINDOWS",
        "estimate": delta_mld,
        "ci_low": delta_mld_ci_low,
        "ci_high": delta_mld_ci_high,
        "p_value": delta_mld_p,
        "units": "m"
    }
])

daily_df.to_csv(OUT_DAILY_CSV, index=False)
annual_df.to_csv(OUT_ANNUAL_CSV, index=False)
summary.to_csv(OUT_SUMMARY_CSV, index=False)
panel_b_df.to_csv(OUT_PANELB_CSV, index=False)
profile_df.to_csv(OUT_PROFILE_CSV, index=False)

daily_df.to_excel(OUT_DAILY_XLSX, index=False)
annual_df.to_excel(OUT_ANNUAL_XLSX, index=False)
summary.to_excel(OUT_SUMMARY_XLSX, index=False)
panel_b_df.to_excel(OUT_PANELB_XLSX, index=False)
profile_df.to_excel(OUT_PROFILE_XLSX, index=False)

with pd.ExcelWriter(OUT_WORKBOOK_XLSX, engine="openpyxl") as writer:
    daily_df.to_excel(writer, sheet_name="daily_metrics", index=False)
    annual_df.to_excel(writer, sheet_name="annual_metrics", index=False)
    summary.to_excel(writer, sheet_name="summary_metrics", index=False)
    panel_b_df.to_excel(writer, sheet_name="panelB_input", index=False)
    profile_df.to_excel(writer, sheet_name="vertical_N2_profiles", index=False)
    candidate_df.to_excel(writer, sheet_name="MLD_candidate_scores", index=False)


# %% Cell 5
# -------------------------
# 18) Figure 1: exact 3-panel reproduction
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.9), constrained_layout=False)
fig.patch.set_facecolor("white")

# Panel A
ax = axes[0]
box = ax.boxplot(
    [high_vals, other_vals],
    tick_labels=["High-exposure\nyears", "Other\nyears"],
    patch_artist=True,
    widths=0.55,
    showfliers=False
)
for patch, color in zip(box["boxes"], ["#d95f02", "#1b9e77"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.78)

ax.set_ylabel("Annual shallow N² (s$^{-2}$)")
ax.set_title("(a) Shallow N² anomaly")
ax.text(
    0.03, 0.97,
    f"ΔN² = {delta_n2:.2e}\n95% CI: {delta_n2_ci_low:.2e} to {delta_n2_ci_high:.2e}\np = {delta_n2_p:.3g}",
    transform=ax.transAxes, va="top", ha="left", fontsize=9
)

# Panel B
ax = axes[1]
if panel_b_ok:
    rng = np.random.default_rng(RANDOM_SEED)
    y_jitter = panel_b_df["y"].values + rng.normal(0, 0.03, len(panel_b_df))
    y_jitter = np.clip(y_jitter, -0.02, 1.02)

    plot_idx = np.arange(len(panel_b_df))
    if len(plot_idx) > 5000:
        plot_idx = rng.choice(plot_idx, size=5000, replace=False)

    ax.scatter(
        panel_b_df["x"].values[plot_idx],
        y_jitter[plot_idx],
        s=7,
        color="#2c7fb8",
        edgecolor="none",
        alpha=0.28,
        zorder=2
    )

    xline = np.linspace(panel_b_df["x"].min(), panel_b_df["x"].max(), 300)
    exog_line = sm.add_constant(xline)
    yline = glm_result.predict(exog_line)
    ax.plot(xline, yline, color="#d95f0e", linewidth=2.1, zorder=3)

    ax.text(
        0.03, 0.97,
        f"Daily binomial GLM\nSlope = {slope_b:.3f}\n95% CI: {slope_low:.3f} to {slope_high:.3f}\n"
        f"pseudo-R² = {pseudo_r2:.3f}\np = {p_b:.3g}\nSpearman ρ = {rho_b:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9
    )
else:
    ax.text(
        0.5, 0.5,
        "Insufficient finite\nGLM samples",
        transform=ax.transAxes, ha="center", va="center", fontsize=13
    )

ax.set_xlabel("Season-standardized shallow N²")
ax.set_ylabel("Daily MHW occurrence")
ax.set_ylim(-0.02, 1.02)
ax.set_title("(b) N²–MHW occurrence scaling")

# Panel C
ax = axes[2]
if panel_c_ok:
    box = ax.boxplot(
        [mhw_mld, non_mhw_mld],
        tick_labels=["MHW\ndays", "Non-MHW\ndays"],
        patch_artist=True,
        widths=0.55,
        showfliers=False
    )
    for patch, color in zip(box["boxes"], ["#ef3b2c", "#6baed6"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)

    rng = np.random.default_rng(RANDOM_SEED + 1)
    mhw_idx_plot = np.arange(len(mhw_mld))
    non_idx_plot = np.arange(len(non_mhw_mld))

    if len(mhw_idx_plot) > 1200:
        mhw_idx_plot = rng.choice(mhw_idx_plot, size=1200, replace=False)
    if len(non_idx_plot) > 1200:
        non_idx_plot = rng.choice(non_idx_plot, size=1200, replace=False)

    ax.scatter(
        np.ones(len(mhw_idx_plot)) * 1 + rng.normal(0, 0.03, len(mhw_idx_plot)),
        mhw_mld[mhw_idx_plot],
        s=5, color="black", alpha=0.12, zorder=2
    )
    ax.scatter(
        np.ones(len(non_idx_plot)) * 2 + rng.normal(0, 0.03, len(non_idx_plot)),
        non_mhw_mld[non_idx_plot],
        s=5, color="black", alpha=0.12, zorder=2
    )

    ax.text(
        0.03, 0.97,
        f"ΔMLD = {delta_mld:.2f} m\n95% CI: {delta_mld_ci_low:.2f} to {delta_mld_ci_high:.2f}\np = {delta_mld_p:.3g}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9
    )
else:
    ax.text(
        0.5, 0.5,
        "Insufficient finite\nMLD samples",
        transform=ax.transAxes, ha="center", va="center", fontsize=13
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["MHW\ndays", "Non-MHW\ndays"])

ax.set_ylabel("MLD (m)")
ax.set_title("(c) MLD contrast")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.45, alpha=0.45)

fig.suptitle(
    "Upper-ocean persistence diagnostics for Bay of Bengal marine heatwaves (1995–2024)",
    fontsize=16, fontweight="bold", y=0.98
)
fig.subplots_adjust(left=0.065, right=0.985, bottom=0.15, top=0.84, wspace=0.32)

savefig_png(fig, OUT_FIG3_PNG, dpi=DPI)
plt.show()
plt.close(fig)

# -------------------------
# 19) Figure 2: stronger 4-panel persistence mechanism figure
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2), constrained_layout=False)
fig.patch.set_facecolor("white")

# (a) same boxplot
ax = axes[0, 0]
box = ax.boxplot(
    [high_vals, other_vals],
    tick_labels=["High-exposure\nyears", "Other\nyears"],
    patch_artist=True,
    widths=0.55,
    showfliers=False
)
for patch, color in zip(box["boxes"], ["#d95f02", "#1b9e77"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.78)

ax.set_ylabel("Annual shallow N² (s$^{-2}$)")
ax.set_title("(a) Shallow N² anomaly")
ax.text(
    0.03, 0.97,
    f"ΔN² = {delta_n2:.2e}\n95% CI: {delta_n2_ci_low:.2e} to {delta_n2_ci_high:.2e}\np = {delta_n2_p:.3g}",
    transform=ax.transAxes, va="top", ha="left", fontsize=9
)

# (b) same GLM
ax = axes[0, 1]
if panel_b_ok:
    rng = np.random.default_rng(RANDOM_SEED)
    y_jitter = panel_b_df["y"].values + rng.normal(0, 0.03, len(panel_b_df))
    y_jitter = np.clip(y_jitter, -0.02, 1.02)

    plot_idx = np.arange(len(panel_b_df))
    if len(plot_idx) > 5000:
        plot_idx = rng.choice(plot_idx, size=5000, replace=False)

    ax.scatter(
        panel_b_df["x"].values[plot_idx],
        y_jitter[plot_idx],
        s=7,
        color="#2c7fb8",
        edgecolor="none",
        alpha=0.25,
        zorder=2
    )

    xline = np.linspace(panel_b_df["x"].min(), panel_b_df["x"].max(), 300)
    exog_line = sm.add_constant(xline)
    yline = glm_result.predict(exog_line)
    ax.plot(xline, yline, color="#d95f0e", linewidth=2.0, zorder=3)

    ax.text(
        0.03, 0.97,
        f"Daily binomial GLM\nSlope = {slope_b:.3f}\n95% CI: {slope_low:.3f} to {slope_high:.3f}\n"
        f"pseudo-R² = {pseudo_r2:.3f}\np = {p_b:.3g}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9
    )
else:
    ax.text(0.5, 0.5, "Insufficient finite\nGLM samples",
            transform=ax.transAxes, ha="center", va="center", fontsize=13)

ax.set_xlabel("Season-standardized shallow N²")
ax.set_ylabel("Daily MHW occurrence")
ax.set_ylim(-0.02, 1.02)
ax.set_title("(b) N²–MHW occurrence scaling")

# (c) same MLD contrast
ax = axes[1, 0]
if panel_c_ok:
    box = ax.boxplot(
        [mhw_mld, non_mhw_mld],
        tick_labels=["MHW\ndays", "Non-MHW\ndays"],
        patch_artist=True,
        widths=0.55,
        showfliers=False
    )
    for patch, color in zip(box["boxes"], ["#ef3b2c", "#6baed6"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)

    rng = np.random.default_rng(RANDOM_SEED + 1)
    mhw_idx_plot = np.arange(len(mhw_mld))
    non_idx_plot = np.arange(len(non_mhw_mld))

    if len(mhw_idx_plot) > 1000:
        mhw_idx_plot = rng.choice(mhw_idx_plot, size=1000, replace=False)
    if len(non_idx_plot) > 1000:
        non_idx_plot = rng.choice(non_idx_plot, size=1000, replace=False)

    ax.scatter(
        np.ones(len(mhw_idx_plot)) * 1 + rng.normal(0, 0.03, len(mhw_idx_plot)),
        mhw_mld[mhw_idx_plot],
        s=5, color="black", alpha=0.10, zorder=2
    )
    ax.scatter(
        np.ones(len(non_idx_plot)) * 2 + rng.normal(0, 0.03, len(non_idx_plot)),
        non_mhw_mld[non_idx_plot],
        s=5, color="black", alpha=0.10, zorder=2
    )

    ax.text(
        0.03, 0.97,
        f"ΔMLD = {delta_mld:.2f} m\n95% CI: {delta_mld_ci_low:.2f} to {delta_mld_ci_high:.2f}\np = {delta_mld_p:.3g}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9
    )
else:
    ax.text(0.5, 0.5, "Insufficient finite\nMLD samples",
            transform=ax.transAxes, ha="center", va="center", fontsize=13)

ax.set_ylabel("MLD (m)")
ax.set_title("(c) MLD contrast")

# (d) vertical N² composite profile
ax = axes[1, 1]
if mhw_mean_prof is not None and non_mean_prof is not None:
    ax.plot(non_mean_prof, depth_mid_0_200, color="#6baed6", linewidth=2.0, label="Non-MHW days")
    ax.fill_betweenx(depth_mid_0_200, non_low_prof, non_high_prof, color="#6baed6", alpha=0.20)

    ax.plot(mhw_mean_prof, depth_mid_0_200, color="#ef3b2c", linewidth=2.0, label="MHW days")
    ax.fill_betweenx(depth_mid_0_200, mhw_low_prof, mhw_high_prof, color="#ef3b2c", alpha=0.20)

    ax.legend(frameon=False, loc="lower right")
else:
    ax.text(0.5, 0.5, "Insufficient finite\nN² profiles",
            transform=ax.transAxes, ha="center", va="center", fontsize=13)

ax.invert_yaxis()
ax.set_xlabel("Brunt–Väisälä frequency, N² (s$^{-2}$)")
ax.set_ylabel("Depth (m)")
ax.set_title("(d) Vertical N² structure during MHW vs non-MHW days")

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", linestyle=":", linewidth=0.40, alpha=0.40)

fig.suptitle(
    "Enhanced persistence mechanism diagnostics for Bay of Bengal marine heatwaves (1995–2024)",
    fontsize=16, fontweight="bold", y=0.98
)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90, wspace=0.28, hspace=0.28)

savefig_png(fig, OUT_FIG4_PNG, dpi=DPI)
plt.show()
plt.close(fig)

# -------------------------
# 20) Console summary
# -------------------------
print("\n================ FINAL METRICS ================\n")
print(summary.to_string(index=False))

print("\nSaved tables:")
print(OUT_DAILY_CSV)
print(OUT_ANNUAL_CSV)
print(OUT_SUMMARY_CSV)
print(OUT_PANELB_CSV)
print(OUT_PROFILE_CSV)

print("\nSaved Excel files:")
print(OUT_WORKBOOK_XLSX)

print("\nSaved figures:")
print(OUT_FIG3_PNG)
print(OUT_FIG4_PNG)
