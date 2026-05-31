#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07 Event Chronology Severity

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
# #**Bay of Bengal MHW integrated chronology**


# %% Cell 2
# ============================================================
# Bay of Bengal MHW integrated chronology figure (1995-2025)
# AUTHENTIC EVENT-SUMMARY-DRIVEN VERSION
# ------------------------------------------------------------
# What this fixes:
# 1) Reads ALL yearly sheets from the authentic Excel workbook
# 2) Uses the Excel event summary as the authoritative event source
# 3) Builds annual panels strictly from the Excel summary
# 4) Shades panel (a) using the Excel event windows, not auto-detected windows
# 5) Derives event severity consistently from daily SST, climatology, and threshold
# 6) Produces clean, publication-ready 1200 dpi PNG + TIFF + PDF outputs
# ============================================================

# -------------------------
# 1) Install dependencies
# -------------------------
# In Colab, uncomment if needed:
# !pip -q install xarray netCDF4 bottleneck cftime scipy pandas matplotlib numpy dask openpyxl

# -------------------------
# 2) Imports
# -------------------------
import os
import gc
import re
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

# -------------------------
# 3) Google Drive mount (Colab)
# -------------------------
try:
# Google Colab Drive import removed; use config/paths_template.yaml and local data folders.
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_COLAB:
    # Google Drive mount removed; configure DATA_ROOT in this script or via environment variable MHW_DATA_ROOT.
    pass

# -------------------------
# 4) USER SETTINGS
# -------------------------
DATA_DIR = str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)/MHWs Category Datasets & Outputs/MHW SST Indices Data (1995-2025)")
EVENT_SUMMARY_XLSX = str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)/MHWs Category Datasets & Outputs/MHW SST Indices Data (1995-2025)/MHWs event summary.xlsx")
OUTDIR = str(DATA_ROOT / "BoB_MHW_FigureOutputs")
os.makedirs(OUTDIR, exist_ok=True)

OUT_FIG_PNG = os.path.join(OUTDIR, "Figure_7_BoB_MHW_Integrated_Chronology_1995_2025_1200dpi.png")
OUT_FIG_TIFF = os.path.join(OUTDIR, "Figure_7_BoB_MHW_Integrated_Chronology_1995_2025_1200dpi.tiff")
OUT_FIG_PDF = os.path.join(OUTDIR, "Figure_7_BoB_MHW_Integrated_Chronology_1995_2025.pdf")
OUT_EVENTS_CSV = os.path.join(OUTDIR, "BoB_MHW_event_summary_cleaned_1995_2025.csv")
OUT_ANNUAL_CSV = os.path.join(OUTDIR, "BoB_MHW_annual_summary_cleaned_1995_2025.csv")
OUT_EVENT_DETAIL_XLSX = os.path.join(OUTDIR, "BoB_MHW_event_summary_enriched_1995_2025.xlsx")

FILE_PATTERNS = [
    "sst.day.anom.*.nc",
    "sst.day.mean.*.nc",
    "*.nc",
]

TIME_CANDIDATES = ["time", "TIME", "t"]
LAT_CANDIDATES = ["lat", "latitude", "LAT", "nav_lat", "y"]
LON_CANDIDATES = ["lon", "longitude", "LON", "nav_lon", "x"]
SST_CANDIDATES = ["sst", "analysed_sst", "sea_surface_temperature", "tos", "temp", "temperature"]

LAT_MIN, LAT_MAX = 5.0, 22.0
LON_MIN, LON_MAX = 80.0, 100.0
START_DATE = "1995-01-01"
END_DATE   = "2025-12-31"
PCTILE = 90
WINDOW_HALF_WIDTH = 5
SMOOTH_WINDOW = 31
AUTO_K_TO_C = True
DPI = 1200


CATEGORY_COLORS = {
    "Moderate": "#f4a582",
    "Strong":   "#d6604d",
    "Severe":   "#b2182b",
    "Extreme":  "#b2182b",
}

# -------------------------
# 5) Plot style
# -------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 20,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

# -------------------------
# 6) Helper functions
# -------------------------
def find_coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims or name in ds.variables:
            return name
    raise ValueError(f"Coordinate not found. Candidates tried: {candidates}")


def find_data_var(ds, candidates):
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise ValueError(f"SST variable not found. Candidates tried: {candidates}")


def ensure_lon_0_360(ds_or_da, lon_name):
    lon = ds_or_da[lon_name].values
    if np.nanmin(lon) < 0:
        ds_or_da = ds_or_da.assign_coords({lon_name: lon % 360}).sortby(lon_name)
    return ds_or_da


def subset_bob(da, lat_name, lon_name):
    if da[lat_name].values[0] > da[lat_name].values[-1]:
        da = da.sortby(lat_name)
    return da.sel({lat_name: slice(LAT_MIN, LAT_MAX), lon_name: slice(LON_MIN, LON_MAX)})


def area_weighted_mean(da, lat_name, lon_name):
    weights = np.cos(np.deg2rad(da[lat_name]))
    weights = weights / weights.mean()
    return da.weighted(weights).mean(dim=[lat_name, lon_name], skipna=True)


def safe_to_datetime(values):
    try:
        return pd.to_datetime(values)
    except Exception:
        return pd.to_datetime([str(v) for v in values], errors="coerce")


def dayofyear_365(dates):
    dates = pd.Series(pd.to_datetime(dates))
    doy = dates.dt.dayofyear.to_numpy().astype(int)
    leap = dates.dt.is_leap_year.to_numpy()
    month = dates.dt.month.to_numpy()
    day = dates.dt.day.to_numpy()

    feb29 = leap & (month == 2) & (day == 29)
    after_feb29 = leap & ((month > 2) | ((month == 2) & (day > 29)))

    doy[feb29] = 59
    doy[after_feb29] -= 1
    return doy


def moving_average_circular(arr, window):
    if window <= 1:
        return arr.copy()
    pad = window // 2
    arr_pad = np.concatenate([arr[-pad:], arr, arr[:pad]])
    kernel = np.ones(window, dtype=float) / window
    smooth = np.convolve(arr_pad, kernel, mode="same")
    return smooth[pad:-pad]


def compute_clim_thresh(dates, sst, pctile=90, window_half_width=5, smooth_window=31):
    doy = dayofyear_365(dates)
    clim_mean = np.full(365, np.nan)
    clim_thresh = np.full(365, np.nan)

    for d in range(1, 366):
        days = [((d - 1 + off) % 365) + 1 for off in range(-window_half_width, window_half_width + 1)]
        mask = np.isin(doy, days) & np.isfinite(sst)
        if np.any(mask):
            clim_mean[d - 1] = np.nanmean(sst[mask])
            clim_thresh[d - 1] = np.nanpercentile(sst[mask], pctile)

    clim_mean = moving_average_circular(clim_mean, smooth_window)
    clim_thresh = moving_average_circular(clim_thresh, smooth_window)
    return clim_mean[doy - 1], clim_thresh[doy - 1]


def load_event_summary_all_sheets(xlsx_path):
    xls = pd.ExcelFile(xlsx_path)
    frames = []

    for sheet in xls.sheet_names:
        if not str(sheet).strip().isdigit():
            continue
        yr = int(str(sheet).strip())
        if yr < 1995 or yr > 2025:
            continue

        tmp = pd.read_excel(xlsx_path, sheet_name=sheet)
        if tmp.empty:
            continue

        tmp.columns = [str(c).strip().lower().replace(" ", "_") for c in tmp.columns]
        tmp = tmp.drop(columns=[c for c in tmp.columns if c.startswith("unnamed")], errors="ignore")

        rename_map = {
            "start": "start_date",
            "end": "end_date",
            "duration": "duration_days",
            "duration_day": "duration_days",
            "peak_intensity": "peak_intensity_c",
            "mean_intensity": "mean_intensity_c",
            "cumulative_intensity": "cumulative_intensity_c",
        }
        tmp = tmp.rename(columns={c: rename_map[c] for c in tmp.columns if c in rename_map})

        required = ["start_date", "end_date", "duration_days", "peak_intensity_c", "mean_intensity_c", "cumulative_intensity_c"]
        missing = [c for c in required if c not in tmp.columns]
        if missing:
            raise ValueError(f"Sheet {sheet} is missing required columns: {missing}")

        tmp["start_date"] = pd.to_datetime(tmp["start_date"], errors="coerce")
        tmp["end_date"] = pd.to_datetime(tmp["end_date"], errors="coerce")
        for c in ["duration_days", "peak_intensity_c", "mean_intensity_c", "cumulative_intensity_c"]:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

        tmp["year"] = yr
        tmp["event_id"] = [f"{yr}_{i+1:02d}" for i in range(len(tmp))]
        frames.append(tmp)

    if not frames:
        raise ValueError("No valid yearly event sheets were found in the Excel workbook.")

    events = pd.concat(frames, ignore_index=True)
    events = events.dropna(subset=["start_date", "end_date"]).copy()
    events = events[(events["start_date"] >= pd.Timestamp(START_DATE)) & (events["end_date"] <= pd.Timestamp(END_DATE))].copy()
    events = events.sort_values(["start_date", "end_date"]).reset_index(drop=True)
    return events


def event_category_from_peak_ratio(peak_intensity, threshold_excess):
    if pd.isna(peak_intensity) or pd.isna(threshold_excess) or threshold_excess <= 0:
        return "Moderate"
    ratio = peak_intensity / threshold_excess
    if ratio < 1:
        return "Moderate"
    if ratio < 2:
        return "Strong"
    if ratio < 3:
        return "Severe"
    return "Extreme"


def add_event_severity_from_daily_context(events, daily_df):
    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.set_index("date")

    cats = []
    threshold_excess_mean = []
    threshold_excess_peak = []
    sst_peak_obs = []

    for _, row in events.iterrows():
        start = pd.Timestamp(row["start_date"])
        end = pd.Timestamp(row["end_date"])
        sub = daily_df.loc[start:end].copy()

        if sub.empty:
            cats.append("Moderate")
            threshold_excess_mean.append(np.nan)
            threshold_excess_peak.append(np.nan)
            sst_peak_obs.append(np.nan)
            continue

        sub["threshold_excess"] = sub["threshold"] - sub["climatology"]
        te_mean = float(np.nanmean(sub["threshold_excess"])) if np.isfinite(sub["threshold_excess"]).any() else np.nan
        te_peak = float(np.nanmax(sub["threshold_excess"])) if np.isfinite(sub["threshold_excess"]).any() else np.nan
        observed_peak_sst = float(np.nanmax(sub["sst"])) if np.isfinite(sub["sst"]).any() else np.nan

        cat = event_category_from_peak_ratio(row["peak_intensity_c"], te_mean)
        cats.append(cat)
        threshold_excess_mean.append(te_mean)
        threshold_excess_peak.append(te_peak)
        sst_peak_obs.append(observed_peak_sst)

    out = events.copy()
    out["category"] = cats
    out["threshold_excess_mean_c"] = threshold_excess_mean
    out["threshold_excess_peak_c"] = threshold_excess_peak
    out["observed_peak_sst_c"] = sst_peak_obs
    return out


def build_annual_summary(events):
    annual = pd.DataFrame({"year": np.arange(1995, 2026)})

    grp = events.groupby("year")
    annual = annual.merge(grp.size().rename("event_count"), how="left", on="year")
    annual = annual.merge(grp["duration_days"].sum().rename("mhw_days"), how="left", on="year")
    annual = annual.merge(grp["peak_intensity_c"].max().rename("max_intensity_c"), how="left", on="year")

    if "category" in events.columns:
        sev_rank = {"Moderate": 1, "Strong": 2, "Severe": 3, "Extreme": 4}
        tmp = events.copy()
        tmp["severity_rank"] = tmp["category"].map(sev_rank)
        annual = annual.merge(tmp.groupby("year")["severity_rank"].max().rename("max_severity_rank"), how="left", on="year")
    else:
        annual["max_severity_rank"] = np.nan

    annual["event_count"] = annual["event_count"].fillna(0).astype(int)
    annual["mhw_days"] = annual["mhw_days"].fillna(0).astype(int)
    return annual


def add_event_shading_from_excel(ax, events):
    for _, row in events.iterrows():
        s = pd.Timestamp(row["start_date"])
        e = pd.Timestamp(row["end_date"])
        cat = row.get("category", "Moderate")
        ax.axvspan(s, e + pd.Timedelta(days=1), color=CATEGORY_COLORS.get(cat, CATEGORY_COLORS["Moderate"]), alpha=0.22, lw=0, zorder=0)


def plot_event_timeline(ax, events):
    cat_to_y = {"Moderate": 1, "Strong": 2, "Severe": 3, "Extreme": 4}
    for _, row in events.iterrows():
        s = pd.Timestamp(row["start_date"])
        e = pd.Timestamp(row["end_date"])
        width_days = max(1, (e - s).days + 1)
        ax.barh(
            y=cat_to_y.get(row["category"], 1),
            width=width_days,
            left=mdates.date2num(s),
            height=0.60,
            color=CATEGORY_COLORS.get(row["category"], CATEGORY_COLORS["Moderate"]),
            edgecolor="none",
            alpha=0.95,
            zorder=3,
        )


def add_panel_label(ax, label):
    ax.text(0.01, 0.92, label, transform=ax.transAxes, fontsize=16, fontweight="bold")

# -------------------------
# 7) Read and trim SST series to 1995-2025
# -------------------------
all_files = []
for pattern in FILE_PATTERNS:
    all_files.extend(glob.glob(os.path.join(DATA_DIR, pattern)))
all_files = sorted(list(set(all_files)))
files = [f for f in all_files if ("ltm" not in os.path.basename(f).lower()) and ("clim" not in os.path.basename(f).lower())]

if len(files) == 0:
    raise FileNotFoundError(f"No usable NetCDF files found in {DATA_DIR}")

print(f"Processing {len(files)} NetCDF files...")
ds = xr.open_mfdataset(files, combine="by_coords", chunks={}, coords="minimal", data_vars="minimal", compat="override", parallel=False)

t_name   = find_coord_name(ds, TIME_CANDIDATES)
lat_name = find_coord_name(ds, LAT_CANDIDATES)
lon_name = find_coord_name(ds, LON_CANDIDATES)
sst_name = find_data_var(ds, SST_CANDIDATES)

da = ds[sst_name]
da = ensure_lon_0_360(da, lon_name)
da = subset_bob(da, lat_name, lon_name)

sample_mean = float(da.isel({t_name: 0}).mean(skipna=True).compute())
if AUTO_K_TO_C and sample_mean > 100:
    da = da - 273.15
    print("Converted SST from Kelvin to Celsius")

sst_series = area_weighted_mean(da, lat_name, lon_name).compute()
times = safe_to_datetime(sst_series[t_name].values)

df = pd.DataFrame({"date": times, "sst": sst_series.values}).dropna().sort_values("date").reset_index(drop=True)
df = df[(df["date"] >= pd.Timestamp(START_DATE)) & (df["date"] <= pd.Timestamp(END_DATE))].copy()
df = df.reset_index(drop=True)

del ds, da, sst_series
gc.collect()
print(f"SST record trimmed to {df['date'].dt.year.min()}-{df['date'].dt.year.max()}")


# %% Cell 3
# -------------------------
# 10) Plot polished figure
# -------------------------
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter

# -------------------------
# Global plot style
# -------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize": 22,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# Wider left margin + a bit more height for cleaner y-label fitting
fig = plt.figure(figsize=(22, 10), constrained_layout=False)
gs = fig.add_gridspec(
    nrows=5,
    ncols=1,
    height_ratios=[3.6, 1.10, 1.00, 1.00, 1.00],
    hspace=0.22
)

axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1], sharex=axA)
axC1 = fig.add_subplot(gs[2], sharex=axA)
axC2 = fig.add_subplot(gs[3], sharex=axA)
axC3 = fig.add_subplot(gs[4], sharex=axA)
axes = [axA, axB, axC1, axC2, axC3]

xmin = pd.Timestamp(START_DATE)
xmax = pd.Timestamp(END_DATE)
for ax in axes:
    ax.set_xlim(xmin, xmax)

# -------------------------
# Panel (a): daily chronology
# -------------------------
add_event_shading_from_excel(axA, event_df)

axA.plot(
    df["date"], df["sst"],
    color="black", linewidth=0.75, label="Daily SST", zorder=3
)
axA.plot(
    df["date"], df["climatology"],
    color="#1f77b4", linestyle="--", linewidth=1.0,
    label="Seasonal climatology", zorder=2
)
axA.plot(
    df["date"], df["threshold"],
    color="#2ca02c", linewidth=1.05,
    label="90th percentile threshold", zorder=2
)

axA.set_ylabel("SST (°C)", fontsize=12, fontweight= "bold")
axA.grid(True, linestyle=":", linewidth=0.45, alpha=0.35)
axA.set_ylim(np.nanmin(df["sst"]) - 0.5, np.nanmax(df["sst"]) + 0.5)
add_panel_label(axA, "(a)")

legend_handles = [
    plt.Line2D([0], [0], color="black", lw=1.0, label="Daily SST"),
    plt.Line2D([0], [0], color="#1f77b4", lw=1.0, ls="--", label="Seasonal climatology"),
    plt.Line2D([0], [0], color="#2ca02c", lw=1.0, label="90th percentile threshold"),
    Patch(facecolor=CATEGORY_COLORS["Moderate"], edgecolor="none", alpha=0.35, label="Moderate"),
    Patch(facecolor=CATEGORY_COLORS["Strong"], edgecolor="none", alpha=0.35, label="Strong"),
    Patch(facecolor=CATEGORY_COLORS["Severe"], edgecolor="none", alpha=0.35, label="Severe"),
]
axA.legend(
    handles=legend_handles,
    ncol=6,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.16),
    frameon=False,
    handlelength=1.8,
    columnspacing=1.2,
    handletextpad=0.5
)

# -------------------------
# Panel (b): event severity timeline
# -------------------------
plot_event_timeline(axB, event_df)
axB.set_ylim(0.5, 3.5)
axB.set_yticks([1, 2, 3])
axB.set_yticklabels(["Moderate", "Strong", "Severe"])
axB.set_ylabel("Severity", fontsize=12 , fontweight="bold")
axB.grid(True, axis="x", linestyle=":", linewidth=0.45, alpha=0.30)
add_panel_label(axB, "(b)")

# -------------------------
# Panel (c1): annual event count
# -------------------------
bar_dates = pd.to_datetime([f"{y}-07-01" for y in annual["year"]])

axC1.bar(
    bar_dates, annual["event_count"],
    width=220, color="#5f5f5f", edgecolor="none"
)
axC1.set_ylabel("Events yr$^{-1}$", fontsize=12 , fontweight="bold")
axC1.grid(True, axis="y", linestyle=":", linewidth=0.45, alpha=0.30)
axC1.yaxis.set_major_locator(MaxNLocator(integer=True))
add_panel_label(axC1, "(c)")

# -------------------------
# Panel (c2): annual MHW days
# Required limit: 20–40
# -------------------------
axC2.bar(
    bar_dates, annual["mhw_days"],
    width=220, color="#2b83ba", edgecolor="none"
)
axC2.set_ylabel("MHW \n days yr$^{-1}$", fontsize=12, fontweight="bold")
axC2.grid(True, axis="y", linestyle=":", linewidth=0.45, alpha=0.30)

# Force axis range and clean ticks
axC2.set_ylim(20, 40)
axC2.yaxis.set_major_locator(FixedLocator([20, 25, 30, 35, 40]))
axC2.yaxis.set_major_formatter(FixedFormatter(["20", "25", "30", "35", "40"]))

# -------------------------
# Panel (c3): annual maximum peak intensity
# Required ticks: 0, 0.5, 1
# -------------------------
axC3.bar(
    bar_dates, annual["max_intensity_c"],
    width=220, color="#ef3b2c", edgecolor="none"
)
axC3.set_ylabel("Max \n intensity (°C)", fontsize=12 , fontweight="bold")
axC3.grid(True, axis="y", linestyle=":", linewidth=0.45, alpha=0.30)

# Force axis range and ticks
axC3.set_ylim(0, 1.0)
axC3.yaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
axC3.yaxis.set_major_formatter(FixedFormatter(["0", "0.5", "1"]))

# -------------------------
# Shared x-axis formatting
# -------------------------
axC3.set_xlabel("Year", fontsize=14)
axC3.xaxis.set_major_locator(mdates.YearLocator(5))
axC3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axC3.tick_params(axis="x", rotation=0)

# Hide upper-panel x tick labels
for ax in [axA, axB, axC1, axC2]:
    plt.setp(ax.get_xticklabels(), visible=False)

# Clean aesthetics
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=11)

# Slightly reduce y-tick label size for panel (b) so it fits neatly
axB.tick_params(axis="y", labelsize=12)

# Footer summary from Excel-authenticated event table
n_events = len(event_df)
max_sst = float(np.nanmax(df["sst"]))
max_intensity = float(np.nanmax(event_df["peak_intensity_c"])) if event_df["peak_intensity_c"].notna().any() else np.nan
peak_year = int(event_df.loc[event_df["peak_intensity_c"].idxmax(), "year"]) if event_df["peak_intensity_c"].notna().any() else np.nan

fig.text(
    0.012, 0.012,
    f"Authenticated Excel summary indicates {n_events} MHW events during 1995–2025. "
    f"Maximum basin-mean SST = {max_sst:.2f} °C; maximum event peak intensity above threshold = {max_intensity:.2f} °C "
    f"(year {peak_year}).",
    fontsize=11
)

# IMPORTANT: better fitting margins
fig.subplots_adjust(
    left=0.11,
    right=0.985,
    top=0.88,
    bottom=0.11,
    hspace=0.22
)

fig.savefig(OUT_FIG_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_FIG_TIFF, dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_FIG_PDF, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)

print(f"[OK] Figure saved to: {OUT_FIG_PNG}")
print(f"[OK] Figure saved to: {OUT_FIG_TIFF}")
print(f"[OK] Figure saved to: {OUT_FIG_PDF}")
print(f"[OK] Cleaned event summary saved to: {OUT_EVENTS_CSV}")
print(f"[OK] Cleaned annual summary saved to: {OUT_ANNUAL_CSV}")
print(f"[OK] Enriched workbook saved to: {OUT_EVENT_DETAIL_XLSX}")
