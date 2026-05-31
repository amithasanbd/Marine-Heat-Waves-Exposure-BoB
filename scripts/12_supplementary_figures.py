#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12 Supplementary Figures

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
# # **S5_BoB_MHW_Expanded_Daily_Chronology_2010_2025**


# %% Cell 2
import os
import re
import gc
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# -------------------------
# 3) Mount Google Drive
# -------------------------
# Google Colab Drive import removed; use config/paths_template.yaml and local data folders.
# Google Drive mount removed; configure DATA_ROOT in this script or via environment variable MHW_DATA_ROOT.

# -------------------------
# 4) USER SETTINGS
# -------------------------
# Daily SST directory used for the main chronology / Figure 8
DATA_DIR = str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)/MHWs Category Datasets & Outputs/MHW SST Indices Data (1995-2025)")

# Authenticated event-summary file used in main chronology
EVENT_SUMMARY_XLSX = str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)/MHWs Category Datasets & Outputs/MHW SST Indices Data (1995-2025)/MHWs event summary.xlsx")

OUTDIR = str(DATA_ROOT / "BoB_Figure_S5_Late_Record_Chronology")
os.makedirs(OUTDIR, exist_ok=True)

OUT_FIG_PNG = os.path.join(OUTDIR, "Figure_S5_BoB_MHW_Expanded_Daily_Chronology_2010_2025_1200dpi.png")
OUT_FIG_PDF = os.path.join(OUTDIR, "Figure_S5_BoB_MHW_Expanded_Daily_Chronology_2010_2025.pdf")
OUT_GAP_CSV = os.path.join(OUTDIR, "Figure_S5_late_record_event_gap_summary.csv")
OUT_ANN_CSV = os.path.join(OUTDIR, "Figure_S5_late_record_annual_metrics.csv")

FILE_PATTERNS = [
    "sst.day.anom.*.nc",
    "sst.day.mean.*.nc",
    "*.nc",
]

TIME_CANDIDATES = ["time", "TIME", "t"]
LAT_CANDIDATES  = ["lat", "latitude", "LAT", "nav_lat", "y"]
LON_CANDIDATES  = ["lon", "longitude", "LON", "nav_lon", "x"]
SST_CANDIDATES  = ["sst", "analysed_sst", "sea_surface_temperature", "tos", "temp", "temperature"]

LAT_MIN, LAT_MAX = 5.0, 22.0
LON_MIN, LON_MAX = 80.0, 100.0

FULL_START = "1995-01-01"
FULL_END   = "2025-12-31"

# Late-record window for Figure S5
PLOT_START = "2010-01-01"
PLOT_END   = "2025-12-31"

PCTILE = 90
WINDOW_HALF_WIDTH = 5
SMOOTH_WINDOW = 31
AUTO_K_TO_C = True
DPI = 1200

# If a recovery gap is shorter than this, annotate it in panel (b)
ANNOTATE_GAPS_LEQ_DAYS = 90

# Only label strongest / shortest gaps if crowded
MAX_GAP_LABELS = 18

FIG_TITLE = "Figure S5. Expanded daily chronology of Bay of Bengal marine heatwaves during the late-record intensification interval"

CATEGORY_COLORS = {
    "Moderate": "#fee8c8",
    "Strong":   "#fdbb84",
    "Severe":   "#e34a33",
    "Extreme":  "#b30000",
}

# -------------------------
# 5) Plot style
# -------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 14,
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
    "savefig.facecolor": "white"
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
    dates = pd.to_datetime(dates)
    doy = dates.dt.dayofyear.to_numpy()
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
    doy = dayofyear_365(pd.to_datetime(dates))
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

    clim_daily = clim_mean[doy - 1]
    thresh_daily = clim_thresh[doy - 1]
    return clim_daily, thresh_daily

def normalize_category(cat):
    if pd.isna(cat):
        return "Moderate"
    c = str(cat).strip().title()
    if c in ["Moderate", "Strong", "Severe", "Extreme"]:
        return c
    return "Moderate"

def find_col(cols, patterns):
    for p in patterns:
        for c in cols:
            if re.search(p, c):
                return c
    return None

# -------------------------
# 7) Read SST and compute basin daily series
# -------------------------
all_files = []
for pattern in FILE_PATTERNS:
    all_files.extend(glob.glob(os.path.join(DATA_DIR, pattern)))
all_files = sorted(list(set(all_files)))

files = [
    f for f in all_files
    if ("ltm" not in os.path.basename(f).lower())
    and ("clim" not in os.path.basename(f).lower())
]

if len(files) == 0:
    raise FileNotFoundError(f"No usable NetCDF files found in {DATA_DIR}")

print(f"Found {len(files)} daily SST netCDF files")

ds = xr.open_mfdataset(
    files,
    combine="by_coords",
    chunks={},
    coords="minimal",
    data_vars="minimal",
    compat="override",
    parallel=False
)

t_name = find_coord_name(ds, TIME_CANDIDATES)
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
df = pd.DataFrame({
    "date": times,
    "sst": sst_series.values
}).dropna().sort_values("date").reset_index(drop=True)

df = df[(df["date"] >= pd.Timestamp(FULL_START)) & (df["date"] <= pd.Timestamp(FULL_END))].copy()
df = df.reset_index(drop=True)

del ds, da, sst_series
gc.collect()

print("Daily SST series:", df["date"].min(), "to", df["date"].max(), "| n =", len(df))

# -------------------------
# 8) Compute climatology and threshold on full record
# -------------------------
clim_daily, thresh_daily = compute_clim_thresh(
    df["date"],
    df["sst"].values,
    pctile=PCTILE,
    window_half_width=WINDOW_HALF_WIDTH,
    smooth_window=SMOOTH_WINDOW
)

df["clim"] = clim_daily
df["thresh"] = thresh_daily
df["year"] = df["date"].dt.year

# -------------------------
# 9) Read authenticated event summary
# -------------------------
xls = pd.ExcelFile(EVENT_SUMMARY_XLSX)
print("Sheets in event workbook:", xls.sheet_names)

event_df_list = [] # Use a list to collect dataframes from all sheets
for sheet in xls.sheet_names:
    tmp = pd.read_excel(EVENT_SUMMARY_XLSX, sheet_name=sheet)
    if len(tmp) > 0 and tmp.shape[1] >= 3:
        event_df_list.append(tmp.copy())

if not event_df_list:
    raise ValueError("Could not read any usable sheets from the event-summary workbook.")

event_df = pd.concat(event_df_list, ignore_index=True)
print(f"Concatenated events from {len(event_df_list)} sheets.")

event_df.columns = [str(c).strip().lower().replace(" ", "_") for c in event_df.columns]

year_col     = find_col(event_df.columns, [r"^year$", r"event_year"])
start_col    = find_col(event_df.columns, [r"start"])
end_col      = find_col(event_df.columns, [r"end"])
duration_col = find_col(event_df.columns, [r"duration"])
cat_col      = find_col(event_df.columns, [r"category", r"severity"])
maxint_col   = find_col(event_df.columns, [r"max_intensity", r"peak_intensity", r"intensity"])

if start_col is None or end_col is None:
    raise ValueError("Event-summary workbook must contain start and end date columns.")

event_df[start_col] = pd.to_datetime(event_df[start_col], errors="coerce")
event_df[end_col]   = pd.to_datetime(event_df[end_col], errors="coerce")

if year_col is None:
    event_df["year"] = event_df[start_col].dt.year
    year_col = "year"

if duration_col is None:
    event_df["duration"] = (event_df[end_col] - event_df[start_col]).dt.days + 1
    duration_col = "duration"

if cat_col is None:
    event_df["category"] = "Moderate"
    cat_col = "category"

event_df[cat_col] = event_df[cat_col].apply(normalize_category)

event_df = event_df.dropna(subset=[start_col, end_col]).copy()
event_df = event_df[(event_df[year_col] >= 2010) & (event_df[year_col] <= 2025)].copy()
event_df = event_df.sort_values(start_col).reset_index(drop=True)

print("Authenticated late-record events:", len(event_df))

# -------------------------
# 10) Build late-record event summary with inter-event gaps
# -------------------------
gap_rows = []
for i in range(len(event_df)):
    row = event_df.iloc[i]
    gap_days = np.nan
    prev_end = pd.NaT
    if i > 0:
        prev_end = event_df.iloc[i - 1][end_col]
        gap_days = (row[start_col] - prev_end).days - 1

    gap_rows.append({
        "event_no": i + 1,
        "year": int(row[year_col]),
        "start_date": row[start_col],
        "end_date": row[end_col],
        "duration_days": int(row[duration_col]),
        "category": row[cat_col],
        "max_intensity": row[maxint_col] if maxint_col is not None else np.nan,
        "gap_since_previous_days": gap_days
    })

gap_df = pd.DataFrame(gap_rows)
gap_df.to_csv(OUT_GAP_CSV, index=False)

# -------------------------
# 11) Late-record annual summary
# -------------------------
# MHW days per year from authenticated events
event_daily_rows = []
for _, row in event_df.iterrows():
    dd = pd.date_range(row[start_col], row[end_col], freq="D")
    tmp = pd.DataFrame({"date": dd})
    tmp["category"] = row[cat_col]
    event_daily_rows.append(tmp)

event_daily = pd.concat(event_daily_rows, ignore_index=True).drop_duplicates(subset=["date"])
event_daily["year"] = event_daily["date"].dt.year

annual_days = event_daily.groupby("year")["date"].nunique().rename("mhw_days")
annual_count = event_df.groupby(year_col).size().rename("event_count")

annual_late = pd.DataFrame({"year": np.arange(2010, 2026)})
annual_late = annual_late.merge(annual_count, on="year", how="left")
annual_late = annual_late.merge(annual_days, on="year", how="left")
annual_late["event_count"] = annual_late["event_count"].fillna(0).astype(int)
annual_late["mhw_days"] = annual_late["mhw_days"].fillna(0).astype(int)

# median annual gap by year of event onset
gap_yearly = gap_df.groupby("year")["gap_since_previous_days"].median().rename("median_gap_days")
annual_late = annual_late.merge(gap_yearly, on="year", how="left")
annual_late.to_csv(OUT_ANN_CSV, index=False)

# -------------------------
# 12) Prepare plotting subset
# -------------------------
plot_df = df[(df["date"] >= pd.Timestamp(PLOT_START)) & (df["date"] <= pd.Timestamp(PLOT_END))].copy()
plot_df = plot_df.reset_index(drop=True)

late_events = event_df[
    (event_df[start_col] <= pd.Timestamp(PLOT_END)) &
    (event_df[end_col] >= pd.Timestamp(PLOT_START))
].copy().reset_index(drop=True)

# -------------------------
# 13) Build figure
# -------------------------
fig = plt.figure(figsize=(17.5, 11.8), constrained_layout=False)
gs = fig.add_gridspec(
    nrows=4, ncols=1,
    height_ratios=[3.2, 1.25, 1.1, 1.1],
    hspace=0.24
)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax4 = fig.add_subplot(gs[3], sharex=ax1)

# -------------------------
# Panel (a): daily SST chronology
# -------------------------
# Shade each authenticated event window
for _, row in late_events.iterrows():
    color = CATEGORY_COLORS.get(row[cat_col], "#fee8c8")
    ax1.axvspan(
        row[start_col], row[end_col],
        color=color, alpha=0.28, lw=0, zorder=0
    )

ax1.plot(plot_df["date"], plot_df["sst"], color="black", lw=0.75, label="Daily SST", zorder=3)
ax1.plot(plot_df["date"], plot_df["clim"], color="#2c7fb8", lw=1.0, ls="--", label="Seasonal climatology", zorder=2)
ax1.plot(plot_df["date"], plot_df["thresh"], color="#33a02c", lw=1.0, label="90th percentile threshold", zorder=2)

ax1.set_ylabel("SST (°C)")
ax1.set_title(FIG_TITLE, fontsize=17, fontweight="bold", pad=12)
ax1.text(0.01, 0.93, "(a)", transform=ax1.transAxes, fontsize=13, fontweight="bold")
ax1.grid(True, linestyle=":", linewidth=0.45, alpha=0.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

legend_handles = [
    plt.Line2D([0], [0], color="black", lw=1.0, label="Daily SST"),
    plt.Line2D([0], [0], color="#2c7fb8", lw=1.0, ls="--", label="Seasonal climatology"),
    plt.Line2D([0], [0], color="#33a02c", lw=1.0, label="90th percentile threshold"),
    Patch(facecolor=CATEGORY_COLORS["Moderate"], edgecolor="none", alpha=0.28, label="Moderate"),
    Patch(facecolor=CATEGORY_COLORS["Strong"], edgecolor="none", alpha=0.28, label="Strong"),
    Patch(facecolor=CATEGORY_COLORS["Severe"], edgecolor="none", alpha=0.28, label="Severe"),
    Patch(facecolor=CATEGORY_COLORS["Extreme"], edgecolor="none", alpha=0.28, label="Extreme"),
]
ax1.legend(
    handles=legend_handles,
    ncol=4,
    loc="upper left",
    bbox_to_anchor=(0.0, 1.02),
    frameon=False,
    columnspacing=1.3,
    handlelength=1.8
)

# -------------------------
# Panel (b): event windows + recovery gaps
# -------------------------
ax2.text(0.01, 0.88, "(b)", transform=ax2.transAxes, fontsize=13, fontweight="bold")

for _, row in late_events.iterrows():
    width = (row[end_col] - row[start_col]).days + 1
    ax2.barh(
        y=1,
        width=width,
        left=mdates.date2num(row[start_col]),
        height=0.42,
        color=CATEGORY_COLORS[row[cat_col]],
        edgecolor="none",
        alpha=0.95
    )

# Gap labels for short recovery intervals
gap_candidates = gap_df.dropna(subset=["gap_since_previous_days"]).copy()
gap_candidates = gap_candidates[(gap_candidates["gap_since_previous_days"] <= ANNOTATE_GAPS_LEQ_DAYS)]
gap_candidates = gap_candidates.sort_values("gap_since_previous_days").head(MAX_GAP_LABELS)

for _, row in gap_candidates.iterrows():
    i = int(row["event_no"]) - 1
    this_start = gap_df.loc[i, "start_date"]
    prev_end = gap_df.loc[i - 1, "end_date"] if i > 0 else pd.NaT
    if pd.notna(prev_end):
        xmid = prev_end + (this_start - prev_end) / 2
        ax2.text(
            xmid, 1.32,
            f"{int(row['gap_since_previous_days'])} d",
            ha="center", va="bottom", fontsize=8, color="0.25", rotation=90
        )
        ax2.plot([prev_end, this_start], [1.18, 1.18], color="0.45", lw=0.8, alpha=0.75)

ax2.set_ylim(0.5, 1.65)
ax2.set_yticks([])
ax2.set_ylabel("Authenticated\nevents")
ax2.grid(True, axis="x", linestyle=":", linewidth=0.45, alpha=0.45)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)

# -------------------------
# Panel (c): annual event count + MHW days
# -------------------------
ax3.text(0.01, 0.86, "(c)", transform=ax3.transAxes, fontsize=13, fontweight="bold")

year_dates = pd.to_datetime([f"{y}-07-01" for y in annual_late["year"]])

bar1 = ax3.bar(
    year_dates,
    annual_late["event_count"],
    width=180,
    color="#6b6b6b",
    alpha=0.9,
    label="Event count"
)

ax3b = ax3.twinx()
bar2 = ax3b.plot(
    year_dates,
    annual_late["mhw_days"],
    color="#3182bd",
    marker="o",
    ms=3.8,
    lw=1.5,
    label="MHW days"
)

ax3.set_ylabel("Events yr$^{-1}")
ax3b.set_ylabel("MHW days yr$^{-1}", color="#3182bd")
ax3b.tick_params(axis="y", colors="#3182bd")

ax3.grid(True, axis="y", linestyle=":", linewidth=0.45, alpha=0.45)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3b.spines["top"].set_visible(False)

handles = [bar1, bar2[0]]
labels = ["Event count", "MHW days"]
ax3.legend(handles, labels, loc="upper left", frameon=False)

# -------------------------
# Panel (d): inter-event gap compression
# -------------------------
ax4.text(0.01, 0.86, "(d)", transform=ax4.transAxes, fontsize=13, fontweight="bold")

gap_plot = gap_df.dropna(subset=["gap_since_previous_days"]).copy()
gap_plot = gap_plot[
    (gap_plot["start_date"] >= pd.Timestamp(PLOT_START)) &
    (gap_plot["start_date"] <= pd.Timestamp(PLOT_END))
].copy()

ax4.plot(
    gap_plot["start_date"],
    gap_plot["gap_since_previous_days"],
    color="#b30000",
    marker="o",
    lw=1.4,
    ms=3.8
)

# rolling median for readability
if len(gap_plot) >= 4:
    gap_plot["gap_roll"] = gap_plot["gap_since_previous_days"].rolling(4, center=True, min_periods=1).median()
    ax4.plot(
        gap_plot["start_date"],
        gap_plot["gap_roll"],
        color="black",
        lw=1.2,
        ls="--",
        alpha=0.8,
        label="4-event rolling median"
    )
    ax4.legend(loc="upper right", frameon=False)

ax4.set_ylabel("Gap since previous\nevent (days)")
ax4.set_xlabel("Year")
ax4.grid(True, axis="both", linestyle=":", linewidth=0.45, alpha=0.45)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# -------------------------
# Shared x formatting
# -------------------------
for ax in [ax1, ax2, ax3]:
    plt.setp(ax.get_xticklabels(), visible=False)

ax4.xaxis.set_major_locator(mdates.YearLocator(1))
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim(pd.Timestamp(PLOT_START), pd.Timestamp(PLOT_END))

# -------------------------
# Footer note
# -------------------------
n_events = len(late_events)
median_gap = float(np.nanmedian(gap_plot["gap_since_previous_days"])) if len(gap_plot) > 0 else np.nan
min_gap = float(np.nanmin(gap_plot["gap_since_previous_days"])) if len(gap_plot) > 0 else np.nan

fig.text(
    0.01, 0.012,
    f"Late-record window (2010–2025): {n_events} authenticated MHW events. "
    f"Median inter-event recovery gap = {median_gap:.0f} days; minimum gap = {min_gap:.0f} days. "
    f"The chronology highlights increasingly clustered post-2010 event windows and compressed recovery intervals.",
    fontsize=9.5
)

fig.savefig(OUT_FIG_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_FIG_PDF, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)

# -------------------------
# 14) Console summary
# -------------------------
print("\n================ FIGURE S5 COMPLETE ==============\n")
print("Saved figure:")
print(OUT_FIG_PNG)
print(OUT_FIG_PDF)

print("\nSaved tables:")
print(OUT_GAP_CSV)
print(OUT_ANN_CSV)

print("\nLate-record annual summary:")
print(annual_late)

print("\nShortest recovery gaps:")
print(gap_plot[["start_date", "gap_since_previous_days"]].sort_values("gap_since_previous_days").head(10))


# %% [markdown]
# # **Figure_S6_teleconnection**


# %% Cell 4
# ============================================================
# FIGURE S6. SUPPLEMENTARY TELECONNECTION ROBUSTNESS SUMMARY
# Bay of Bengal marine heatwaves (1995-2025)
#
# FULLY INTEGRATED FINAL MASTER SCRIPT
# Nature / Q1-style | Google Colab-ready | 1200 dpi
#
# PURPOSE
#   Generate publication-ready supplementary teleconnection
#   robustness figures supporting the main ENSO-IOD interpretation.
#
# FIGURE OUTPUTS
#   1) Figure S6A: clean 2x2 heatmaps
#   2) Figure S6B: enhanced 4x2 summary
#      - top: detrended zero-lag correlation heatmaps
#      - bottom: climate-phase composite forest plots
#
# KEY IMPROVEMENTS
#   - no legend/title collision
#   - dedicated colorbar row
#   - non-overlapping axis text
#   - balanced Nature-style whitespace
#   - left-axis labels only on left-column panels
#   - right-column panels suppress redundant y labels
#   - 1200 dpi export
#
# INPUT DATA
#   A) Seasonal annual sub-basin workbooks:
#      - Seasonal_Yearly_MHW & Teleconnection_Nbob_1995_2025.xlsx
#      - Seasonal_Yearly_MHW & Teleconnection_CBoB_1995_2025.xlsx
#      - Seasonal_Yearly_MHW & Teleconnection_SBoB_1995_2025.xlsx
#
#   B) Optional composite workbook:
#      - Composite_ElNino_vs_LaNina,Pos IOD Vs Neg IOD_Exposure,Intensity.xlsx
#
# OUTPUT
#   - Figure_S6_teleconnection_robustness_heatmaps_1200dpi.png/pdf
#   - Figure_S6_teleconnection_robustness_enhanced_1200dpi.png/pdf
#   - Teleconnection_summary_S5_recomputed.csv
#   - Teleconnection_summary_S5_recomputed.xlsx
# ============================================================

# -------------------------
# 1) Install dependencies
# -------------------------
# Colab shell command removed for repository reproducibility: !pip -q install pandas numpy scipy matplotlib openpyxl xlsxwriter statsmodels

# -------------------------
# 2) Imports
# -------------------------
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# -------------------------
# 3) Mount Google Drive
# -------------------------
# Google Colab Drive import removed; use config/paths_template.yaml and local data folders.
# Google Drive mount removed; configure DATA_ROOT in this script or via environment variable MHW_DATA_ROOT.

# -------------------------
# 4) USER SETTINGS
# -------------------------
BASE_DIR = Path(str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)"))

TELE_DIR = BASE_DIR / "Teleconnection (1995-2025)"
COMPOSITE_XLSX = BASE_DIR / "Composite_ElNino_vs_LaNina,Pos IOD Vs Neg IOD_Exposure,Intensity.xlsx"

NORTH_XLSX = TELE_DIR / "Seasonal_Yearly_MHW & Teleconnection_Nbob_1995_2025.xlsx"
CENTRAL_XLSX = TELE_DIR / "Seasonal_Yearly_MHW & Teleconnection_CBoB_1995_2025.xlsx"
SOUTH_XLSX = TELE_DIR / "Seasonal_Yearly_MHW & Teleconnection_SBoB_1995_2025.xlsx"

OUTDIR = Path(str(DATA_ROOT / "BoB_Figure_S6_Teleconnection_Robustness_FINAL"))
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY_CSV = OUTDIR / "Teleconnection_summary_S5_recomputed.csv"
OUT_SUMMARY_XLSX = OUTDIR / "Teleconnection_summary_S5_recomputed.xlsx"

OUT_FIG_HEAT_PNG = OUTDIR / "Figure_S6_teleconnection_robustness_heatmaps_1200dpi.png"
OUT_FIG_HEAT_PDF = OUTDIR / "Figure_S6_teleconnection_robustness_heatmaps.pdf"

OUT_FIG_ENH_PNG = OUTDIR / "Figure_S6_teleconnection_robustness_enhanced_1200dpi.png"
OUT_FIG_ENH_PDF = OUTDIR / "Figure_S6_teleconnection_robustness_enhanced.pdf"

DPI = 1200
FIGURE_MODE = "both"   # options: "heatmaps_only", "enhanced_only", "both"

# climate thresholds for phase composites
ONI_POS = 0.5
ONI_NEG = -0.5
DMI_POS = 0.4
DMI_NEG = -0.4

MIN_SAMPLES_COMPOSITE = 4
N_BOOT = 5000
RANDOM_SEED = 42

SEASONS = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
SUBBASINS = ["Northern", "Central", "Southern"]

SUBBASIN_FILEMAP = {
    "Northern": NORTH_XLSX,
    "Central": CENTRAL_XLSX,
    "Southern": SOUTH_XLSX
}

SUBBASIN_COLORS = {
    "Northern": "#d73027",
    "Central": "#4575b4",
    "Southern": "#1a9850"
}

# -------------------------
# 5) Nature-style plotting
# -------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 18,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "grid.color": "#d9d9d9",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white"
})

# -------------------------
# 6) Helper functions
# -------------------------
def savefig_master(fig, png_path, pdf_path):
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

def normalize_season_name(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    mapping = {
        "Pre-Monsoon": "Pre-monsoon",
        "Post-Monsoon": "Post-monsoon",
        "premonsoon": "Pre-monsoon",
        "postmonsoon": "Post-monsoon"
    }
    return mapping.get(s, s)

def read_season_sheet(xlsx_path, season, subbasin):
    df = pd.read_excel(xlsx_path, sheet_name=season).copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        cl = c.lower().strip()

        if cl == "year":
            rename_map[c] = "Year"
        elif cl in ["intensity_mean", "mean_intensity", "intensity", "average mean mhw intensity"]:
            rename_map[c] = "Intensity"
        elif cl in ["no_of_events", "mhw_exposure", "exposure", "mhw_days", "days", "frequency", "event_frequency"]:
            rename_map[c] = "Exposure"
        elif cl in ["oni"]:
            rename_map[c] = "ONI"
        elif cl in ["iod", "dmi", "dipole_mode_index"]:
            rename_map[c] = "DMI"

    df = df.rename(columns=rename_map)

    required = ["Year", "Intensity", "Exposure", "ONI", "DMI"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{xlsx_path.name} / {season} missing columns: {missing}")

    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Sub-basin"] = subbasin
    df["Season"] = normalize_season_name(season)
    return df

def linear_detrend(series):
    s = pd.Series(series).astype(float)
    x = np.arange(len(s), dtype=float)
    mask = np.isfinite(s.values)

    out = np.full(len(s), np.nan, dtype=float)
    if mask.sum() < 5:
        return out

    fit = stats.linregress(x[mask], s.values[mask])
    trend = fit.intercept + fit.slope * x[mask]
    out[mask] = s.values[mask] - trend
    return out

def pearson_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 5 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return len(x), np.nan, np.nan

    r, p = stats.pearsonr(x, y)
    return len(x), float(r), float(p)

def mean_diff_ci_welch(a, b, alpha=0.05, n_boot=N_BOOT, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < MIN_SAMPLES_COMPOSITE or len(b) < MIN_SAMPLES_COMPOSITE:
        return len(a), len(b), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    mean1 = np.mean(a)
    mean2 = np.mean(b)
    diff = mean1 - mean2

    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        boots[i] = np.mean(aa) - np.mean(bb)

    ci_low, ci_high = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")

    return len(a), len(b), mean1, mean2, diff, ci_low, ci_high, float(p)

def significance_marker(q):
    if pd.isna(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""

def build_heatmap_matrix(df, variable, index_name):
    vals = []
    qvals = []

    for sub in SUBBASINS:
        row_vals = []
        row_qs = []
        for season in SEASONS:
            row = df[
                (df["Sub-basin"] == sub) &
                (df["Season"] == season) &
                (df["Metric"] == variable) &
                (df["Index"] == index_name)
            ]
            if len(row) == 0:
                row_vals.append(np.nan)
                row_qs.append(np.nan)
            else:
                row_vals.append(row.iloc[0]["r"])
                row_qs.append(row.iloc[0]["q_fdr"])
        vals.append(row_vals)
        qvals.append(row_qs)

    return np.array(vals, dtype=float), np.array(qvals, dtype=float)

def make_phase_subset(comp_df, index_name, metric_name):
    tmp = comp_df[(comp_df["Index"] == index_name) & (comp_df["Metric"] == metric_name)].copy()
    tmp["season_order"] = tmp["Season"].map({s: i for i, s in enumerate(SEASONS)})
    tmp["sub_order"] = tmp["Sub-basin"].map({s: i for i, s in enumerate(SUBBASINS)})
    return tmp.sort_values(["season_order", "sub_order"]).reset_index(drop=True)

def set_sym_xlim_from_data(ax, tmp, min_half_range=None):
    vals = []
    for col in ["phase_difference", "ci_low_95", "ci_high_95"]:
        arr = pd.to_numeric(tmp[col], errors="coerce").values
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            vals.extend(arr.tolist())

    if len(vals) == 0:
        half = 1.0 if min_half_range is None else min_half_range
        ax.set_xlim(-half, half)
        return

    vmax = np.nanmax(np.abs(vals))
    if min_half_range is not None:
        vmax = max(vmax, min_half_range)
    ax.set_xlim(-1.12 * vmax, 1.12 * vmax)

def plot_corr_heatmap_clean(ax, mat, qmat, title, show_y=True):
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")

    ax.set_xticks(np.arange(len(SEASONS)))
    ax.set_xticklabels(SEASONS, rotation=0)

    ax.set_yticks(np.arange(len(SUBBASINS)))
    if show_y:
        ax.set_yticklabels(SUBBASINS)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            qv = qmat[i, j]
            if np.isfinite(val):
                star = significance_marker(qv)
                txt = f"{val:+.2f}" + (star if star else "")
                txt_color = "white" if abs(val) >= 0.38 else "black"
                ax.text(
                    j, i, txt,
                    ha="center", va="center",
                    fontsize=8.8,
                    color=txt_color,
                    fontweight="bold" if star else "normal"
                )

    ax.set_title(title, fontweight="bold", pad=7)

    ax.set_xticks(np.arange(-0.5, len(SEASONS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SUBBASINS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    return im

def plot_forest_clean(ax, tmp, title, xlabel, show_y=True):
    season_to_base = {s: i for i, s in enumerate(SEASONS[::-1])}
    offsets = {"Northern": -0.22, "Central": 0.0, "Southern": 0.22}

    ax.axvline(0, color="0.45", linestyle="--", linewidth=1.0, zorder=0)

    for _, row in tmp.iterrows():
        season = row["Season"]
        sub = row["Sub-basin"]
        y = season_to_base[season] + offsets[sub]

        d = row["phase_difference"]
        lo = row["ci_low_95"]
        hi = row["ci_high_95"]
        p = row["phase_p"]

        if np.isfinite(d):
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(
                    d, y,
                    xerr=[[d - lo], [hi - d]],
                    fmt="o",
                    color=SUBBASIN_COLORS[sub],
                    ecolor=SUBBASIN_COLORS[sub],
                    elinewidth=1.4,
                    capsize=3,
                    markersize=5.5,
                    markeredgecolor="black" if (np.isfinite(p) and p < 0.05) else SUBBASIN_COLORS[sub],
                    markeredgewidth=0.8 if (np.isfinite(p) and p < 0.05) else 0.0,
                    zorder=3
                )
            else:
                ax.plot(
                    d, y, "o",
                    color=SUBBASIN_COLORS[sub],
                    markersize=5.5,
                    markeredgecolor="black" if (np.isfinite(p) and p < 0.05) else SUBBASIN_COLORS[sub],
                    markeredgewidth=0.8 if (np.isfinite(p) and p < 0.05) else 0.0,
                    zorder=3
                )

    ax.set_yticks(np.arange(len(SEASONS)))
    if show_y:
        ax.set_yticklabels(SEASONS[::-1])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)

    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, labelpad=6)
    ax.grid(True, axis="x")

    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

# -------------------------
# 7) Read and combine seasonal workbooks
# -------------------------
frames = []
for sub, xlsx in SUBBASIN_FILEMAP.items():
    for season in SEASONS:
        frames.append(read_season_sheet(xlsx, season, sub))

seasonal_df = pd.concat(frames, ignore_index=True)
seasonal_df["Season"] = seasonal_df["Season"].map(normalize_season_name)
seasonal_df = seasonal_df.sort_values(["Sub-basin", "Season", "Year"]).reset_index(drop=True)

# Detrend within each sub-basin × season
seasonal_df["Exposure_dt"] = np.nan
seasonal_df["Intensity_dt"] = np.nan
seasonal_df["ONI_dt"] = np.nan
seasonal_df["DMI_dt"] = np.nan

for (sub, season), idx in seasonal_df.groupby(["Sub-basin", "Season"]).groups.items():
    ix = list(idx)
    seasonal_df.loc[ix, "Exposure_dt"] = linear_detrend(seasonal_df.loc[ix, "Exposure"].values)
    seasonal_df.loc[ix, "Intensity_dt"] = linear_detrend(seasonal_df.loc[ix, "Intensity"].values)
    seasonal_df.loc[ix, "ONI_dt"] = linear_detrend(seasonal_df.loc[ix, "ONI"].values)
    seasonal_df.loc[ix, "DMI_dt"] = linear_detrend(seasonal_df.loc[ix, "DMI"].values)

# -------------------------
# 8) Compute zero-lag teleconnection summary
# -------------------------
corr_rows = []

for sub in SUBBASINS:
    for season in SEASONS:
        sdf = seasonal_df[(seasonal_df["Sub-basin"] == sub) & (seasonal_df["Season"] == season)].copy()

        for metric_name, metric_col in [("Exposure", "Exposure_dt"), ("Intensity", "Intensity_dt")]:
            for index_name, index_col in [("ONI", "ONI_dt"), ("DMI", "DMI_dt")]:
                n, r, p = pearson_safe(sdf[metric_col].values, sdf[index_col].values)
                corr_rows.append({
                    "Sub-basin": sub,
                    "Season": season,
                    "Metric": metric_name,
                    "Index": index_name,
                    "n": n,
                    "r": r,
                    "p": p
                })

corr_df = pd.DataFrame(corr_rows)

valid = np.isfinite(corr_df["p"].values)
corr_df["q_fdr"] = np.nan
corr_df["sig_fdr_0_05"] = False

if valid.sum() > 0:
    _, qvals, _, _ = multipletests(corr_df.loc[valid, "p"].values, alpha=0.05, method="fdr_bh")
    corr_df.loc[valid, "q_fdr"] = qvals
    corr_df.loc[valid, "sig_fdr_0_05"] = qvals < 0.05

# -------------------------
# 9) Compute phase-composite contrasts
# -------------------------
comp_rows = []

for sub in SUBBASINS:
    for season in SEASONS:
        sdf = seasonal_df[(seasonal_df["Sub-basin"] == sub) & (seasonal_df["Season"] == season)].copy()

        for metric_name, metric_col in [("Exposure", "Exposure"), ("Intensity", "Intensity")]:

            pos = sdf.loc[sdf["ONI"] >= ONI_POS, metric_col].values
            neg = sdf.loc[sdf["ONI"] <= ONI_NEG, metric_col].values
            n1, n2, m1, m2, d, lo, hi, p = mean_diff_ci_welch(pos, neg, seed=RANDOM_SEED)
            comp_rows.append({
                "Sub-basin": sub,
                "Season": season,
                "Metric": metric_name,
                "Index": "ONI",
                "Positive_phase": "El Niño",
                "Negative_phase": "La Niña",
                "n_positive": n1,
                "n_negative": n2,
                "mean_positive": m1,
                "mean_negative": m2,
                "phase_difference": d,
                "ci_low_95": lo,
                "ci_high_95": hi,
                "phase_p": p
            })

            pos = sdf.loc[sdf["DMI"] >= DMI_POS, metric_col].values
            neg = sdf.loc[sdf["DMI"] <= DMI_NEG, metric_col].values
            n1, n2, m1, m2, d, lo, hi, p = mean_diff_ci_welch(pos, neg, seed=RANDOM_SEED + 42)
            comp_rows.append({
                "Sub-basin": sub,
                "Season": season,
                "Metric": metric_name,
                "Index": "DMI",
                "Positive_phase": "Positive IOD",
                "Negative_phase": "Negative IOD",
                "n_positive": n1,
                "n_negative": n2,
                "mean_positive": m1,
                "mean_negative": m2,
                "phase_difference": d,
                "ci_low_95": lo,
                "ci_high_95": hi,
                "phase_p": p
            })

comp_df = pd.DataFrame(comp_rows)

# -------------------------
# 10) Merge summary and save outputs
# -------------------------
summary_df = corr_df.merge(
    comp_df,
    on=["Sub-basin", "Season", "Metric", "Index"],
    how="left"
)

summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

with pd.ExcelWriter(OUT_SUMMARY_XLSX, engine="xlsxwriter") as writer:
    seasonal_df.to_excel(writer, sheet_name="seasonal_input_merged", index=False)
    corr_df.to_excel(writer, sheet_name="zero_lag_correlations", index=False)
    comp_df.to_excel(writer, sheet_name="phase_composites", index=False)
    summary_df.to_excel(writer, sheet_name="summary_S5_like", index=False)

# -------------------------
# 11) Build matrices
# -------------------------
mat_oni_exp, q_oni_exp = build_heatmap_matrix(corr_df, "Exposure", "ONI")
mat_oni_int, q_oni_int = build_heatmap_matrix(corr_df, "Intensity", "ONI")
mat_dmi_exp, q_dmi_exp = build_heatmap_matrix(corr_df, "Exposure", "DMI")
mat_dmi_int, q_dmi_int = build_heatmap_matrix(corr_df, "Intensity", "DMI")

# -------------------------
# 12) Figure S6A: clean 2x2 heatmaps
# -------------------------
if FIGURE_MODE in ["heatmaps_only", "both"]:
    fig = plt.figure(figsize=(13.8, 8.6), facecolor="white")
    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[1, 1, 0.10],
        left=0.08, right=0.985, bottom=0.11, top=0.90,
        wspace=0.20, hspace=0.34
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[2, :])

    im1 = plot_corr_heatmap_clean(ax1, mat_oni_exp, q_oni_exp, "(a) ONI vs MHW exposure", show_y=True)
    im2 = plot_corr_heatmap_clean(ax2, mat_oni_int, q_oni_int, "(b) ONI vs MHW intensity", show_y=False)
    im3 = plot_corr_heatmap_clean(ax3, mat_dmi_exp, q_dmi_exp, "(c) DMI vs MHW exposure", show_y=True)
    im4 = plot_corr_heatmap_clean(ax4, mat_dmi_int, q_dmi_int, "(d) DMI vs MHW intensity", show_y=False)

    cbar = fig.colorbar(im1, cax=cax, orientation="horizontal")
    cbar.set_label("Detrended zero-lag Pearson correlation (r)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#cccccc", markeredgecolor="none",
               label="Cell values = detrended r", markersize=8),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#cccccc", markeredgecolor="none",
               label="*, **, *** = FDR-adjusted q < 0.05, 0.01, 0.001", markersize=8)
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.6
    )

    savefig_master(fig, OUT_FIG_HEAT_PNG, OUT_FIG_HEAT_PDF)
    plt.show()
    plt.close(fig)

# -------------------------
# 13) Figure S6B: enhanced 4x2 summary
# -------------------------
if FIGURE_MODE in ["enhanced_only", "both"]:
    fig = plt.figure(figsize=(16, 12.5), facecolor="white")
    gs = GridSpec(
        5, 2, figure=fig,
        height_ratios=[1, 1, 0.11, 1.05, 1.05],
        left=0.08, right=0.985, bottom=0.10, top=0.92,
        wspace=0.20, hspace=0.46
    )

    # top heatmaps
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[2, :])

    im1 = plot_corr_heatmap_clean(ax1, mat_oni_exp, q_oni_exp, "(a) ONI vs MHW exposure", show_y=True)
    im2 = plot_corr_heatmap_clean(ax2, mat_oni_int, q_oni_int, "(b) ONI vs MHW intensity", show_y=False)
    im3 = plot_corr_heatmap_clean(ax3, mat_dmi_exp, q_dmi_exp, "(c) DMI vs MHW exposure", show_y=True)
    im4 = plot_corr_heatmap_clean(ax4, mat_dmi_int, q_dmi_int, "(d) DMI vs MHW intensity", show_y=False)

    #Colorbar
    cbar = fig.colorbar(im1, cax=cax, orientation="horizontal")
    cbar.set_label("Detrended zero-lag Pearson correlation (r)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # bottom forest panels
    ax5 = fig.add_subplot(gs[3, 0])
    ax6 = fig.add_subplot(gs[3, 1])
    ax7 = fig.add_subplot(gs[4, 0])
    ax8 = fig.add_subplot(gs[4, 1])

    tmp_e = make_phase_subset(comp_df, "ONI", "Exposure")
    tmp_f = make_phase_subset(comp_df, "ONI", "Intensity")
    tmp_g = make_phase_subset(comp_df, "DMI", "Exposure")
    tmp_h = make_phase_subset(comp_df, "DMI", "Intensity")

    plot_forest_clean(
        ax5, tmp_e,
        "(e) El Niño − La Niña effects on exposure",
        "Composite difference", # Added missing xlabel
        show_y=True
    )
    plot_forest_clean(
        ax6, tmp_f,
        "(f) El Niño − La Niña effects on intensity",
        "Composite difference", # Added missing xlabel
        show_y=False
    )
    plot_forest_clean(
        ax7, tmp_g,
        "(g) Positive IOD − Negative IOD effects on exposure",
        "Composite difference",
        show_y=True
    )
    plot_forest_clean(
        ax8, tmp_h,
        "(h) Positive IOD − Negative IOD effects on intensity",
        "Composite difference",
        show_y=False
    )

    set_sym_xlim_from_data(ax5, tmp_e, min_half_range=20)
    set_sym_xlim_from_data(ax6, tmp_f, min_half_range=0.15)
    set_sym_xlim_from_data(ax7, tmp_g, min_half_range=20)
    set_sym_xlim_from_data(ax8, tmp_h, min_half_range=0.10)

    forest_legend = [
        Line2D([0], [0], marker="o", color="#d73027", linestyle="None", markersize=6, label="Northern"),
        Line2D([0], [0], marker="o", color="#4575b4", linestyle="None", markersize=6, label="Central"),
        Line2D([0], [0], marker="o", color="#1a9850", linestyle="None", markersize=6, label="Southern"),
        Line2D([0], [0], marker="o", color="white", markeredgecolor="black", linestyle="None", markersize=7, label="p < 0.05")
    ]

    fig.legend(
        handles=forest_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.6
    )

    savefig_master(fig, OUT_FIG_ENH_PNG, OUT_FIG_ENH_PDF)
    plt.show()
    plt.close(fig)

# -------------------------
# 14) Console summary
# -------------------------
print("\n================ TELECONNECTION ROBUSTNESS OUTPUTS ================\n")
print("Saved summary tables:")
print(OUT_SUMMARY_CSV)
print(OUT_SUMMARY_XLSX)

if FIGURE_MODE in ["heatmaps_only", "both"]:
    print("\nSaved heatmap figure:")
    print(OUT_FIG_HEAT_PNG)
    print(OUT_FIG_HEAT_PDF)

if FIGURE_MODE in ["enhanced_only", "both"]:
    print("\nSaved enhanced figure:")
    print(OUT_FIG_ENH_PNG)
    print(OUT_FIG_ENH_PDF)

print("\nTop nominal teleconnections (sorted by p):")
display(
    corr_df.sort_values("p", na_position="last")
           [["Sub-basin", "Season", "Metric", "Index", "n", "r", "p", "q_fdr", "sig_fdr_0_05"]]
           .head(20)
)

print("\nTop phase-composite contrasts (sorted by p):")
display(
    comp_df.sort_values("phase_p", na_position="last")
           [["Sub-basin", "Season", "Metric", "Index", "n_positive", "n_negative",
             "phase_difference", "ci_low_95", "ci_high_95", "phase_p"]]
           .head(20)
)

print("\nDone.")


# %% [markdown]
# # **S7**


# %% Cell 6
# ==============================================================================
# Figure S7: Data-Availability and Analysis-Period Matrix
# Optimized for Google Colab | Q1 Publication Standard
# ==============================================================================

# 1. Mount Google Drive
# Google Colab Drive import removed; use config/paths_template.yaml and local data folders.
import os

print("Mounting Google Drive...")
# Google Drive mount removed; configure DATA_ROOT in this script or via environment variable MHW_DATA_ROOT.

# Define the output directory (Modify this path if needed)
output_dir = str(DATA_ROOT / "BoB_MHW_Paper_Outputs")
os.makedirs(output_dir, exist_ok=True)

# 2. Import required libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# 3. Configure Q1-standard plotting aesthetics (Clean, sans-serif, high legibility)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# 4. Define the Datasets and Periods based on Supplementary Table S1
data = [
    # Group 1: Core Thermal & Detection
    {"Product": "NOAA OISST v2.1 (SST)", "Start": 1995, "End": 2025, "Group": "SST & MHW Climatology"},
    {"Product": "Derived MHW Event Catalogue", "Start": 1995, "End": 2025, "Group": "SST & MHW Climatology"},

    # Group 2: Climate Modes
    {"Product": "Oceanic Niño Index (ONI)", "Start": 1995, "End": 2025, "Group": "Climate Teleconnections"},
    {"Product": "Dipole Mode Index (DMI)", "Start": 1995, "End": 2025, "Group": "Climate Teleconnections"},

    # Group 3: Upper-Ocean Physics
    {"Product": "CMEMS ARMOR3D (T, S, MLD)", "Start": 1995, "End": 2024, "Group": "Upper-Ocean Stratification"},
    {"Product": "Density & N² Reconstruction", "Start": 1995, "End": 2024, "Group": "Upper-Ocean Stratification"},

    # Group 4: Biogeochemistry
    {"Product": "CMEMS BGC Hindcast (DO, NO3, PO4, Si)", "Start": 1995, "End": 2025, "Group": "Ecosystem Exposure"}
]

df = pd.DataFrame(data)
# Reverse dataframe so it plots top-to-bottom logically
df = df.iloc[::-1].reset_index(drop=True)

# 5. Define a professional, colorblind-friendly palette
color_map = {
    "SST & MHW Climatology": "#b2182b",        # Deep Red
    "Climate Teleconnections": "#ef8a62",      # Soft Orange
    "Upper-Ocean Stratification": "#2166ac",   # Deep Blue
    "Ecosystem Exposure": "#1b7837"            # Forest Green
}

# 6. Initialize Figure
fig, ax = plt.subplots(figsize=(12, 6), dpi=1200)

# 7. Plot horizontal bars
bar_height = 0.6
for i, row in df.iterrows():
    duration = row['End'] - row['Start']
    color = color_map[row['Group']]

    # Draw the bar
    ax.barh(y=row['Product'], width=duration, left=row['Start'],
            height=bar_height, color=color, edgecolor='white', linewidth=1.5, alpha=0.9, zorder=3)

    # Add exact year text directly on the bar for absolute clarity
    ax.text(x=row['Start'] + (duration / 2), y=i,
            s=f"{row['Start']} – {row['End']}",
            va='center', ha='center', color='white', fontweight='bold', fontsize=10, zorder=4)

# 8. Axis Formatting
ax.set_xlim(1993, 2027)
ax.set_xticks(np.arange(1995, 2026, 5)) # Major ticks every 5 years
ax.set_xticks(np.arange(1995, 2026, 1), minor=True) # Minor ticks every 1 year
ax.set_xlabel("Analysis Period (Years)", fontweight='bold', labelpad=10)

# Add faint grid lines behind the bars for readability
ax.grid(axis='x', which='major', linestyle='-', color='#d3d3d3', alpha=0.7, zorder=0)
ax.grid(axis='x', which='minor', linestyle=':', color='#e0e0e0', alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# 9. Create a Custom Legend
legend_patches = [mpatches.Patch(color=color, label=group) for group, color in color_map.items()]
ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.3, -0.18),
          ncol=4, frameon=False, fontsize=11, handlelength=1.5)

# 10. Title and Layout adjustments

plt.tight_layout()

# 11. Save the figure in high resolution
output_path_png = os.path.join(output_dir, "Figure_S7_Data_Matrix_600dpi.png")
output_path_pdf = os.path.join(output_dir, "Figure_S7_Data_Matrix.pdf")

plt.savefig(output_path_png, dpi=600, bbox_inches='tight', format='png')
plt.savefig(output_path_pdf, bbox_inches='tight', format='pdf') # Q1 Journals prefer PDFs for final typesetting

print(f"\n✅ Figure successfully generated and saved to: {output_dir}")

# Display the plot in Colab
plt.show()
