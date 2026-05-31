#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05 Trend Regime Shift Diagnostics

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
# #**BAY OF BENGAL MHW TREND + REGIME-SHIFT DIAGNOSTICS**


# %% [markdown]
# # **FINAL: BAY OF BENGAL MHW TREND + REGIME-SHIFT ANALYSIS**


# %% Cell 3
# ============================================================
# BAY OF BENGAL MHW TREND + REGIME-SHIFT ANALYSIS
# REFINED Q1-PUBLICATION-READY COLAB SCRIPT
#
# PURPOSE
#   Supports Sections 3.1 and 3.2 using:
#     (1) annual basin and sub-basin MHW metrics
#     (2) Mann-Kendall trend significance
#     (3) Sen-slope trend magnitude
#     (4) Pettitt regime-shift detection
#     (5) publication-ready 2x2 annual time-series figure
#     (6) publication-ready spatial trend map figure
#
# INPUT
#   Excel workbook with sheet "MHW"
#   Required columns:
#       event, lon, lat, date_start, date_peak, date_end,
#       intensity_mean, year, month, Season
#
# OUTPUT
#   - annual metrics tables
#   - trend summary tables
#   - grid-cell trend tables
#   - Figure_3A_Annual_TimeSeries_Q1_2x2_1200dpi.png / .pdf
#   - Figure_3B_Spatial_Trend_Maps_Q1_1200dpi.png / .pdf
#
# NOTES
#   - Uses uploaded event catalogue directly
#   - MHW days computed from expanded daily event windows
#   - Exposure metrics are zero-filled for no-event years
#   - Mean intensity is computed only from active-event years
# ============================================================

# -------------------------
# 1) Install dependencies
# -------------------------
# Colab shell command removed for repository reproducibility: !pip -q install pandas numpy scipy matplotlib openpyxl xlsxwriter cartopy

# -------------------------
# 2) Imports
# -------------------------
import os
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import theilslopes, norm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

OUTDIR = Path(str(DATA_ROOT / "BoB_MHW_trend_regimeshift_outputs_Q1_final_refined"))
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_ANNUAL_CSV   = OUTDIR / "BoB_MHW_annual_basin_subbasin_metrics_refined.csv"
OUT_TREND_CSV    = OUTDIR / "BoB_MHW_trend_summary_refined.csv"
OUT_GRID_CSV     = OUTDIR / "BoB_MHW_gridcell_trends_refined.csv"
OUT_XLSX         = OUTDIR / "BoB_MHW_trend_regimeshift_outputs_refined.xlsx"

OUT_FIG_A_PNG    = OUTDIR / "Figure_3A_Annual_TimeSeries_Q1_2x2_1200dpi.png"
OUT_FIG_A_PDF    = OUTDIR / "Figure_3A_Annual_TimeSeries_Q1_2x2.pdf"
OUT_FIG_B_PNG    = OUTDIR / "Figure_3B_Spatial_Trend_Maps_Q1_1200dpi.png"
OUT_FIG_B_PDF    = OUTDIR / "Figure_3B_Spatial_Trend_Maps_Q1.pdf"

# -------------------------
# 5) STUDY SETTINGS
# -------------------------
START_YEAR = 1995
END_YEAR   = 2025

LAT_MIN, LAT_MAX = 5.0, 22.0
LON_MIN, LON_MAX = 80.0, 100.0

SUBBASIN_BOUNDS = {
    "Northern BoB": (16.0, 22.0),
    "Central BoB":  (10.0, 16.0),
    "Southern BoB": (5.0, 10.0),
}

REGION_ORDER = ["Basin", "Northern BoB", "Central BoB", "Southern BoB"]

REGION_COLORS = {
    "Basin": "#1a1a1a",
    "Northern BoB": "#d73027",
    "Central BoB": "#4575b4",
    "Southern BoB": "#1a9850",
}

REGION_LABELS_SHORT = {
    "Basin": "Basin",
    "Northern BoB": "Northern",
    "Central BoB": "Central",
    "Southern BoB": "Southern",
}

DPI = 1200
MIN_YEARS_FOR_GRID_TREND = 8
MIN_EVENT_YEARS_FOR_INTENSITY_TREND = 6

# -------------------------
# 6) Plot style
# -------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.titlesize": 19,
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

# -------------------------
# 7) Helper functions
# -------------------------
def save_figure(fig, png_path, pdf_path=None):
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

def detect_subbasin(lat):
    if 16.0 <= lat <= 22.0:
        return "Northern BoB"
    elif 10.0 <= lat < 16.0:
        return "Central BoB"
    elif 5.0 <= lat < 10.0:
        return "Southern BoB"
    return np.nan

def mann_kendall_test(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)

    if n < 3:
        return {"n": n, "S": np.nan, "tau": np.nan, "z": np.nan, "p": np.nan}

    s = 0
    for i in range(n - 1):
        s += np.sum(np.sign(y[i + 1:] - y[i]))

    _, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if var_s <= 0:
        z = 0.0
        p = 1.0
    else:
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0.0
        p = 2 * (1 - norm.cdf(abs(z)))

    tau = s / (0.5 * n * (n - 1))
    return {"n": n, "S": float(s), "tau": float(tau), "z": float(z), "p": float(p)}

def sen_slope_with_ci(years, values, alpha=0.95):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(years) & np.isfinite(values)

    if mask.sum() < 3:
        return {
            "slope_per_year": np.nan,
            "slope_per_decade": np.nan,
            "ci_low_per_decade": np.nan,
            "ci_high_per_decade": np.nan,
            "intercept": np.nan
        }

    res = theilslopes(values[mask], years[mask], alpha=alpha)
    return {
        "slope_per_year": float(res.slope),
        "slope_per_decade": float(res.slope * 10.0),
        "ci_low_per_decade": float(res.low_slope * 10.0),
        "ci_high_per_decade": float(res.high_slope * 10.0),
        "intercept": float(res.intercept)
    }

def pettitt_test(y, years):
    y = np.asarray(y, dtype=float)
    years = np.asarray(years, dtype=int)
    mask = np.isfinite(y)
    y = y[mask]
    years = years[mask]
    n = len(y)

    if n < 5:
        return {"change_index": np.nan, "change_year": np.nan, "K": np.nan, "p": np.nan}

    U = np.zeros(n)
    for t in range(n):
        left = y[:t + 1]
        right = y[t + 1:]
        if len(right) == 0:
            U[t] = 0
        else:
            s = 0
            for xi in left:
                s += np.sum(np.sign(xi - right))
            U[t] = s

    K = np.max(np.abs(U))
    idx = int(np.argmax(np.abs(U)))
    p = 2 * np.exp((-6 * K**2) / (n**3 + n**2))

    return {
        "change_index": idx,
        "change_year": int(years[idx]),
        "K": float(K),
        "p": float(min(p, 1.0))
    }

def area_weighted_mean(values, lats):
    values = np.asarray(values, dtype=float)
    lats = np.asarray(lats, dtype=float)
    mask = np.isfinite(values) & np.isfinite(lats)
    if mask.sum() == 0:
        return np.nan
    weights = np.cos(np.deg2rad(lats[mask]))
    return np.average(values[mask], weights=weights)

def build_complete_cell_year_panel(cell_df, years):
    panel = (
        pd.MultiIndex.from_product(
            [cell_df.index.values, years],
            names=["cell_idx", "year"]
        )
        .to_frame(index=False)
        .merge(
            cell_df.reset_index().rename(columns={"index": "cell_idx"}),
            on="cell_idx",
            how="left"
        )
    )
    return panel

def pretty_metric_name(metric):
    mapping = {
        "event_frequency_mean": "Event frequency",
        "mhw_days_mean": "MHW days",
        "mean_intensity_active_cells": "Mean intensity",
        "cumulative_intensity_mean": "Cumulative intensity"
    }
    return mapping.get(metric, metric)

def pretty_metric_unit(metric):
    mapping = {
        "event_frequency_mean": "events yr$^{-1}$",
        "mhw_days_mean": "days yr$^{-1}$",
        "mean_intensity_active_cells": "°C",
        "cumulative_intensity_mean": "°C·days yr$^{-1}$"
    }
    return mapping.get(metric, "")

def annotate_trend_box(ax, trend_row):
    if pd.isna(trend_row["sen_slope_per_decade"]):
        txt = "Trend statistics unavailable"
    else:
        txt = (
            f"Sen slope = {trend_row['sen_slope_per_decade']:.2f} {trend_row['unit']} decade$^{{-1}}$\n"
            f"MK p = {trend_row['mk_p']:.3g}\n"
            f"Pettitt year = {int(trend_row['pettitt_change_year']) if pd.notna(trend_row['pettitt_change_year']) else 'NA'}"
        )
    ax.text(
        0.015, 0.965, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.70", alpha=0.92)
    )

def make_grid(df, value_col):
    lons = np.sort(df["lon"].unique())
    lats = np.sort(df["lat"].unique())
    grid = df.pivot(index="lat", columns="lon", values=value_col).reindex(index=lats, columns=lons)
    xx, yy = np.meshgrid(lons, lats)
    return xx, yy, grid.values

def add_subbasin_lines(ax):
    ax.plot([LON_MIN, LON_MAX], [10, 10], transform=ccrs.PlateCarree(), color="0.45", lw=0.55, ls="--", zorder=5)
    ax.plot([LON_MIN, LON_MAX], [16, 16], transform=ccrs.PlateCarree(), color="0.45", lw=0.55, ls="--", zorder=5)

def robust_vmax(series, pct=95, fallback=1.0):
    vals = np.asarray(series, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return fallback
    vmax = np.nanpercentile(np.abs(vals), pct)
    if not np.isfinite(vmax) or vmax == 0:
        return fallback
    return float(vmax)

# -------------------------
# 8) Read workbook
# -------------------------
print("Reading MHW workbook...")
df = pd.read_excel(MHW_XLSX, sheet_name="MHW")

required_cols = [
    "event", "lon", "lat", "date_start", "date_peak", "date_end",
    "intensity_mean", "year", "month", "Season"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for c in ["date_start", "date_peak", "date_end"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")

df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["intensity_mean"] = pd.to_numeric(df["intensity_mean"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")

df = df.dropna(subset=["lon", "lat", "date_start", "date_end", "intensity_mean", "year"]).copy()

df = df[
    (df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX) &
    (df["lon"] >= LON_MIN) & (df["lon"] <= LON_MAX)
].copy()

df["subbasin"] = df["lat"].apply(detect_subbasin)
df = df.dropna(subset=["subbasin"]).copy()

df["year"] = df["year"].astype(int)
df["duration_days"] = (df["date_end"] - df["date_start"]).dt.days + 1
df["duration_days"] = df["duration_days"].clip(lower=1)
df["cumulative_intensity"] = df["intensity_mean"] * df["duration_days"]

df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

print("Rows:", len(df))
print("Year range:", df["year"].min(), "to", df["year"].max())
print(df["subbasin"].value_counts())

# -------------------------
# 9) Expand to daily event windows
# -------------------------
print("\nExpanding events to daily windows...")

expanded = []
for row in df.itertuples(index=False):
    dr = pd.date_range(row.date_start, row.date_end, freq="D")
    expanded.append(
        pd.DataFrame({
            "lon": row.lon,
            "lat": row.lat,
            "subbasin": row.subbasin,
            "event_id": f"{row.event}_{row.lon}_{row.lat}_{row.date_start.strftime('%Y%m%d')}",
            "date": dr
        })
    )

daily = pd.concat(expanded, ignore_index=True)
daily["year"] = daily["date"].dt.year
daily = daily.drop_duplicates(subset=["lon", "lat", "date"]).copy()

print("Expanded daily rows:", len(daily))

# -------------------------
# 10) Cell-year metrics
# -------------------------
print("\nBuilding cell-year metrics...")

cells = df[["lon", "lat", "subbasin"]].drop_duplicates().copy()
years = np.arange(START_YEAR, END_YEAR + 1)

cell_year_event = (
    df.groupby(["lon", "lat", "subbasin", "year"], as_index=False)
      .agg(
          event_frequency=("event", "size"),
          mean_intensity=("intensity_mean", "mean"),
          cumulative_intensity=("cumulative_intensity", "sum")
      )
)

cell_year_days = (
    daily.groupby(["lon", "lat", "subbasin", "year"], as_index=False)
         .agg(mhw_days=("date", "nunique"))
)

cell_year = cell_year_event.merge(
    cell_year_days,
    on=["lon", "lat", "subbasin", "year"],
    how="outer"
)

panel = build_complete_cell_year_panel(cells, years)
cell_year_full = panel.merge(
    cell_year,
    on=["lon", "lat", "subbasin", "year"],
    how="left"
)

cell_year_full["event_frequency"] = cell_year_full["event_frequency"].fillna(0.0)
cell_year_full["mhw_days"] = cell_year_full["mhw_days"].fillna(0.0)
cell_year_full["cumulative_intensity"] = cell_year_full["cumulative_intensity"].fillna(0.0)
cell_year_full["area_weight"] = np.cos(np.deg2rad(cell_year_full["lat"]))

print("Cell-year rows:", len(cell_year_full))

# -------------------------
# 11) Annual basin and sub-basin metrics
# -------------------------
print("\nComputing annual regional metrics...")

annual_rows = []

for region in REGION_ORDER:
    for yr in years:
        sub = cell_year_full[cell_year_full["year"] == yr].copy()
        if region != "Basin":
            sub = sub[sub["subbasin"] == region].copy()

        freq_mean = area_weighted_mean(sub["event_frequency"], sub["lat"])
        days_mean = area_weighted_mean(sub["mhw_days"], sub["lat"])
        cumi_mean = area_weighted_mean(sub["cumulative_intensity"], sub["lat"])

        active = sub[np.isfinite(sub["mean_intensity"])].copy()
        intensity_mean = area_weighted_mean(active["mean_intensity"], active["lat"]) if len(active) > 0 else np.nan

        annual_rows.append({
            "region": region,
            "year": yr,
            "event_frequency_mean": freq_mean,
            "mhw_days_mean": days_mean,
            "mean_intensity_active_cells": intensity_mean,
            "cumulative_intensity_mean": cumi_mean,
            "n_total_cells": len(sub),
            "n_active_cells": len(active)
        })

annual_df = pd.DataFrame(annual_rows)
annual_df.to_csv(OUT_ANNUAL_CSV, index=False)

# -------------------------
# 12) Formal trend and regime-shift statistics
# -------------------------
print("\nComputing trend statistics...")

trend_rows = []
metrics = [
    "event_frequency_mean",
    "mhw_days_mean",
    "mean_intensity_active_cells",
    "cumulative_intensity_mean"
]

for region in REGION_ORDER:
    reg = annual_df[annual_df["region"] == region].copy()
    for metric in metrics:
        y = reg[metric].values
        yrs = reg["year"].values

        mk = mann_kendall_test(y)
        sen = sen_slope_with_ci(yrs, y)
        pet = pettitt_test(y, yrs)

        trend_rows.append({
            "region": region,
            "metric": metric,
            "metric_pretty": pretty_metric_name(metric),
            "unit": pretty_metric_unit(metric),
            "n": mk["n"],
            "mk_tau": mk["tau"],
            "mk_z": mk["z"],
            "mk_p": mk["p"],
            "sen_slope_per_year": sen["slope_per_year"],
            "sen_slope_per_decade": sen["slope_per_decade"],
            "sen_ci_low_per_decade": sen["ci_low_per_decade"],
            "sen_ci_high_per_decade": sen["ci_high_per_decade"],
            "pettitt_change_year": pet["change_year"],
            "pettitt_K": pet["K"],
            "pettitt_p": pet["p"]
        })

trend_df = pd.DataFrame(trend_rows)
trend_df.to_csv(OUT_TREND_CSV, index=False)

# -------------------------
# 13) Grid-cell trends
# -------------------------
print("\nComputing grid-cell trends...")

grid_rows = []

for (lon, lat, subbasin), sub in cell_year_full.groupby(["lon", "lat", "subbasin"]):
    sub = sub.sort_values("year")

    y_days = sub["mhw_days"].values
    yrs = sub["year"].values

    mk_days = mann_kendall_test(y_days)
    sen_days = sen_slope_with_ci(yrs, y_days)

    active = sub[np.isfinite(sub["mean_intensity"])].copy()
    if len(active) >= MIN_EVENT_YEARS_FOR_INTENSITY_TREND:
        mk_int = mann_kendall_test(active["mean_intensity"].values)
        sen_int = sen_slope_with_ci(active["year"].values, active["mean_intensity"].values)
    else:
        mk_int = {"tau": np.nan, "z": np.nan, "p": np.nan, "n": len(active)}
        sen_int = {
            "slope_per_year": np.nan,
            "slope_per_decade": np.nan,
            "ci_low_per_decade": np.nan,
            "ci_high_per_decade": np.nan,
            "intercept": np.nan
        }

    grid_rows.append({
        "lon": lon,
        "lat": lat,
        "subbasin": subbasin,
        "n_years_days": len(sub),
        "mhw_days_sen_per_decade": sen_days["slope_per_decade"],
        "mhw_days_mk_p": mk_days["p"],
        "mhw_days_mk_tau": mk_days["tau"],
        "mhw_days_sig_p05": bool(np.isfinite(mk_days["p"]) and mk_days["p"] < 0.05),
        "n_years_intensity": len(active),
        "intensity_sen_per_decade": sen_int["slope_per_decade"],
        "intensity_mk_p": mk_int["p"],
        "intensity_mk_tau": mk_int["tau"],
        "intensity_sig_p05": bool(np.isfinite(mk_int["p"]) and mk_int["p"] < 0.05)
    })

grid_df = pd.DataFrame(grid_rows)
grid_df.to_csv(OUT_GRID_CSV, index=False)

# -------------------------
# 14) Save Excel workbook
# -------------------------
with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
    annual_df.to_excel(writer, sheet_name="annual_metrics", index=False)
    trend_df.to_excel(writer, sheet_name="trend_summary", index=False)
    grid_df.to_excel(writer, sheet_name="gridcell_trends", index=False)
    cell_year_full.to_excel(writer, sheet_name="cell_year_metrics", index=False)

print("\nSaved tables:")
print(OUT_ANNUAL_CSV)
print(OUT_TREND_CSV)
print(OUT_GRID_CSV)
print(OUT_XLSX)

# -------------------------
# 15) FIGURE A: 2x2 annual basin and sub-basin time series
# -------------------------
print("\nBuilding refined Figure A (2x2 layout, legend-fixed)...")

fig, axes = plt.subplots(2, 2, figsize=(22, 12), sharex=True, constrained_layout=False)
axes = axes.flatten()

panel_labels = ["(a)", "(b)", "(c)", "(d)"]

for ax, metric, plab in zip(axes, metrics, panel_labels):
    meta_name = pretty_metric_name(metric)
    meta_unit = pretty_metric_unit(metric)

    basin_tr = trend_df[
        (trend_df["region"] == "Basin") &
        (trend_df["metric"] == metric)
    ].iloc[0]

    cp_year = basin_tr["pettitt_change_year"] if pd.notna(basin_tr["pettitt_change_year"]) else None

    # Pettitt post-change shading
    if cp_year is not None:
        ax.axvspan(cp_year, END_YEAR + 0.3, color="#f2f2f2", alpha=0.85, zorder=0)
        ax.axvline(cp_year, color="0.40", ls=":", lw=1.15, zorder=1)

    # Plot Basin + sub-basins
    for region in REGION_ORDER:
        reg = annual_df[annual_df["region"] == region].copy()

        lw = 2.6 if region == "Basin" else 1.8
        alpha = 1.0 if region == "Basin" else 0.96

        ax.plot(
            reg["year"], reg[metric],
            color=REGION_COLORS[region],
            linewidth=lw,
            alpha=alpha,
            zorder=4 if region == "Basin" else 3
        )

        # Sen trend line
        sen = sen_slope_with_ci(reg["year"].values, reg[metric].values)
        if np.isfinite(sen["slope_per_year"]):
            yfit = sen["slope_per_year"] * reg["year"].values + sen["intercept"]
            ax.plot(
                reg["year"], yfit,
                color=REGION_COLORS[region],
                linestyle="--",
                linewidth=1.0,
                alpha=0.42,
                zorder=2
            )

    # Panel label
    ax.text(
        0.005, 1.02, plab,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom"
    )

    ax.set_title(meta_name, pad=7)
    ax.set_ylabel(meta_unit)
    ax.grid(True, linestyle=":", linewidth=0.50, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    annotate_trend_box(ax, basin_tr)
    ax.set_xlim(START_YEAR - 0.5, END_YEAR + 0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

for ax in axes[2:]:
    ax.set_xlabel("Year")

legend_handles = [
    Line2D([0], [0], color=REGION_COLORS["Basin"], lw=2.6, label="Basin"),
    Line2D([0], [0], color=REGION_COLORS["Northern BoB"], lw=1.8, label="Northern"),
    Line2D([0], [0], color=REGION_COLORS["Central BoB"], lw=1.8, label="Central"),
    Line2D([0], [0], color=REGION_COLORS["Southern BoB"], lw=1.8, label="Southern"),
    Patch(facecolor="#f2f2f2", edgecolor="0.65", label="Post-change regime")
]

# Legend placed BELOW title, ABOVE subplot area
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.952),
    ncol=5,
    frameon=False,
    columnspacing=1.4,
    handlelength=1.9,
    handletextpad=0.55,
    borderaxespad=0.0
)

# Extra top margin so title + legend never collide with panels
fig.subplots_adjust(
    left=0.08,
    right=0.985,
    bottom=0.07,
    top=0.89,
    wspace=0.17,
    hspace=0.23
)

save_figure(fig, OUT_FIG_A_PNG, OUT_FIG_A_PDF)
plt.show()
plt.close(fig)

# -------------------------
# 16) FIGURE B: Spatial trend maps
# -------------------------
print("\nBuilding refined Figure B...")

fig = plt.figure(figsize=(12, 6))
proj = ccrs.PlateCarree()

ax1 = fig.add_subplot(1, 2, 1, projection=proj)
ax2 = fig.add_subplot(1, 2, 2, projection=proj)

for ax in [ax1, ax2]:
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", edgecolor="black", linewidth=0.35, zorder=4)
    ax.coastlines(resolution="10m", linewidth=0.55)
    add_subbasin_lines(ax)

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, linestyle="--", color="0.6", alpha=0.45)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}
    gl.xlocator = mticker.FixedLocator(np.arange(80, 101, 5))
    gl.ylocator = mticker.FixedLocator(np.arange(5, 23, 5))

# panel (a): days trend
xx_d, yy_d, zz_d = make_grid(grid_df, "mhw_days_sen_per_decade")
vmax_days = robust_vmax(grid_df["mhw_days_sen_per_decade"], pct=95, fallback=10.0)
norm_days = TwoSlopeNorm(vmin=-vmax_days, vcenter=0, vmax=vmax_days)

pcm1 = ax1.pcolormesh(
    xx_d, yy_d, zz_d,
    cmap="RdBu_r",
    norm=norm_days,
    shading="nearest",
    transform=proj,
    zorder=1
)

sig_days = grid_df[grid_df["mhw_days_sig_p05"] == True]
ax1.scatter(
    sig_days["lon"], sig_days["lat"],
    s=7, c="black", marker="o", alpha=0.65,
    transform=proj, zorder=5
)
ax1.set_title("(a) Trend in annual MHW days", fontsize=13.5, pad=8)

# panel (b): intensity trend
xx_i, yy_i, zz_i = make_grid(grid_df, "intensity_sen_per_decade")
vmax_int = robust_vmax(grid_df["intensity_sen_per_decade"], pct=95, fallback=0.03)
norm_int = TwoSlopeNorm(vmin=-vmax_int, vcenter=0, vmax=vmax_int)

pcm2 = ax2.pcolormesh(
    xx_i, yy_i, zz_i,
    cmap="RdBu_r",
    norm=norm_int,
    shading="nearest",
    transform=proj,
    zorder=1
)

sig_int = grid_df[grid_df["intensity_sig_p05"] == True]
ax2.scatter(
    sig_int["lon"], sig_int["lat"],
    s=7, c="black", marker="o", alpha=0.65,
    transform=proj, zorder=5
)
ax2.set_title("(b) Trend in mean MHW intensity", fontsize=13.5, pad=8)

# colorbars
cbar1 = fig.colorbar(pcm1, ax=ax1, orientation="horizontal", fraction=0.052, pad=0.07)
cbar1.set_label("Trend in MHW days (days decade$^{-1}$)")

cbar2 = fig.colorbar(pcm2, ax=ax2, orientation="horizontal", fraction=0.052, pad=0.07)
cbar2.set_label("Trend in mean intensity (°C decade$^{-1}$)")

legend_sig = [
    Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=4.2, label="p < 0.05")
]

fig.legend(
    handles=legend_sig,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.952),
    ncol=1,
    frameon=False
)

fig.subplots_adjust(left=0.04, right=0.985, bottom=0.08, top=0.92, wspace=0.08)

save_figure(fig, OUT_FIG_B_PNG, OUT_FIG_B_PDF)
plt.show()
plt.close(fig)

# -------------------------
# 17) Ready-to-paste basin statistics
# -------------------------
print("\n================ BASIN-LEVEL FORMAL STATISTICS ================\n")

basin_only = trend_df[trend_df["region"] == "Basin"].copy()
for _, r in basin_only.iterrows():
    print(
        f"{r['metric_pretty']}: Sen slope = {r['sen_slope_per_decade']:.3f} {r['unit']} decade^-1 "
        f"(95% CI {r['sen_ci_low_per_decade']:.3f} to {r['sen_ci_high_per_decade']:.3f}); "
        f"Mann-Kendall p = {r['mk_p']:.4g}; "
        f"Pettitt change year = {int(r['pettitt_change_year']) if pd.notna(r['pettitt_change_year']) else 'NA'} "
        f"(p = {r['pettitt_p']:.4g})"
    )

basin_int = basin_only[basin_only["metric"] == "mean_intensity_active_cells"].iloc[0]
print("\nREADY-TO-PASTE SECTION 3.1 SENTENCE:\n")
print(
    f"Formal statistical testing of the basin-averaged mean-intensity time series indicates "
    f"a Sen-slope trend of {basin_int['sen_slope_per_decade']:.3f} °C decade⁻¹ "
    f"(95% CI {basin_int['sen_ci_low_per_decade']:.3f} to {basin_int['sen_ci_high_per_decade']:.3f}), "
    f"a Mann–Kendall p value of {basin_int['mk_p']:.3g}, "
    f"and a Pettitt change point centered on {int(basin_int['pettitt_change_year']) if pd.notna(basin_int['pettitt_change_year']) else 'NA'} "
    f"(p = {basin_int['pettitt_p']:.3g})."
)

print("\nSaved figures:")
print(OUT_FIG_A_PNG)
print(OUT_FIG_B_PNG)

print("\nDone.")


# %% [markdown]
# # **With Trend Summary**


# %% Cell 5
# ============================================================
# BAY OF BENGAL MHW FIGURE 4 (REFINED JOURNAL-STYLE 8-PANEL)
# From previously generated outputs only
#
# INPUT FILES:
#   1) BoB_annual_basin_subbasin_metrics.csv
#   2) BoB_trend_summary_basin_subbasin.xlsx
#
# OUTPUT:
#   Figure_4_BoB_MHW_8Panel_JournalStyle_1200dpi.png
#   Figure_4_BoB_MHW_8Panel_JournalStyle_1200dpi.pdf
#
# Designed for Google Colab
# ============================================================

# -------------------------
# 1) Install dependencies
# -------------------------
# Colab shell command removed for repository reproducibility: !pip -q install pandas openpyxl numpy scipy matplotlib

# -------------------------
# 2) Imports
# -------------------------
import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# -------------------------
# 3) Mount Google Drive
# -------------------------
# Google Colab Drive import removed; use config/paths_template.yaml and local data folders.
# Google Drive mount removed; configure DATA_ROOT in this script or via environment variable MHW_DATA_ROOT.

# -------------------------
# 4) USER SETTINGS
# -------------------------
INDIR = str(DATA_ROOT / "BoB_MHW_Trend_Outputs_Section_3_1_3_2")

ANNUAL_CSV = os.path.join(INDIR, "BoB_annual_basin_subbasin_metrics.csv")
TREND_XLSX = os.path.join(INDIR, "BoB_trend_summary_basin_subbasin.xlsx")

OUT_PNG = os.path.join(INDIR, "Figure_4_BoB_MHW_8Panel_JournalStyle_1200dpi.png")
OUT_PDF = os.path.join(INDIR, "Figure_4_BoB_MHW_8Panel_JournalStyle_1200dpi.pdf")
OUT_SLOPE_TABLE = os.path.join(INDIR, "Figure_4_panel_trend_summary.csv")

DPI = 1200

# -------------------------
# 5) Plot style
# -------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "figure.titlesize": 20,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "grid.alpha": 0.20,
    "grid.linestyle": "--",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white"
})

# -------------------------
# 6) Read data
# -------------------------
annual_all = pd.read_csv(ANNUAL_CSV)

trend_summary = pd.read_excel(TREND_XLSX, sheet_name="trend_summary")

# enforce types
annual_all["year"] = pd.to_numeric(annual_all["year"], errors="coerce")
for c in ["event_frequency", "mhw_days", "mean_intensity", "cumulative_intensity"]:
    annual_all[c] = pd.to_numeric(annual_all[c], errors="coerce")

trend_summary["sen_slope_per_decade"] = pd.to_numeric(trend_summary["sen_slope_per_decade"], errors="coerce")
trend_summary["mk_p_value"] = pd.to_numeric(trend_summary["mk_p_value"], errors="coerce")
trend_summary["pettitt_change_year"] = pd.to_numeric(trend_summary["pettitt_change_year"], errors="coerce")
trend_summary["pettitt_p_value"] = pd.to_numeric(trend_summary["pettitt_p_value"], errors="coerce")

regions = ["Basin", "Northern", "Central", "Southern"]
region_colors = {
    "Basin": "black",
    "Northern": "#d73027",
    "Central": "#4575b4",
    "Southern": "#1a9850"
}

metric_order = [
    "event_frequency",
    "mhw_days",
    "mean_intensity",
    "cumulative_intensity"
]

metric_labels = {
    "event_frequency": "Event frequency (events yr$^{-1}$)",
    "mhw_days": "MHW days (days yr$^{-1}$)",
    "mean_intensity": "Mean intensity (°C)",
    "cumulative_intensity": "Cumulative intensity (°C·days)"
}

metric_titles = {
    "event_frequency": "Event frequency",
    "mhw_days": "MHW days",
    "mean_intensity": "Mean intensity",
    "cumulative_intensity": "Cumulative intensity"
}

panel_letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

# -------------------------
# 7) Optional bootstrap CI for slopes
#    This improves the right-column trend-summary panels
# -------------------------
def bootstrap_linear_slope_per_decade(x, y, n_boot=4000, seed=42):
    """
    Bootstrap CI around ordinary least squares slope per decade.
    Used only for plotting uncertainty in the summary panels.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 8:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    slopes = []

    idx = np.arange(len(x))
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        xb = x[b]
        yb = y[b]
        try:
            fit = stats.linregress(xb, yb)
            slopes.append(fit.slope * 10.0)
        except Exception:
            continue

    if len(slopes) < 100:
        return np.nan, np.nan

    ci_low, ci_high = np.percentile(slopes, [2.5, 97.5])
    return float(ci_low), float(ci_high)

slope_rows = []
for region in regions:
    sub = annual_all[annual_all["region"] == region].sort_values("year")
    for metric in metric_order:
        ci_low, ci_high = bootstrap_linear_slope_per_decade(
            sub["year"].values,
            sub[metric].values,
            n_boot=3000,
            seed=42
        )

        tr = trend_summary[
            (trend_summary["region"] == region) &
            (trend_summary["metric"] == metric)
        ]

        if len(tr) == 0:
            continue

        tr = tr.iloc[0]
        slope_rows.append({
            "region": region,
            "metric": metric,
            "sen_slope_per_decade": tr["sen_slope_per_decade"],
            "mk_p_value": tr["mk_p_value"],
            "pettitt_change_year": tr["pettitt_change_year"],
            "pettitt_p_value": tr["pettitt_p_value"],
            "slope_ci_low": ci_low,
            "slope_ci_high": ci_high
        })

slope_plot_df = pd.DataFrame(slope_rows)
slope_plot_df.to_csv(OUT_SLOPE_TABLE, index=False)

# -------------------------
# 8) Helper functions
# -------------------------
def add_timeseries_panel(ax, metric, letter):
    """
    Left-column panels: annual basin + sub-basin time series.
    """
    for region in regions:
        sub = annual_all[annual_all["region"] == region].sort_values("year")
        ax.plot(
            sub["year"], sub[metric],
            color=region_colors[region],
            lw=2.0 if region == "Basin" else 1.6,
            label=region,
            zorder=3 if region == "Basin" else 2
        )

        # linear trend overlay (for visual guidance only)
        xx = sub["year"].values.astype(float)
        yy = sub[metric].values.astype(float)
        m = np.isfinite(xx) & np.isfinite(yy)
        if m.sum() >= 6:
            fit = stats.linregress(xx[m], yy[m])
            xfit = np.array([xx[m].min(), xx[m].max()])
            yfit = fit.intercept + fit.slope * xfit
            ax.plot(
                xfit, yfit,
                color=region_colors[region],
                lw=1.1,
                ls="--",
                alpha=0.45,
                zorder=1
            )

    # basin Pettitt line
    basin_tr = trend_summary[
        (trend_summary["region"] == "Basin") &
        (trend_summary["metric"] == metric)
    ]
    if len(basin_tr) > 0:
        basin_tr_series = basin_tr.iloc[0] # Renamed variable to avoid confusion
        cp_year = basin_tr_series["pettitt_change_year"]
        if pd.notna(cp_year):
            ax.axvline(cp_year, color="0.45", lw=1.2, ls=":", zorder=0)

    # compact basin annotation
    if len(basin_tr) > 0:
        # basin_tr = basin_tr.iloc[0] # REMOVED: This line caused the error
        slope_txt = basin_tr_series["sen_slope_per_decade"]
        mkp = basin_tr_series["mk_p_value"]
        cp = basin_tr_series["pettitt_change_year"]

        ann = f"Basin Sen slope = {slope_txt:.2f} decade$^{{-1}}$\nMK p = {mkp:.3g}"
        if pd.notna(cp):
            ann += f"\nPettitt = {int(cp)}"

        ax.text(
            0.015, 0.96,
            ann,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8.7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.5)
        )

    ax.set_title(f"{letter} {metric_titles[metric]}", loc="left", fontweight="bold")
    ax.set_ylabel(metric_labels[metric])
    ax.grid(True, axis="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

def add_trend_summary_panel(ax, metric, letter):
    """
    Right-column panels: Sen slopes with CI, MK significance, and change-point labels.
    """
    sub = slope_plot_df[slope_plot_df["metric"] == metric].copy()

    y_positions = np.arange(len(regions))[::-1]  # Basin at top
    region_to_y = {r: y for r, y in zip(regions, y_positions)}

    # zero line
    ax.axvline(0, color="0.4", lw=1.0, ls="--", zorder=0)

    for _, row in sub.iterrows():
        reg = row["region"]
        y = region_to_y[reg]
        x = row["sen_slope_per_decade"]
        lo = row["slope_ci_low"]
        hi = row["slope_ci_high"]
        p = row["mk_p_value"]
        cp = row["pettitt_change_year"]
        cpp = row["pettitt_p_value"]

        color = region_colors[reg]

        if np.isfinite(lo) and np.isfinite(hi):
            ax.hlines(y, lo, hi, color=color, lw=2.0, alpha=0.85, zorder=2)

        ax.plot(
            x, y,
            marker="o",
            ms=6.5 if reg == "Basin" else 5.7,
            color=color,
            mec="black" if (np.isfinite(p) and p < 0.05) else color,
            mew=0.8 if (np.isfinite(p) and p < 0.05) else 0.0,
            zorder=3
        )

        # significance star
        sig_txt = ""
        if np.isfinite(p):
            if p < 0.001:
                sig_txt = "***"
            elif p < 0.01:
                sig_txt = "**"
            elif p < 0.05:
                sig_txt = "*"

        if sig_txt != "":
            ax.text(x, y + 0.17, sig_txt, color=color, ha="center", va="bottom", fontsize=10, fontweight="bold")

        # change-point year label
        if pd.notna(cp):
            cp_label = f"{int(cp)}"
            if np.isfinite(cpp):
                if cpp < 0.05:
                    cp_label += "†"
            ax.text(
                ax.get_xlim()[1] if np.isfinite(ax.get_xlim()[1]) else 1,
                y,
                cp_label,
                ha="right",
                va="center",
                fontsize=8.3,
                color="0.35"
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(regions)
    ax.set_title(f"{letter} Trend summary", loc="left", fontweight="bold")
    ax.set_xlabel(f"Sen slope per decade ({metric_titles[metric].lower()})")
    ax.grid(True, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def set_metric_specific_xlim(ax, metric):
    vals = slope_plot_df[slope_plot_df["metric"] == metric]["sen_slope_per_decade"].dropna()
    lows = slope_plot_df[slope_plot_df["metric"] == metric]["slope_ci_low"].dropna()
    highs = slope_plot_df[slope_plot_df["metric"] == metric]["slope_ci_high"].dropna()

    allv = []
    allv.extend(vals.tolist())
    allv.extend(lows.tolist())
    allv.extend(highs.tolist())

    if len(allv) == 0:
        return

    mn = np.nanmin(allv)
    mx = np.nanmax(allv)

    if np.isfinite(mn) and np.isfinite(mx):
        span = mx - mn
        if span == 0:
            span = abs(mx) * 0.4 + 1
        pad = 0.22 * span
        ax.set_xlim(mn - pad, mx + pad)

# -------------------------
# 9) Create refined 8-panel figure
# -------------------------
fig = plt.figure(figsize=(16.2, 18.2), constrained_layout=False)
gs = fig.add_gridspec(
    nrows=4, ncols=2,
    width_ratios=[2.5, 1.45],
    hspace=0.18, wspace=0.16
)

axes = []
for i in range(4):
    ax_left = fig.add_subplot(gs[i, 0])
    ax_right = fig.add_subplot(gs[i, 1])
    axes.append((ax_left, ax_right))

for i, metric in enumerate(metric_order):
    add_timeseries_panel(axes[i][0], metric, panel_letters[2 * i])
    set_metric_specific_xlim(axes[i][1], metric)
    add_trend_summary_panel(axes[i][1], metric, panel_letters[2 * i + 1])

# x labels only on bottom row
for i in range(3):
    axes[i][0].tick_params(labelbottom=False)
    axes[i][1].tick_params(labelbottom=False)

axes[-1][0].set_xlabel("Year")
axes[-1][1].set_xlabel("Sen slope per decade")

# legend for left column
legend_handles = [
    Line2D([0], [0], color=region_colors[r], lw=2.0 if r == "Basin" else 1.8, label=r)
    for r in regions
]
axes[0][0].legend(
    handles=legend_handles,
    ncol=4,
    loc="upper left",
    bbox_to_anchor=(0.0, 1.01),
    frameon=False,
    handlelength=2.0,
    columnspacing=1.4
)

# explanatory note for right column
fig.text(
    0.73, 0.975,
    "Right-column panels show Sen slopes per decade; black-edged points indicate Mann–Kendall p < 0.05;\n"
    "† denotes Pettitt change point significant at p < 0.05.",
    ha="center", va="top", fontsize=9.5
)

fig.suptitle(
    "Figure 4. Intensification and regime-shift diagnostics of Bay of Bengal marine heatwaves (1995–2025)",
    fontweight="bold",
    y=0.995
)

fig.subplots_adjust(top=0.955, left=0.08, right=0.97, bottom=0.05)

fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PDF, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)

# -------------------------
# 10) Print compact manuscript-ready summary
# -------------------------
print("\n================ FIGURE 4 GENERATED ================\n")
print("Saved files:")
print(OUT_PNG)
print(OUT_PDF)
print(OUT_SLOPE_TABLE)

print("\nBasin-level summary:")
for metric in metric_order:
    row = trend_summary[
        (trend_summary["region"] == "Basin") &
        (trend_summary["metric"] == metric)
    ].iloc[0]

    slope = row["sen_slope_per_decade"]
    mkp = row["mk_p_value"]
    cp = row["pettitt_change_year"]
    cpp = row["pettitt_p_value"]

    msg = (
        f"{metric}: Sen slope = {slope:.3f} per decade, "
        f"MK p = {mkp:.3g}"
    )
    if pd.notna(cp):
        msg += f", Pettitt = {int(cp)} (p = {cpp:.3g})"
    print(msg)


# %% [markdown]
# # **Without Trend Summary, Spatial Plot**


# %% Cell 7
# ============================================================
# BAY OF BENGAL MHW TREND + REGIME-SHIFT ANALYSIS
# REFINED Q1-PUBLICATION-READY COLAB SCRIPT
#
# PURPOSE
#   Supports Sections 3.1 and 3.2 with:
#     (1) annual basin and sub-basin MHW metrics
#     (2) Mann-Kendall trend significance
#     (3) Sen-slope trend magnitude
#     (4) Pettitt regime-shift detection
#     (5) professional annual time-series figure
#     (6) professional grid-cell trend maps
#
# INPUT
#   Excel workbook with sheet "MHW"
#   Required columns:
#       event, lon, lat, date_start, date_peak, date_end,
#       intensity_mean, year, month, Season
#
# OUTPUT
#   - annual metrics tables
#   - trend summary tables
#   - grid-cell trend tables
#   - Figure_3A_Annual_TimeSeries_Q1_1200dpi.png / .pdf
#   - Figure_3B_Spatial_Trend_Maps_Q1_1200dpi.png / .pdf
#
# NOTES
#   - Uses uploaded event catalogue directly
#   - MHW days computed from expanded daily event windows
#   - Exposure metrics are zero-filled for no-event years
#   - Mean intensity is computed from active-event years only
# ============================================================

# -------------------------
# 1) Install dependencies
# -------------------------
# Colab shell command removed for repository reproducibility: !pip -q install pandas numpy scipy matplotlib openpyxl xlsxwriter cartopy

# -------------------------
# 2) Imports
# -------------------------
import os
import gc
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm

from scipy.stats import theilslopes, norm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

OUTDIR = Path(str(DATA_ROOT / "BoB_MHW_trend_regimeshift_outputs_Q1_refined"))
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_ANNUAL_CSV   = OUTDIR / "BoB_MHW_annual_basin_subbasin_metrics_refined.csv"
OUT_TREND_CSV    = OUTDIR / "BoB_MHW_trend_summary_refined.csv"
OUT_GRID_CSV     = OUTDIR / "BoB_MHW_gridcell_trends_refined.csv"
OUT_XLSX         = OUTDIR / "BoB_MHW_trend_regimeshift_outputs_refined.xlsx"

OUT_FIG_A_PNG    = OUTDIR / "Figure_3A_Annual_TimeSeries_Q1_1200dpi.png"
OUT_FIG_A_PDF    = OUTDIR / "Figure_3A_Annual_TimeSeries_Q1.pdf"
OUT_FIG_B_PNG    = OUTDIR / "Figure_3B_Spatial_Trend_Maps_Q1_1200dpi.png"
OUT_FIG_B_PDF    = OUTDIR / "Figure_3B_Spatial_Trend_Maps_Q1.pdf"

# -------------------------
# 5) STUDY SETTINGS
# -------------------------
START_YEAR = 1995
END_YEAR   = 2025

LAT_MIN, LAT_MAX = 5.0, 22.0
LON_MIN, LON_MAX = 80.0, 100.0

SUBBASIN_BOUNDS = {
    "Northern BoB": (16.0, 22.0),
    "Central BoB":  (10.0, 16.0),
    "Southern BoB": (5.0, 10.0),
}
REGION_ORDER = ["Basin", "Northern BoB", "Central BoB", "Southern BoB"]

DPI = 1200
MIN_YEARS_FOR_GRID_TREND = 8
MIN_EVENT_YEARS_FOR_INTENSITY_TREND = 6

# -------------------------
# 6) Plot style
# -------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.titlesize": 18,
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
# 7) Helper functions
# -------------------------
def save_figure(fig, png_path, pdf_path=None):
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

def detect_subbasin(lat):
    if 16.0 <= lat <= 22.0:
        return "Northern BoB"
    elif 10.0 <= lat < 16.0:
        return "Central BoB"
    elif 5.0 <= lat < 10.0:
        return "Southern BoB"
    return np.nan

def mann_kendall_test(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)

    if n < 3:
        return {"n": n, "S": np.nan, "tau": np.nan, "z": np.nan, "p": np.nan}

    s = 0
    for i in range(n - 1):
        s += np.sum(np.sign(y[i + 1:] - y[i]))

    _, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if var_s <= 0:
        z = 0.0
        p = 1.0
    else:
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0.0
        p = 2 * (1 - norm.cdf(abs(z)))

    tau = s / (0.5 * n * (n - 1))
    return {"n": n, "S": float(s), "tau": float(tau), "z": float(z), "p": float(p)}

def sen_slope_with_ci(years, values, alpha=0.95):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(years) & np.isfinite(values)

    if mask.sum() < 3:
        return {
            "slope_per_year": np.nan,
            "slope_per_decade": np.nan,
            "ci_low_per_decade": np.nan,
            "ci_high_per_decade": np.nan,
            "intercept": np.nan
        }

    res = theilslopes(values[mask], years[mask], alpha=alpha)
    return {
        "slope_per_year": float(res.slope),
        "slope_per_decade": float(res.slope * 10.0),
        "ci_low_per_decade": float(res.low_slope * 10.0),
        "ci_high_per_decade": float(res.high_slope * 10.0),
        "intercept": float(res.intercept)
    }

def pettitt_test(y, years):
    y = np.asarray(y, dtype=float)
    years = np.asarray(years, dtype=int)
    mask = np.isfinite(y)
    y = y[mask]
    years = years[mask]
    n = len(y)

    if n < 5:
        return {"change_index": np.nan, "change_year": np.nan, "K": np.nan, "p": np.nan}

    U = np.zeros(n)
    for t in range(n):
        left = y[:t + 1]
        right = y[t + 1:]
        if len(right) == 0:
            U[t] = 0
        else:
            s = 0
            for xi in left:
                s += np.sum(np.sign(xi - right))
            U[t] = s

    K = np.max(np.abs(U))
    idx = int(np.argmax(np.abs(U)))
    p = 2 * np.exp((-6 * K**2) / (n**3 + n**2))
    return {
        "change_index": idx,
        "change_year": int(years[idx]),
        "K": float(K),
        "p": float(min(p, 1.0))
    }

def area_weighted_mean(values, lats):
    values = np.asarray(values, dtype=float)
    lats = np.asarray(lats, dtype=float)
    mask = np.isfinite(values) & np.isfinite(lats)
    if mask.sum() == 0:
        return np.nan
    weights = np.cos(np.deg2rad(lats[mask]))
    return np.average(values[mask], weights=weights)

def build_complete_cell_year_panel(cell_df, years):
    panel = (
        pd.MultiIndex.from_product(
            [cell_df.index.values, years],
            names=["cell_idx", "year"]
        )
        .to_frame(index=False)
        .merge(
            cell_df.reset_index().rename(columns={"index": "cell_idx"}),
            on="cell_idx",
            how="left"
        )
    )
    return panel

def pretty_metric_name(metric):
    mapping = {
        "event_frequency_mean": "Event frequency",
        "mhw_days_mean": "MHW days",
        "mean_intensity_active_cells": "Mean intensity",
        "cumulative_intensity_mean": "Cumulative intensity"
    }
    return mapping.get(metric, metric)

def pretty_metric_unit(metric):
    mapping = {
        "event_frequency_mean": "events yr$^{-1}$",
        "mhw_days_mean": "days yr$^{-1}$",
        "mean_intensity_active_cells": "°C",
        "cumulative_intensity_mean": "°C·days yr$^{-1}$"
    }
    return mapping.get(metric, "")

def annotate_trend_box(ax, trend_row):
    if pd.isna(trend_row["sen_slope_per_decade"]):
        txt = "Trend statistics unavailable"
    else:
        txt = (
            f"Sen slope = {trend_row['sen_slope_per_decade']:.2f} {trend_row['unit']} decade$^{{-1}}$\n"
            f"MK p = {trend_row['mk_p']:.3g}\n"
            f"Pettitt year = {int(trend_row['pettitt_change_year']) if pd.notna(trend_row['pettitt_change_year']) else 'NA'}"
        )
    ax.text(
        0.015, 0.97, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.92)
    )

def make_grid(df, value_col):
    lons = np.sort(df["lon"].unique())
    lats = np.sort(df["lat"].unique())
    grid = df.pivot(index="lat", columns="lon", values=value_col).reindex(index=lats, columns=lons)
    xx, yy = np.meshgrid(lons, lats)
    return xx, yy, grid.values, lons, lats

# -------------------------
# 8) Read workbook
# -------------------------
print("Reading MHW workbook...")
df = pd.read_excel(MHW_XLSX, sheet_name="MHW")

required_cols = [
    "event", "lon", "lat", "date_start", "date_peak", "date_end",
    "intensity_mean", "year", "month", "Season"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for c in ["date_start", "date_peak", "date_end"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")

df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["intensity_mean"] = pd.to_numeric(df["intensity_mean"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")

df = df.dropna(subset=["lon", "lat", "date_start", "date_end", "intensity_mean", "year"]).copy()

df = df[
    (df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX) &
    (df["lon"] >= LON_MIN) & (df["lon"] <= LON_MAX)
].copy()

df["subbasin"] = df["lat"].apply(detect_subbasin)
df = df.dropna(subset=["subbasin"]).copy()

df["year"] = df["year"].astype(int)
df["duration_days"] = (df["date_end"] - df["date_start"]).dt.days + 1
df["duration_days"] = df["duration_days"].clip(lower=1)
df["cumulative_intensity"] = df["intensity_mean"] * df["duration_days"]

df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

print("Rows:", len(df))
print("Year range:", df["year"].min(), "to", df["year"].max())
print(df["subbasin"].value_counts())

# -------------------------
# 9) Expand to daily event windows
# -------------------------
print("\nExpanding events to daily windows...")

expanded = []
for row in df.itertuples(index=False):
    dr = pd.date_range(row.date_start, row.date_end, freq="D")
    expanded.append(
        pd.DataFrame({
            "lon": row.lon,
            "lat": row.lat,
            "subbasin": row.subbasin,
            "event_id": f"{row.event}_{row.lon}_{row.lat}_{row.date_start.strftime('%Y%m%d')}",
            "date": dr
        })
    )

daily = pd.concat(expanded, ignore_index=True)
daily["year"] = daily["date"].dt.year
daily = daily.drop_duplicates(subset=["lon", "lat", "date"]).copy()

print("Expanded daily rows:", len(daily))

# -------------------------
# 10) Cell-year metrics
# -------------------------
print("\nBuilding cell-year metrics...")

cells = df[["lon", "lat", "subbasin"]].drop_duplicates().copy()
years = np.arange(START_YEAR, END_YEAR + 1)

cell_year_event = (
    df.groupby(["lon", "lat", "subbasin", "year"], as_index=False)
      .agg(
          event_frequency=("event", "size"),
          mean_intensity=("intensity_mean", "mean"),
          cumulative_intensity=("cumulative_intensity", "sum")
      )
)

cell_year_days = (
    daily.groupby(["lon", "lat", "subbasin", "year"], as_index=False)
         .agg(mhw_days=("date", "nunique"))
)

cell_year = cell_year_event.merge(
    cell_year_days,
    on=["lon", "lat", "subbasin", "year"],
    how="outer"
)

panel = build_complete_cell_year_panel(cells, years)
cell_year_full = panel.merge(
    cell_year,
    on=["lon", "lat", "subbasin", "year"],
    how="left"
)

cell_year_full["event_frequency"] = cell_year_full["event_frequency"].fillna(0.0)
cell_year_full["mhw_days"] = cell_year_full["mhw_days"].fillna(0.0)
cell_year_full["cumulative_intensity"] = cell_year_full["cumulative_intensity"].fillna(0.0)
cell_year_full["area_weight"] = np.cos(np.deg2rad(cell_year_full["lat"]))

print("Cell-year rows:", len(cell_year_full))

# -------------------------
# 11) Annual basin and sub-basin metrics
# -------------------------
print("\nComputing annual regional metrics...")

annual_rows = []

for region in REGION_ORDER:
    for yr in years:
        sub = cell_year_full[cell_year_full["year"] == yr].copy()
        if region != "Basin":
            sub = sub[sub["subbasin"] == region].copy()

        freq_mean = area_weighted_mean(sub["event_frequency"], sub["lat"])
        days_mean = area_weighted_mean(sub["mhw_days"], sub["lat"])
        cumi_mean = area_weighted_mean(sub["cumulative_intensity"], sub["lat"])

        active = sub[np.isfinite(sub["mean_intensity"])].copy()
        intensity_mean = area_weighted_mean(active["mean_intensity"], active["lat"]) if len(active) > 0 else np.nan

        annual_rows.append({
            "region": region,
            "year": yr,
            "event_frequency_mean": freq_mean,
            "mhw_days_mean": days_mean,
            "mean_intensity_active_cells": intensity_mean,
            "cumulative_intensity_mean": cumi_mean,
            "n_total_cells": len(sub),
            "n_active_cells": len(active)
        })

annual_df = pd.DataFrame(annual_rows)
annual_df.to_csv(OUT_ANNUAL_CSV, index=False)

# -------------------------
# 12) Formal trend and regime-shift statistics
# -------------------------
print("\nComputing trend statistics...")

trend_rows = []
metrics = [
    "event_frequency_mean",
    "mhw_days_mean",
    "mean_intensity_active_cells",
    "cumulative_intensity_mean"
]

for region in REGION_ORDER:
    reg = annual_df[annual_df["region"] == region].copy()
    for metric in metrics:
        y = reg[metric].values
        yrs = reg["year"].values

        mk = mann_kendall_test(y)
        sen = sen_slope_with_ci(yrs, y)
        pet = pettitt_test(y, yrs)

        trend_rows.append({
            "region": region,
            "metric": metric,
            "metric_pretty": pretty_metric_name(metric),
            "unit": pretty_metric_unit(metric),
            "n": mk["n"],
            "mk_tau": mk["tau"],
            "mk_z": mk["z"],
            "mk_p": mk["p"],
            "sen_slope_per_year": sen["slope_per_year"],
            "sen_slope_per_decade": sen["slope_per_decade"],
            "sen_ci_low_per_decade": sen["ci_low_per_decade"],
            "sen_ci_high_per_decade": sen["ci_high_per_decade"],
            "pettitt_change_year": pet["change_year"],
            "pettitt_K": pet["K"],
            "pettitt_p": pet["p"]
        })

trend_df = pd.DataFrame(trend_rows)
trend_df.to_csv(OUT_TREND_CSV, index=False)

# -------------------------
# 13) Grid-cell trends
# -------------------------
print("\nComputing grid-cell trends...")

grid_rows = []
for (lon, lat, subbasin), sub in cell_year_full.groupby(["lon", "lat", "subbasin"]):
    sub = sub.sort_values("year")

    y_days = sub["mhw_days"].values
    yrs = sub["year"].values

    mk_days = mann_kendall_test(y_days)
    sen_days = sen_slope_with_ci(yrs, y_days)

    active = sub[np.isfinite(sub["mean_intensity"])].copy()
    if len(active) >= MIN_EVENT_YEARS_FOR_INTENSITY_TREND:
        mk_int = mann_kendall_test(active["mean_intensity"].values)
        sen_int = sen_slope_with_ci(active["year"].values, active["mean_intensity"].values)
    else:
        mk_int = {"tau": np.nan, "z": np.nan, "p": np.nan, "n": len(active)}
        sen_int = {
            "slope_per_year": np.nan,
            "slope_per_decade": np.nan,
            "ci_low_per_decade": np.nan,
            "ci_high_per_decade": np.nan,
            "intercept": np.nan
        }

    grid_rows.append({
        "lon": lon,
        "lat": lat,
        "subbasin": subbasin,
        "n_years_days": len(sub),
        "mhw_days_sen_per_decade": sen_days["slope_per_decade"],
        "mhw_days_mk_p": mk_days["p"],
        "mhw_days_mk_tau": mk_days["tau"],
        "mhw_days_sig_p05": bool(np.isfinite(mk_days["p"]) and mk_days["p"] < 0.05),

        "n_years_intensity": len(active),
        "intensity_sen_per_decade": sen_int["slope_per_decade"],
        "intensity_mk_p": mk_int["p"],
        "intensity_mk_tau": mk_int["tau"],
        "intensity_sig_p05": bool(np.isfinite(mk_int["p"]) and mk_int["p"] < 0.05)
    })

grid_df = pd.DataFrame(grid_rows)
grid_df.to_csv(OUT_GRID_CSV, index=False)

# -------------------------
# 14) Save Excel workbook
# -------------------------
with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
    annual_df.to_excel(writer, sheet_name="annual_metrics", index=False)
    trend_df.to_excel(writer, sheet_name="trend_summary", index=False)
    grid_df.to_excel(writer, sheet_name="gridcell_trends", index=False)
    cell_year_full.to_excel(writer, sheet_name="cell_year_metrics", index=False)

print("\nSaved tables:")
print(OUT_ANNUAL_CSV)
print(OUT_TREND_CSV)
print(OUT_GRID_CSV)
print(OUT_XLSX)

# -------------------------
# 15) FIGURE A: Annual basin and sub-basin time series
# -------------------------
print("\nBuilding refined Figure A...")

region_colors = {
    "Basin": "#1a1a1a",
    "Northern BoB": "#d73027",
    "Central BoB": "#4575b4",
    "Southern BoB": "#1a9850"
}

fig, axes = plt.subplots(4, 1, figsize=(14.5, 15.5), sharex=True, constrained_layout=False)

panel_labels = ["(a)", "(b)", "(c)", "(d)"]

for ax, metric, plab in zip(axes, metrics, panel_labels):
    meta_name = pretty_metric_name(metric)
    meta_unit = pretty_metric_unit(metric)

    basin_tr = trend_df[(trend_df["region"] == "Basin") & (trend_df["metric"] == metric)].iloc[0]
    cp_year = basin_tr["pettitt_change_year"] if pd.notna(basin_tr["pettitt_change_year"]) else None

    # Pettitt regime shading
    if cp_year is not None:
        ax.axvspan(cp_year, END_YEAR + 0.3, color="#f0f0f0", alpha=0.85, zorder=0)

    # plot all regions
    for region in REGION_ORDER:
        reg = annual_df[annual_df["region"] == region].copy()

        lw = 2.5 if region == "Basin" else 1.8
        alpha = 1.0 if region == "Basin" else 0.95

        ax.plot(
            reg["year"], reg[metric],
            color=region_colors[region],
            linewidth=lw,
            alpha=alpha,
            zorder=3 if region == "Basin" else 2
        )

        # Sen trend fit
        sen = sen_slope_with_ci(reg["year"].values, reg[metric].values)
        if np.isfinite(sen["slope_per_year"]):
            yfit = sen["slope_per_year"] * reg["year"].values + sen["intercept"]
            ax.plot(
                reg["year"], yfit,
                color=region_colors[region],
                linestyle="--",
                linewidth=1.1,
                alpha=0.45
            )

    if cp_year is not None:
        ax.axvline(cp_year, color="0.45", ls=":", lw=1.2)

    ax.text(0.005, 1.02, plab, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    ax.set_ylabel(meta_unit)
    ax.set_title(meta_name, pad=6)
    ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    annotate_trend_box(ax, basin_tr)

axes[-1].set_xlabel("Year")
axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(5))
axes[-1].set_xlim(START_YEAR - 0.5, END_YEAR + 0.5)

legend_lines = [
    Line2D([0], [0], color=region_colors["Basin"], lw=2.5, label="Basin"),
    Line2D([0], [0], color=region_colors["Northern BoB"], lw=1.8, label="Northern"),
    Line2D([0], [0], color=region_colors["Central BoB"], lw=1.8, label="Central"),
    Line2D([0], [0], color=region_colors["Southern BoB"], lw=1.8, label="Southern"),
    Patch(facecolor="#f0f0f0", edgecolor="0.6", label="Post-change regime")
]

fig.legend(
    handles=legend_lines,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.975),
    ncol=5,
    frameon=False
)

fig.suptitle(
    "Annual basin and sub-basin evolution of Bay of Bengal marine heatwave metrics (1995–2025)",
    fontsize=20,
    fontweight="bold",
    y=0.995
)

fig.subplots_adjust(left=0.10, right=0.98, bottom=0.06, top=0.955, hspace=0.16)

save_figure(fig, OUT_FIG_A_PNG, OUT_FIG_A_PDF)
plt.show()
plt.close(fig)

# -------------------------
# 16) FIGURE B: Spatial trend maps
# -------------------------
print("\nBuilding refined Figure B...")

fig = plt.figure(figsize=(15.8, 7.2))
proj = ccrs.PlateCarree()

ax1 = fig.add_subplot(1, 2, 1, projection=proj)
ax2 = fig.add_subplot(1, 2, 2, projection=proj)

for ax in [ax1, ax2]:
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", edgecolor="black", linewidth=0.35, zorder=3)
    ax.coastlines(resolution="10m", linewidth=0.55)
    gl = ax.gridlines(draw_labels=True, linewidth=0.35, linestyle="--", color="0.6", alpha=0.45)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}
    gl.xlocator = mticker.FixedLocator(np.arange(80, 101, 5))
    gl.ylocator = mticker.FixedLocator(np.arange(5, 23, 5))

# --- MHW days map
days_df = grid_df.copy()
xx_d, yy_d, zz_d, _, _ = make_grid(days_df, "mhw_days_sen_per_decade")
vmax_days = np.nanpercentile(np.abs(days_df["mhw_days_sen_per_decade"]), 95)
norm_days = TwoSlopeNorm(vmin=-vmax_days, vcenter=0, vmax=vmax_days)

pcm1 = ax1.pcolormesh(
    xx_d, yy_d, zz_d,
    cmap="RdBu_r",
    norm=norm_days,
    shading="nearest",
    transform=proj,
    zorder=1
)

sig_days = days_df[days_df["mhw_days_sig_p05"] == True]
ax1.scatter(
    sig_days["lon"], sig_days["lat"],
    s=6, c="k", marker="o", alpha=0.65,
    transform=proj, zorder=4
)

ax1.set_title("(a) Trend in annual MHW days", fontsize=13, pad=8)

# --- Intensity map
int_df = grid_df.copy()
xx_i, yy_i, zz_i, _, _ = make_grid(int_df, "intensity_sen_per_decade")
vmax_int = np.nanpercentile(np.abs(int_df["intensity_sen_per_decade"]), 95)
norm_int = TwoSlopeNorm(vmin=-vmax_int, vcenter=0, vmax=vmax_int)

pcm2 = ax2.pcolormesh(
    xx_i, yy_i, zz_i,
    cmap="RdBu_r",
    norm=norm_int,
    shading="nearest",
    transform=proj,
    zorder=1
)

sig_int = int_df[int_df["intensity_sig_p05"] == True]
ax2.scatter(
    sig_int["lon"], sig_int["lat"],
    s=6, c="k", marker="o", alpha=0.65,
    transform=proj, zorder=4
)

ax2.set_title("(b) Trend in mean MHW intensity", fontsize=13, pad=8)

# colorbars
cbar1 = fig.colorbar(pcm1, ax=ax1, orientation="horizontal", fraction=0.050, pad=0.08)
cbar1.set_label("Trend in MHW days (days decade$^{-1}$)")

cbar2 = fig.colorbar(pcm2, ax=ax2, orientation="horizontal", fraction=0.050, pad=0.08)
cbar2.set_label("Trend in mean intensity (°C decade$^{-1}$)")

fig.legend(
    handles=[Line2D([0], [0], marker="o", color="k", linestyle="None", markersize=4, label="p < 0.05")],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.945),
    ncol=1,
    frameon=False
)

fig.suptitle(
    "Grid-cell trends in Bay of Bengal marine heatwave exposure and intensity (1995–2025)",
    fontsize=20,
    fontweight="bold",
    y=0.985
)

fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.91, wspace=0.08)

save_figure(fig, OUT_FIG_B_PNG, OUT_FIG_B_PDF)
plt.show()
plt.close(fig)

# -------------------------
# 17) Ready-to-paste basin statistics
# -------------------------
print("\n================ BASIN-LEVEL FORMAL STATISTICS ================\n")

basin_only = trend_df[trend_df["region"] == "Basin"].copy()
for _, r in basin_only.iterrows():
    print(
        f"{r['metric_pretty']}: Sen slope = {r['sen_slope_per_decade']:.3f} {r['unit']} decade^-1 "
        f"(95% CI {r['sen_ci_low_per_decade']:.3f} to {r['sen_ci_high_per_decade']:.3f}); "
        f"Mann-Kendall p = {r['mk_p']:.4g}; "
        f"Pettitt change year = {int(r['pettitt_change_year']) if pd.notna(r['pettitt_change_year']) else 'NA'} "
        f"(p = {r['pettitt_p']:.4g})"
    )

# Specific line for manuscript Section 3.1
basin_int = basin_only[basin_only["metric"] == "mean_intensity_active_cells"].iloc[0]
print("\nREADY-TO-PASTE SECTION 3.1 SENTENCE:\n")
print(
    f"Formal statistical testing of the basin-averaged mean-intensity time series indicates "
    f"a Sen-slope trend of {basin_int['sen_slope_per_decade']:.3f} °C decade⁻¹ "
    f"(95% CI {basin_int['sen_ci_low_per_decade']:.3f} to {basin_int['sen_ci_high_per_decade']:.3f}), "
    f"a Mann–Kendall p value of {basin_int['mk_p']:.3g}, "
    f"and a Pettitt change point centered on {int(basin_int['pettitt_change_year']) if pd.notna(basin_int['pettitt_change_year']) else 'NA'} "
    f"(p = {basin_int['pettitt_p']:.3g})."
)

print("\nSaved figures:")
print(OUT_FIG_A_PNG)
print(OUT_FIG_B_PNG)

print("\nDone.")
