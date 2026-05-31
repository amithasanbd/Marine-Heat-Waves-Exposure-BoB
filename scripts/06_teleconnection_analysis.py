#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06 Teleconnection Analysis

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
# # **Figure_7_BoB_MHW_phase_composites**


# %% Cell 2
# ============================================================
# BAY OF BENGAL MHW TELECONNECTION FIGURES
# Figure 6: Climate-mode lag correlations with BoB MHW metrics
# Figure 7: Climate-phase composite differences in BoB MHW activity
#
# Refined Nature/Q1 style version
# Fixes:
# - overlapping legends / colorbars / axis labels
# - tighter but clean journal layout
# - better typography and spacing
# - stronger visual balance
# ============================================================

# -------------------------
# 1) Install dependencies
# -------------------------
# Colab shell command removed for repository reproducibility: !pip -q install pandas numpy scipy matplotlib openpyxl statsmodels seaborn

# -------------------------
# 2) Imports
# -------------------------
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib import gridspec
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# -------------------------
# 3) Mount Google Drive
# -------------------------
# Google Colab Drive import removed; use config/paths_template.yaml and local data folders.
# Google Drive mount removed; configure DATA_ROOT in this script or via environment variable MHW_DATA_ROOT.

# -------------------------
# 4) USER PATHS
# -------------------------
BASE_DIR = Path(str(DATA_ROOT / "Marine Heat Waves Dataset (2000-24)"))

MHW_SEASONAL_XLSX = BASE_DIR / "BoB_MHW_Seasonal_1995_2025.xlsx"

TELE_DIR = BASE_DIR / "Teleconnection (1995-2025)"
NORTH_XLSX = TELE_DIR / "Seasonal_Yearly_MHW & Teleconnection_Nbob_1995_2025.xlsx"
CENTRAL_XLSX = TELE_DIR / "Seasonal_Yearly_MHW & Teleconnection_CBoB_1995_2025.xlsx"
SOUTH_XLSX = TELE_DIR / "Seasonal_Yearly_MHW & Teleconnection_SBoB_1995_2025.xlsx"

COMPOSITE_XLSX = BASE_DIR / "Composite_ElNino_vs_LaNina,Pos IOD Vs Neg IOD_Exposure,Intensity.xlsx"

OUTDIR = Path(str(DATA_ROOT / "BoB_MHW_Fig6_Fig7_Teleconnection_Q1"))
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_FIG6_PNG = OUTDIR / "Figure_6_BoB_MHW_lag_correlations_Q1_1200dpi_refined.png"
OUT_FIG6_PDF = OUTDIR / "Figure_6_BoB_MHW_lag_correlations_Q1_refined.pdf"

OUT_FIG7_PNG = OUTDIR / "Figure_7_BoB_MHW_phase_composites_Q1_1200dpi_refined.png"
OUT_FIG7_PDF = OUTDIR / "Figure_7_BoB_MHW_phase_composites_Q1_refined.pdf"

OUT_LAG_STATS_CSV = OUTDIR / "BoB_Figure6_lag_correlation_statistics.csv"
OUT_PHASE_STATS_CSV = OUTDIR / "BoB_Figure7_phase_composite_statistics.csv"

DPI = 1200
RANDOM_SEED = 42
LAGS = np.arange(-3, 4, 1)

ONI_POS = 0.5
ONI_NEG = -0.5
DMI_POS = 0.4
DMI_NEG = -0.4

# -------------------------
# 5) Plot style (REFINED)
# -------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",

    "font.size": 10.5,
    "axes.titlesize": 13.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 19,

    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.direction": "out",
    "ytick.direction": "out",

    "grid.color": "#d7d7d7",
    "grid.linestyle": "--",
    "grid.linewidth": 0.45,
    "grid.alpha": 0.6,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",

    "axes.spines.top": False,
    "axes.spines.right": False
})

# -------------------------
# 6) Helper functions
# -------------------------
def save_figure(fig, png_path, pdf_path=None):
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

def season_sort_key(season_name):
    order = {
        "Winter": 0,
        "Pre-monsoon": 1,
        "Pre-Monsoon": 1,
        "Monsoon": 2,
        "Post-monsoon": 3,
        "Post-Monsoon": 3
    }
    return order.get(season_name, 99)

def normalize_season_name(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    mapping = {
        "Pre-Monsoon": "Pre-monsoon",
        "Post-Monsoon": "Post-monsoon"
    }
    return mapping.get(s, s)

def detrend_series(y):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    out = np.full_like(y, np.nan, dtype=float)
    if m.sum() < 3:
        return out
    slope, intercept, *_ = stats.linregress(x[m], y[m])
    out[m] = y[m] - (intercept + slope * x[m])
    return out

def effective_sample_size(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = len(x)
    if n < 4:
        return np.nan

    def lag1_autocorr(a):
        if len(a) < 3:
            return 0.0
        a1 = a[:-1]
        a2 = a[1:]
        if np.nanstd(a1) == 0 or np.nanstd(a2) == 0:
            return 0.0
        r = np.corrcoef(a1, a2)[0, 1]
        return 0.0 if not np.isfinite(r) else r

    r1x = lag1_autocorr(x)
    r1y = lag1_autocorr(y)

    neff = n * (1 - r1x * r1y) / (1 + r1x * r1y)
    neff = max(3, min(n, neff))
    return neff

def pearson_r_eff_p(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = len(x)

    if n < 4:
        return np.nan, np.nan, n

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan, n

    r, _ = stats.pearsonr(x, y)
    neff = effective_sample_size(x, y)

    if not np.isfinite(neff) or neff <= 2 or not np.isfinite(r):
        return r, np.nan, n

    t = r * np.sqrt((neff - 2) / (1 - r**2 + 1e-12))
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=neff - 2))
    return r, p, n

def shifted_pair(index_vals, mhw_vals, lag):
    """
    Positive lag => climate index leads MHW response.
    lag = +1: correlate index[t] with mhw[t+1]
    lag = -1: correlate index[t] with mhw[t-1]
    """
    index_vals = np.asarray(index_vals, dtype=float)
    mhw_vals = np.asarray(mhw_vals, dtype=float)

    if lag > 0:
        return index_vals[:-lag], mhw_vals[lag:]
    elif lag < 0:
        return index_vals[-lag:], mhw_vals[:lag]
    else:
        return index_vals, mhw_vals

def bootstrap_mean_diff(a, b, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
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
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue

def format_p(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def significance_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def symmetric_limit_from_data(*arrays, floor=0.2, step=None):
    vals = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (-floor, floor)
    vmax = np.max(np.abs(vals))
    vmax = max(vmax, floor)
    if step is not None and step > 0:
        vmax = np.ceil(vmax / step) * step
    return (-vmax, vmax)

def nice_limit(min_val, max_val, padding_frac=0.10):
    vals = np.array([min_val, max_val], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (-1, 1)
    lo = vals.min()
    hi = vals.max()
    if lo == hi:
        pad = 0.2 if lo == 0 else abs(lo) * 0.25
        return lo - pad, hi + pad
    span = hi - lo
    pad = span * padding_frac
    return lo - pad, hi + pad

def add_panel_label(ax, text):
    ax.text(0.0, 1.03, text, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=13.5, fontweight="bold")

# -------------------------
# 7) Read teleconnection seasonal files
# -------------------------
subbasin_files = {
    "Northern": NORTH_XLSX,
    "Central": CENTRAL_XLSX,
    "Southern": SOUTH_XLSX
}

season_order = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
tele_rows = []

for subbasin, xlsx_path in subbasin_files.items():
    xls = pd.ExcelFile(xlsx_path)

    for sheet in xls.sheet_names:
        season = normalize_season_name(sheet)
        df = pd.read_excel(xlsx_path, sheet_name=sheet).copy()
        df.columns = [str(c).strip() for c in df.columns]

        required = ["Year", "Intensity_Mean", "No_of_Events", "ONI", "IOD"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{xlsx_path.name} / {sheet}: missing columns {missing}")

        df["Sub_basin"] = subbasin
        df["Season"] = season
        df["Exposure"] = pd.to_numeric(df["No_of_Events"], errors="coerce")
        df["Intensity"] = pd.to_numeric(df["Intensity_Mean"], errors="coerce")
        df["ONI"] = pd.to_numeric(df["ONI"], errors="coerce")
        df["DMI"] = pd.to_numeric(df["IOD"], errors="coerce")
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

        tele_rows.append(df[["Year", "Sub_basin", "Season", "Exposure", "Intensity", "ONI", "DMI"]])

tele_df = pd.concat(tele_rows, ignore_index=True)
tele_df["Season"] = tele_df["Season"].map(normalize_season_name)

# -------------------------
# 8) Figure 6 statistics
# -------------------------
lag_rows = []

for subbasin in ["Northern", "Central", "Southern"]:
    for season in season_order:
        sub = tele_df[(tele_df["Sub_basin"] == subbasin) & (tele_df["Season"] == season)].sort_values("Year").copy()

        if len(sub) < 8:
            continue

        sub["Exposure_dt"] = detrend_series(sub["Exposure"].values)
        sub["Intensity_dt"] = detrend_series(sub["Intensity"].values)
        sub["ONI_dt"] = detrend_series(sub["ONI"].values)
        sub["DMI_dt"] = detrend_series(sub["DMI"].values)

        for index_name in ["ONI", "DMI"]:
            idx_vals = sub[f"{index_name}_dt"].values

            for metric in ["Exposure", "Intensity"]:
                mhw_vals = sub[f"{metric}_dt"].values

                for lag in LAGS:
                    x_lag, y_lag = shifted_pair(idx_vals, mhw_vals, lag)
                    r, p, n = pearson_r_eff_p(x_lag, y_lag)

                    lag_rows.append({
                        "Sub_basin": subbasin,
                        "Season": season,
                        "Index": index_name,
                        "Metric": metric,
                        "Lag": lag,
                        "n": n,
                        "r": r,
                        "p_eff": p
                    })

lag_df = pd.DataFrame(lag_rows)

mask = np.isfinite(lag_df["p_eff"].values)
qvals = np.full(len(lag_df), np.nan, dtype=float)
if mask.sum() > 0:
    _, q_corr, _, _ = multipletests(
        lag_df.loc[mask, "p_eff"].values,
        alpha=0.05,
        method="fdr_bh"
    )
    qvals[mask] = q_corr
lag_df["q_fdr"] = qvals
lag_df.to_csv(OUT_LAG_STATS_CSV, index=False)

# -------------------------
# 9) Build Figure 6 matrices
# -------------------------
row_labels = [f"{sb} | {ss}" for sb in ["Northern", "Central", "Southern"] for ss in season_order]

def build_matrix(index_name, metric_name, value_col="r"):
    arr = np.full((len(row_labels), len(LAGS)), np.nan)
    star = np.full((len(row_labels), len(LAGS)), "", dtype=object)

    for i, row_lab in enumerate(row_labels):
        sb, ss = row_lab.split(" | ")
        sub = lag_df[
            (lag_df["Sub_basin"] == sb) &
            (lag_df["Season"] == ss) &
            (lag_df["Index"] == index_name) &
            (lag_df["Metric"] == metric_name)
        ].sort_values("Lag")

        for j, lag in enumerate(LAGS):
            rrow = sub[sub["Lag"] == lag]
            if len(rrow) == 0:
                continue
            arr[i, j] = rrow.iloc[0][value_col]
            p_here = rrow.iloc[0]["p_eff"]
            if pd.notna(p_here) and p_here < 0.05:
                star[i, j] = "*"
    return arr, star

mat_ONI_exp, star_ONI_exp = build_matrix("ONI", "Exposure")
mat_ONI_int, star_ONI_int = build_matrix("ONI", "Intensity")
mat_DMI_exp, star_DMI_exp = build_matrix("DMI", "Exposure")
mat_DMI_int, star_DMI_int = build_matrix("DMI", "Intensity")

# -------------------------
# 10) Draw Figure 6 (REFINED)
# -------------------------
def draw_fig6():
    fig = plt.figure(figsize=(13.8, 11.2), facecolor="white")
    gs = gridspec.GridSpec(
        nrows=3, ncols=2,
        height_ratios=[1, 1, 0.10],
        hspace=0.42, wspace=0.16
    )

    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ])
    cax = fig.add_subplot(gs[2, :])

    panels = [
        ("(a) ONI vs MHW exposure", mat_ONI_exp, star_ONI_exp),
        ("(b) ONI vs MHW intensity", mat_ONI_int, star_ONI_int),
        ("(c) DMI vs MHW exposure", mat_DMI_exp, star_DMI_exp),
        ("(d) DMI vs MHW intensity", mat_DMI_int, star_DMI_int)
    ]

    vmax = np.nanmax(np.abs(np.concatenate([
        mat_ONI_exp.flatten(), mat_ONI_int.flatten(),
        mat_DMI_exp.flatten(), mat_DMI_int.flatten()
    ])))
    if not np.isfinite(vmax):
        vmax = 0.6
    vmax = max(vmax, 0.40)
    vmax = np.ceil(vmax / 0.05) * 0.05
    vmin = -vmax

    cmap = mpl.cm.get_cmap("RdBu_r")
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    im = None
    for k, (ax, (title, mat, stars)) in enumerate(zip(axes.flat, panels)):
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto", interpolation="none")

        ax.set_title(title, loc="left", pad=8, fontweight="bold")
        ax.set_xticks(np.arange(len(LAGS)))
        ax.set_xticklabels([f"{l:+d}" for l in LAGS])
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)

        ax.set_xlabel("Lag (positive = climate index leads)", labelpad=4)

        if k % 2 == 1:
            ax.tick_params(axis="y", labelleft=False)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isfinite(val):
                    txt_color = "white" if abs(val) > 0.52 * vmax else "black"
                    ax.text(
                        j, i, f"{val:.2f}{stars[i, j]}",
                        ha="center", va="center",
                        fontsize=8.2, color=txt_color
                    )

        ax.set_xlim(-0.5, len(LAGS) - 0.5)
        ax.set_ylim(len(row_labels) - 0.5, -0.5)
        ax.tick_params(axis="x", pad=2)
        ax.tick_params(axis="y", pad=2)

        for spine in ax.spines.values():
            spine.set_visible(False)

    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Detrended Pearson correlation coefficient (r)", fontsize=11.5, labelpad=2)
    cb.ax.tick_params(labelsize=10, pad=2)

    fig.suptitle(
        "Figure 6 | Climate-mode lag correlations with Bay of Bengal marine heatwave metrics",
        fontsize=19, fontweight="bold", y=0.97
    )

    fig.text(
        0.5, 0.035,
        "Rows show sub-basin–season combinations; columns show lead–lag relationships. "
        "Asterisks denote p < 0.05 using effective-sample-size-adjusted significance.",
        ha="center", va="center", fontsize=10.5
    )

    fig.subplots_adjust(left=0.24, right=0.985, top=0.92, bottom=0.12)
    save_figure(fig, OUT_FIG6_PNG, OUT_FIG6_PDF)
    plt.show()
    plt.close(fig)

draw_fig6()

# -------------------------
# 11) Figure 7 statistics
# -------------------------
phase_rows = []

for subbasin in ["Northern", "Central", "Southern"]:
    for season in season_order:
        sub = tele_df[(tele_df["Sub_basin"] == subbasin) & (tele_df["Season"] == season)].copy()

        for index_name, pos_thr, neg_thr in [("ONI", ONI_POS, ONI_NEG), ("DMI", DMI_POS, DMI_NEG)]:
            idx = sub[index_name].values

            pos_mask = idx >= pos_thr
            neg_mask = idx <= neg_thr

            for metric in ["Exposure", "Intensity"]:
                pos_vals = sub.loc[pos_mask, metric].values
                neg_vals = sub.loc[neg_mask, metric].values

                delta, lo, hi = bootstrap_mean_diff(pos_vals, neg_vals, n_boot=5000, seed=RANDOM_SEED)
                p = welch_p(pos_vals, neg_vals)

                phase_rows.append({
                    "Sub_basin": subbasin,
                    "Season": season,
                    "Index": index_name,
                    "Metric": metric,
                    "n_positive": int(np.sum(pos_mask)),
                    "n_negative": int(np.sum(neg_mask)),
                    "mean_positive": np.nanmean(pos_vals) if len(pos_vals) > 0 else np.nan,
                    "mean_negative": np.nanmean(neg_vals) if len(neg_vals) > 0 else np.nan,
                    "mean_difference": delta,
                    "ci_low_95": lo,
                    "ci_high_95": hi,
                    "p_value": p
                })

phase_df = pd.DataFrame(phase_rows)
phase_df.to_csv(OUT_PHASE_STATS_CSV, index=False)

# -------------------------
# 12) Draw Figure 7 (REFINED)
# -------------------------
subbasin_colors = {
    "Northern": "#d73027",
    "Central": "#4575b4",
    "Southern": "#1a9850"
}

plot_specs = [
    ("(a) El Niño − La Niña: exposure", "ONI", "Exposure"),
    ("(b) El Niño − La Niña: intensity", "ONI", "Intensity"),
    ("(c) Positive IOD − Negative IOD: exposure", "DMI", "Exposure"),
    ("(d) Positive IOD − Negative IOD: intensity", "DMI", "Intensity")
]

ytick_positions = [0,1,2,3.5,4.5,5.5,7,8,9,10.5,11.5,12.5]
ytick_labels = [
    "Winter | Northern", "Winter | Central", "Winter | Southern",
    "Pre-monsoon | Northern", "Pre-monsoon | Central", "Pre-monsoon | Southern",
    "Monsoon | Northern", "Monsoon | Central", "Monsoon | Southern",
    "Post-monsoon | Northern", "Post-monsoon | Central", "Post-monsoon | Southern"
]

def draw_panel_composite(ax, df_sub, title, metric_name, show_left_labels=True):
    for season_idx, season in enumerate(season_order):
        for sb_idx, sb in enumerate(["Northern", "Central", "Southern"]):
            row = df_sub[(df_sub["Season"] == season) & (df_sub["Sub_basin"] == sb)]
            if len(row) == 0:
                continue
            row = row.iloc[0]

            y = season_idx * 3.5 + sb_idx
            x = row["mean_difference"]
            lo = row["ci_low_95"]
            hi = row["ci_high_95"]
            p = row["p_value"]

            if np.isfinite(lo) and np.isfinite(hi):
                ax.hlines(y, lo, hi, color=subbasin_colors[sb], linewidth=1.9, zorder=2)

            ax.plot(
                x, y,
                marker="o",
                markersize=6.8,
                color=subbasin_colors[sb],
                markeredgecolor="black" if pd.notna(p) and p < 0.05 else subbasin_colors[sb],
                markeredgewidth=0.85 if pd.notna(p) and p < 0.05 else 0.0,
                zorder=3
            )

    vals = np.concatenate([
        df_sub["ci_low_95"].values.astype(float),
        df_sub["ci_high_95"].values.astype(float),
        df_sub["mean_difference"].values.astype(float)
    ])
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        xmin, xmax = -1, 1
    else:
        xmin, xmax = nice_limit(vals.min(), vals.max(), padding_frac=0.12)

        if metric_name == "Intensity":
            xmin = min(xmin, -0.05)
            xmax = max(xmax, 0.05)

        rng = xmax - xmin
        if rng < 0.25 and metric_name == "Intensity":
            mid = 0.5 * (xmin + xmax)
            xmin, xmax = mid - 0.15, mid + 0.15

    ax.set_xlim(xmin, xmax)
    ax.axvline(0, color="0.45", linestyle="--", linewidth=1.0, zorder=1)

    ax.set_title(title, loc="left", pad=8, fontweight="bold")
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels if show_left_labels else [])

    ax.set_ylim(-0.6, 13.1)
    ax.grid(True, axis="x")

    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", pad=2)

    if metric_name == "Exposure":
        ax.set_xlabel("Composite difference in MHW exposure\n(positive phase − negative phase)", labelpad=5)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    else:
        ax.set_xlabel("Composite difference in MHW intensity (°C)\n(positive phase − negative phase)", labelpad=5)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(6))

def draw_fig7():
    fig = plt.figure(figsize=(13.6, 11.1), facecolor="white")
    gs = gridspec.GridSpec(
        nrows=2, ncols=2,
        hspace=0.28, wspace=0.18
    )

    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ])

    for idx, (ax, (title, index_name, metric_name)) in enumerate(zip(axes.flat, plot_specs)):
        sub = phase_df[(phase_df["Index"] == index_name) & (phase_df["Metric"] == metric_name)].copy()
        show_left = idx in [0, 2]
        draw_panel_composite(ax, sub, title, metric_name, show_left_labels=show_left)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", color="none",
                   markerfacecolor=subbasin_colors["Northern"], markeredgecolor=subbasin_colors["Northern"],
                   markersize=7, label="Northern"),
        plt.Line2D([0], [0], marker="o", linestyle="None", color="none",
                   markerfacecolor=subbasin_colors["Central"], markeredgecolor=subbasin_colors["Central"],
                   markersize=7, label="Central"),
        plt.Line2D([0], [0], marker="o", linestyle="None", color="none",
                   markerfacecolor=subbasin_colors["Southern"], markeredgecolor=subbasin_colors["Southern"],
                   markersize=7, label="Southern"),
        plt.Line2D([0], [0], marker="o", linestyle="None", color="none",
                   markerfacecolor="white", markeredgecolor="black",
                   markersize=7, label="p < 0.05")
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
        handletextpad=0.6,
        columnspacing=1.8,
        borderaxespad=0.0
    )

    fig.suptitle(
        "Figure 7 | Climate-phase composite differences in Bay of Bengal marine heatwave activity",
        fontsize=19, fontweight="bold", y=0.975
    )

    fig.text(
        0.5, 0.04,
        "Points show composite mean differences; horizontal bars indicate 95% bootstrap confidence intervals. "
        "Black-edged points denote Welch p < 0.05.",
        ha="center", va="center", fontsize=10.5
    )

    fig.subplots_adjust(left=0.18, right=0.985, top=0.875, bottom=0.105)
    save_figure(fig, OUT_FIG7_PNG, OUT_FIG7_PDF)
    plt.show()
    plt.close(fig)

draw_fig7()

# -------------------------
# 13) Console summary
# -------------------------
print("\n================ OUTPUTS GENERATED ================\n")
print("Figure 6:")
print(OUT_FIG6_PNG)
print(OUT_FIG6_PDF)

print("\nFigure 7:")
print(OUT_FIG7_PNG)
print(OUT_FIG7_PDF)

print("\nStatistics tables:")
print(OUT_LAG_STATS_CSV)
print(OUT_PHASE_STATS_CSV)

print("\nTop nominally significant lag-correlation results (p < 0.05):")
print(
    lag_df.loc[lag_df["p_eff"] < 0.05, ["Sub_basin","Season","Index","Metric","Lag","r","p_eff","q_fdr"]]
    .sort_values(["Index","Metric","p_eff"])
    .head(20)
    .to_string(index=False)
)

print("\nTop phase-composite results (p < 0.05):")
print(
    phase_df.loc[phase_df["p_value"] < 0.05, ["Sub_basin","Season","Index","Metric","mean_difference","ci_low_95","ci_high_95","p_value"]]
    .sort_values(["Index","Metric","p_value"])
    .to_string(index=False)
)
