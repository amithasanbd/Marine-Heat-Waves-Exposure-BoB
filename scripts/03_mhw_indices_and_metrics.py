#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03 Mhw Indices And Metrics

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
# # **Library Installation**


# %% Cell 2
# Colab shell command removed for repository reproducibility: !pip install netCDF4 # Install the netCDF4 library
# Colab shell command removed for repository reproducibility: !pip install cartopy
# Colab shell command removed for repository reproducibility: !pip install xarray matplotlib geopandas shapely
# Colab shell command removed for repository reproducibility: !pip install cftime


# %% Cell 3
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
import scipy as sp
import datetime
from datetime import date
import time
import pandas as pd
import xarray as xr
import netCDF4 # Now this import should work!
from netCDF4 import Dataset
import sys
import cartopy.crs as ccrs # Now this import should work!
import scipy.ndimage as ndimage
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.cbook as cbook
import scipy.ndimage as ndimage

# Install 'Times New Roman' font (or a suitable serif alternative like TeX Gyre Termes)
# Colab shell command removed for repository reproducibility: !apt-get update
# Colab shell command removed for repository reproducibility: !apt-get install -y texlive-fonts-recommended fonts-dejavu-core

# Clear matplotlib's font cache and rebuild
import matplotlib.font_manager as fm
fm._get_fontconfig_fonts.cache_clear()
fm.fontManager.findfont('Times New Roman', rebuild_if_missing=True)
print("Matplotlib font cache cleared and rebuilt. Please restart the runtime to apply changes.")


# %% [markdown]
# # **System Integration**


# %% Cell 5
# Google Colab Drive import removed; use config/paths_template.yaml and local data folders.

# Mount Google Drive
if not os.path.exists(str(OUTPUT_ROOT / "drive")):
    print("Mounting Google Drive...")
    # NOTE: The user will execute this cell, triggering the interactive mount.
# Google Drive mount removed; configure DATA_ROOT in this script or via environment variable MHW_DATA_ROOT.


# %% Cell 6
# Google Colab Drive import removed; use config/paths_template.yaml and local data folders.

# ==========================================
# 1. SYSTEM CONFIGURATION & MOUNTING
# ==========================================

# Configuration Dictionary
CONFIG = {
    "clim_path": str(DATA_ROOT / "MHWs/Copy of Sst.day.mean.ltm.1991-2020.nc"),
    "anom_path": str(DATA_ROOT / "MHWs/sst.day.anom"),
    "save_dir": str(DATA_ROOT / "MHWs"),
    "regions": { # Added 'regions' key
        "Bay of Bengal": {
            "lat_bounds": slice(23, 5),  # Note: Check NetCDF orientation. Usually N->S or S->N.
            "lon_bounds": slice(78, 100)
        }
    },
    "smooth_window": 15,         # Days for rolling average smoothing (increased from 5 to 15 for better visuals)
    "dpi": 600,
    "font": "Times New Roman"
}

# Ensure save directory exists
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# Set Global Plotting Styles
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [CONFIG['font']],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'grid.alpha': 0.3,
    'mathtext.fontset': 'stix'
})


# %% [markdown]
# # **MHWs Index**


# %% [markdown]
# 
# 
# ---
# 


# %% Cell 9
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# Google Colab Drive import removed; use config/paths_template.yaml and local data folders.
import cftime
import datetime as dt

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
CONFIG = {
    # Paths (update to your exact file locations)
    "clim_path": str(DATA_ROOT / "MHWs/Copy of Sst.day.mean.ltm.1991-2020.nc"),
    "anom_path": str(DATA_ROOT / "MHWs/sst.day.anom"),
    "save_dir": str(DATA_ROOT / "MHWs"),
    "regions": {
        "Bay of Bengal": { "lat_bounds": slice(23, 5), "lon_bounds": slice(78, 100) },
        "Arabian Sea":   { "lat_bounds": slice(25, 5), "lon_bounds": slice(50, 75) }
    },
    "mhw_min_duration": 5,   # Minimum days to qualify as an MHW event
    "dpi": 600,
    "font": "Times New Roman",
    # visual tunables
    "event_vline_color": "#2b6cb0",
    "event_vline_width": 1.25,
    "event_span_alpha": 0.08,
    "label_x_fraction": 0.15,  # fraction of x-axis from left where category labels appear
    "label_rotation": 32,      # degrees for slanted category labels
    "label_fontsize": 11
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [CONFIG["font"]],
    "axes.labelsize": 12,
    "axes.titlesize": 15,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 12,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.shadow": True,
    "font.weight": "bold",
    "grid.alpha": 0.3,
    "mathtext.fontset": "stix"
})

# =============================================================================
# 2. HELPERS
# =============================================================================
def _handle_cftime_conversion(time_values):
    """
    Convert cftime/datetime-like objects to Matplotlib numeric dates.
    """
    time_values = np.asarray(time_values)
    if time_values.size == 0:
        return np.array([])

    first = time_values[0]

    # cftime-like objects
    if isinstance(first, (
        cftime.DatetimeNoLeap,
        cftime.Datetime360Day,
        cftime.DatetimeGregorian,
        cftime.DatetimeProlepticGregorian
    )) or (hasattr(first, "year") and hasattr(first, "month") and hasattr(first, "day")
           and not isinstance(first, (np.datetime64,))):
        py_dates = [dt.datetime(int(t.year), int(t.month), int(t.day)) for t in time_values]
        return mdates.date2num(py_dates)

    # numpy datetime64 or pandas timestamps
    py_dates = pd.to_datetime(time_values).to_pydatetime()
    return mdates.date2num(py_dates)


def compute_dayofyear(time_coord):
    """Return day-of-year integers for cftime/pandas/numpy time arrays."""
    vals = time_coord.values
    doy = []
    for t in vals:
        if isinstance(t, (
            cftime.DatetimeNoLeap,
            cftime.Datetime360Day,
            cftime.DatetimeGregorian,
            cftime.DatetimeProlepticGregorian
        )) or hasattr(t, "timetuple"):
            doy.append(t.timetuple().tm_yday)
        else:
            doy.append(pd.to_datetime(t).dayofyear)
    return np.array(doy, dtype=int)


def _to_ordinal_any(t):
    """Convert various time types to ordinal for safe comparisons."""
    if isinstance(t, (
        cftime.DatetimeNoLeap,
        cftime.Datetime360Day,
        cftime.DatetimeGregorian,
        cftime.DatetimeProlepticGregorian
    )):
        return t.toordinal()
    if hasattr(t, "toordinal"):
        return t.toordinal()
    return pd.to_datetime(t).to_ordinal()


def is_climatology_like(time_coord):
    """Detect single-year climatology-style time axis (e.g., year 0001)"""
    vals = time_coord.values
    if vals.size == 0:
        return False
    years = []
    for t in vals:
        if isinstance(t, (
            cftime.DatetimeNoLeap,
            cftime.Datetime360Day,
            cftime.DatetimeGregorian,
            cftime.DatetimeProlepticGregorian
        )) or hasattr(t, "year"):
            years.append(int(t.year))
        else:
            years.append(pd.to_datetime(t).year)
    unique = sorted(set(years))
    if len(unique) == 1 and unique[0] <= 10 and 100 <= len(vals) <= 400:
        return True
    return False


def build_repeated_from_climatology(anom_subset, start_date_str, end_date_str):
    """
    Repeat a daily climatology/anomaly (single early year) over a requested period
    by selecting by day-of-year.
    """
    src_time = anom_subset.time
    src_doy = compute_dayofyear(src_time)
    anom_doy = anom_subset.assign_coords(doy=("time", src_doy)).swap_dims({"time": "doy"})
    anom_doy = anom_doy.drop_vars("time", errors="ignore")

    req_start = pd.to_datetime(start_date_str)
    req_end   = pd.to_datetime(end_date_str)
    new_dates = pd.date_range(req_start, req_end, freq="D")
    new_doy   = new_dates.dayofyear.values

    repeated = anom_doy.sel(doy=new_doy)
    repeated = repeated.rename({"doy": "time"})
    repeated = repeated.assign_coords(time=new_dates.to_pydatetime())
    return repeated


def detect_mhw_events(time_array, sst_values, thresh_values, min_duration=5):
    """
    Detect contiguous MHW events where sst_values >= thresh_values for at least
    min_duration days. Returns list of (start_num, end_num) pairs (matplotlib date numbers).
    """
    sst_vals = np.asarray(sst_values).astype(float)
    thr_vals = np.asarray(thresh_values).astype(float)
    mask = np.isfinite(sst_vals) & np.isfinite(thr_vals) & (sst_vals >= thr_vals)
    if mask.sum() == 0:
        return []

    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]

    dates_num = _handle_cftime_conversion(time_array)
    events = []
    for s, e in zip(starts, ends):
        duration = e - s
        if duration >= min_duration:
            events.append((dates_num[s], dates_num[e-1]))
    return events


def add_category_labels(ax, dates_num, label_x_fraction,
                        thresh, cat2, cat3, cat4,
                        rotation=32, fontsize=11):
    """
    Place slanted labels along each category threshold line.
    - dates_num : numeric x-axis dates (matplotlib date numbers)
    - label_x_fraction : fraction from left to place label (0..1)
    - thresh, cat2, cat3, cat4 : xarray DataArray or arrays aligned to same time axis
    """
    n = len(dates_num)
    if n == 0:
        return

    idx = max(0, int(np.round(n * float(label_x_fraction))))  # safe int index
    # retrieve y-values at idx (fall back to mean if missing)
    def y_at(da):
        try:
            return float(np.asarray(da.values)[idx])
        except Exception:
            # fallback
            arr = np.asarray(da.values).astype(float)
            arr = arr[~np.isnan(arr)]
            return float(np.mean(arr)) if arr.size > 0 else np.nan

    x_pos = dates_num[idx]

    # Style and text for each category line
    label_specs = [
        (thresh,  "category-I",   (0.0, 0.10)),  # small offset in axes fraction for visibility
        (cat2,    "category-II",  (0.0, 0.10)),
        (cat3,    "category-III", (0.0, 0.10)),
        (cat4,    "category-IV",  (0.0, 0.10))
    ]

    # Use a consistent green shade to match threshold lines
    text_color = "#2ca02c"
    for da, txt, offset_axes in label_specs:
        y = y_at(da)
        if np.isnan(y):
            continue
        # convert axes offset (fraction) to data coordinates for y offset
        # compute small vertical shift in data units: fraction * y-range
        ylim = ax.get_ylim()
        yrange = ylim[1] - ylim[0]
        y_off = offset_axes[1] * yrange
        # Place text on the line and rotate to match example
        ax.text(x_pos, y + y_off, txt,
                fontsize=fontsize, fontstyle='italic', rotation=rotation,
                color=text_color, alpha=0.9,
                ha='left', va='center', zorder=11,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.0, pad=0.5))


# =============================================================================
# 3. USER I/O
# =============================================================================
def get_user_inputs(config):
    print("\n--- MHW Analysis Setup ---")
    s_date = input("Enter Start Date (YYYY-MM-DD, e.g., 2020-01-01): ").strip()
    e_date = input("Enter End Date   (YYYY-MM-DD, e.g., 2020-12-31): ").strip()
    pd.to_datetime(s_date); pd.to_datetime(e_date)

    print("\n--- Data Source ---")
    anom_path = input(f"Enter Observed SST File Path (or press Enter for default):\n").strip(" '\"")
    if not anom_path:
        anom_path = config["anom_path"]
    return s_date, e_date, anom_path


def select_region(config):
    print("\n--- Select Region ---")
    regions = list(config["regions"].keys())
    for i, r in enumerate(regions, 1):
        print(f"{i}. {r}")
    while True:
        try:
            choice = int(input("Region Number: "))
            if 1 <= choice <= len(regions):
                name = regions[choice - 1]
                return name, config["regions"][name]["lat_bounds"], config["regions"][name]["lon_bounds"]
        except Exception:
            pass
        print("Invalid selection.")


# =============================================================================
# 4. CORE PROCESSING & PLOTTING
# =============================================================================
def process_mhw_data(region_name, lat_bounds, lon_bounds, start_date, end_date, anom_path):
    print(f"\nProcessing {region_name} | {start_date} to {end_date}...")

    # Load datasets
    try:
        coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds_clim = xr.open_dataset(CONFIG["clim_path"], decode_times=coder)
        ds_anom = xr.open_dataset(anom_path, decode_times=coder)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Try find sensible variable names
    clim_vars = list(ds_clim.data_vars)
    anom_vars = list(ds_anom.data_vars)
    clim_var = next((v for v in clim_vars if 'sst' in v or 'temp' in v), clim_vars[0])
    anom_var = next((v for v in anom_vars if 'anom' in v or 'sst' in v), anom_vars[0])

    # Spatial subsetting (robust)
    lat_min = min(lat_bounds.start, lat_bounds.stop)
    lat_max = max(lat_bounds.start, lat_bounds.stop)
    lon_min = min(lon_bounds.start, lon_bounds.stop)
    lon_max = max(lon_bounds.start, lon_bounds.stop)

    clim_sub_pre = ds_clim[clim_var].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    anom_sub_pre = ds_anom[anom_var].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    if clim_sub_pre.size == 0 or anom_sub_pre.size == 0:
        print("FATAL: Spatial subset returned empty data. Check coordinates and slices.")
        return

    clim_sub = clim_sub_pre.mean(dim=["lat", "lon"])
    anom_sub = anom_sub_pre.mean(dim=["lat", "lon"])

    # Handle climatology-like anomaly inputs
    if is_climatology_like(anom_sub.time):
        print("Observed/anomaly file looks like a climatology (single-year). Repeating by day-of-year.")
        anom_time_slice = build_repeated_from_climatology(anom_sub, start_date, end_date)
    else:
        anom_time_slice = anom_sub.sel(time=slice(start_date, end_date))
        if anom_time_slice.size == 0:
            t_min = anom_sub.time.min().item()
            t_max = anom_sub.time.max().item()
            print("FATAL: Temporal slice returned empty anomaly data.")
            print(f"Available: {t_min} to {t_max}")
            return

    # Reconstruct absolute SST
    doy_clim = compute_dayofyear(clim_sub.time)
    clim_doy = clim_sub.assign_coords(doy=("time", doy_clim)).swap_dims({"time": "doy"})
    clim_doy = clim_doy.drop_vars("time", errors="ignore")

    doy_obs = compute_dayofyear(anom_time_slice.time)
    clim_aligned_vals = clim_doy.reindex(doy=doy_obs, method='nearest')
    clim_aligned = xr.DataArray(clim_aligned_vals.values, coords={"time": anom_time_slice.time}, dims="time")

    sst_obs = clim_aligned + anom_time_slice

    # Thresholds & categories
    anom_90th = anom_time_slice.quantile(0.9)
    thresh_curve = clim_aligned + anom_90th
    delta = thresh_curve - clim_aligned

    cat1 = thresh_curve
    cat2 = thresh_curve + 1.0* delta
    cat3 = thresh_curve + 2.0 * delta
    cat4 = thresh_curve + 3.0 * delta

    # Detect events
    events = detect_mhw_events(anom_time_slice.time.values, sst_obs.values, thresh_curve.values,
                               min_duration=CONFIG["mhw_min_duration"])
    print(f"Detected {len(events)} MHW event(s) (min duration {CONFIG['mhw_min_duration']} days).")

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8), dpi=CONFIG["dpi"])
    dates_num = _handle_cftime_conversion(sst_obs.time.values)

    # Core lines
    ax.plot(dates_num, sst_obs, color="k", linewidth=2.2, label="SST (Observed)", zorder=6)
    ax.plot(dates_num, clim_aligned, color="#1f77b4", linewidth=1.8, linestyle="--",
            label="Seasonal Climatology", alpha=0.9, zorder=3)
    ax.plot(dates_num, thresh_curve, color="#2ca02c", linewidth=1.8, label="category-I threshold", zorder=4)

    # Faint category lines
    ax.plot(dates_num, cat2, color='#2ca02c', linewidth=1, linestyle='--', alpha=0.5, label='category-II')
    ax.plot(dates_num, cat3, color='#2ca02c', linewidth=1, linestyle='-.', alpha=0.4, label='category-III')
    ax.plot(dates_num, cat4, color='#2ca02c', linewidth=1, linestyle=':', alpha=0.3, label='category-IV')

    # Shading categories (layered so higher categories overpaint lower)
    ax.fill_between(dates_num, thresh_curve, sst_obs,
                    where=(sst_obs >= cat1),
                    interpolate=True, color='#FFDD99', alpha=0.95, zorder=2)   # light gold for Cat I

    ax.fill_between(dates_num, cat2, sst_obs,
                    where=(sst_obs >= cat2),
                    interpolate=True, color='#FFB266', alpha=0.95, zorder=3)   # orange for Cat II

    ax.fill_between(dates_num, cat3, sst_obs,
                    where=(sst_obs >= cat3),
                    interpolate=True, color='#FF6666', alpha=0.95, zorder=4)   # red for Cat III

    ax.fill_between(dates_num, cat4, sst_obs,
                    where=(sst_obs >= cat4),
                    interpolate=True, color='#8B0000', alpha=0.95, zorder=5)   # dark red for Cat IV

    # Event vertical lines and spans
    for i, (start_x, end_x) in enumerate(events):
        ax.axvline(x=start_x, color=CONFIG["event_vline_color"], linewidth=CONFIG["event_vline_width"], alpha=0.9, zorder=7)
        ax.axvline(x=end_x,   color=CONFIG["event_vline_color"], linewidth=CONFIG["event_vline_width"], alpha=0.9, zorder=7)
        ax.axvspan(start_x, end_x, color=CONFIG["event_vline_color"], alpha=CONFIG["event_span_alpha"], zorder=1)
        # optional label above span (kept minimal to avoid clutter)
        if i < 20:
            mid = (start_x + end_x) / 2.0
            ylim = ax.get_ylim()
            y_text = ylim[1] - 0.06 * (ylim[1] - ylim[0])
            ax.text(mid, y_text, f"Event {i+1}", ha='center', va='top', fontsize=9, zorder=10,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # Add slanted category labels
    add_category_labels(ax, dates_num, CONFIG["label_x_fraction"],
                        cat1, cat2, cat3, cat4,
                        rotation=CONFIG["label_rotation"],
                        fontsize=CONFIG["label_fontsize"])

    # Formatting
    ax.set_ylabel("SST [°C]", fontweight='bold')
    ax.set_title(f"Marine Heatwave Events & Thresholds — {region_name}\nPeriod: {start_date} to {end_date}",
                 fontweight='bold', pad=12)

    # Limits
    try:
        y_min = float(np.nanmin(sst_obs.values)) - 0.6
        y_max = float(np.nanmax(cat4.values)) + 0.6
        ax.set_ylim(y_min, y_max)
    except Exception:
        pass

    ax.set_xlim(dates_num[0], dates_num[-1])

    # X axis formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    ax.grid(True, linestyle='-', alpha=0.25, color='gray')

    # Legend: show concise descriptors
    legend_elements = [
        Line2D([0], [0], color="k", lw=2, label="SST (Observed)"),
        Line2D([0], [0], color="#1f77b4", lw=1.8, ls='--', label="Climatology"),
        Line2D([0], [0], color="#2ca02c", lw=1.8, label="Threshold"),
        Line2D([0], [0], color=CONFIG["event_vline_color"], lw=1.5, label="Event Durations"),
        Patch(facecolor='#FFDD99', label='Cat I (Moderate)'),
        Patch(facecolor='#FFB266', label='Cat II (Strong)'),
        Patch(facecolor='#FF6666', label='Cat III (Severe)'),
        Patch(facecolor='#8B0000', label='Cat IV (Extreme)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.16),
              ncol=4, frameon=True, fancybox=True, shadow=True, title="Legend")

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save figure
    fname = f"MHW_{region_name.replace(' ', '_')}_CategoryLabels_{start_date}_to_{end_date}.png"
    save_path = os.path.join(CONFIG["save_dir"], fname)
    plt.savefig(save_path, bbox_inches='tight', dpi=CONFIG["dpi"])
    print(f"\nSaved figure to: {save_path}")
    plt.show()


# =============================================================================
# 5. DRIVER
# =============================================================================
if __name__ == "__main__":
    # Mount Drive for Colab if needed
    if not os.path.exists(str(OUTPUT_ROOT / "drive")):
        try:
            print("Mounting Google Drive...")
# Google Drive mount removed; configure DATA_ROOT in this script or via environment variable MHW_DATA_ROOT.
        except Exception:
            print("Could not mount Drive; running locally may be fine.")

    region_name, lat_bounds, lon_bounds = select_region(CONFIG)
    start_date, end_date, anom_path = get_user_inputs(CONFIG)
    process_mhw_data(region_name, lat_bounds, lon_bounds, start_date, end_date, anom_path)


# %% [markdown]



# %% [markdown]
# ## Calculate MHW Metrics and Identify Events
# 
# ### Subtask:
# Re-execute the code to prepare the data, identify individual Marine Heatwave events, and calculate their duration, peak intensity, mean intensity, and cumulative intensity. This will ensure the `mhw_summary_df` is correctly populated.


# %% [markdown]
# **Reasoning**:
# The previous code block for calculating MHW metrics failed to retain the `mhw_summary_df` in the runtime for displaying, leading to a 'Runtime no longer has a reference' error. To ensure the `mhw_summary_df` is correctly populated and accessible for subsequent steps as requested, I will re-execute the full code block that performs data preparation, MHW event identification, and metric calculation.
# 


# %% Cell 13
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# Google Colab Drive import removed; use config/paths_template.yaml and local data folders.
import cftime
import datetime as dt

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
CONFIG = {
    # Paths (update to your exact file locations)
    "clim_path": str(DATA_ROOT / "MHWs/Copy of Sst.day.mean.ltm.1991-2020.nc"),
    "anom_path": str(DATA_ROOT / "MHWs/sst.day.anom"),
    "save_dir": str(DATA_ROOT / "MHWs"),
    "regions": {
        "Bay of Bengal": { "lat_bounds": slice(23, 5), "lon_bounds": slice(78, 100) },
        "Arabian Sea":   { "lat_bounds": slice(25, 5), "lon_bounds": slice(50, 75) }
    },
    "mhw_min_duration": 5,   # Minimum days to qualify as an MHW event
    "smooth_window": 15,         # Days for rolling average smoothing (increased from 5 to 15 for better visuals)
    "dpi": 600,
    "font": "Times New Roman",
    # visual tunables
    "event_vline_color": "#2b6cb0",
    "event_vline_width": 1.25,
    "event_span_alpha": 0.08,
    "label_x_fraction": 0.15,  # fraction of x-axis from left where category labels appear
    "label_rotation": 32,      # degrees for slanted category labels
    "label_fontsize": 11
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [CONFIG["font"]],
    "axes.labelsize": 12,
    "axes.titlesize": 15,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 12,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.shadow": True,
    "font.weight": "bold",
    "grid.alpha": 0.3,
    "mathtext.fontset": "stix"
})

# =============================================================================
# 2. HELPERS
# =============================================================================
def _handle_cftime_conversion(time_values):
    """
    Convert cftime/datetime-like objects to Matplotlib numeric dates.
    """
    time_values = np.asarray(time_values)
    if time_values.size == 0:
        return np.array([])

    first = time_values[0]

    # cftime-like objects
    if isinstance(first, (
        cftime.DatetimeNoLeap,
        cftime.Datetime360Day,
        cftime.DatetimeGregorian,
        cftime.DatetimeProlepticGregorian
    )) or (hasattr(first, "year") and hasattr(first, "month") and hasattr(first, "day")
           and not isinstance(first, (np.datetime64,))):
        py_dates = [dt.datetime(int(t.year), int(t.month), int(t.day)) for t in time_values]
        return mdates.date2num(py_dates)

    # numpy datetime64 or pandas timestamps
    py_dates = pd.to_datetime(time_values).to_pydatetime()
    return mdates.date2num(py_dates)


def compute_dayofyear(time_coord):
    """Return day-of-year integers for cftime/pandas/numpy time arrays."""
    vals = time_coord.values
    doy = []
    for t in vals:
        if isinstance(t, (
            cftime.DatetimeNoLeap,
            cftime.Datetime360Day,
            cftime.DatetimeGregorian,
            cftime.DatetimeProlepticGregorian
        )) or hasattr(t, "timetuple"):
            doy.append(t.timetuple().tm_yday)
        else:
            doy.append(pd.to_datetime(t).dayofyear)
    return np.array(doy, dtype=int)


def _to_ordinal_any(t):
    """
    Convert various time types to ordinal for safe comparisons.
    """
    if isinstance(t, (
        cftime.DatetimeNoLeap,
        cftime.Datetime360Day,
        cftime.DatetimeGregorian,
        cftime.DatetimeProlepticGregorian
    )):
        return t.toordinal()
    if hasattr(t, "toordinal"):
        return t.toordinal()
    return pd.to_datetime(t).to_ordinal()


def is_climatology_like(time_coord):
    """
    Detect single-year climatology-style time axis (e.g., year 0001)
    """
    vals = time_coord.values
    if vals.size == 0:
        return False
    years = []
    for t in vals:
        if isinstance(t, (
            cftime.DatetimeNoLeap,
            cftime.Datetime360Day,
            cftime.DatetimeGregorian,
            cftime.DatetimeProlepticGregorian
        )) or hasattr(t, "year"):
            years.append(int(t.year))
        else:
            years.append(pd.to_datetime(t).year)
    unique = sorted(set(years))
    if len(unique) == 1 and unique[0] <= 10 and 100 <= len(vals) <= 400:
        return True
    return False


def build_repeated_from_climatology(anom_subset, start_date_str, end_date_str):
    """
    Repeat a daily climatology/anomaly (single early year) over a requested period
    by selecting by day-of-year.
    """
    src_time = anom_subset.time
    src_doy = compute_dayofyear(src_time)
    anom_doy = anom_subset.assign_coords(doy=("time", src_doy)).swap_dims({"time": "doy"})
    anom_doy = anom_doy.drop_vars("time", errors="ignore")

    req_start = pd.to_datetime(start_date_str)
    req_end   = pd.to_datetime(end_date_str)
    new_dates = pd.date_range(req_start, req_end, freq="D")
    new_doy   = new_dates.dayofyear.values

    repeated = anom_doy.sel(doy=new_doy)
    repeated = repeated.rename({"doy": "time"})
    repeated = repeated.assign_coords(time=new_dates.to_pydatetime())
    return repeated


def detect_mhw_events(time_array, sst_values, thresh_values, min_duration=5):
    """
    Detect contiguous MHW events where sst_values >= thresh_values for at least
    min_duration days. Returns list of (start_num, end_num) pairs (matplotlib date numbers).
    """
    sst_vals = np.asarray(sst_values).astype(float)
    thr_vals = np.asarray(thresh_values).astype(float)
    mask = np.isfinite(sst_vals) & np.isfinite(thr_vals) & (sst_vals >= thr_vals)
    if mask.sum() == 0:
        return []

    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]

    dates_num = _handle_cftime_conversion(time_array)
    events = []
    for s, e in zip(starts, ends):
        duration = e - s
        if duration >= min_duration:
            events.append((dates_num[s], dates_num[e-1]))
    return events


def add_category_labels(ax, dates_num, label_x_fraction,
                        thresh, cat2, cat3, cat4,
                        rotation=32, fontsize=11):
    """
    Place slanted labels along each category threshold line.
    - dates_num : numeric x-axis dates (matplotlib date numbers)
    - label_x_fraction : fraction from left to place label (0..1)
    - thresh, cat2, cat3, cat4 : xarray DataArray or arrays aligned to same time axis
    """
    n = len(dates_num)
    if n == 0:
        return

    idx = max(0, int(np.round(n * float(label_x_fraction))))  # safe int index
    # retrieve y-values at idx (fall back to mean if missing)
    def y_at(da):
        try:
            return float(np.asarray(da.values)[idx])
        except Exception:
            # fallback
            arr = np.asarray(da.values).astype(float)
            arr = arr[~np.isnan(arr)]
            return float(np.mean(arr)) if arr.size > 0 else np.nan

    x_pos = dates_num[idx]

    # Style and text for each category line
    label_specs = [
        (thresh,  "category-I",   (0.0, 0.10)),  # small offset in axes fraction for visibility
        (cat2,    "category-II",  (0.0, 0.10)),
        (cat3,    "category-III", (0.0, 0.10)),
        (cat4,    "category-IV",  (0.0, 0.10))
    ]

    # Use a consistent green shade to match threshold lines
    text_color = "#2ca02c"
    for da, txt, offset_axes in label_specs:
        y = y_at(da)
        if np.isnan(y):
            continue
        # convert axes offset (fraction) to data coordinates for y offset
        # compute small vertical shift in data units: fraction * y-range
        ylim = ax.get_ylim()
        yrange = ylim[1] - ylim[0]
        y_off = offset_axes[1] * yrange
        # Place text on the line and rotate to match example
        ax.text(x_pos, y + y_off, txt,
                fontsize=fontsize, fontstyle='italic', rotation=rotation,
                color=text_color, alpha=0.9,
                ha='left', va='center', zorder=11,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.0, pad=0.5))


# =============================================================================
# 3. USER I/O
# =============================================================================
def get_user_inputs(config):
    print("\n--- MHW Analysis Setup ---")
    s_date = input("Enter Start Date (YYYY-MM-DD, e.g., 2020-01-01): ").strip()
    e_date = input("Enter End Date   (YYYY-MM-DD, e.g., 2020-12-31): ").strip()
    pd.to_datetime(s_date); pd.to_datetime(e_date)

    print("\n--- Data Source ---")
    anom_path = input(f"Enter Observed SST File Path (or press Enter for default):\n").strip(" '\"")
    if not anom_path:
        anom_path = config["anom_path"]
    return s_date, e_date, anom_path


def select_region(config):
    print("\n--- Select Region ---")
    regions = list(config["regions"].keys())
    for i, r in enumerate(regions, 1):
        print(f"{i}. {r}")
    while True:
        try:
            choice = int(input("Region Number: "))
            if 1 <= choice <= len(regions):
                name = regions[choice - 1]
                return name, config["regions"][name]["lat_bounds"], config["regions"][name]["lon_bounds"]
        except Exception:
            pass
        print("Invalid selection.")


# =============================================================================
# 4. CORE PROCESSING & PLOTTING
# =============================================================================
def process_mhw_data(region_name, lat_bounds, lon_bounds, start_date, end_date, anom_path):
    print(f"\nProcessing {region_name} | {start_date} to {end_date}...")

    # Load datasets
    try:
        coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds_clim = xr.open_dataset(CONFIG["clim_path"], decode_times=coder)
        ds_anom = xr.open_dataset(anom_path, decode_times=coder)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None, None, None, None, None

    # Try find sensible variable names
    clim_vars = list(ds_clim.data_vars)
    anom_vars = list(ds_anom.data_vars)
    clim_var = next((v for v in clim_vars if 'sst' in v or 'temp' in v), clim_vars[0])
    anom_var = next((v for v in anom_vars if 'anom' in v or 'sst' in v), anom_vars[0])

    if not clim_var or not anom_var:
        print(f"Error: Could not identify variables in NetCDF. Found: Clim={list(ds_clim.keys())}, Anom={list(ds_anom.keys())}")
        return None, None, None, None, None, None

    # Spatial subsetting (robust)
    lat_min = min(lat_bounds.start, lat_bounds.stop)
    lat_max = max(lat_bounds.start, lat_bounds.stop)
    lon_min = min(lon_bounds.start, lon_bounds.stop)
    lon_max = max(lon_bounds.start, lon_bounds.stop)

    clim_sub_pre = ds_clim[clim_var].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    anom_sub_pre = ds_anom[anom_var].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    if clim_sub_pre.size == 0 or anom_sub_pre.size == 0:
        print("FATAL: Spatial subset returned empty data. Check coordinates and slices.")
        return None, None, None, None, None, None

    clim_sub = clim_sub_pre.mean(dim=["lat", "lon"])
    anom_sub = anom_sub_pre.mean(dim=["lat", "lon"])

    # Handle climatology-like anomaly inputs
    if is_climatology_like(anom_sub.time):
        print("Observed/anomaly file looks like a climatology (single-year). Repeating by day-of-year.")
        anom_time_slice = build_repeated_from_climatology(anom_sub, start_date, end_date)
    else:
        anom_time_slice = anom_sub.sel(time=slice(start_date, end_date))
        if anom_time_slice.size == 0:
            t_min = anom_sub.time.min().item()
            t_max = anom_sub.time.max().item()
            print("FATAL: Temporal slice returned empty anomaly data.")
            print(f"Available: {t_min} to {t_max}")
            return None, None, None, None, None, None

    # Reconstruct absolute SST
    doy_clim = compute_dayofyear(clim_sub.time)
    clim_doy = clim_sub.assign_coords(doy=("time", doy_clim)).swap_dims({"time": "doy"})
    clim_doy = clim_doy.drop_vars("time", errors="ignore")

    doy_obs = compute_dayofyear(anom_time_slice.time)
    clim_aligned_vals = clim_doy.reindex(doy=doy_obs, method='nearest')
    clim_aligned = xr.DataArray(clim_aligned_vals.values, coords={"time": anom_time_slice.time}, dims="time")

    sst_obs = clim_aligned + anom_time_slice

    # Thresholds & categories
    anom_90th = anom_time_slice.quantile(0.9)
    thresh_curve = clim_aligned + anom_90th
    delta = thresh_curve - clim_aligned

    cat1 = thresh_curve
    cat2 = thresh_curve + 1.0* delta
    cat3 = thresh_curve + 2.0 * delta
    cat4 = thresh_curve + 3.0 * delta

    # Detect events
    events = detect_mhw_events(anom_time_slice.time.values, sst_obs.values, thresh_curve.values,
                               min_duration=CONFIG["mhw_min_duration"])
    print(f"Detected {len(events)} MHW event(s) (min duration {CONFIG['mhw_min_duration']} days).")

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8), dpi=CONFIG["dpi"])
    dates_num = _handle_cftime_conversion(sst_obs.time.values)

    # Core lines
    ax.plot(dates_num, sst_obs, color="k", linewidth=2.2, label="SST (Observed)", zorder=6)
    ax.plot(dates_num, clim_aligned, color="#1f77b4", linewidth=1.8, linestyle="--",
            label="Seasonal Climatology", alpha=0.9, zorder=3)
    ax.plot(dates_num, thresh_curve, color="#2ca02c", linewidth=1.8, label="category-I threshold", zorder=4)

    # Faint category lines
    ax.plot(dates_num, cat2, color='#2ca02c', linewidth=1, linestyle='--', alpha=0.5, label='category-II')
    ax.plot(dates_num, cat3, color='#2ca02c', linewidth=1, linestyle='-.', alpha=0.4, label='category-III')
    ax.plot(dates_num, cat4, color='#2ca02c', linewidth=1, linestyle=':', alpha=0.3, label='category-IV')

    # Shading categories (layered so higher categories overpaint lower)
    ax.fill_between(dates_num, thresh_curve, sst_obs,
                    where=(sst_obs >= cat1),
                    interpolate=True, color='#FFDD99', alpha=0.95, zorder=2)   # light gold for Cat I

    ax.fill_between(dates_num, cat2, sst_obs,
                    where=(sst_obs >= cat2),
                    interpolate=True, color='#FFB266', alpha=0.95, zorder=3)   # orange for Cat II

    ax.fill_between(dates_num, cat3, sst_obs,
                    where=(sst_obs >= cat3),
                    interpolate=True, color='#FF6666', alpha=0.95, zorder=4)   # red for Cat III

    ax.fill_between(dates_num, cat4, sst_obs,
                    where=(sst_obs >= cat4),
                    interpolate=True, color='#8B0000', alpha=0.95, zorder=5)   # dark red for Cat IV

    # Event vertical lines and spans
    for i, (start_x, end_x) in enumerate(events):
        ax.axvline(x=start_x, color=CONFIG["event_vline_color"], linewidth=CONFIG["event_vline_width"], alpha=0.9, zorder=7)
        ax.axvline(x=end_x,   color=CONFIG["event_vline_color"], linewidth=CONFIG["event_vline_width"], alpha=0.9, zorder=7)
        ax.axvspan(start_x, end_x, color=CONFIG["event_vline_color"], alpha=CONFIG["event_span_alpha"], zorder=1)
        # optional label above span (kept minimal to avoid clutter)
        if i < 20:
            mid = (start_x + end_x) / 2.0
            ylim = ax.get_ylim()
            y_text = ylim[1] - 0.06 * (ylim[1] - ylim[0])
            ax.text(mid, y_text, f"Event {i+1}", ha='center', va='top', fontsize=9, zorder=10,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # Add slanted category labels
    add_category_labels(ax, dates_num, CONFIG["label_x_fraction"],
                        cat1, cat2, cat3, cat4,
                        rotation=CONFIG["label_rotation"],
                        fontsize=CONFIG["label_fontsize"])

    # Formatting
    ax.set_ylabel("SST [°C]", fontweight='bold')
    ax.set_title(f"Marine Heatwave Events & Thresholds — {region_name}\nPeriod: {start_date} to {end_date}",
                 fontweight='bold', pad=12)

    # Limits
    try:
        y_min = float(np.nanmin(sst_obs.values)) - 0.6
        y_max = float(np.nanmax(cat4.values)) + 0.6
        ax.set_ylim(y_min, y_max)
    except Exception:
        pass

    ax.set_xlim(dates_num[0], dates_num[-1])

    # X axis formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    ax.grid(True, linestyle='-', alpha=0.25, color='gray')

    # Legend: show concise descriptors
    legend_elements = [
        Line2D([0], [0], color="k", lw=2, label="SST (Observed)"),
        Line2D([0], [0], color="#1f77b4", lw=1.8, ls='--', label="Climatology"),
        Line2D([0], [0], color="#2ca02c", lw=1.8, label="Threshold"),
        Line2D([0], [0], color=CONFIG["event_vline_color"], lw=1.5, label="Event Durations"),
        Patch(facecolor='#FFDD99', label='Cat I (Moderate)'),
        Patch(facecolor='#FFB266', label='Cat II (Strong)'),
        Patch(facecolor='#FF6666', label='Cat III (Severe)'),
        Patch(facecolor='#8B0000', label='Cat IV (Extreme)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.16),
              ncol=4, frameon=True, fancybox=True, shadow=True, title="Legend")

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save figure
    fname = f"MHW_{region_name.replace(' ', '_')}_CategoryLabels_{start_date}_to_{end_date}.png"
    save_path = os.path.join(CONFIG["save_dir"], fname)
    plt.savefig(save_path, bbox_inches='tight', dpi=CONFIG["dpi"])
    print(f"\nSaved figure to: {save_path}")
    plt.show()

    return sst_obs, thresh_curve, clim_aligned, cat1, cat2, cat3, cat4 # Return these for metrics calculation


# =============================================================================
# 5. DRIVER
# =============================================================================
if __name__ == "__main__":
    # Mount Drive for Colab if needed
    if not os.path.exists(str(OUTPUT_ROOT / "drive")):
        try:
            print("Mounting Google Drive...")
# Google Drive mount removed; configure DATA_ROOT in this script or via environment variable MHW_DATA_ROOT.
        except Exception:
            print("Could not mount Drive; running locally may be fine.")

    region_name, lat_bounds, lon_bounds = select_region(CONFIG)
    start_date, end_date, anom_path = get_user_inputs(CONFIG)

    sst_obs, thresh_curve, clim_aligned, cat1, cat2, cat3, cat4 = process_mhw_data(region_name, lat_bounds, lon_bounds, start_date, end_date, anom_path)

    if sst_obs is not None and thresh_curve is not None:
        # 1. Convert xarray.DataArray objects into a pandas DataFrame
        df_mhw = pd.DataFrame({
            'SST': sst_obs.to_pandas(),
            'Threshold': thresh_curve.to_pandas() # Use thresh_curve (not smoothed) for event detection based on standard def
        })
        df_mhw.index.name = 'time'

        # 2. Create a new boolean column `is_mhw`
        df_mhw['is_mhw'] = df_mhw['SST'] >= df_mhw['Threshold']

        # 3. Identify individual Marine Heatwave events.
        # Calculate the cumulative sum of changes in the is_mhw status to group consecutive True values
        df_mhw['event_id'] = (df_mhw['is_mhw'] != df_mhw['is_mhw'].shift(1)).cumsum()

        # Filter out non-MHW periods and groups shorter than min_duration
        mhw_groups = df_mhw[df_mhw['is_mhw']].groupby('event_id')

        # 4. Initialize an empty list to store the details of each Marine Heatwave event.
        mhw_events_list = []

        # 5. Iterate through the identified Marine Heatwave events.
        for event_id, group in mhw_groups:
            duration = len(group)
            if duration >= CONFIG["mhw_min_duration"]:
                # a. Determine its start_date and end_date
                start_date_event = group.index.min()
                end_date_event = group.index.max()

                # c. Calculate the daily intensity
                daily_intensity = group['SST'] - group['Threshold']

                # d. Determine peak, mean, and cumulative intensity
                peak_intensity = daily_intensity.max()
                mean_intensity = daily_intensity.mean()
                cumulative_intensity = daily_intensity.sum()

                # e. Store all calculated metrics
                mhw_events_list.append({
                    'start_date': start_date_event,
                    'end_date': end_date_event,
                    'duration_days': duration,
                    'peak_intensity_c': peak_intensity,
                    'mean_intensity_c': mean_intensity,
                    'cumulative_intensity_c': cumulative_intensity
                })

        # 6. Convert the list of MHW event dictionaries into a pandas DataFrame
        mhw_summary_df = pd.DataFrame(mhw_events_list)

        print("\n--- MHW Event Summary ---")
        if not mhw_summary_df.empty:
            display(mhw_summary_df.head())
            # 7. Calculate the overall frequency of Marine Heatwaves
            mhw_frequency = len(mhw_summary_df)
            print(f"\nTotal number of MHW events (frequency): {mhw_frequency}")
        else:
            print("No Marine Heatwave events detected in the specified period.")

        # Display descriptive statistics for MHW metrics if events exist
        if not mhw_summary_df.empty:
            print("\nDescriptive statistics for MHW metrics:")
            display(mhw_summary_df.describe())


# %% Cell 14
def _prepare_mhw_data(start_date_str, end_date_str):
    """
    Loads and prepares SST and threshold data for MHW calculation.
    Returns sst_obs and thresh_curve (the MHW threshold).
    """
    # 2. Load Datasets
    try:
        time_decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds_clim = xr.open_dataset(CONFIG["clim_path"], decode_times=time_decoder)
        ds_anom = xr.open_dataset(CONFIG["anom_path"], decode_times=time_decoder)
    except FileNotFoundError as e:
        print(f"Error: File not found. Check paths.\n{e}")
        return None, None

    # 3. Identify Variables (Robust variable finding)
    clim_var_name = next((v for v in ['sst', 'temp', 'mean'] if v in ds_clim), None)
    anom_var_name = next((v for v in ['sst', 'anom', 'anomaly'] if v in ds_anom), None)

    if not clim_var_name or not anom_var_name:
        print(f"Error: Could not identify variables in NetCDF. Found: Clim={list(ds_clim.keys())}, Anom={list(ds_anom.keys())}")
        return None, None

    # 4. Spatial Averaging (Lat 5-23 N, Lon 78-100 E)
    lat_slice = slice(min(5, 23), max(5, 23))
    lon_slice = slice(78, 100)

    clim_subset = ds_clim[clim_var_name].sel(
        lat=lat_slice,
        lon=lon_slice
    ).mean(dim=['lat', 'lon'])

    anom_subset = ds_anom[anom_var_name].sel(
        lat=lat_slice,
        lon=lon_slice
    ).mean(dim=['lat', 'lon'])

    # 5. Time Filtering (Select the requested dates from the Anomaly/Target file)
    anom_time_slice = anom_subset.sel(time=slice(start_date_str, end_date_str))

    if len(anom_time_slice) == 0:
        print("Error: No data found for the selected time range. Check input dates or data coverage.")
        return None, None

    # 6. Reconstruct Absolute SST
    doy_values_clim = xr.CFTimeIndex(clim_subset.time.values).dayofyear
    clim_doy = clim_subset.assign_coords(doy=('time', doy_values_clim)).swap_dims({'time': 'doy'})
    clim_doy = clim_doy.drop_vars('time', errors='ignore')

    anom_doy_values = xr.CFTimeIndex(anom_time_slice.time.values).dayofyear
    clim_aligned_values = clim_doy.sel(doy=anom_doy_values, method='nearest')

    clim_aligned = clim_aligned_values.rename({'doy': 'time'})
    clim_aligned['time'] = anom_time_slice.time

    sst_obs = clim_aligned + anom_time_slice

    # 7. Define Thresholds
    anom_90th = anom_time_slice.quantile(0.9)
    thresh_curve = clim_aligned + anom_90th

    # 8. Smoothing (Rolling Mean)
    window = CONFIG['smooth_window']
    clim_smooth = clim_aligned.rolling(time=window, center=True).mean()
    thresh_smooth = thresh_curve.rolling(time=window, center=True).mean()

    return sst_obs, thresh_smooth

# Get user input dates from the previously executed cell
start_date_str = start_date
end_date_str = end_date

# Prepare data
sst_obs, thresh_smooth = _prepare_mhw_data(start_date_str, end_date_str)

if sst_obs is None or thresh_smooth is None:
    print("Failed to prepare MHW data. Exiting.")
else:
    # 1. Convert xarray.DataArray objects into a pandas DataFrame
    df_mhw = pd.DataFrame({
        'SST': sst_obs.to_pandas(),
        'Threshold': thresh_smooth.to_pandas()
    })
    df_mhw.index.name = 'time'

    # 2. Create a new boolean column `is_mhw`
    df_mhw['is_mhw'] = df_mhw['SST'] >= df_mhw['Threshold']

    # 3. Identify individual Marine Heatwave events
    # Calculate the cumulative sum of changes in the is_mhw status to group consecutive True values
    df_mhw['event_id'] = (df_mhw['is_mhw'] != df_mhw['is_mhw'].shift(1)).cumsum()

    # 4. Initialize an empty list to store the details of each Marine Heatwave event.
    mhw_events_list = []

    # 5. Iterate through the identified Marine Heatwave events.
    for event_id, group in df_mhw.groupby('event_id'):
        if group['is_mhw'].iloc[0]:  # Only process groups that are MHWs
            # a. Determine its start_date and end_date
            start_date = group.index.min()
            end_date = group.index.max()

            # b. Calculate its duration in days
            duration = (end_date - start_date).days + 1

            # c. Calculate the daily intensity
            daily_intensity = group['SST'] - group['Threshold']

            # d. Determine peak, mean, and cumulative intensity
            peak_intensity = daily_intensity.max()
            mean_intensity = daily_intensity.mean()
            cumulative_intensity = daily_intensity.sum()

            # e. Store all calculated metrics
            mhw_events_list.append({
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': duration,
                'peak_intensity_c': peak_intensity,
                'mean_intensity_c': mean_intensity,
                'cumulative_intensity_c': cumulative_intensity
            })

    # 6. Convert the list of MHW event dictionaries into a pandas DataFrame
    mhw_summary_df = pd.DataFrame(mhw_events_list)

    print("\n--- MHW Event Summary ---")
    if not mhw_summary_df.empty:
        display(mhw_summary_df.head())
        # 7. Calculate the overall frequency of Marine Heatwaves
        mhw_frequency = len(mhw_summary_df)
        print(f"\nTotal number of MHW events (frequency): {mhw_frequency}")
    else:
        print("No Marine Heatwave events detected in the specified period.")

    # Display descriptive statistics for MHW metrics if events exist
    if not mhw_summary_df.empty:
        print("\nDescriptive statistics for MHW metrics:")
        display(mhw_summary_df.describe())
