#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02 Mhw Detection Main

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
# # **MHWs Event Analysis**


# %% [markdown]
# **Installation and Import Libaries**


# %% Cell 3
# Colab shell command removed for repository reproducibility: !pip install netCDF4 # Install the netCDF4 library
# Colab shell command removed for repository reproducibility: !pip install cartopy
# Colab shell command removed for repository reproducibility: !pip install xarray matplotlib geopandas shapely
# Colab shell command removed for repository reproducibility: !pip install cftime


# %% Cell 4
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
import matplotlib.pyplot as plt
import scipy as sp
import datetime
from datetime import date
import time
import pandas as pd
import netCDF4 # Now this import should work!
from netCDF4 import Dataset
import sys
import cartopy.crs as ccrs # Now this import should work!
import scipy.ndimage as ndimage
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import scipy.ndimage as ndimage


# %% Cell 5
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# Google Colab Drive import removed; use config/paths_template.yaml and local data folders.

# ==========================================
# 1. SYSTEM CONFIGURATION & MOUNTING
# ==========================================

# Mount Google Drive
if not os.path.exists(str(OUTPUT_ROOT / "drive")):
    print("Mounting Google Drive...")
    # NOTE: The user will execute this cell, triggering the interactive mount.
# Google Drive mount removed; configure DATA_ROOT in this script or via environment variable MHW_DATA_ROOT.

# Configuration Dictionary
CONFIG = {
    "clim_path": str(DATA_ROOT / "(OCN-04) /OCN-04 Research /Research 2025/MHW & CYCLONE 2025/MHWs All Datasets 2025/WQ Datasets 2025/Sst.day.mean.ltm.1991-2020.nc"),
    "anom_path": str(DATA_ROOT / "(OCN-04) /OCN-04 Research /Research 2025/MHW & CYCLONE 2025/MHWs All Datasets 2025/WQ Datasets 2025/Sst.day.mean.ltm.1991-2020.nc"),
    "save_dir": str(DATA_ROOT / "(OCN-04) /OCN-04 Research /Research 2025/MHW & CYCLONE 2025/MHWs All Datasets 2025/WQ Datasets 2025"),
    "lat_bounds": slice(23, 5),  # Note: Check NetCDF orientation. Usually N->S or S->N.
    "lon_bounds": slice(78, 100),
    "smooth_window": 15,         # Days for rolling average smoothing (increased from 5 to 15 for better visuals)
    "dpi": 300,
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

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def add_label_follow_line(ax, x_da, y_da, text, frac=0.18, offset_points=10, **kwargs):
    """
    Adds a label that follows the slope of the line curve.
    """
    # Convert xarray to numpy
    y_vals = np.asarray(y_da.values)

    # Handle datetime x-axis for slope calculation
    if np.issubdtype(x_da.values.dtype, np.datetime64):
        # Convert to float days for calculation
        x_vals = mdates.date2num(x_da.values)
    else:
        # Convert cftime objects to numerical dates for plotting and slope calculation
        x_vals = mdates.date2num(x_da.values)

    n = len(y_vals)
    if n < 2: return

    # Index calculation
    i = int(frac * (n - 2))
    i = max(0, min(i, n - 2))

    # Calculate Slope
    dy = y_vals[i+1] - y_vals[i]
    dx = x_vals[i+1] - x_vals[i]

    # Heuristic adjustment for aspect ratio (often needed for time series)
    angle = 0.0 if dx == 0 else np.degrees(np.arctan2(dy, dx))

    # Use the original x_da.values[i] for annotation placement if it's already numerical or datetime
    # If it's a cftime object, mdates.date2num will be used for calculation, but for display
    # Matplotlib usually handles it if the axes are configured with a DateFormatter.
    # However, to be safe and consistent with slope calculation, we'll use numerical x for xy.
    ax.annotate(
        text,
        xy=(mdates.date2num(x_da.values[i]), y_vals[i]), # Convert cftime here too for consistency
        xytext=(0, offset_points),
        textcoords="offset points",
        ha="left",
        va="bottom",
        rotation=angle * 0.5, # Dampen the rotation slightly for readability
        rotation_mode="anchor",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5),
        **kwargs
    )

def get_user_input_dates():
    """
    Gets start and end dates from console in YYYY-MM-DD format.
    The X-axis will be formatted as Month-Year (%b-%Y).
    """
    print("\n--- MHW Plotting Configuration ---")
    print("X-Axis Labels will display in Month-Year format (e.g., Jan-2024).")
    try:
        s_date = input("Enter Plotting Start Date (Format YYYY-MM-DD, e.g., 2024-03-01): ")
        e_date = input("Enter Plotting End Date (Format YYYY-MM-DD, e.g., 2024-09-30): ")
        # Validate date strings (raises error on fail)
        pd.to_datetime(s_date)
        pd.to_datetime(e_date)
        return s_date, e_date
    except Exception as e:
        print(f"Invalid date input format or value: {e}. Defaulting to 2024-01-01 to 2024-12-31.")
        return "2024-01-01", "2024-12-31"

# ==========================================
# 3. CORE PROCESSING LOGIC
# ==========================================

def process_mhw_data():
    # 1. Get Time Range
    start_date_str, end_date_str = get_user_input_dates()

    print(f"Loading datasets and slicing data from {start_date_str} to {end_date_str}...")

    # 2. Load Datasets
    try:
        # Use CFDatetimeCoder as recommended for robust cftime handling
        time_decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds_clim = xr.open_dataset(CONFIG["clim_path"], decode_times=time_decoder)
        ds_anom = xr.open_dataset(CONFIG["anom_path"], decode_times=time_decoder)
    except FileNotFoundError as e:
        print(f"Error: File not found. Check paths.\n{e}")
        return

    # 3. Identify Variables (Robust variable finding)
    clim_var_name = next((v for v in ['sst', 'temp', 'mean'] if v in ds_clim), None)
    anom_var_name = next((v for v in ['sst', 'anom', 'anomaly'] if v in ds_anom), None)

    if not clim_var_name or not anom_var_name:
        print(f"Error: Could not identify variables in NetCDF. Found: Clim={list(ds_clim.keys())}, Anom={list(ds_anom.keys())}")
        return

    # 4. Spatial Averaging (Lat 5-23 N, Lon 78-100 E)
    print("Performing geospatial subsetting and averaging...")

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
        return

    # 6. Reconstruct Absolute SST - FIXING THE ATTRIBUTE ERROR HERE
    print("Reconstructing Absolute SST and calculating Thresholds...")

    # --- FIX START ---

    # Robust way to handle LTM alignment, avoiding xarray DataArray.dt attribute errors

    # 6a. Prepare Climatology for DoY indexing
    # Extract Day of Year for Climatology (LTM) by explicitly creating a CFTimeIndex
    doy_values_clim = xr.CFTimeIndex(clim_subset.time.values).dayofyear

    # Assign the DoY values as a new coordinate/dimension
    # Use tuple assignment ('time', values) to create a DataArray coordinate
    clim_doy = clim_subset.assign_coords(doy=('time', doy_values_clim)).swap_dims({'time': 'doy'})
    clim_doy = clim_doy.drop_vars('time', errors='ignore')

    # 6b. Align Climatology (indexed by DoY) with Anomaly's DoY values
    # Extract DoY values from the anomaly file by explicitly creating a CFTimeIndex
    anom_doy_values = xr.CFTimeIndex(anom_time_slice.time.values).dayofyear

    # Select the LTM value corresponding to each DoY in the anomaly data
    clim_aligned_values = clim_doy.sel(doy=anom_doy_values, method='nearest')

    # 6c. Assign the correct target time coordinate back
    clim_aligned = clim_aligned_values.rename({'doy': 'time'})
    clim_aligned['time'] = anom_time_slice.time

    # --- FIX END ---

    # Observed SST = Climatology + Anomaly
    sst_obs = clim_aligned + anom_time_slice

    # 7. Define Thresholds & Categories
    # Threshold Line = Climatology + (90th percentile of Anomaly)
    anom_90th = anom_time_slice.quantile(0.9)

    clim_curve = clim_aligned
    thresh_curve = clim_aligned + anom_90th # The Green Line (Threshold)

    # Calculate Delta (Difference between Threshold and Climatology)
    diff = thresh_curve - clim_curve

    # Define Categories based on User Formulas
    cat1 = thresh_curve
    cat2 = thresh_curve + diff      # Strong (Thresh + 2*Delta from Clim)
    cat3 = thresh_curve + 2 * diff  # Severe (Thresh + 3*Delta from Clim)
    cat4 = thresh_curve + 3 * diff  # Extreme (Thresh + 4*Delta from Clim)

    # 8. Smoothing (Rolling Mean)
    window = CONFIG['smooth_window']
    # sst_smooth = sst_obs.rolling(time=window, center=True).mean() # Removed smoothing for observed SST
    clim_smooth = clim_curve.rolling(time=window, center=True).mean()
    thresh_smooth = thresh_curve.rolling(time=window, center=True).mean()
    cat2_smooth = cat2.rolling(time=window, center=True).mean()
    cat3_smooth = cat3.rolling(time=window, center=True).mean()
    cat4_smooth = cat4.rolling(time=window, center=True).mean()

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    print("Generating Publication-Ready Plot...")

    fig, ax = plt.subplots(figsize=(16, 9), dpi=CONFIG['dpi'])

    # Convert cftime objects to Matplotlib's numerical dates for plotting
    dates_num = mdates.date2num(sst_obs.time.values) # Using sst_obs for dates

    # A. Plot Main Lines
    ax.plot(dates_num, sst_obs, color='k', linewidth=2.5, label='Observed SST', zorder=5) # Plotting original sst_obs
    ax.plot(dates_num, clim_smooth, color='#1f77b4', linewidth=2, linestyle='--', label='Climatology', alpha=0.8)
    ax.plot(dates_num, thresh_smooth, color='#2ca02c', linewidth=2, label='MHW Threshold', zorder=3)

    # B. Plot Category Lines (Faintly, for context)
    # Note: Cat1 is the Threshold line itself, so plotting Cat2, Cat3, Cat4 below.
    ax.plot(dates_num, cat2_smooth, color='#2ca02c', linewidth=1, linestyle='--', alpha=0.5)
    ax.plot(dates_num, cat3_smooth, color='#2ca02c', linewidth=1, linestyle='-.', alpha=0.5)
    ax.plot(dates_num, cat4_smooth, color='#2ca02c', linewidth=1, linestyle=':', alpha=0.5)

    # C. Fill MHW Events (The Red Fill)
    # Logic: Fill where SST > Threshold
    ax.fill_between(
        dates_num,
        thresh_smooth,
        sst_obs, # Using sst_obs for fill
        where=(sst_obs >= thresh_smooth), # Using sst_obs for condition
        interpolate=True,
        color='red',
        alpha=0.6,
        label='MHW Event'
    )

    # D. Add "Follow Line" Labels for Categories
    # We use the Cat1 line (Thresh_smooth) for Cat I label
    add_label_follow_line(ax, sst_obs.time, thresh_smooth, "Cat I (Moderate)", frac=0.15, offset_points=10, color='#2ca02c', fontsize=10, fontweight='bold') # Use sst_obs.time
    add_label_follow_line(ax, sst_obs.time, cat2_smooth, "Cat II (Strong)", frac=0.35, offset_points=10, color='#2ca02c', fontsize=10) # Use sst_obs.time
    add_label_follow_line(ax, sst_obs.time, cat3_smooth, "Cat III (Severe)", frac=0.55, offset_points=10, color='#2ca02c', fontsize=10) # Use sst_obs.time
    add_label_follow_line(ax, sst_obs.time, cat4_smooth, "Cat IV (Extreme)", frac=0.75, offset_points=10, color='#2ca02c', fontsize=10) # Use sst_obs.time

    # E. Formatting
    ax.set_ylabel('Sea Surface Temperature (°C)', fontsize=16, fontname=CONFIG['font'])
    ax.set_title(f'Marine Heatwave Events: Bay of Bengal ({start_date_str} to {end_date_str})\nArea Averaged: Lat 5-23°N, Lon 78-100°E',
                 fontsize=20, fontweight='bold', fontname=CONFIG['font'], pad=15)

    # Date Formatting: Enforcing Month-Year format (%b-%Y) as requested
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)) # Set to show every 2 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Dynamic Y-axis Limits
    y_max = float(max(sst_obs.max(), cat4_smooth.max())) + 0.5 # Using sst_obs.max()
    y_min = float(min(sst_obs.min(), clim_smooth.min())) - 0.5 # Using sst_obs.min()
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(dates_num[0], dates_num[-1])

    # Grid
    ax.grid(True, which='major', linestyle='-', alpha=0.2, color='gray')

    # F. Custom Legend (Best Position: Below the plot for clarity)
    legend_elements = [
        Line2D([0], [0], color='k', lw=2.5, label='Observed SST'), # Removed (Smoothed)
        Line2D([0], [0], color='#1f77b4', lw=2, linestyle='--', label='Climatology (1991-2020)'),
        Line2D([0], [0], color='#2ca02c', lw=2, label='MHW Threshold (90th %ile)'),
        Patch(facecolor='red', edgecolor='none', alpha=0.6, label='MHW Event'),
    ]

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.2), ncol=4,
              frameon=True, fancybox=True, shadow=True, fontsize=14, title="Legend")

    # Layout Adjustment
    plt.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust for legend below

    # Save
    save_path = os.path.join(CONFIG["save_dir"], f"MHW_BoB_Plot_unsmoothed_SST_{start_date_str}_to_{end_date_str}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Success! Figure saved to: {save_path}")
    plt.show()

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    process_mhw_data()


# %% [markdown]
# 
# 
# ---
# 


# %% [markdown]
# # **Font Update**


# %% Cell 8
# Install Liberation fonts, which include 'Liberation Serif' (a Times-like font)
# Colab shell command removed for repository reproducibility: !apt-get update -qq > /dev/null
# Colab shell command removed for repository reproducibility: !apt-get install -qq fonts-liberation > /dev/null

# Update matplotlib font cache
import matplotlib.font_manager as fm
fm.findSystemFonts()

print("Liberation fonts installed and Matplotlib font cache rebuilt.")


# %% [markdown]
# **MHW Line Plot**


# %% Cell 10
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Force a rebuild of the font cache before checking
fm.findSystemFonts()

# Verify if 'Times New Roman' or a suitable alternative is now available
font_found = False
chosen_font = "serif" # Default to generic serif

for font_path in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    try:
        fprop = fm.FontProperties(fname=font_path)
        font_name_lower = fprop.get_name().lower()
        if 'times new roman' in font_name_lower:
            print(f"Found Times New Roman: {fprop.get_name()} at {font_path}")
            chosen_font = fprop.get_name()
            font_found = True
            break
        elif 'liberation serif' in font_name_lower:
            print(f"Found Liberation Serif: {fprop.get_name()} at {font_path}")
            chosen_font = fprop.get_name()
            font_found = True
            # Don't break yet, in case Times New Roman is also found later
            # If Times New Roman is preferred, ensure it's prioritized.

    except RuntimeError:
        # Some font files might be corrupted or unreadable
        continue

if not font_found or chosen_font == "serif":
    print("Warning: 'Times New Roman' or 'Liberation Serif' not explicitly found. Matplotlib will fall back to a generic serif font.")
    CONFIG['font'] = 'serif'
else:
    CONFIG['font'] = chosen_font

# Set Global Plotting Styles with potentially updated font
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


# Re-run the main plotting function to apply the new font settings
process_mhw_data()


# %% [markdown]
# #**3D MHW Digital Twin**


# %% Cell 12
# -*- coding: utf-8 -*-
"""
Ocean Digital Twin: 3D MHW Reconstruction System (Satellite -> Subsurface)
Target: 3D Reconstruction of Temperature, Salinity, and MLD (0-500m)
Method: Random Forest Regression (Surface Features -> Vertical Profiles)

Author: Gemini AI (Based on User Architecture)
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import os # Import os for os.makedirs and os.path.join
# Google Colab Drive import removed; use config/paths_template.yaml and local data folders.
from datetime import datetime

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# Mount Google Drive - This is from GBHUeSXVMJp4, crucial for save_dir to work
if not os.path.exists(str(OUTPUT_ROOT / "drive")):
    print("Mounting Google Drive...")
# Google Drive mount removed; configure DATA_ROOT in this script or via environment variable MHW_DATA_ROOT.

# Explicitly define CONFIG here, including general parameters and save_dir
# This ensures CONFIG is always correctly initialized within this cell.
CONFIG = {
    "clim_path": str(DATA_ROOT / "(OCN-04) /OCN-04 Research /Research 2025/MHW & CYCLONE 2025/MHWs All Datasets 2025/WQ Datasets 2025/Sst.day.mean.ltm.1991-2020.nc"),
    "anom_path": str(DATA_ROOT / "(OCN-04) /OCN-04 Research /Research 2025/MHW & CYCLONE 2025/MHWs All Datasets 2025/WQ Datasets 2025/Sst.day.mean.ltm.1991-2020.nc"),
    "save_dir": str(DATA_ROOT / "(OCN-04) /OCN-04 Research /Research 2025/MHW & CYCLONE 2025/MHWs All Datasets 2025/WQ Datasets 2025"),
    "lat_bounds": slice(23, 5),  # Note: Check NetCDF orientation. Usually N->S or S->N.
    "lon_bounds": slice(78, 100),
    "smooth_window": 15,
    "dpi": 300,
    "font": "Times New Roman"
}

# Now, update it with digital twin specific parameters
CONFIG.update({
    "depth_levels": np.arange(0, 505, 10),  # 0 to 500m every 10m
    "features": ['sst', 'sss', 'sla', 'u_wind', 'v_wind', 'chl_a', 'lat', 'lon', 'doy'],
    "target_var": "temperature_profile",    # Predicting T(z)
    "n_estimators": 100,                    # RF Trees
    "random_state": 42
})

# Ensure save directory exists (re-checking in case CONFIG was reset)
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# Set Global Plotting Styles (re-setting in case CONFIG was reset)
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

# =============================================================================
# 2. DATA GENERATION (DIGITAL TWIN SIMULATOR)
# =============================================================================
def generate_mock_data(n_samples=2000):
    """
    Generates synthetic 'Ground Truth' data (like Argo floats matched with Satellite).
    Replace this function with real data loading from NetCDF/Argo CSVs.
    """
    print(f"--- Generating {n_samples} synthetic training profiles (Simulating Argo/Satellite match-ups) ---")

    np.random.seed(CONFIG['random_state'])

    # 1. Generate Surface Features (Satellite Proxies)
    data = {
        'sst': np.random.uniform(26, 31, n_samples),      # Deg C
        'sss': np.random.uniform(32, 36, n_samples),      # PSU
        'sla': np.random.uniform(-0.2, 0.2, n_samples),   # m
        'u_wind': np.random.normal(0, 5, n_samples),      # m/s
        'v_wind': np.random.normal(0, 5, n_samples),      # m/s
        'chl_a': np.random.exponential(0.5, n_samples),   # mg/m3
        'lat': np.random.uniform(5, 23, n_samples),       # BoB N
        'lon': np.random.uniform(78, 100, n_samples),     # BoB E
        'doy': np.random.randint(1, 366, n_samples)       # Day of Year
    }

    X = pd.DataFrame(data)

    # 2. Generate Vertical Profiles (Targets) based on physics-like heuristic
    # T(z) = SST * decay_function + thermocline_effect
    depths = CONFIG['depth_levels']
    Y = []

    for i in range(n_samples):
        sst = X['sst'].iloc[i]
        sla = X['sla'].iloc[i]

        # Physical proxy: SLA correlates with Thermocline Depth (positive SLA -> deeper thermocline)
        mld_proxy = 50 + (sla * 100)
        mld_proxy = np.clip(mld_proxy, 20, 100)

        profile = []
        for z in depths:
            if z <= mld_proxy:
                # Mixed Layer: Temp is roughly SST
                temp = sst - (z * 0.005)
            else:
                # Thermocline/Deep: Exponential decay
                decay = np.exp(-(z - mld_proxy) / 150)
                temp = 10 + (sst - 10) * decay

            # Add some noise
            temp += np.random.normal(0, 0.1)
            profile.append(temp)
        Y.append(profile)

    Y = np.array(Y)
    return X, Y, depths

# =============================================================================
# 3. MODEL TRAINING (Machine Learning Core)
# =============================================================================
class OceanDigitalTwin:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=CONFIG['n_estimators'],
            n_jobs=-1,
            random_state=CONFIG['random_state']
        )
        self.scaler_X = StandardScaler()

    def train(self, X, Y):
        """Train the function f: {Satellite} -> {Subsurface}"""
        print("\n--- Training Digital Twin Model (Random Forest) ---")

        # Normalize features
        X_scaled = self.scaler_X.fit_transform(X)

        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

        # Fit Model
        self.model.fit(X_train, Y_train)

        # Evaluate
        score = self.model.score(X_test, Y_test)
        Y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

        print(f"Model Training Complete.")
        print(f"R2 Score (Accuracy): {score:.3f}")
        print(f"RMSE (Avg Temp Error): {rmse:.3f} °C")

        return Y_pred, Y_test

    def predict_field(self, satellite_data):
        """
        Reconstruct 3D field from satellite inputs.
        satellite_data: DataFrame containing surface features for a grid.
        """
        X_new = self.scaler_X.transform(satellite_data[CONFIG['features']])
        Y_pred_3d = self.model.predict(X_new) # Shape: (n_grid_points, n_depth_levels)
        return Y_pred_3d

# =============================================================================
# 4. 3D VISUALIZATION (Eye-Catching Scientific Plot)
# =============================================================================
def plot_3d_mhw_interactive(grid_df, predicted_profiles, depths, date_str):
    """
    Creates a high-end interactive 3D Volume plot using Plotly.
    """
    print(f"\n--- Generating 3D MHW Visualization for {date_str} ---")

    # 1. Restructure Data for Plotting
    # We have profiles at Lat/Lon points. We need to unstack to (Lat, Lon, Depth)

    lats = sorted(grid_df['lat'].unique())
    lons = sorted(grid_df['lon'].unique())

    # Meshgrid for visualization
    X_grid, Y_grid = np.meshgrid(lons, lats)

    # To plot volume, we essentially flatten everything into coordinate lists
    # x_flat, y_flat, z_flat, value_flat

    x_vals = []
    y_vals = []
    z_vals = []
    temp_vals = []

    # MHW Threshold (simplified for visual: anything > 29C at depth is 'hot')
    # In real app, calculate climatology per voxel.

    print("Building 3D Volumetric Data...")

    for idx, row in grid_df.iterrows():
        lat = row['lat']
        lon = row['lon']
        profile = predicted_profiles[idx]

        for depth_idx, temp in enumerate(profile):
            z = depths[depth_idx]

            # Optimization: Only plot upper 300m where MHWs matter most
            if z > 300: continue

            x_vals.append(lon)
            y_vals.append(lat)
            z_vals.append(-z) # Negative depth for plotting
            temp_vals.append(temp)

    # 2. Create Plotly 3D Scene
    fig = go.Figure(data=go.Volume(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        value=temp_vals,
        isomin=20,    # Minimum Temp to show (transparency cutoff)
        isomax=31,    # Max Temp
        opacity=0.1,  # General transparency
        surface_count=15, # Number of contour layers
        colorscale='Thermal',
        caps=dict(x_show=False, y_show=False, z_show=True),
        colorbar=dict(title='Temperature (°C)')
    ))

    # Add an Isosurface for the "Core" of the Heatwave (e.g., > 29°C)
    fig.add_trace(go.Isosurface(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        value=temp_vals,
        isomin=29.0,
        isomax=32.0,
        caps=dict(x_show=False, y_show=False),
        surface=dict(fill=0.8, pattern='all'),
        colorscale='Hot',
        showlegend=True,
        name='MHW Core (>29°C)'
    ))

    # 3. Aesthetics
    fig.update_layout(
        title=dict(
            text=f"3D MHW Digital Twin Reconstruction: {date_str}<br>Bay of Bengal (0-300m)",
            font=dict(size=20, family="Times New Roman")
        ),
        scene=dict(
            xaxis_title='Longitude (°E)',
            yaxis_title='Latitude (°N)',
            zaxis_title='Depth (m)',
            aspectratio=dict(x=1, y=1, z=0.4), # Flatten z-axis slightly
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        font=dict(family="Times New Roman"),
        template="plotly_dark" # Eye-catching dark theme
    )

    # Save HTML for interactivity
    save_path = os.path.join(CONFIG['save_dir'], f"MHW_3D_Twin_{date_str}.html")
    fig.write_html(save_path)
    print(f"Interactive 3D plot saved to: {save_path}")

    # Show static screenshot in colab (optional)
    try:
        fig.show()
    except:
        print("Interactive plot requires browser. Check the saved HTML.")

# =============================================================================
# 5. MAIN WORKFLOW
# =============================================================================
def run_digital_twin_for_date():
    """
    Prompts user for a date and runs the 3D MHW Digital Twin.
    """
    print("\n--- 3D MHW Digital Twin Configuration ---")
    while True:
        try:
            date_str = input("Enter the date for MHW reconstruction (YYYY-MM-DD, e.g., 2024-05-15): ")
            year, month, day = map(int, date_str.split('-'))
            target_date = datetime(year, month, day)
            doy = target_date.timetuple().tm_yday
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    print(f"Simulating MHW for {date_str} (Day of Year: {doy})")

    # A. Load Data (Replacing with Mock Generator for demonstration)
    # In production: Load Argo match-ups here
    X, Y, depths = generate_mock_data(n_samples=5000)

    # B. Train the Digital Twin Model
    twin_model = OceanDigitalTwin()
    _, _ = twin_model.train(X, Y)

    # C. Reconstruction (Inference on a specific date)
    print(f"\n--- Reconstructing 3D Field for {date_str} ---")

    # Create a synthetic grid of the Bay of Bengal for one day
    lat_range = np.linspace(10, 20, 20)  # Coarse grid for demo speed
    lon_range = np.linspace(80, 90, 20)
    lons_grid, lats_grid = np.meshgrid(lon_range, lat_range)

    grid_flat = pd.DataFrame({
        'lat': lats_grid.flatten(),
        'lon': lons_grid.flatten()
    })

    # Fill with 'Satellite' data for this specific day (Simulated MHW conditions)
    n_grid = len(grid_flat)
    grid_flat['sst'] = np.random.uniform(29.5, 31.0, n_grid) # High SST for MHW
    grid_flat['sss'] = 33.0
    grid_flat['sla'] = 0.1
    grid_flat['u_wind'] = 2.0
    grid_flat['v_wind'] = 1.5
    grid_flat['chl_a'] = 0.2
    grid_flat['doy'] = doy # Use the user-specified Day of Year

    # D. Predict Subsurface Structure
    predicted_profiles = twin_model.predict_field(grid_flat)

    # E. Calculate MLD (Mixed Layer Depth)
    # Definition: Depth where T = SST - 0.5
    print("Calculating Derived Variables (MLD)...")
    mld_list = []
    for prof in predicted_profiles:
        sst = prof[0]
        mld_indices = np.where(prof < (sst - 0.5))[0]
        if len(mld_indices) > 0:
            mld_list.append(depths[mld_indices[0]])
        else:
            mld_list.append(depths[-1]) # Deep MLD

    grid_flat['predicted_mld'] = mld_list
    print(f"Average Reconstructed MLD: {np.mean(mld_list):.1f}m")

    # F. Visualize
    plot_3d_mhw_interactive(grid_flat, predicted_profiles, depths, date_str)

if __name__ == "__main__":
    run_digital_twin_for_date()


# %% Cell 13
# Locate the 'main' function in the cell above (XPpZVKQuX4nL)
# Inside the 'main' function, find the section where 'grid_flat' is created and populated:

# ... (lines before this)

# Fill with 'Satellite' data for this specific day (Simulated MHW conditions)
n_grid = len(grid_flat)
grid_flat['sst'] = np.random.uniform(29.5, 31.0, n_grid) # High SST for MHW
grid_flat['sss'] = 33.0
grid_flat['sla'] = 0.1
grid_flat['u_wind'] = 2.0
grid_flat['v_wind'] = 1.5
grid_flat['chl_a'] = 0.2
grid_flat['doy'] = 135 # <-- CHANGE THIS LINE for a different Day of Year

# ... (lines after this)

# To change the date, modify the 'doy' value:
# For example:
# - For January 1st, set doy = 1
# - For March 1st, set doy = 60 (approx. accounting for leap year if needed)
# - For September 1st, set doy = 244 (approx.)

# After changing 'doy', you might also want to update the date_str argument
# in the plot_3d_mhw_interactive call to reflect the new simulated date:
# plot_3d_mhw_interactive(grid_flat, predicted_profiles, depths, "YYYY-MM-DD")
# For example, if doy is 1 (Jan 1st), change "2024-05-15" to "2024-01-01"

# After making these changes, re-run the entire cell (XPpZVKQuX4nL) to generate the new 3D plot.


# %% [markdown]
# ### Examining Time Dimensions


# %% Cell 15
print("Climatology Time (first 5 values):")
display(ds_clim_full.time.values[:5])

print("Anomaly Time (first 5 values):")
display(ds_anom_full.time.values[:5])


# %% [markdown]
# # MHW event statistics and frequencies, visualizing
# Perform a comprehensive Marine Heatwave analysis for the Bay of Bengal, including calculating and summarizing MHW event statistics and frequencies, visualizing the distribution of key MHW metrics, enhancing the time series plot to highlight MHW events by intensity categories, and summarizing the key findings from the calculated metrics and visualizations.


# %% [markdown]
# ## Calculate MHW Metrics and Identify Events
# 
# ### Subtask:
# Re-execute the code to prepare the data, identify individual Marine Heatwave events, and calculate their duration, peak intensity, mean intensity, and cumulative intensity. This will ensure the `mhw_summary_df` is correctly populated.


# %% [markdown]
# **Reasoning**:
# The previous code block for calculating MHW metrics failed to retain the `mhw_summary_df` in the runtime for displaying, leading to a 'Runtime no longer has a reference' error. To ensure the `mhw_summary_df` is correctly populated and accessible for subsequent steps as requested, I will re-execute the full code block that performs data preparation, MHW event identification, and metric calculation.
# 


# %% Cell 19
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

# Get user input dates
start_date_str, end_date_str = get_user_input_dates()

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


# %% [markdown]
# ## Visualize MHW Metrics Distribution
# 
# ### Subtask:
# Generate visualizations, such as histograms or box plots, to show the distribution of key MHW metrics (e.g., MHW duration, maximum intensity) over the analyzed period. Provide appropriate legends and labels.


# %% [markdown]
# ## Summarize MHW Frequency and Statistics
# 
# ### Subtask:
# Display the total number of identified MHW events (frequency) and provide descriptive statistics (e.g., mean, median, max, min, standard deviation) for the calculated MHW metrics.


# %% [markdown]
# ## Summarize MHW Frequency and Statistics
# 
# ### Subtask:
# Display the total number of identified MHW events (frequency) and provide descriptive statistics (e.g., mean, median, max, min, standard deviation) for the calculated MHW metrics.
# 
# ### MHW Event Frequency and Statistics Summary
# 
# **Total Number of MHW Events (Frequency):** 4
# 
# **Descriptive Statistics for MHW Metrics:**
# 
# | Metric                   | Count | Mean     | Std Dev  | Min      | 25th %ile | Median   | 75th %ile | Max      |
# |:-------------------------|:------|:---------|:---------|:---------|:----------|:---------|:----------|:---------|
# | **Duration (days)**      | 4     | 9.25     | 12.98    | 1.00     | 2.50      | 4.00     | 10.75     | 28.00    |
# | **Peak Intensity (°C)**  | 4     | 0.197    | 0.301    | 0.018    | 0.040     | 0.049    | 0.206     | 0.643    |
# | **Mean Intensity (°C)**  | 4     | 0.138    | 0.156    | 0.018    | 0.040     | 0.049    | 0.147     | 0.339    |
# | **Cumulative Intensity (°C days)** | 4     | 2.056    | 4.767    | 0.018    | 0.040     | 0.049    | 2.065     | 9.497    |


# %% [markdown]
# ## Visualize MHW Events on Time Series
# 
# ### Subtask:
# Enhance the existing time series plot to clearly highlight identified Marine Heatwave events by intensity categories, and ensure clear labels and legends are provided.


# %% [markdown]
# **Reasoning**:
# The user wants to enhance the time series plot to show MHW intensity categories. This requires modifying the `_prepare_mhw_data` function to return unsmoothed data (`sst_obs`, `clim_aligned`, `thresh_curve`) and creating a new `plot_mhw_time_series` function that handles the category calculations, smoothing, and plotting, including the `fill_between` logic for different intensity levels.
# 


# %% Cell 25
def _prepare_mhw_data(start_date_str, end_date_str):
    """
    Loads and prepares SST and threshold data for MHW calculation.
    Returns sst_obs, clim_aligned, and thresh_curve (all unsmoothed).
    """
    # 2. Load Datasets
    try:
        time_decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds_clim = xr.open_dataset(CONFIG["clim_path"], decode_times=time_decoder)
        ds_anom = xr.open_dataset(CONFIG["anom_path"], decode_times=time_decoder)
    except FileNotFoundError as e:
        print(f"Error: File not found. Check paths.\n{e}")
        return None, None, None

    # 3. Identify Variables (Robust variable finding)
    clim_var_name = next((v for v in ['sst', 'temp', 'mean'] if v in ds_clim), None)
    anom_var_name = next((v for v in ['sst', 'anom', 'anomaly'] if v in ds_anom), None)

    if not clim_var_name or not anom_var_name:
        print(f"Error: Could not identify variables in NetCDF. Found: Clim={list(ds_clim.keys())}, Anom={list(ds_anom.keys())}")
        return None, None, None

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
        return None, None, None

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

    return sst_obs, clim_aligned, thresh_curve

def plot_mhw_time_series(sst_obs, clim_aligned, thresh_curve, start_date_str, end_date_str, CONFIG):
    """
    Generates an enhanced time series plot showing MHW events by intensity categories.
    """
    # Calculate Delta (Difference between Threshold and Climatology)
    diff = thresh_curve - clim_aligned

    # Define Categories based on User Formulas (unsmoothed)
    # Cat1 is the threshold itself, used as the base for filling.
    cat2 = thresh_curve + diff      # Strong (Thresh + 2*Delta from Clim)
    cat3 = thresh_curve + 2 * diff  # Severe (Thresh + 3*Delta from Clim)
    cat4 = thresh_curve + 3 * diff  # Extreme (Thresh + 4*Delta from Clim)

    # Smoothing for plotting purposes
    window = CONFIG['smooth_window']
    sst_obs_smooth = sst_obs.rolling(time=window, center=True).mean()
    clim_smooth = clim_aligned.rolling(time=window, center=True).mean()
    thresh_smooth = thresh_curve.rolling(time=window, center=True).mean()
    cat2_smooth = cat2.rolling(time=window, center=True).mean()
    cat3_smooth = cat3.rolling(time=window, center=True).mean()
    cat4_smooth = cat4.rolling(time=window, center=True).mean()

    print("Generating Enhanced Publication-Ready Plot...")

    fig, ax = plt.subplots(figsize=(16, 9), dpi=CONFIG['dpi'])

    # Convert cftime objects to Matplotlib's numerical dates for plotting
    dates_num = mdates.date2num(sst_obs.time.values)

    # A. Plot Main Lines
    ax.plot(dates_num, sst_obs_smooth, color='k', linewidth=2.5, label='Observed SST (Smoothed)', zorder=5)
    ax.plot(dates_num, clim_smooth, color='#1f77b4', linewidth=2, linestyle='--', label='Climatology (1991-2020)', alpha=0.8)
    ax.plot(dates_num, thresh_smooth, color='#2ca02c', linewidth=2, label='MHW Threshold (90th %ile)', zorder=3)

    # B. Fill MHW Events by Intensity Categories
    # Extreme: sst >= Cat4
    ax.fill_between(
        dates_num,
        cat4_smooth,
        sst_obs_smooth,
        where=(sst_obs_smooth >= cat4_smooth),
        interpolate=True,
        color='darkred',
        alpha=0.7,
        label='MHW Extreme'
    )
    # Severe: Cat3 <= sst < Cat4
    ax.fill_between(
        dates_num,
        cat3_smooth,
        sst_obs_smooth,
        where=(sst_obs_smooth >= cat3_smooth) & (sst_obs_smooth < cat4_smooth),
        interpolate=True,
        color='red',
        alpha=0.7,
        label='MHW Severe'
    )
    # Strong: Cat2 <= sst < Cat3
    ax.fill_between(
        dates_num,
        cat2_smooth,
        sst_obs_smooth,
        where=(sst_obs_smooth >= cat2_smooth) & (sst_obs_smooth < cat3_smooth),
        interpolate=True,
        color='orange',
        alpha=0.7,
        label='MHW Strong'
    )
    # Moderate: Thresh <= sst < Cat2
    ax.fill_between(
        dates_num,
        thresh_smooth,
        sst_obs_smooth,
        where=(sst_obs_smooth >= thresh_smooth) & (sst_obs_smooth < cat2_smooth),
        interpolate=True,
        color='gold',
        alpha=0.7,
        label='MHW Moderate'
    )

    # C. Add "Follow Line" Labels for Categories (optional, if desired)
    # add_label_follow_line(ax, sst_obs.time, thresh_smooth, "Moderate", frac=0.15, offset_points=10, color='gold', fontsize=10, fontweight='bold')
    # add_label_follow_line(ax, sst_obs.time, cat2_smooth, "Strong", frac=0.35, offset_points=10, color='orange', fontsize=10)
    # add_label_follow_line(ax, sst_obs.time, cat3_smooth, "Severe", frac=0.55, offset_points=10, color='red', fontsize=10)
    # add_label_follow_line(ax, sst_obs.time, cat4_smooth, "Extreme", frac=0.75, offset_points=10, color='darkred', fontsize=10)

    # D. Formatting
    ax.set_ylabel('Sea Surface Temperature (°C)', fontsize=16, fontname=CONFIG['font'])
    ax.set_title(f'Marine Heatwave Events by Intensity: Bay of Bengal ({start_date_str} to {end_date_str})\nArea Averaged: Lat 5-23°N, Lon 78-100°E',
                 fontsize=20, fontweight='bold', fontname=CONFIG['font'], pad=15)

    # Date Formatting: Enforcing Month-Year format (%b-%Y)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Dynamic Y-axis Limits
    y_max = float(sst_obs_smooth.max()) + 0.5
    y_min = float(clim_smooth.min()) - 0.5
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(dates_num[0], dates_num[-1])

    # Grid
    ax.grid(True, which='major', linestyle='-', alpha=0.2, color='gray')

    # E. Custom Legend
    legend_elements = [
        Line2D([0], [0], color='k', lw=2.5, label='Observed SST (Smoothed)'),
        Line2D([0], [0], color='#1f77b4', lw=2, linestyle='--', label='Climatology (1991-2020)'),
        Line2D([0], [0], color='#2ca02c', lw=2, label='MHW Threshold (90th %ile)'),
        Patch(facecolor='gold', edgecolor='none', alpha=0.7, label='MHW Moderate'),
        Patch(facecolor='orange', edgecolor='none', alpha=0.7, label='MHW Strong'),
        Patch(facecolor='red', edgecolor='none', alpha=0.7, label='MHW Severe'),
        Patch(facecolor='darkred', edgecolor='none', alpha=0.7, label='MHW Extreme'),
    ]

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.2), ncol=4,
              frameon=True, fancybox=True, shadow=True, fontsize=12, title="MHW Categories")

    # Layout Adjustment
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save
    save_path = os.path.join(CONFIG["save_dir"], f"MHW_BoB_TimeSeries_Categorized_{start_date_str}_to_{end_date_str}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Success! Categorized MHW figure saved to: {save_path}")
    plt.show()

# Get user input dates
start_date_str, end_date_str = get_user_input_dates()

# Prepare data (now returns sst_obs, clim_aligned, thresh_curve unsmoothed)
sst_obs, clim_aligned, thresh_curve = _prepare_mhw_data(start_date_str, end_date_str)

if sst_obs is None or clim_aligned is None or thresh_curve is None:
    print("Failed to prepare MHW data. Exiting.")
else:
    # Call the new plotting function
    plot_mhw_time_series(sst_obs, clim_aligned, thresh_curve, start_date_str, end_date_str, CONFIG)


# %% [markdown]
# ## Comprehensive Summary of Marine Heatwave Analysis in the Bay of Bengal
# 
# This analysis aimed to identify, characterize, and visualize Marine Heatwave (MHW) events in the Bay of Bengal for the specified period (2024-01-01 to 2024-12-31).
# 
# ### Key Findings from MHW Metrics and Summary Statistics:
# 
# *   **Total MHW Events (Frequency):** Over the analyzed period (2024), a total of **4 distinct Marine Heatwave events** were identified in the Bay of Bengal.
# 
# *   **Duration:**
#     *   The duration of MHW events varied significantly, ranging from a minimum of **1 day** to a maximum of **28 days**.
#     *   The average duration was approximately **9.25 days**, with a median of **4 days**, indicating that while some MHWs were relatively short-lived, there was at least one prolonged event.
# 
# *   **Peak Intensity:**
#     *   The maximum temperature anomaly during an MHW (peak intensity) ranged from **0.018 °C to 0.643 °C** above the seasonal threshold.
#     *   The mean peak intensity was around **0.197 °C**, highlighting that even though some events were moderate, one event stood out with a significantly higher peak intensity.
# 
# *   **Mean Intensity:**
#     *   The average intensity over the duration of an MHW (mean intensity) varied from **0.018 °C to 0.339 °C**.
#     *   The mean of these average intensities was approximately **0.138 °C**, suggesting that most MHWs in the region were of relatively low to moderate intensity on average.
# 
# *   **Cumulative Intensity:**
#     *   Cumulative intensity, which represents the total heat stress, ranged from **0.018 °C days to 9.497 °C days**.
#     *   The mean cumulative intensity was approximately **2.056 °C days**, which is largely influenced by the single longer-duration event.
# 
# ### Insights from Visualizations (Time Series Plot with Intensity Categories):
# 
# The enhanced time series plot clearly illustrates the occurrences and intensity categories of MHWs:
# 
# *   **Observed SST vs. Climatology/Threshold:** The plot clearly shows the seasonal cycle of SST, with the observed SST (smoothed) fluctuating around the climatological mean. MHW events are visible as periods where the observed SST rises above the MHW Threshold.
# 
# *   **Intensity Categorization:** The use of `fill_between` with distinct colors (gold for Moderate, orange for Strong, red for Severe, dark red for Extreme) effectively highlights the varying intensity levels of MHWs. This visual representation quickly conveys the severity of each event over time.
# 
# *   **Prominent Events:** One particularly strong or severe MHW event appears to have occurred around April-May, characterized by a significant red/dark red fill area. Other events, particularly towards the end of the year, appear to be of lower intensity (moderate to strong), consistent with the `mhw_summary_df` showing one event with much higher duration and intensity metrics.
# 
# *   **Overall Pattern:** While there were several detected MHW events, a majority seemed to be in the 'Moderate' or 'Strong' categories, with at least one 'Severe' or 'Extreme' event driving the higher end of the intensity and duration statistics.
# 
# ### Conclusion:
# 
# This analysis successfully quantified and visualized Marine Heatwave characteristics in the Bay of Bengal for 2024. The region experienced a few MHW events, with one notable event in the spring showing higher duration and intensity. The visual categorizations provide an immediate understanding of the heat stress experienced. These findings are crucial for understanding the ecological and climatic impacts on the marine environment of the Bay of Bengal.


# %% [markdown]
# ## Final Task
# 
# ### Subtask:
# Provide a comprehensive summary of the Marine Heatwave analysis, including key findings from the calculated metrics and visualizations. Discuss insights gained regarding the characteristics and occurrences of MHWs in the Bay of Bengal for the analyzed period.


# %% [markdown]
# ## Summary:
# 
# ### Q&A
# The comprehensive summary of the Marine Heatwave analysis for the Bay of Bengal from 2024-01-01 to 2024-12-31 reveals key characteristics and occurrences of MHWs in the region:
# 
# ### Data Analysis Key Findings
# *   **Total Events:** A total of **4 distinct Marine Heatwave events** were identified in the Bay of Bengal during the analyzed period (2024).
# *   **Duration:** MHW durations ranged from a minimum of **1 day** to a maximum of **28 days**, with an average duration of **9.25 days** and a median of **4 days**.
# *   **Peak Intensity:** The highest temperature anomaly during an MHW (peak intensity) varied from **0.018 °C to 0.643 °C** above the seasonal threshold, with a mean peak intensity of approximately **0.197 °C**.
# *   **Mean Intensity:** The average intensity over an MHW event ranged from **0.018 °C to 0.339 °C**, averaging about **0.138 °C**.
# *   **Cumulative Intensity:** The total heat stress (cumulative intensity) for events ranged from **0.018 °C days to 9.497 °C days**, with a mean of approximately **2.056 °C days**.
# *   **Intensity Distribution:** The time series visualization confirmed varying intensity categories, with one prominent event around April-May exhibiting 'Severe' or 'Extreme' characteristics, while other detected events tended to be 'Moderate' or 'Strong'.
# 
# ### Insights or Next Steps
# *   The significant difference between mean and median duration (9.25 vs. 4 days) and the large standard deviation in duration (\~13 days) indicate that while most MHWs were short-lived, there was at least one substantially longer event driving up the average. This suggests a bimodal distribution of MHW duration or the presence of an outlier event.
# *   Further investigation into the specific atmospheric and oceanic drivers of the single long-duration, high-intensity MHW event (likely around April-May 2024) could provide crucial insights into the mechanisms behind extreme heatwaves in the Bay of Bengal.


# %% [markdown]
# # Comparative Marine Heatwaves analysis
# The task is complete. I have provided a comprehensive summary of the Marine Heatwave analysis for the Bay of Bengal, including key findings from calculated metrics and visualizations, and discussed insights gained regarding the characteristics and occurrences of MHWs for the analyzed period.


# %% [markdown]
# ### Deeper Exploration of `regional_comparison` DataFrame


# %% Cell 31
import pandas as pd

# Ensure regional_mhw_metrics is available
if 'regional_mhw_metrics' not in globals():
    print("regional_mhw_metrics not found in current session. Recalculating...")
    # This assumes calculate_regional_mhw_metrics() is defined in a previous cell.
    # We need to make sure CONFIG is also correctly set for regions and dates.
    # For simplicity and to avoid re-prompting, we'll try to run the calculation.
    # In a real scenario, you might want a more sophisticated way to get dates.
    try:
        regional_mhw_metrics = calculate_regional_mhw_metrics()
    except NameError:
        print("Error: calculate_regional_mhw_metrics() function not found. Please ensure it's defined and executed.")
        regional_mhw_metrics = {}

# 1. Initialize an empty list to store aggregated MHW summary DataFrames
all_mhw_summary_dfs = []

# 2. Iterate through the regional_mhw_metrics dictionary
for region_name, mhw_summary_df in regional_mhw_metrics.items():
    # 3. For each region's MHW summary DataFrame:
    if not mhw_summary_df.empty:
        # a. Add a new column named 'region'
        mhw_summary_df['region'] = region_name
        # b. Append this modified DataFrame to the list
        all_mhw_summary_dfs.append(mhw_summary_df)
    else:
        print(f"No MHW events found for {region_name}, skipping.")

# 4. Concatenate all DataFrames into a single, consolidated pandas DataFrame
if all_mhw_summary_dfs:
    all_regions_mhw_df = pd.concat(all_mhw_summary_dfs, ignore_index=True)

    # 5. Group by 'region' and calculate descriptive statistics
    comparison_metrics = ['duration_days', 'peak_intensity_c', 'mean_intensity_c', 'cumulative_intensity_c']
    regional_comparison = all_regions_mhw_df.groupby('region')[comparison_metrics].agg(
        ['count', 'mean', 'median', 'std', 'min', 'max']
    )

    print("Displaying the full regional_comparison DataFrame:")
    display(regional_comparison)
else:
    print("No MHW events were recorded across any of the selected regions for the specified period, so no regional_comparison DataFrame can be created or displayed.")


# %% [markdown]
# ### Deeper Insights from `regional_comparison`
# 
# Based on the `regional_comparison` DataFrame, here are some deeper insights into the Marine Heatwave characteristics across the Bay of Bengal and the Arabian Sea for the analyzed period (March 1, 2024 to September 30, 2024):
# 
# **1. MHW Frequency and Dominance of Single Events:**
# *   **Bay of Bengal:** Only 1 MHW event was recorded. The fact that its `mean`, `median`, `min`, `max`, and `std` for all metrics are identical (or NaN for std due to single observation) implies this *single event* completely defines the region's MHW characteristics for this period. This singular event must have been quite significant to drive the overall regional impact.
# *   **Arabian Sea:** Experienced 3 MHW events. This higher count suggests that MHW-triggering conditions were more frequent or spatially heterogeneous, leading to multiple, distinct events rather than one prolonged one.
# 
# **2. Divergent MHW Strategies (Duration vs. Frequency):**
# *   The **Bay of Bengal** exhibits a 'high-impact, low-frequency' strategy, with a single, very long (22 days) and intense MHW. This event, despite being unique, contributes substantially to the overall heat stress.
# *   The **Arabian Sea** shows a 'lower-impact, higher-frequency' strategy, with shorter-duration events (4 to 9 days) but more occurrences. The mean duration of ~7.33 days in the Arabian Sea is notably lower than the 22 days in the Bay of Bengal.
# 
# **3. Intensity Differences and Implications:**
# *   **Peak Intensity:** The Bay of Bengal's peak intensity (~0.478 °C) is about **3.3 times higher** than the Arabian Sea's mean peak intensity (~0.145 °C). This suggests that when MHWs occur in the Bay of Bengal, they can reach much higher temperature anomalies, posing a greater immediate threat to marine ecosystems.
# *   **Mean Intensity:** Similar to peak intensity, the mean intensity in the Bay of Bengal (~0.201 °C) is roughly **2.4 times higher** than in the Arabian Sea (~0.082 °C). This implies that the single Bay of Bengal event sustained a stronger anomaly throughout its duration.
# 
# **4. Cumulative Heat Stress (The 'Total Burden'):**
# *   The most striking difference is in **Cumulative Intensity**. The Bay of Bengal's single event delivered ~4.43 °C days of heat stress, which is roughly **6.7 times greater** than the *mean* cumulative intensity of events in the Arabian Sea (~0.66 °C days). More importantly, it's significantly higher than even the *maximum* cumulative intensity recorded for any single event in the Arabian Sea (0.895 °C days).
# *   This highlights that even with fewer events, a single prolonged and intense MHW can result in a much larger total heat burden on marine organisms and ecosystems, which is often a more critical indicator of ecological impact.
# 
# **5. Variability within Regions:**
# *   **Bay of Bengal:** The `std` for all metrics is `NaN` because there's only one event, so no variability can be calculated from the sample. All observed values are also `mean`, `median`, `min`, and `max`.
# *   **Arabian Sea:** The standard deviations (e.g., duration std ~2.89 days) show some variability in the characteristics of its three events, but these variations are much smaller in magnitude compared to the stark difference between the two regions.
# 
# **Overall Insight:**
# While the Arabian Sea experiences more frequent MHW events, the Bay of Bengal can be subjected to singularly impactful events that are significantly longer, hotter, and deliver a far greater cumulative heat stress. This suggests different management and monitoring strategies might be required for each region, focusing on frequency in the Arabian Sea versus the potential for extreme, prolonged events in the Bay of Bengal. These findings underscore the regional diversity in MHW characteristics and their potential ecological consequences.


# %% [markdown]
# ## Define Regions for Comparison
# 
# ### Subtask:
# Determine the new geographical regions you want to compare MHWs for. This will involve defining new latitude and longitude bounds for each region. You might need to update the CONFIG dictionary or create a new input mechanism to specify these regions.


# %% [markdown]
# **Reasoning**:
# The subtask requires defining multiple geographical regions within the `CONFIG` dictionary. I will update the `CONFIG` dictionary by adding a new 'regions' key that stores the latitude and longitude bounds for predefined areas (Bay of Bengal and Arabian Sea). I will also add a `selected_region_name` key to control which region's bounds are actively used by the existing `lat_bounds` and `lon_bounds` keys in `CONFIG`.
# 


# %% Cell 35
import os

# Define the new regions and update the CONFIG dictionary
CONFIG.update({
    "regions": {
        "Bay of Bengal": {
            "lat_bounds": slice(23, 5),  # North to South
            "lon_bounds": slice(78, 100) # West to East
        },
        "Arabian Sea": {
            "lat_bounds": slice(25, 5),  # North to South
            "lon_bounds": slice(50, 75)  # West to East
        }
    },
    "selected_region_name": "Bay of Bengal" # Default region
})

# Update CONFIG's top-level lat_bounds and lon_bounds based on the selected_region_name
selected_region_data = CONFIG["regions"][CONFIG["selected_region_name"]]
CONFIG["lat_bounds"] = selected_region_data["lat_bounds"]
CONFIG["lon_bounds"] = selected_region_data["lon_bounds"]

print("CONFIG dictionary updated with new regions and a default selected region.")
print(f"Currently selected region: {CONFIG['selected_region_name']}")
print(f"Latitude bounds for selected region: {CONFIG['lat_bounds']}")
print(f"Longitude bounds for selected region: {CONFIG['lon_bounds']}")


# %% [markdown]
# **Reasoning**:
# The next logical step is to create a function that allows the user to dynamically select a region for analysis from the newly defined regions in the `CONFIG` dictionary. This function will update the `lat_bounds` and `lon_bounds` in `CONFIG` based on the user's choice and then re-run the `process_mhw_data` function with the newly selected region's boundaries, effectively preparing the data for the new region.
# 


# %% Cell 37
def select_and_process_region():
    """
    Allows the user to select a region for MHW analysis from the predefined regions
    in the CONFIG dictionary and then re-runs the MHW processing.
    """
    print("\n--- Select Region for MHW Analysis ---")
    print("Available regions:")
    for i, region_name in enumerate(CONFIG["regions"].keys()):
        print(f"  {i+1}. {region_name}")

    while True:
        try:
            choice = input(f"Enter the number of the region you want to analyze (1-{len(CONFIG['regions'])}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(CONFIG["regions"]):
                selected_region_name = list(CONFIG["regions"].keys())[choice_idx]
                break
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Update CONFIG with the selected region's data
    CONFIG["selected_region_name"] = selected_region_name
    selected_region_data = CONFIG["regions"][selected_region_name]
    CONFIG["lat_bounds"] = selected_region_data["lat_bounds"]
    CONFIG["lon_bounds"] = selected_region_data["lon_bounds"]

    print(f"\nRegion '{CONFIG['selected_region_name']}' selected for analysis.")
    print(f"Latitude bounds: {CONFIG['lat_bounds']}")
    print(f"Longitude bounds: {CONFIG['lon_bounds']}")

    # Re-run the main MHW data processing with the new region
    # We need to call the process_mhw_data function, which will handle plotting.
    process_mhw_data()

# Call the function to allow user to select a region and run the analysis
select_and_process_region()


# %% [markdown]
# ## Adapt Data Loading and Processing for Multiple Regions
# 
# ### Subtask:
# Modify the existing data preparation and plotting functions to accept region-specific latitude and longitude bounds and region names for accurate processing and visualization across different geographical areas.


# %% [markdown]
# **Reasoning**:
# The subtask requires modifying existing functions to accept region-specific parameters. I will update the `_prepare_mhw_data` function to take `lat_bounds` and `lon_bounds` as arguments, adjust the `plot_mhw_time_series` function to accept `region_name`, and then modify the `select_and_process_region` function to correctly pass these new parameters to the data preparation and plotting functions.
# 


# %% [markdown]
# **Reasoning**:
# The `plot_mhw_time_series` function produced a `NameError` because `lat_bounds_param` and `lon_bounds_param` were not passed to it. I will modify the function signature to accept these parameters and update the call in `select_and_process_region` to pass them.
# 


# %% Cell 41
def _prepare_mhw_data(start_date_str, end_date_str, lat_bounds_param, lon_bounds_param):
    """
    Loads and prepares SST and threshold data for MHW calculation.
    Returns sst_obs, clim_aligned, and thresh_curve (all unsmoothed).
    Accepts region-specific lat/lon bounds.
    """
    # 2. Load Datasets
    try:
        time_decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds_clim = xr.open_dataset(CONFIG["clim_path"], decode_times=time_decoder)
        ds_anom = xr.open_dataset(CONFIG["anom_path"], decode_times=time_decoder)
    except FileNotFoundError as e:
        print(f"Error: File not found. Check paths.\n{e}")
        return None, None, None

    # 3. Identify Variables (Robust variable finding)
    clim_var_name = next((v for v in ['sst', 'temp', 'mean'] if v in ds_clim), None)
    anom_var_name = next((v for v in ['sst', 'anom', 'anomaly'] if v in ds_anom), None)

    if not clim_var_name or not anom_var_name:
        print(f"Error: Could not identify variables in NetCDF. Found: Clim={list(ds_clim.keys())}, Anom={list(ds_anom.keys())}")
        return None, None, None

    # 4. Spatial Averaging (Using passed lat/lon bounds)
    # Ensure slices are in correct order for selection (e.g., if lat_bounds is slice(23, 5), ensure min/max)
    lat_slice = slice(min(lat_bounds_param.start, lat_bounds_param.stop), max(lat_bounds_param.start, lat_bounds_param.stop))
    lon_slice = slice(lon_bounds_param.start, lon_bounds_param.stop)

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
        return None, None, None

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

    return sst_obs, clim_aligned, thresh_curve

def plot_mhw_time_series(sst_obs, clim_aligned, thresh_curve, start_date_str, end_date_str, CONFIG, region_name, lat_bounds_param, lon_bounds_param):
    """
    Generates an enhanced time series plot showing MHW events by intensity categories.
    Includes region_name in the title.
    """
    # Calculate Delta (Difference between Threshold and Climatology)
    diff = thresh_curve - clim_aligned

    # Define Categories based on User Formulas (unsmoothed)
    cat2 = thresh_curve + diff
    cat3 = thresh_curve + 2 * diff
    cat4 = thresh_curve + 3 * diff

    # Smoothing for plotting purposes
    window = CONFIG['smooth_window']
    sst_obs_smooth = sst_obs.rolling(time=window, center=True).mean()
    clim_smooth = clim_aligned.rolling(time=window, center=True).mean()
    thresh_smooth = thresh_curve.rolling(time=window, center=True).mean()
    cat2_smooth = cat2.rolling(time=window, center=True).mean()
    cat3_smooth = cat3.rolling(time=window, center=True).mean()
    cat4_smooth = cat4.rolling(time=window, center=True).mean()

    print("Generating Enhanced Publication-Ready Plot...")

    fig, ax = plt.subplots(figsize=(16, 9), dpi=CONFIG['dpi'])

    # Convert cftime objects to Matplotlib's numerical dates for plotting
    dates_num = mdates.date2num(sst_obs.time.values)

    # A. Plot Main Lines
    ax.plot(dates_num, sst_obs_smooth, color='k', linewidth=2.5, label='Observed SST (Smoothed)', zorder=5)
    ax.plot(dates_num, clim_smooth, color='#1f77b4', linewidth=2, linestyle='--', label='Climatology (1991-2020)', alpha=0.8)
    ax.plot(dates_num, thresh_smooth, color='#2ca02c', linewidth=2, label='MHW Threshold (90th %ile)', zorder=3)

    # B. Fill MHW Events by Intensity Categories
    ax.fill_between(
        dates_num,
        cat4_smooth,
        sst_obs_smooth,
        where=(sst_obs_smooth >= cat4_smooth),
        interpolate=True,
        color='darkred',
        alpha=0.7,
        label='MHW Extreme'
    )
    ax.fill_between(
        dates_num,
        cat3_smooth,
        sst_obs_smooth,
        where=(sst_obs_smooth >= cat3_smooth) & (sst_obs_smooth < cat4_smooth),
        interpolate=True,
        color='red',
        alpha=0.7,
        label='MHW Severe'
    )
    ax.fill_between(
        dates_num,
        cat2_smooth,
        sst_obs_smooth,
        where=(sst_obs_smooth >= cat2_smooth) & (sst_obs_smooth < cat3_smooth),
        interpolate=True,
        color='orange',
        alpha=0.7,
        label='MHW Strong'
    )
    ax.fill_between(
        dates_num,
        thresh_smooth,
        sst_obs_smooth,
        where=(sst_obs_smooth >= thresh_smooth) & (sst_obs_smooth < cat2_smooth),
        interpolate=True,
        color='gold',
        alpha=0.7,
        label='MHW Moderate'
    )

    # D. Formatting
    ax.set_ylabel('Sea Surface Temperature (°C)', fontsize=16, fontname=CONFIG['font'])
    ax.set_title(f'Marine Heatwave Events by Intensity: {region_name} ({start_date_str} to {end_date_str})\nArea Averaged: Lat {lat_bounds_param.start}-{lat_bounds_param.stop}°N, Lon {lon_bounds_param.start}-{lon_bounds_param.stop}°E',
                 fontsize=20, fontweight='bold', fontname=CONFIG['font'], pad=15)

    # Date Formatting: Enforcing Month-Year format (%b-%Y)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Dynamic Y-axis Limits
    y_max = float(sst_obs_smooth.max()) + 0.5
    y_min = float(clim_smooth.min()) - 0.5
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(dates_num[0], dates_num[-1])

    # Grid
    ax.grid(True, which='major', linestyle='-', alpha=0.2, color='gray')

    # E. Custom Legend
    legend_elements = [
        Line2D([0], [0], color='k', lw=2.5, label='Observed SST (Smoothed)'),
        Line2D([0], [0], color='#1f77b4', lw=2, linestyle='--', label='Climatology (1991-2020)'),
        Line2D([0], [0], color='#2ca02c', lw=2, label='MHW Threshold (90th %ile)'),
        Patch(facecolor='gold', edgecolor='none', alpha=0.7, label='MHW Moderate'),
        Patch(facecolor='orange', edgecolor='none', alpha=0.7, label='MHW Strong'),
        Patch(facecolor='red', edgecolor='none', alpha=0.7, label='MHW Severe'),
        Patch(facecolor='darkred', edgecolor='none', alpha=0.7, label='MHW Extreme'),
    ]

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.2), ncol=4,
              frameon=True, fancybox=True, shadow=True, fontsize=12, title="MHW Categories")

    # Layout Adjustment
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save
    save_path = os.path.join(CONFIG["save_dir"], f"MHW_{region_name.replace(' ', '_')}_TimeSeries_Categorized_{start_date_str}_to_{end_date_str}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Success! Categorized MHW figure saved to: {save_path}")
    plt.show()

def select_and_process_region():
    """
    Allows the user to select a region for MHW analysis from the predefined regions
    in the CONFIG dictionary and then re-runs the MHW processing.
    """
    print("\n--- Select Region for MHW Analysis ---")
    print("Available regions:")
    for i, region_name_iter in enumerate(CONFIG["regions"].keys()):
        print(f"  {i+1}. {region_name_iter}")

    while True:
        try:
            choice = input(f"Enter the number of the region you want to analyze (1-{len(CONFIG['regions'])}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(CONFIG["regions"]):
                selected_region_name = list(CONFIG["regions"].keys())[choice_idx]
                break
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Update CONFIG with the selected region's data (for global access if needed, but functions will use passed params)
    CONFIG["selected_region_name"] = selected_region_name
    selected_region_data = CONFIG["regions"][selected_region_name]
    # These lines are now primarily for logging/display within this function
    CONFIG["lat_bounds"] = selected_region_data["lat_bounds"]
    CONFIG["lon_bounds"] = selected_region_data["lon_bounds"]

    print(f"\nRegion '{CONFIG['selected_region_name']}' selected for analysis.")
    print(f"Latitude bounds: {CONFIG['lat_bounds']}")
    print(f"Longitude bounds: {CONFIG['lon_bounds']}")

    # Get user input dates
    start_date_str, end_date_str = get_user_input_dates()

    # Prepare data, passing region-specific bounds
    sst_obs, clim_aligned, thresh_curve = _prepare_mhw_data(start_date_str, end_date_str,
                                                          selected_region_data["lat_bounds"],
                                                          selected_region_data["lon_bounds"])

    if sst_obs is None or clim_aligned is None or thresh_curve is None:
        print("Failed to prepare MHW data. Exiting.")
    else:
        # Call the new plotting function, passing the region name and bounds
        plot_mhw_time_series(sst_obs, clim_aligned, thresh_curve, start_date_str, end_date_str, CONFIG, selected_region_name, selected_region_data["lat_bounds"], selected_region_data["lon_bounds"])

# Call the function to allow user to select a region and run the analysis
select_and_process_region()


# %% [markdown]
# **Reasoning**:
# The previous code block resulted in a `NameError` because `lat_bounds_param` and `lon_bounds_param` were not correctly passed to `plot_mhw_time_series`. The provided code block correctly updates the function signature of `plot_mhw_time_series` and its call within `select_and_process_region` to pass these parameters, resolving the error and fulfilling the subtask requirements.
# 


# %% Cell 43
def _prepare_mhw_data(start_date_str, end_date_str, lat_bounds_param, lon_bounds_param):
    """
    Loads and prepares SST and threshold data for MHW calculation.
    Returns sst_obs, clim_aligned, and thresh_curve (all unsmoothed).
    Accepts region-specific lat/lon bounds.
    """
    # 2. Load Datasets
    try:
        time_decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds_clim = xr.open_dataset(CONFIG["clim_path"], decode_times=time_decoder)
        ds_anom = xr.open_dataset(CONFIG["anom_path"], decode_times=time_decoder)
    except FileNotFoundError as e:
        print(f"Error: File not found. Check paths.\n{e}")
        return None, None, None

    # 3. Identify Variables (Robust variable finding)
    clim_var_name = next((v for v in ['sst', 'temp', 'mean'] if v in ds_clim), None)
    anom_var_name = next((v for v in ['sst', 'anom', 'anomaly'] if v in ds_anom), None)

    if not clim_var_name or not anom_var_name:
        print(f"Error: Could not identify variables in NetCDF. Found: Clim={list(ds_clim.keys())}, Anom={list(ds_anom.keys())}")
        return None, None, None

    # 4. Spatial Averaging (Using passed lat/lon bounds)
    # Ensure slices are in correct order for selection (e.g., if lat_bounds is slice(23, 5), ensure min/max)
    lat_slice = slice(min(lat_bounds_param.start, lat_bounds_param.stop), max(lat_bounds_param.start, lat_bounds_param.stop))
    lon_slice = slice(lon_bounds_param.start, lon_bounds_param.stop)

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
        return None, None, None

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

    return sst_obs, clim_aligned, thresh_curve

def plot_mhw_time_series(sst_obs, clim_aligned, thresh_curve, start_date_str, end_date_str, CONFIG, region_name, lat_bounds_param, lon_bounds_param):
    """
    Generates an enhanced time series plot showing MHW events by intensity categories.
    Includes region_name in the title.
    """
    # Calculate Delta (Difference between Threshold and Climatology)
    diff = thresh_curve - clim_aligned

    # Define Categories based on User Formulas (unsmoothed)
    cat2 = thresh_curve + diff
    cat3 = thresh_curve + 2 * diff
    cat4 = thresh_curve + 3 * diff

    # Smoothing for plotting purposes
    window = CONFIG['smooth_window']
    sst_obs_smooth = sst_obs.rolling(time=window, center=True).mean()
    clim_smooth = clim_aligned.rolling(time=window, center=True).mean()
    thresh_smooth = thresh_curve.rolling(time=window, center=True).mean()
    cat2_smooth = cat2.rolling(time=window, center=True).mean()
    cat3_smooth = cat3.rolling(time=window, center=True).mean()
    cat4_smooth = cat4.rolling(time=window, center=True).mean()

    print("Generating Enhanced Publication-Ready Plot...")

    fig, ax = plt.subplots(figsize=(16, 9), dpi=CONFIG['dpi'])

    # Convert cftime objects to Matplotlib's numerical dates for plotting
    dates_num = mdates.date2num(sst_obs.time.values)

    # A. Plot Main Lines
    ax.plot(dates_num, sst_obs_smooth, color='k', linewidth=2.5, label='Observed SST (Smoothed)', zorder=5)
    ax.plot(dates_num, clim_smooth, color='#1f77b4', linewidth=2, linestyle='--', label='Climatology (1991-2020)', alpha=0.8)
    ax.plot(dates_num, thresh_smooth, color='#2ca02c', linewidth=2, label='MHW Threshold (90th %ile)', zorder=3)

    # B. Fill MHW Events by Intensity Categories
    ax.fill_between(
        dates_num,
        cat4_smooth,
        sst_obs_smooth,
        where=(sst_obs_smooth >= cat4_smooth),
        interpolate=True,
        color='darkred',
        alpha=0.7,
        label='MHW Extreme'
    )
    ax.fill_between(
        dates_num,
        cat3_smooth,
        sst_obs_smooth,
        where=(sst_obs_smooth >= cat3_smooth) & (sst_obs_smooth < cat4_smooth),
        interpolate=True,
        color='red',
        alpha=0.7,
        label='MHW Severe'
    )
    ax.fill_between(
        dates_num,
        cat2_smooth,
        sst_obs_smooth,
        where=(sst_obs_smooth >= cat2_smooth) & (sst_obs_smooth < cat3_smooth),
        interpolate=True,
        color='orange',
        alpha=0.7,
        label='MHW Strong'
    )
    ax.fill_between(
        dates_num,
        thresh_smooth,
        sst_obs_smooth,
        where=(sst_obs_smooth >= thresh_smooth) & (sst_obs_smooth < cat2_smooth),
        interpolate=True,
        color='gold',
        alpha=0.7,
        label='MHW Moderate'
    )

    # D. Formatting
    ax.set_ylabel('Sea Surface Temperature (°C)', fontsize=16, fontname=CONFIG['font'])
    ax.set_title(f'Marine Heatwave Events by Intensity: {region_name} ({start_date_str} to {end_date_str})\nArea Averaged: Lat {lat_bounds_param.start}-{lat_bounds_param.stop}°N, Lon {lon_bounds_param.start}-{lon_bounds_param.stop}°E',
                 fontsize=20, fontweight='bold', fontname=CONFIG['font'], pad=15)

    # Date Formatting: Enforcing Month-Year format (%b-%Y)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Dynamic Y-axis Limits
    y_max = float(sst_obs_smooth.max()) + 0.5
    y_min = float(clim_smooth.min()) - 0.5
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(dates_num[0], dates_num[-1])

    # Grid
    ax.grid(True, which='major', linestyle='-', alpha=0.2, color='gray')

    # E. Custom Legend
    legend_elements = [
        Line2D([0], [0], color='k', lw=2.5, label='Observed SST (Smoothed)'),
        Line2D([0], [0], color='#1f77b4', lw=2, linestyle='--', label='Climatology (1991-2020)'),
        Line2D([0], [0], color='#2ca02c', lw=2, label='MHW Threshold (90th %ile)'),
        Patch(facecolor='gold', edgecolor='none', alpha=0.7, label='MHW Moderate'),
        Patch(facecolor='orange', edgecolor='none', alpha=0.7, label='MHW Strong'),
        Patch(facecolor='red', edgecolor='none', alpha=0.7, label='MHW Severe'),
        Patch(facecolor='darkred', edgecolor='none', alpha=0.7, label='MHW Extreme'),
    ]

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.2), ncol=4,
              frameon=True, fancybox=True, shadow=True, fontsize=12, title="MHW Categories")

    # Layout Adjustment
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save
    save_path = os.path.join(CONFIG["save_dir"], f"MHW_{region_name.replace(' ', '_')}_TimeSeries_Categorized_{start_date_str}_to_{end_date_str}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Success! Categorized MHW figure saved to: {save_path}")
    plt.show()

def select_and_process_region():
    """
    Allows the user to select a region for MHW analysis from the predefined regions
    in the CONFIG dictionary and then re-runs the MHW processing.
    """
    print("\n--- Select Region for MHW Analysis ---")
    print("Available regions:")
    for i, region_name_iter in enumerate(CONFIG["regions"].keys()):
        print(f"  {i+1}. {region_name_iter}")

    while True:
        try:
            choice = input(f"Enter the number of the region you want to analyze (1-{len(CONFIG['regions'])}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(CONFIG["regions"]):
                selected_region_name = list(CONFIG["regions"].keys())[choice_idx]
                break
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Update CONFIG with the selected region's data (for global access if needed, but functions will use passed params)
    CONFIG["selected_region_name"] = selected_region_name
    selected_region_data = CONFIG["regions"][selected_region_name]
    # These lines are now primarily for logging/display within this function
    CONFIG["lat_bounds"] = selected_region_data["lat_bounds"]
    CONFIG["lon_bounds"] = selected_region_data["lon_bounds"]

    print(f"\nRegion '{CONFIG['selected_region_name']}' selected for analysis.")
    print(f"Latitude bounds: {CONFIG['lat_bounds']}")
    print(f"Longitude bounds: {CONFIG['lon_bounds']}")

    # Get user input dates
    start_date_str, end_date_str = get_user_input_dates()

    # Prepare data, passing region-specific bounds
    sst_obs, clim_aligned, thresh_curve = _prepare_mhw_data(start_date_str, end_date_str,
                                                          selected_region_data["lat_bounds"],
                                                          selected_region_data["lon_bounds"])

    if sst_obs is None or clim_aligned is None or thresh_curve is None:
        print("Failed to prepare MHW data. Exiting.")
    else:
        # Call the new plotting function, passing the region name and bounds
        plot_mhw_time_series(sst_obs, clim_aligned, thresh_curve, start_date_str, end_date_str, CONFIG, selected_region_name, selected_region_data["lat_bounds"], selected_region_data["lon_bounds"])

# Call the function to allow user to select a region and run the analysis
select_and_process_region()


# %% [markdown]
# ## Calculate MHW Metrics for Each Region
# 
# ### Subtask:
# Iterate through the defined regions, calculate Marine Heatwave (MHW) metrics for each, and store them for comparative analysis.


# %% [markdown]
# **Reasoning**:
# The subtask requires defining a new function `calculate_regional_mhw_metrics` that will iterate through the defined regions, obtain user input dates, prepare MHW data, calculate MHW metrics for each region, and store the results in a dictionary for comparative analysis. This function will encapsulate the logic described in the instructions.
# 


# %% Cell 46
def calculate_regional_mhw_metrics():
    """
    Iterates through defined regions, calculates Marine Heatwave (MHW) metrics for each,
    and stores them for comparative analysis.
    """
    all_regions_mhw_metrics = {}

    print("\n--- Calculating MHW Metrics for Each Region ---")
    start_date_str, end_date_str = get_user_input_dates() # Get dates once for all regions

    for region_name, region_data in CONFIG["regions"].items():
        print(f"\nProcessing region: {region_name}")
        lat_bounds = region_data["lat_bounds"]
        lon_bounds = region_data["lon_bounds"]

        # Prepare data for the current region
        sst_obs, clim_aligned, thresh_curve = _prepare_mhw_data(start_date_str, end_date_str,
                                                              lat_bounds, lon_bounds)

        if sst_obs is None or clim_aligned is None or thresh_curve is None:
            print(f"Skipping {region_name} due to data preparation failure.")
            continue

        # 1. Convert xarray.DataArray objects into a pandas DataFrame
        df_mhw = pd.DataFrame({
            'SST': sst_obs.to_pandas(),
            'Threshold': thresh_curve.to_pandas()
        })
        df_mhw.index.name = 'time'

        # 2. Create a new boolean column `is_mhw`
        df_mhw['is_mhw'] = df_mhw['SST'] >= df_mhw['Threshold']

        # 3. Identify individual Marine Heatwave events
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

        # Store the summary DataFrame for the current region
        all_regions_mhw_metrics[region_name] = mhw_summary_df
        print(f"Calculated {len(mhw_summary_df)} MHW events for {region_name}.")

    print("\nMHW metrics calculated for all specified regions.")
    return all_regions_mhw_metrics

# Execute the function to calculate metrics for all regions
regional_mhw_metrics = calculate_regional_mhw_metrics()


# %% [markdown]
# ## Perform Comparative Analysis of MHW Metrics
# 
# ### Subtask:
# Once MHW metrics are calculated for all regions, perform statistical comparisons. This could involve creating a consolidated DataFrame of key metrics (e.g., mean duration, mean peak intensity, total frequency) for all regions. Analyze these statistics to identify similarities and differences across regions.


# %% [markdown]
# **Reasoning**:
# The subtask requires consolidating MHW metrics from all regions into a single DataFrame and then performing statistical comparisons. I will initialize a list, iterate through the regional MHW metrics, add a 'region' column to each DataFrame, concatenate them, and finally calculate and display descriptive statistics grouped by region.
# 


# %% Cell 49
import pandas as pd

# 1. Initialize an empty list to store aggregated MHW summary DataFrames
all_mhw_summary_dfs = []

# 2. Iterate through the regional_mhw_metrics dictionary
for region_name, mhw_summary_df in regional_mhw_metrics.items():
    # 3. For each region's MHW summary DataFrame:
    if not mhw_summary_df.empty:
        # a. Add a new column named 'region'
        mhw_summary_df['region'] = region_name
        # b. Append this modified DataFrame to the list
        all_mhw_summary_dfs.append(mhw_summary_df)
    else:
        print(f"No MHW events found for {region_name}, skipping.")

# 4. Concatenate all DataFrames into a single, consolidated pandas DataFrame
if all_mhw_summary_dfs:
    all_regions_mhw_df = pd.concat(all_mhw_summary_dfs, ignore_index=True)
    print("\nConsolidated MHW metrics DataFrame created.")
    display(all_regions_mhw_df.head())

    # 5. Group by 'region' and calculate descriptive statistics
    comparison_metrics = ['duration_days', 'peak_intensity_c', 'mean_intensity_c', 'cumulative_intensity_c']
    regional_comparison = all_regions_mhw_df.groupby('region')[comparison_metrics].agg(
        ['count', 'mean', 'median', 'std', 'min', 'max']
    )

    # 6. Display the consolidated descriptive statistics
    print("\n--- Comparative MHW Statistics Across Regions ---")
    display(regional_comparison)
else:
    print("No MHW events were recorded across any of the selected regions for the specified period.")


# %% [markdown]
# ### Comparative Analysis of MHW Metrics Across Regions
# 
# **Summary of Findings:**
# 
# The `regional_comparison` DataFrame provides a detailed statistical comparison of Marine Heatwave (MHW) metrics between the **Bay of Bengal** and the **Arabian Sea** for the specified period (March 1, 2024 to September 30, 2024).
# 
# **Key Observations:**
# 
# 1.  **MHW Frequency:**
#     *   **Bay of Bengal:** Experienced 1 MHW event.
#     *   **Arabian Sea:** Experienced 3 MHW events.
#     *   **Insight:** The Arabian Sea had a higher frequency of MHW events during this period compared to the Bay of Bengal, suggesting that MHW conditions were more common or fragmented in the Arabian Sea.
# 
# 2.  **Duration (duration_days):**
#     *   **Bay of Bengal:** The single MHW event had a duration of **22 days**.
#     *   **Arabian Sea:** MHW durations ranged from **4 to 9 days**, with a mean of approximately **7.33 days** and a median of **9 days**. The standard deviation was 2.89 days.
#     *   **Insight:** The MHW event in the Bay of Bengal was significantly longer than any individual event recorded in the Arabian Sea during the same period. This indicates that while less frequent, the Bay of Bengal can experience more prolonged heat stress.
# 
# 3.  **Peak Intensity (°C):**
#     *   **Bay of Bengal:** The single event had a peak intensity of **0.478 °C**.
#     *   **Arabian Sea:** Peak intensities ranged from **0.099 °C to 0.174 °C**, with a mean of **0.145 °C** and a median of **0.163 °C**. The standard deviation was 0.04 °C.
#     *   **Insight:** The MHW event in the Bay of Bengal reached a considerably higher peak intensity, almost three times that of the average peak intensity in the Arabian Sea. This suggests the Bay of Bengal event was more intense at its peak.
# 
# 4.  **Mean Intensity (°C):**
#     *   **Bay of Bengal:** The single event had a mean intensity of **0.201 °C**.
#     *   **Arabian Sea:** Mean intensities ranged from **0.052 °C to 0.099 °C**, with a mean of **0.082 °C** and a median of **0.096 °C**. The standard deviation was 0.026 °C.
#     *   **Insight:** Similar to peak intensity, the mean intensity of the Bay of Bengal event was much higher than the average mean intensity in the Arabian Sea, indicating a stronger average heat anomaly throughout its duration.
# 
# 5.  **Cumulative Intensity (°C days):**
#     *   **Bay of Bengal:** The single event had a cumulative intensity of **4.425 °C days**.
#     *   **Arabian Sea:** Cumulative intensities ranged from **0.210 °C days to 0.895 °C days**, with a mean of **0.655 °C days** and a median of **0.861 °C days**. The standard deviation was 0.386 °C days.
#     *   **Insight:** The cumulative heat stress in the Bay of Bengal, despite having only one event, was significantly greater than the total cumulative stress from all three events in the Arabian Sea. This highlights the substantial impact of the longer, more intense event in the Bay of Bengal.
# 
# **Overall Conclusion for the Analyzed Period:**
# 
# During the analyzed period (March-September 2024), the **Arabian Sea** experienced more frequent, but generally shorter and less intense Marine Heatwave events. In contrast, the **Bay of Bengal**, while having fewer events, experienced a single, highly prolonged and significantly more intense MHW event that contributed to a much higher overall cumulative heat stress.


# %% [markdown]
# ### Comparative Analysis of MHW Metrics Across Regions
# 
# **Summary of Findings:**
# 
# The `regional_comparison` DataFrame provides a detailed statistical comparison of Marine Heatwave (MHW) metrics between the **Bay of Bengal** and the **Arabian Sea** for the specified period (March 1, 2024 to September 30, 2024).
# 
# **Key Observations:**
# 
# 1.  **MHW Frequency:**
#     *   **Bay of Bengal:** Experienced 1 MHW event.
#     *   **Arabian Sea:** Experienced 3 MHW events.
#     *   **Insight:** The Arabian Sea had a higher frequency of MHW events during this period compared to the Bay of Bengal, suggesting that MHW conditions were more common or fragmented in the Arabian Sea.
# 
# 2.  **Duration (duration_days):**
#     *   **Bay of Bengal:** The single MHW event had a duration of **22 days**.
#     *   **Arabian Sea:** MHW durations ranged from **4 to 9 days**, with a mean of approximately **7.33 days** and a median of **9 days**. The standard deviation was 2.89 days.
#     *   **Insight:** The MHW event in the Bay of Bengal was significantly longer than any individual event recorded in the Arabian Sea during the same period. This indicates that while less frequent, the Bay of Bengal can experience more prolonged heat stress.
# 
# 3.  **Peak Intensity (°C):**
#     *   **Bay of Bengal:** The single event had a peak intensity of **0.478 °C**.
#     *   **Arabian Sea:** Peak intensities ranged from **0.099 °C to 0.174 °C**, with a mean of **0.145 °C** and a median of **0.163 °C**. The standard deviation was 0.04 °C.
#     *   **Insight:** The MHW event in the Bay of Bengal reached a considerably higher peak intensity, almost three times that of the average peak intensity in the Arabian Sea. This suggests the Bay of Bengal event was more intense at its peak.
# 
# 4.  **Mean Intensity (°C):**
#     *   **Bay of Bengal:** The single event had a mean intensity of **0.201 °C**.
#     *   **Arabian Sea:** Mean intensities ranged from **0.052 °C to 0.099 °C**, with a mean of **0.082 °C** and a median of **0.096 °C**. The standard deviation was 0.026 °C.
#     *   **Insight:** Similar to peak intensity, the mean intensity of the Bay of Bengal event was much higher than the average mean intensity in the Arabian Sea, indicating a stronger average heat anomaly throughout its duration.
# 
# 5.  **Cumulative Intensity (°C days):**
#     *   **Bay of Bengal:** The single event had a cumulative intensity of **4.425 °C days**.
#     *   **Arabian Sea:** Cumulative intensities ranged from **0.210 °C days to 0.895 °C days**, with a mean of **0.655 °C days** and a median of **0.861 °C days**. The standard deviation was 0.386 °C days.
#     *   **Insight:** The cumulative heat stress in the Bay of Bengal, despite having only one event, was significantly greater than the total cumulative stress from all three events in the Arabian Sea. This highlights the substantial impact of the longer, more intense event in the Bay of Bengal.
# 
# **Overall Conclusion for the Analyzed Period:**
# 
# During the analyzed period (March-September 2024), the **Arabian Sea** experienced more frequent, but generally shorter and less intense Marine Heatwave events. In contrast, the **Bay of Bengal**, while having fewer events, experienced a single, highly prolonged and significantly more intense MHW event that contributed to a much higher overall cumulative heat stress.


# %% [markdown]
# ### Comparative Analysis of MHW Metrics Across Regions
# 
# **Summary of Findings:**
# 
# The `regional_comparison` DataFrame provides a detailed statistical comparison of Marine Heatwave (MHW) metrics between the **Bay of Bengal** and the **Arabian Sea** for the specified period (March 1, 2024 to September 30, 2024).
# 
# **Key Observations:**
# 
# 1.  **MHW Frequency:**
#     *   **Bay of Bengal:** Experienced 1 MHW event.
#     *   **Arabian Sea:** Experienced 3 MHW events.
#     *   **Insight:** The Arabian Sea had a higher frequency of MHW events during this period compared to the Bay of Bengal, suggesting that MHW conditions were more common or fragmented in the Arabian Sea.
# 
# 2.  **Duration (duration_days):**
#     *   **Bay of Bengal:** The single MHW event had a duration of **22 days**.
#     *   **Arabian Sea:** MHW durations ranged from **4 to 9 days**, with a mean of approximately **7.33 days** and a median of **9 days**. The standard deviation was 2.89 days.
#     *   **Insight:** The MHW event in the Bay of Bengal was significantly longer than any individual event recorded in the Arabian Sea during the same period. This indicates that while less frequent, the Bay of Bengal can experience more prolonged heat stress.
# 
# 3.  **Peak Intensity (°C):**
#     *   **Bay of Bengal:** The single event had a peak intensity of **0.478 °C**.
#     *   **Arabian Sea:** Peak intensities ranged from **0.099 °C to 0.174 °C**, with a mean of **0.145 °C** and a median of **0.163 °C**. The standard deviation was 0.04 °C.
#     *   **Insight:** The MHW event in the Bay of Bengal reached a considerably higher peak intensity, almost three times that of the average peak intensity in the Arabian Sea. This suggests the Bay of Bengal event was more intense at its peak.
# 
# 4.  **Mean Intensity (°C):**
#     *   **Bay of Bengal:** The single event had a mean intensity of **0.201 °C**.
#     *   **Arabian Sea:** Mean intensities ranged from **0.052 °C to 0.099 °C**, with a mean of **0.082 °C** and a median of **0.096 °C**. The standard deviation was 0.026 °C.
#     *   **Insight:** Similar to peak intensity, the mean intensity of the Bay of Bengal event was much higher than the average mean intensity in the Arabian Sea, indicating a stronger average heat anomaly throughout its duration.
# 
# 5.  **Cumulative Intensity (°C days):**
#     *   **Bay of Bengal:** The single event had a cumulative intensity of **4.425 °C days**.
#     *   **Arabian Sea:** Cumulative intensities ranged from **0.210 °C days to 0.895 °C days**, with a mean of **0.655 °C days** and a median of **0.861 °C days**. The standard deviation was 0.386 °C days.
#     *   **Insight:** The cumulative heat stress in the Bay of Bengal, despite having only one event, was significantly greater than the total cumulative stress from all three events in the Arabian Sea. This highlights the substantial impact of the longer, more intense event in the Bay of Bengal.
# 
# **Overall Conclusion for the Analyzed Period:**
# 
# During the analyzed period (March-September 2024), the **Arabian Sea** experienced more frequent, but generally shorter and less intense Marine Heatwave events. In contrast, the **Bay of Bengal**, while having fewer events, experienced a single, highly prolonged and significantly more intense MHW event that contributed to a much higher overall cumulative heat stress.


# %% [markdown]
# ## Visualize Cross-Regional MHW Metrics
# 
# ### Subtask:
# Generate comparative visualizations to highlight differences and similarities between regions, including bar charts and box plots for key MHW metrics. Ensure all plots are clearly labeled and include legends.


# %% [markdown]
# **Reasoning**:
# To visualize the comparative MHW metrics, I will generate bar charts for the mean values and box plots for the distributions of key MHW metrics across different regions, ensuring clear labels and titles for each plot as requested.
# 


# %% Cell 55
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics to plot
metrics_to_plot = [
    ('duration_days', 'Duration (days)'),
    ('peak_intensity_c', 'Peak Intensity (°C)'),
    ('mean_intensity_c', 'Mean Intensity (°C)'),
    ('cumulative_intensity_c', 'Cumulative Intensity (°C days)')
]

print("Generating comparative bar charts...")
# 1. Bar Charts for Mean Metrics
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12), dpi=300)
axes = axes.flatten()

for i, (metric, ylabel) in enumerate(metrics_to_plot):
    ax = axes[i]
    # Corrected indexing: Access the specific metric's 'mean' level
    regional_comparison[(metric, 'mean')].plot(kind='bar', ax=ax, rot=0, color=['skyblue', 'lightcoral'])
    ax.set_title(f'Mean {ylabel} by Region', fontsize=14, fontweight='bold', fontname=CONFIG['font'])
    ax.set_xlabel('Region', fontsize=12, fontname=CONFIG['font'])
    ax.set_ylabel(f'Mean {ylabel}', fontsize=12, fontname=CONFIG['font'])
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=9)

plt.tight_layout()
plt.suptitle('Comparative Analysis of Mean MHW Metrics by Region', y=1.02, fontsize=18, fontweight='bold', fontname=CONFIG['font'])
plt.show()

print("Generating comparative box plots...")
# 2. Box Plots for Metric Distributions
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12), dpi=300)
axes = axes.flatten()

for i, (metric, ylabel) in enumerate(metrics_to_plot):
    ax = axes[i]
    sns.boxplot(data=all_regions_mhw_df, x='region', y=metric, ax=ax, palette='Set2')
    ax.set_title(f'Distribution of MHW {ylabel} by Region', fontsize=14, fontweight='bold', fontname=CONFIG['font'])
    ax.set_xlabel('Region', fontsize=12, fontname=CONFIG['font'])
    ax.set_ylabel(ylabel, fontsize=12, fontname=CONFIG['font'])
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.suptitle('Distribution of MHW Metrics by Region (Box Plots)', y=1.02, fontsize=18, fontweight='bold', fontname=CONFIG['font'])
plt.show()


# %% [markdown]
# ## Final Task
# 
# ### Subtask:
# Summarize the findings from the cross-regional MHW analysis. Discuss the key differences and similarities observed in MHW characteristics (duration, intensity, frequency) across the compared regions, and provide insights into potential regional drivers or impacts.


# %% [markdown]
# ## Summary:
# 
# ### Q&A
# The cross-regional Marine Heatwave (MHW) analysis revealed significant differences in MHW characteristics between the Bay of Bengal and the Arabian Sea for the period of March 1, 2024, to September 30, 2024.
# 
# *   **Frequency**: The Arabian Sea experienced a higher frequency of MHW events, with 3 events recorded, compared to the Bay of Bengal which had only 1 MHW event.
# *   **Duration**: The single MHW event in the Bay of Bengal was significantly longer, lasting 22 days. In contrast, MHW events in the Arabian Sea were shorter, ranging from 4 to 9 days, with a mean duration of approximately 7.33 days.
# *   **Intensity**:
#     *   **Peak Intensity**: The Bay of Bengal's MHW event reached a considerably higher peak intensity of 0.478 °C, which is almost three times the average peak intensity in the Arabian Sea (mean of 0.145 °C).
#     *   **Mean Intensity**: Similarly, the mean intensity of the Bay of Bengal event was 0.201 °C, notably higher than the average mean intensity in the Arabian Sea (mean of 0.082 °C).
# *   **Cumulative Intensity**: The overall heat stress in the Bay of Bengal was substantially greater, with a cumulative intensity of 4.425 °C days from its single event. This far surpassed the cumulative intensity of all three Arabian Sea events combined (mean of 0.655 °C days).
# 
# ### Data Analysis Key Findings
# *   Two new geographical regions, the Bay of Bengal (lat\_bounds: slice(23, 5), lon\_bounds: slice(78, 100)) and the Arabian Sea (lat\_bounds: slice(25, 5), lon\_bounds: slice(50, 75)), were successfully defined for cross-regional MHW analysis.
# *   The data processing and plotting functions were successfully adapted to dynamically process and visualize MHW data for user-selected regions, including region-specific latitude/longitude bounds and names in plot titles.
# *   MHW metrics (duration, peak intensity, mean intensity, cumulative intensity) were calculated for both regions for the period of March 1, 2024, to September 30, 2024.
# *   The Arabian Sea experienced 3 MHW events, while the Bay of Bengal experienced 1 MHW event during the analyzed period.
# *   The single MHW event in the Bay of Bengal was notably longer (22 days) and more intense (peak intensity of 0.478 °C, mean intensity of 0.201 °C, cumulative intensity of 4.425 °C days) compared to the Arabian Sea's events (mean duration of $\sim$7.33 days, mean peak intensity of $\sim$0.145 °C, mean mean intensity of $\sim$0.082 °C, mean cumulative intensity of $\sim$0.655 °C days).
# *   Comparative bar charts and box plots were generated to effectively visualize these differences in MHW metrics across the regions.
# 
# ### Insights or Next Steps
# *   **Insights**: The Bay of Bengal appears to be prone to less frequent but significantly more prolonged and intense MHW events, leading to higher overall cumulative heat stress, while the Arabian Sea experiences more frequent but generally shorter and less intense events. This difference could be driven by regional oceanographic conditions, such as stronger stratification or less mixing in the Bay of Bengal, or influence from monsoon patterns.
# *   **Next Steps**: Investigate the underlying physical oceanographic and atmospheric drivers (e.g., ocean currents, stratification, monsoon variability, wind patterns) contributing to the observed differences in MHW characteristics between the two regions. This could involve correlating MHW metrics with other environmental parameters.
