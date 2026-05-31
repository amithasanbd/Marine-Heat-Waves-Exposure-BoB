#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 Spatial Mhw Maps

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
# Colab shell command removed for repository reproducibility: !pip install cartopy
import cartopy


# %% [markdown]
# # **MHW Intensity Spatial Maps**


# %% Cell 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as mticker
import matplotlib as mpl


# 1) Load Excel file

file_path = str(OUTPUT_ROOT / "BoB_All_Events_1995_2025.xlsx")
df_raw = pd.read_excel(file_path, sheet_name="All_Events_1995_2025")


# 2) Settings

seasons = ['Pre-Monsoon', 'Monsoon', 'Post-Monsoon', 'Winter']
periods = ['1995-2000', '2001-2005', '2006-2010', '2011-2015', '2016-2020', '2021-2025']

# BoB spatial extent
bo_lat_min, bo_lat_max = 5, 23
bo_lon_min, bo_lon_max = 80, 100

lon_grid = np.linspace(bo_lon_min, bo_lon_max, 300)
lat_grid = np.linspace(bo_lat_min, bo_lat_max, 300)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# Intensity color levels
levels = np.arange(0, 2.25, 0.25)
norm = BoundaryNorm(levels, ncolors=256)
cbar_ticks = np.array([0, 0.5, 1.0, 1.5, 2.0])
cmap = "hot_r"

# Tick + label styling
tick_font = 12
tick_weight = "bold"
title_font = 18
rowlabel_font = 15


# 3) Functions

def year_to_period(y):
    y = int(y)
    if 1995 <= y <= 2000:
        return "1995-2000"
    elif 2001 <= y <= 2005:
        return "2001-2005"
    elif 2006 <= y <= 2010:
        return "2006-2010"
    elif 2011 <= y <= 2015:
        return "2011-2015"
    elif 2016 <= y <= 2020:
        return "2016-2020"
    elif 2021 <= y <= 2025:
        return "2021-2025"
    else:
        return np.nan

def month_to_season(month):
    # Pre-monsoon = Mar, Apr, May
    if month in [3, 4, 5]:
        return "Pre-Monsoon"
    # Monsoon = Jun, Jul, Aug
    elif month in [6, 7, 8]:
        return "Monsoon"
    # Post-monsoon = Oct, Nov
    elif month in [10, 11]:
        return "Post-Monsoon"
    # Winter = Dec, Jan, Feb
    elif month in [12, 1, 2]:
        return "Winter"
    # September excluded as per your definition
    else:
        return np.nan

def season_from_event_dates(start_date, end_date):
    # Use midpoint date of the event duration
    midpoint = start_date + (end_date - start_date) / 2
    return month_to_season(midpoint.month)


# 4) Prepare data

# Keep only needed columns
df = df_raw[['lon', 'lat', 'date_start', 'date_end', 'intensity_mean']].copy()

# Convert dates
df['date_start'] = pd.to_datetime(df['date_start'], errors='coerce')
df['date_end'] = pd.to_datetime(df['date_end'], errors='coerce')

# Drop missing rows
df = df.dropna(subset=['lon', 'lat', 'date_start', 'date_end', 'intensity_mean'])

# Create year from start date
df['year'] = df['date_start'].dt.year.astype(int)

# Create season from event midpoint between start and end
df['Season'] = df.apply(lambda row: season_from_event_dates(row['date_start'], row['date_end']), axis=1)

# Create period
df['period'] = df['year'].apply(year_to_period)

# Drop rows outside required periods and rows with excluded months
df = df.dropna(subset=['period', 'Season'])

print("Available years:", sorted(df['year'].unique()))
print("Available seasons:", df['Season'].unique())

# 1) Mean intensity per (lon, lat, season, year)
annual = (
    df.groupby(['lon', 'lat', 'Season', 'period', 'year'], as_index=False)['intensity_mean']
      .mean()
)

# 2) Mean across years within each period
df_period = (
    annual.groupby(['lon', 'lat', 'Season', 'period'], as_index=False)['intensity_mean']
          .mean()
)


# %% Cell 4
# 5) Plot

nrows = len(periods)
ncols = len(seasons)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(20, 26),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

fig.subplots_adjust(
    left=0.06, right=0.90, top=0.95, bottom=0.06,
    wspace=0.12, hspace=0.10
)

# Colorbar handle
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

for row_idx, period in enumerate(periods):
    for col_idx, season in enumerate(seasons):
        ax = axes[row_idx, col_idx]
        ax.set_extent([bo_lon_min, bo_lon_max, bo_lat_min, bo_lat_max], crs=ccrs.PlateCarree())

        df_block = df_period[
            (df_period['period'] == period) &
            (df_period['Season'] == season)
        ]
        df_points = df_block[['lon', 'lat', 'intensity_mean']].dropna()

        if not df_points.empty:
            grid_cubic = griddata(
                (df_points['lon'], df_points['lat']),
                df_points['intensity_mean'],
                (lon_mesh, lat_mesh),
                method='cubic'
            )

            grid_nearest = griddata(
                (df_points['lon'], df_points['lat']),
                df_points['intensity_mean'],
                (lon_mesh, lat_mesh),
                method='nearest'
            )

            grid_filled = np.where(np.isnan(grid_cubic), grid_nearest, grid_cubic)

            ax.contourf(
                lon_mesh, lat_mesh, grid_filled,
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend='both',
                transform=ccrs.PlateCarree()
            )
        else:
            ax.text(
                0.5, 0.5, 'No Data',
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=12,
                weight='bold'
            )

        # Land
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                'physical', 'land', '10m',
                facecolor='lightgray',
                edgecolor='black'
            )
        )

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (col_idx == 0)
        gl.bottom_labels = (row_idx == nrows - 1)
        gl.xlabel_style = {'size': tick_font, 'weight': tick_weight}
        gl.ylabel_style = {'size': tick_font, 'weight': tick_weight}
        gl.xlocator = mticker.FixedLocator(np.arange(bo_lon_min, bo_lon_max + 1, 5))
        gl.ylocator = mticker.FixedLocator(np.arange(bo_lat_min, bo_lat_max + 1, 5))

        # Column titles
        if row_idx == 0:
            ax.set_title(season, fontsize=title_font, weight='bold', pad=10)

        # Row labels
        if col_idx == 0:
            ax.text(
                -0.22, 0.5, period,
                transform=ax.transAxes,
                rotation=90,
                fontsize=rowlabel_font,
                weight='bold',
                ha='center',
                va='center'
            )

        ax.set_xticks([])
        ax.set_yticks([])


# 6) Colorbar

cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.64])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=cbar_ticks, spacing='uniform', extend='both')
cbar.set_label('Average Mean MHWs Intensity (°C) Per Year', fontsize=15, weight='bold')
cbar.ax.tick_params(labelsize=12)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')

# 7) Save

out_png = str(OUTPUT_ROOT / "BoB_MHW_Intensity_Seasonal_5yr_1200dpi.png")
plt.savefig(out_png, dpi=1200, bbox_inches="tight", facecolor="white")

out_pdf = str(OUTPUT_ROOT / "BoB_MHW_Intensity_Seasonal_5yr_1080dpi.pdf")
plt.savefig(out_pdf, dpi=1080, bbox_inches="tight")

plt.show()

print("Saved PNG:", out_png)
print("Saved PDF:", out_pdf)


# %% [markdown]
# # **MHW Frequency Spatial Maps**


# %% Cell 6
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as mticker
import matplotlib as mpl


# 1) Load Excel

file_path = str(OUTPUT_ROOT / "BoB_All_Events_1995_2025.xlsx")
df_raw = pd.read_excel(file_path, sheet_name="All_Events_1995_2025")


# 2) Settings

seasons = ['Pre-monsoon', 'Monsoon', 'Post-monsoon', 'Winter']
periods = ['1995-2000', '2001-2005', '2006-2010', '2011-2015', '2016-2020', '2021-2025']

# BoB extent
bo_lat_min, bo_lat_max = 5, 23
bo_lon_min, bo_lon_max = 80, 100

lon_grid = np.linspace(bo_lon_min, bo_lon_max, 300)
lat_grid = np.linspace(bo_lat_min, bo_lat_max, 300)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# Frequency color levels
levels = np.arange(0, 13, 1)
norm = BoundaryNorm(levels, ncolors=256)
cbar_ticks = np.array([0, 2, 4, 6, 8, 10, 12])

# Fonts
tick_font = 12
title_font = 18
rowlabel_font = 15
tick_weight = 'bold'


# 3) Helpers

def year_to_period(y):
    y = int(y)
    if 1995 <= y <= 2000:
        return "1995-2000"
    if 2001 <= y <= 2005:
        return "2001-2005"
    if 2006 <= y <= 2010:
        return "2006-2010"
    if 2011 <= y <= 2015:
        return "2011-2015"
    if 2016 <= y <= 2020:
        return "2016-2020"
    if 2021 <= y <= 2025:
        return "2021-2025"
    return np.nan

def month_to_season(month):
    # Pre-monsoon = Mar, Apr, May
    if month in [3, 4, 5]:
        return "Pre-monsoon"
    # Monsoon = Jun, Jul, Aug
    elif month in [6, 7, 8]:
        return "Monsoon"
    # Post-monsoon = Oct, Nov
    elif month in [10, 11]:
        return "Post-monsoon"
    # Winter = Dec, Jan, Feb
    elif month in [12, 1, 2]:
        return "Winter"
    # September excluded
    else:
        return np.nan

def season_from_event_dates(start_date, end_date):
    midpoint = start_date + (end_date - start_date) / 2
    return month_to_season(midpoint.month)


# 4) Prepare data (frequency)

needed_cols = ['lon', 'lat', 'date_start', 'date_end']
df = df_raw[needed_cols].copy()

# Parse dates
df['date_start'] = pd.to_datetime(df['date_start'], errors='coerce')
df['date_end'] = pd.to_datetime(df['date_end'], errors='coerce')

# Drop missing rows
df = df.dropna(subset=['lon', 'lat', 'date_start', 'date_end'])

# Build year and season from dates
df['year'] = df['date_start'].dt.year.astype(int)
df['Season'] = df.apply(
    lambda row: season_from_event_dates(row['date_start'], row['date_end']),
    axis=1
)

# Build period
df['period'] = df['year'].apply(year_to_period)
df = df.dropna(subset=['period', 'Season'])

print("Available years:", sorted(df['year'].unique()))
print("Available seasons:", df['Season'].unique())

# Frequency = count of rows (events) at each lon/lat/season/period
df_period = (
    df.groupby(['lon', 'lat', 'Season', 'period'])
      .size()
      .reset_index(name='frequency')
)


# %% Cell 7
# 5) Plot

nrows = len(periods)
ncols = len(seasons)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(20, 26),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

fig.subplots_adjust(
    left=0.06, right=0.90, top=0.95, bottom=0.06,
    wspace=0.12, hspace=0.10
)

# Colorbar handle
sm = mpl.cm.ScalarMappable(norm=norm, cmap='turbo')
sm.set_array([])

for row_idx, period in enumerate(periods):
    for col_idx, season_title in enumerate(seasons):
        ax = axes[row_idx, col_idx]
        ax.set_extent([bo_lon_min, bo_lon_max, bo_lat_min, bo_lat_max], crs=ccrs.PlateCarree())

        df_block = df_period[
            (df_period['period'] == period) &
            (df_period['Season'] == season_title)
        ]
        df_points = df_block[['lon', 'lat', 'frequency']].dropna()

        if not df_points.empty:
            grid_cubic = griddata(
                (df_points['lon'], df_points['lat']),
                df_points['frequency'],
                (lon_mesh, lat_mesh),
                method='cubic'
            )
            grid_nearest = griddata(
                (df_points['lon'], df_points['lat']),
                df_points['frequency'],
                (lon_mesh, lat_mesh),
                method='nearest'
            )
            grid_filled = np.where(np.isnan(grid_cubic), grid_nearest, grid_cubic)

            ax.contourf(
                lon_mesh, lat_mesh, grid_filled,
                levels=levels, cmap='turbo', norm=norm,
                extend='both', transform=ccrs.PlateCarree()
            )
        else:
            ax.text(
                0.5, 0.5, 'No Data',
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=12,
                weight='bold'
            )

        # Land overlay
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                'physical', 'land', '10m',
                facecolor='lightgray',
                edgecolor='black'
            )
        )

        # Gridlines + label control
        gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (col_idx == 0)
        gl.bottom_labels = (row_idx == nrows - 1)

        gl.xlabel_style = {'size': tick_font, 'weight': tick_weight}
        gl.ylabel_style = {'size': tick_font, 'weight': tick_weight}
        gl.xlocator = mticker.FixedLocator(np.arange(bo_lon_min, bo_lon_max + 1, 5))
        gl.ylocator = mticker.FixedLocator(np.arange(bo_lat_min, bo_lat_max + 1, 5))

        # Column titles
        if row_idx == 0:
            ax.set_title(season_title, fontsize=title_font, weight='bold', pad=10)

        # Row labels
        if col_idx == 0:
            ax.text(
                -0.22, 0.5, period,
                transform=ax.transAxes, rotation=90,
                fontsize=rowlabel_font, weight='bold',
                ha='center', va='center'
            )

        ax.set_xticks([])
        ax.set_yticks([])


# 6) Colorbar

cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.64])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=cbar_ticks, spacing='uniform', extend='both')
cbar.set_label('MHWs Frequency (days) Per Year', fontsize=15, weight='bold')

cbar.ax.tick_params(labelsize=12)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')


# 7) Save

out_png = str(OUTPUT_ROOT / "BoB_MHW_Frequency_Seasonal_5yr_1200dpi.png")
plt.savefig(out_png, dpi=1200, bbox_inches="tight", facecolor="white")

out_pdf = str(OUTPUT_ROOT / "BoB_MHW_Frequency_Seasonal_5yr_1080dpi.pdf")
plt.savefig(out_pdf, dpi=1080, bbox_inches="tight")

plt.show()

print("Saved PNG:", out_png)
print("Saved PDF:", out_pdf)


# %% Cell 8

