#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01 Study Area Map

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
# # **FINAL WITH CURRENTS**


# %% Cell 2
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LightSource
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean

# =============================================================================
# 1. TYPOGRAPHY & PUBLICATION STYLE SETTINGS (Nature Journal Standards)
# =============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "figure.dpi": 150,        # Colab display DPI
    "savefig.dpi": 1200,      # Output DPI for print publication
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05
})

# Bay of Bengal domain bounds
lon_min, lon_max = 79.5, 100.5
lat_min, lat_max = 4.5, 23.5

# =============================================================================
# 2. DATA IMPORT & PRE-PROCESSING
# =============================================================================
# Path to the mounted GEBCO 2025 Bathymetry data
file_path = str(OUTPUT_ROOT / "gebco_2025_n23.0_s5.0_w80.0_e100.0.nc")

try:
    ds = xr.open_dataset(file_path)
    elevation = ds['elevation']
    lons = ds['lon']
    lats = ds['lat']

    # Mask land values to apply colors strictly to the ocean
    ocean_bathy = elevation.where(elevation <= 0)

except FileNotFoundError:
    print(f"Error: Data file not found at {file_path}. Please mount your Google Drive.")
    # Safe fallback for structural testing
    lons = np.linspace(lon_min, lon_max, 500)
    lats = np.linspace(lat_min, lat_max, 500)
    Lons, Lats = np.meshgrid(lons, lats)
    ocean_bathy = xr.DataArray(np.random.uniform(-4500, 0, Lons.shape), coords=[lats, lons], dims=['lat', 'lon'])

# =============================================================================
# 3. FIGURE SETUP & MAP PROJECTION
# =============================================================================
fig = plt.figure(figsize=(9, 10), facecolor='white')
proj = ccrs.Mercator() # Mercator projection gives the classic ArcGIS Pro proportion
data_proj = ccrs.PlateCarree()
ax = plt.axes(projection=proj)
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_proj)

# Text path effects (white halo) for absolute readability against 3D backgrounds
txt_effect = [PathEffects.withStroke(linewidth=3.0, foreground='white')]
red_effect = [PathEffects.withStroke(linewidth=3.0, foreground='#ffe6e6')]

# =============================================================================
# 4. CARTOGRAPHIC LAYERING (ArcGIS Pro 3D Hillshade Vibe)
# =============================================================================
# A. Elegant Coastal Lands
land_feature = cfeature.NaturalEarthFeature(
    'physical', 'land', '10m',
    edgecolor='#333333', facecolor='#E3E1D3', linewidth=0.6
)
ax.add_feature(land_feature, zorder=4)
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':', edgecolor='#777777', linewidth=0.5, zorder=5)

# Explicitly add cfeature.COASTLINE with distinct styling
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#555555', zorder=4.5)

# Add Rivers Feature
river_feature = cfeature.NaturalEarthFeature(
    'physical', 'rivers_lake_centerlines', '10m',
    edgecolor='#005b96', # Dark blue for rivers
    facecolor='none',
    linewidth=0.0,
    zorder=6 # Ensure rivers are visible above land and bathy, below text
)
ax.add_feature(river_feature)

# Add Lakes Feature
lake_feature = cfeature.NaturalEarthFeature(
    'physical', 'lakes', '10m',
    edgecolor='#005b96', # Dark blue outline
    facecolor='#ADD8E6', # Light blue for lake bodies
    linewidth=0.0,
    zorder=5 # Below rivers, above land
)
ax.add_feature(lake_feature)

# Add Administrative Boundaries
admin_boundary_feature = cfeature.NaturalEarthFeature(
    'cultural', 'admin_1_states_provinces_lines', '10m',
    edgecolor='#8B4513', # SaddleBrown for admin lines
    facecolor='none',
    linestyle=':', # Dotted lines for distinction
    linewidth=0.0,
    zorder=7 # Above rivers and lakes
)
ax.add_feature(admin_boundary_feature)

# B. 3D Hillshade Generation (The "Vibe")
# Calculate terrain lighting to give physical depth to trenches and shelves
ls = LightSource(azdeg=315, altdeg=45)
# Fill NaNs with 0 temporarily just for the structural shading algorithm
bathy_for_shade = ocean_bathy.fillna(0).values
# Create an RGB shaded array blending physical shadows with cmocean colors
shaded_rgb = ls.shade(bathy_for_shade, cmap=cmocean.cm.deep_r, vmin=-4500, vmax=0, blend_mode='soft')

# Plot the 3D RGB array
img = ax.imshow(shaded_rgb, extent=[lons.min().item(), lons.max().item(), lats.min().item(), lats.max().item()],
                transform=data_proj, origin='lower', zorder=1)

# C. 100m to 500m Contour Distribution (Shelf tracking)
contours = ax.contour(lons, lats, ocean_bathy, levels=[-500, -200, -100],
                      colors=['#ffffff', '#ffcc00', '#ff6600'], linewidths=[0.5, 0.7, 0.9],
                      linestyles='solid', alpha=0.85, transform=data_proj, zorder=2)
ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f m', colors='black')

# Custom legend proxies for depth contours
ax.plot([], [], color='#ff6600', linewidth=0.9, label='100m Coastal Shelf')
ax.plot([], [], color='#ffcc00', linewidth=0.7, label='200m Break')
ax.plot([], [], color='#ffffff', linewidth=0.5, label='500m Deep Margin')

# =============================================================================
# 5. SUB-BASIN BOUNDARIES & MHW HOTSPOT MAPPING
# =============================================================================
# Basin latitudinal dividers
ax.plot([lon_min, lon_max], [16.0, 16.0], color='#d9534f', linestyle='--', linewidth=1.5, alpha=0.9, transform=data_proj, zorder=6)
ax.plot([lon_min, lon_max], [10.0, 10.0], color='#d9534f', linestyle='--', linewidth=1.5, alpha=0.9, transform=data_proj, zorder=6)

# Basin Annotations
ax.text(98.5, 19.5, 'NORTHERN BoB\n(River-Influenced)', transform=data_proj, fontsize=9,
        fontweight='bold', color='#1a1a1a', ha='center', va='center', path_effects=txt_effect, zorder=7)
ax.text(98.5, 13.0, 'CENTRAL BoB\n(Mixing Zone)', transform=data_proj, fontsize=9,
        fontweight='bold', color='#1a1a1a', ha='center', va='center', path_effects=txt_effect, zorder=7)
ax.text(98.5, 7.5, 'SOUTHERN BoB\n(Open Ocean)', transform=data_proj, fontsize=9,
        fontweight='bold', color='#1a1a1a', ha='center', va='center', path_effects=txt_effect, zorder=7)

# --- Major Spatial Findings Annotations ---
# 1. Core Post-2010 MHW Hotspot
theta = np.linspace(0, 2*np.pi, 200)
x_hot = 89.5 + 4.0 * np.cos(theta)
y_hot = 13.5 + 2.0 * np.sin(theta)
ax.plot(x_hot, y_hot, color='#cc0000', linewidth=2.5, linestyle='-.', transform=data_proj, zorder=6)
ax.text(89.5, 13.5, 'Region 3:\nPermanent Post-2010\nMHW Hotspot\n(Max Thermal Exposure)',
        transform=data_proj, fontsize=8, fontweight='bold', color='#cc0000', ha='center', va='center', path_effects=red_effect, zorder=7)

# 2. Coral & Andaman Ecosystem Exposure
ax.text(81.5, 8.0, 'Region 1:\nCoral Reef\nThermal Vulnerability', transform=data_proj, fontsize=7.5,
        fontweight='bold', color='#0044cc', ha='center', path_effects=txt_effect, zorder=7)
ax.text(94.5, 11.5, 'Region 2:\nAndaman Ecosystem\nStress Zone', transform=data_proj, fontsize=7.5,
        fontweight='bold', color='#0044cc', ha='center', path_effects=txt_effect, zorder=7)

# =============================================================================
# 5.5. OCEAN CURRENTS
# =============================================================================
# East India Coastal Current (EICC) - approximate path
eicc_lons = [80.5, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 91.5]
eicc_lats = [8.0, 10.0, 12.0, 13.5, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 22.5]
ax.plot(eicc_lons, eicc_lats, color='purple', linewidth=2, linestyle='-', transform=data_proj, zorder=9)
# EICC arrows
for i in [1, 4, 7, 10]: # Add arrows along the path
    ax.plot(eicc_lons[i], eicc_lats[i], marker='>', markersize=7, color='purple', transform=data_proj, zorder=9)
ax.text(85.0, 14.5, 'East India Coastal Current', transform=data_proj, fontsize=8, fontweight='bold',
        color='purple', ha='center', va='bottom', path_effects=txt_effect, zorder=9)

# Southwest Monsoon Surface Current (SWMSC) - approximate path
swmsc_lons = [80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0]
swmsc_lats = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
ax.plot(swmsc_lons, swmsc_lats, color='darkgreen', linewidth=2, linestyle='-', transform=data_proj, zorder=9)
# SWMSC arrows
for i in [1, 3, 5]: # Add arrows along the path
    ax.plot(swmsc_lons[i], swmsc_lats[i], marker='>', markersize=7, color='darkgreen', transform=data_proj, zorder=9)
ax.text(85.0, 6.0, 'Southwest Monsoon Surface Current', transform=data_proj, fontsize=8, fontweight='bold',
        color='darkgreen', ha='center', va='top', path_effects=txt_effect, zorder=9)

# =============================================================================
# 6. MAJOR RIVER DISCHARGE LOCATIONS
# =============================================================================
rivers = {
    'Ganges-Brahmaputra': (89.75, 21.68),
    'Mahanadi': (86.73, 20.29),
    'Godavari': (81.99, 16.29),
    'Krishna': (80.95, 15.72),
    'Cauvery': (79.86, 11.14),
    'Irrawaddy': (95.25, 15.80)
}

for name, (lon, lat) in rivers.items():
    ax.plot(lon, lat, marker='v', color='#005b96', markersize=8, markeredgecolor='white',
            markeredgewidth=1.0, transform=data_proj, zorder=8)
    # Dynamic text offset
    offset_x = -0.8 if lon < 85 else 0.8
    offset_y = 0.3 if lat > 20 else -0.3
    if name == 'Ganges-Brahmaputra': offset_x, offset_y = -0.5, -0.6

    ax.text(lon + offset_x, lat + offset_y, f"{name}\nDischarge", transform=data_proj, fontsize=7.5,
            fontstyle='italic', color='#003b6f', ha='center', path_effects=txt_effect, zorder=8)

# =============================================================================
# 7. GRIDLINES, AXES, & 3D COLORBAR
# =============================================================================
gl = ax.gridlines(crs=data_proj, draw_labels=True, linewidth=0.5, color='gray', alpha=0.4, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 9, 'color': 'black'}
gl.ylabel_style = {'size': 9, 'color': 'black'}

# Create a proxy mapping for the colorbar since we used imshow for the 3D RGB array
sm = plt.cm.ScalarMappable(cmap=cmocean.cm.deep_r, norm=plt.Normalize(vmin=-4500, vmax=0))
sm._A = []
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.03, shrink=0.7, aspect=25)
cbar.set_label('3D Topographic Bathymetry Depth (m)', rotation=270, labelpad=20, weight='bold', fontsize=10)
cbar.ax.tick_params(labelsize=9)

# =============================================================================
# 8. MAP SCALE & NORTH ARROW
# =============================================================================
# Professional North Arrow
ax.text(0.06, 0.94, 'N\n▲', transform=ax.transAxes, ha='center', va='center',
        fontsize=14, weight='bold', color='#111111', path_effects=txt_effect, zorder=10)

# Map Scale (~300 km representation)
sb_lon, sb_lat = 90, 6.0
ax.plot([sb_lon, sb_lon + 2.7], [sb_lat, sb_lat], color='k', linewidth=3, transform=data_proj, zorder=9)
ax.plot([sb_lon, sb_lon + 2.7], [sb_lat, sb_lat], color='w', linewidth=1.5, linestyle='--', transform=data_proj, zorder=10)
ax.text(sb_lon + 1.35, sb_lat - 0.4, '300 km', transform=data_proj, fontsize=8, weight='bold', ha='center', path_effects=txt_effect, zorder=9)

# =============================================================================
# 9. FINAL TOUCHES & Q1 EXPORT
# =============================================================================
plt.title('Synthesis of Bay of Bengal Marine Heatwaves (1995–2025)\nSpatial Hotspots, Riverine Inputs, and 3D Basin Morphology',
          fontsize=12, weight='bold', loc='center', pad=15)

ax.legend(loc='lower left', fontsize=8, framealpha=0.95, edgecolor='black', fancybox=False)

# Export as both PNG (for easy viewing) and TIFF (for publication submission)
plt.savefig('BoB_MHW_Synthesis_3D_Map_1200dpi.png', dpi=1200, bbox_inches='tight', facecolor='white')
plt.savefig('BoB_MHW_Synthesis_3D_Map_1200dpi.tif', dpi=1200, format='tiff', pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')

print("✅ Map Generated Successfully: 1200 DPI Outputs saved.")
plt.show()


# %% [markdown]
# # **Final 3D Study Area Map**


# %% Cell 4
# ===============================
# Install Scientific Mapping Libraries
# ===============================

# Colab shell command removed for repository reproducibility: !pip install cmocean
# Colab shell command removed for repository reproducibility: !pip install cartopy
# Colab shell command removed for repository reproducibility: !pip install netCDF4
# Colab shell command removed for repository reproducibility: !pip install xarray
# Colab shell command removed for repository reproducibility: !pip install shapely
# Colab shell command removed for repository reproducibility: !pip install geopandas
# Colab shell command removed for repository reproducibility: !pip install matplotlib-scalebar


# %% Cell 5
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LightSource
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean

# =============================================================================
# 1. TYPOGRAPHY & PUBLICATION STYLE SETTINGS (Nature Journal Standards)
# =============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 13,
    "figure.dpi": 300,        # Colab display DPI
    "savefig.dpi": 1200,      # Output DPI for print publication
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05
})

# Bay of Bengal domain bounds
lon_min, lon_max = 79.5, 100.5
lat_min, lat_max = 4.5, 23.5

# =============================================================================
# 2. DATA IMPORT & PRE-PROCESSING
# =============================================================================
file_path = str(OUTPUT_ROOT / "gebco.nc")

try:
    ds = xr.open_dataset(file_path)
    elevation = ds['elevation']
    lons = ds['lon']
    lats = ds['lat']
    ocean_bathy = elevation.where(elevation <= 0)
except FileNotFoundError:
    print(f"Error: Data file not found at {file_path}. Using fallback array for testing.")
    lons = np.linspace(lon_min, lon_max, 500)
    lats = np.linspace(lat_min, lat_max, 500)
    Lons, Lats = np.meshgrid(lons, lats)
    ocean_bathy = xr.DataArray(np.random.uniform(-4500, 0, Lons.shape), coords=[lats, lons], dims=['lat', 'lon'])

# =============================================================================
# 3. FIGURE SETUP & MAP PROJECTION
# =============================================================================
fig = plt.figure(figsize=(10, 11), facecolor='white')
proj = ccrs.Mercator()
data_proj = ccrs.PlateCarree()
ax = plt.axes(projection=proj)
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_proj)

txt_effect = [PathEffects.withStroke(linewidth=3.0, foreground='white')]

# =============================================================================
# 4. CARTOGRAPHIC LAYERING (3D Bathymetry & Coastal Features)
# =============================================================================
# Land and borders
land_feature = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='#333333', facecolor='#EAE8DC', linewidth=0.6)
ax.add_feature(land_feature, zorder=4)
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':', edgecolor='#888888', linewidth=0.5, zorder=5)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#444444', zorder=4.5)

# 3D Hillshade bathymetry
ls = LightSource(azdeg=315, altdeg=45)
bathy_for_shade = ocean_bathy.fillna(0).values
shaded_rgb = ls.shade(bathy_for_shade, cmap=cmocean.cm.deep_r, vmin=-4500, vmax=0, blend_mode='soft')
img = ax.imshow(shaded_rgb, extent=[lons.min().item(), lons.max().item(), lats.min().item(), lats.max().item()],
                transform=data_proj, origin='lower', zorder=1)

# Bathymetric contours (Shelf tracking)
contours = ax.contour(lons, lats, ocean_bathy, levels=[-500, -200, -100],
                      colors=['#E0E0E0', '#ffcc00', '#ff6600'], linewidths=[0.5, 0.7, 1.0],
                      linestyles='solid', alpha=0.85, transform=data_proj, zorder=2)
ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f m', colors='black')

# =============================================================================
# 5. ZONAL BUFFERS, BORDERS & MHW HOTSPOT MAPPING
# =============================================================================
# A. Zonal Buffers (Faint translucent color blocks representing study regions)
ax.fill_between([lon_min, lon_max], 16.0, 23.5, color='#f0ad4e', alpha=0.08, transform=data_proj, zorder=2) # Northern Red hint
ax.fill_between([lon_min, lon_max], 10.0, 16.0, color='#d9534f', alpha=0.08, transform=data_proj, zorder=2) # Central Orange hint
ax.fill_between([lon_min, lon_max], 4.5, 10.0,  color='#5bc0de', alpha=0.08, transform=data_proj, zorder=2) # Southern Blue hint

# B. Zonal separator lines
ax.plot([lon_min, lon_max], [16.0, 16.0], color='#d35400', linestyle='--', linewidth=1.5, alpha=0.8, transform=data_proj, zorder=6)
ax.plot([lon_min, lon_max], [10.0, 10.0], color='#2980b9', linestyle='--', linewidth=1.5, alpha=0.8, transform=data_proj, zorder=6)

# C. Q1 Style Thematic Annotations (using bounding boxes for pure readability)
bbox_n = dict(boxstyle="round,pad=0.5", facecolor="#fffcf5", edgecolor="#f0ad4e", alpha=0.9, linewidth=1.2)
bbox_c = dict(boxstyle="round,pad=0.5", facecolor="#fff5f5", edgecolor="#d9534f", alpha=0.9, linewidth=1.2)
bbox_s = dict(boxstyle="round,pad=0.5", facecolor="#f5fafd", edgecolor="#5bc0de", alpha=0.9, linewidth=1.2)

ax.text(98, 19.5, 'Northern BoB (16–22° N):\nStratified Biogeochemical\nCrisis Zone',
        transform=data_proj, fontsize=8.5, fontweight='bold', color='#b35900', ha='center', va='center', bbox=bbox_n, zorder=10)

ax.text(98, 13.0, 'Central BoB (10–16° N):\nPersistent Post-2010\nMHW Hotspot',
        transform=data_proj, fontsize=8.5, fontweight='bold', color='#8b0000', ha='center', va='center', bbox=bbox_c, zorder=10)

ax.text(98, 7.5, 'Southern BoB (5–10° N):\nEquatorial Teleconnection\nCorridor',
        transform=data_proj, fontsize=8.5, fontweight='bold', color='#004080', ha='center', va='center', bbox=bbox_s, zorder=10)

# =============================================================================
# 6. HOTSPOT GLOW & ECOSYSTEM CALLOUTS
# =============================================================================
# Hotspot Glow Ellipse
hotspot = patches.Ellipse((89.5, 13.5), width=8.0, height=4.0, angle=0,
                          linewidth=2, edgecolor='#cc0000', facecolor='#ff4d4d',
                          alpha=0.3, linestyle='-.', transform=data_proj, zorder=6)
ax.add_patch(hotspot)

ax.text(89.5, 13.5, 'Core Thermal Exposure\n(Max Intensity)',
        transform=data_proj, fontsize=8.5, fontweight='bold', color='#990000', ha='center', va='center', path_effects=txt_effect, zorder=11)

# Ecosystem text
ax.text(82, 9.2, 'Coral Reef\nThermal Vulnerability', transform=data_proj, fontsize=7.5,
        fontweight='bold', color='#0044cc', ha='center', path_effects=txt_effect, zorder=11)
ax.text(94.5, 11, 'Andaman Ecosystem\nStress Zone', transform=data_proj, fontsize=7.5,
        fontweight='bold', color='#0044cc', ha='center', path_effects=txt_effect, zorder=11)

# =============================================================================
# 7. MAJOR RIVER DISCHARGE LOCATIONS
# =============================================================================
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m', edgecolor='#005b96', facecolor='none', linewidth=0.7, zorder=6))

rivers = {
    'Ganges-Brahmaputra-Meghna': (90.5, 21.68),
    'Godavari': (81.99, 16.29),
    'Irrawaddy': (95.25, 15.80)
}
for name, (lon, lat) in rivers.items():
    ax.plot(lon, lat, marker='v', color='#005b96', markersize=10, markeredgecolor='white', markeredgewidth=1.0, transform=data_proj, zorder=9)
    ax.text(lon, lat - 0.5, f"{name}\nDischarge", transform=data_proj, fontsize=7.5,
            fontstyle='italic', fontweight='bold', color='#003b6f', ha='center', path_effects=txt_effect, zorder=10)

# =============================================================================
# 8. GRIDLINES, AXES, COLORBAR & MASTER LEGEND
# =============================================================================
gl = ax.gridlines(crs=data_proj, draw_labels=True, linewidth=0.4, color='gray', alpha=0.3, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 9, 'color': '#333333'}; gl.ylabel_style = {'size': 9, 'color': '#333333'}

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmocean.cm.deep_r, norm=plt.Normalize(vmin=-4500, vmax=0))
sm._A = []
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.03, shrink=0.68, aspect=25)
cbar.set_label('3D Topographic Bathymetry Depth (m)', rotation=270, labelpad=18, weight='bold', fontsize=10)
cbar.ax.tick_params(labelsize=8)

# Unified Master Legend
legend_elements = [
    patches.Patch(facecolor='#e67e22', alpha=0.3, label='Northern Zone Buffer'),
    patches.Patch(facecolor='#e74c3c', alpha=0.2, label='Central Zone Buffer'),
    patches.Patch(facecolor='#2980b9', alpha=0.3, label='Southern Zone Buffer'),
    plt.Line2D([0], [0], color='#ff6600', lw=1.0, label='100m Coastal Shelf'),
    plt.Line2D([0], [0], color='#ffcc00', lw=0.7, label='200m Continental Break'),
    plt.Line2D([0], [0], color='#E0E0E0', lw=0.5, label='500m Deep Margin'),
    plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#005b96', markersize=8, label='Major River Discharge')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=7.5, framealpha=0.95, edgecolor='#aaaaaa', title='Map Features', title_fontsize=8)

# =============================================================================
# 9. MAP SCALE & NORTH ARROW
# =============================================================================
ax.text(0.06, 0.94, '▲\nN', transform=ax.transAxes, ha='center', va='center', fontsize=14, weight='bold', color='#111111', path_effects=txt_effect, zorder=12)

sb_lon, sb_lat = 89, 5.7
ax.plot([sb_lon, sb_lon + 2.7], [sb_lat, sb_lat], color='k', linewidth=5, transform=data_proj, zorder=15)
ax.plot([sb_lon, sb_lon + 2.7], [sb_lat, sb_lat], color='w', linewidth=2.5, linestyle='--', transform=data_proj, zorder=20)
ax.text(sb_lon + 1.35, sb_lat - 0.4, '300 km', transform=data_proj, fontsize=10, weight='bold', ha='center', path_effects=txt_effect, zorder=15)

# =============================================================================
# 10. FINAL TITLE & Q1 EXPORT
# =============================================================================
plt.title('Zonal Anatomy of Bay of Bengal Marine Heatwaves (1995–2025)\nHotspots, Teleconnections, and Biogeochemical Stress Regimes',
          fontsize=12, weight='bold', loc='center', pad=15)

plt.savefig('BoB_MHW_Synthesis_Zonal_Map_1200dpi.png', dpi=1200, bbox_inches='tight', facecolor='white')
plt.savefig('BoB_MHW_Synthesis_Zonal_Map_1200dpi.tif', dpi=1200, format='tiff', pil_kwargs={"compression": "tiff_lzw"}, bbox_inches='tight')

print("✅ Zonal Map Generated Successfully: 1200 DPI Outputs saved.")
plt.show()
