# Source Data Access

This repository is designed to be paired with a Zenodo dataset containing processed and figure-ready data. Large raw source products should not be committed to GitHub.

## Public source datasets used in the manuscript

1. NOAA OISST v2.1 daily sea-surface temperature, used for MHW detection and SST anomalies.
2. NOAA CPC Oceanic Niño Index, used for ENSO teleconnection analysis.
3. NOAA PSL Dipole Mode Index, used for IOD teleconnection analysis.
4. Copernicus Marine physical product / ARMOR3D / MULTIOBS_GLO_PHY_TSUV_3D, used for temperature, salinity, mixed-layer depth, density, and Brunt–Väisälä frequency diagnostics.
5. Copernicus Marine biogeochemical product, used for nitrate, phosphate, silicate, chlorophyll-a, and dissolved oxygen composites.

## Recommended Zenodo processed-data archive

Archive the following in a separate Zenodo dataset record:

- Bay of Bengal domain and sub-basin masks.
- Marine heatwave event catalogue, 1995–2025.
- Annual and seasonal MHW metrics.
- Trend, Mann-Kendall, Sen-slope, and Pettitt outputs.
- Teleconnection and phase-composite summary tables.
- Upper-ocean stratification diagnostics.
- Biogeochemical event-composite statistics.
- Figure-ready data for all main and supplementary figures.

## Path configuration

Set `MHW_DATA_ROOT` and `MHW_OUTPUT_ROOT` environment variables if your data are stored outside this repository.
