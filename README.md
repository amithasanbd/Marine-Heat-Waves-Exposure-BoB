# Bay of Bengal Marine Heatwave Exposure Regime

This repository contains the refined Python/Colab code package prepared for GitHub + Zenodo release for the manuscript:

**An Emerging Marine Heatwave Exposure Regime in the Bay of Bengal: Stratification-Mediated Persistence and Compound Heat-Oxygen Risk**

## Authors
Md. Amit Hasan; Mohan Kumar Das; Sheikh Fahim Faysal Sowrav; Farjana Akther Koly; Md. Zuhaib Kabir; S.M. Mustafizur Rahman; Muhammad Sajid Anam Hoque; Maisha Salwoa Haque.

## Repository purpose
This package provides a professional, publication-oriented organization of the manuscript code set for AGU/Earth's Future-style Open Research compliance. The original Google Colab-exported files were converted into clean `.py` scripts and cleaned `.ipynb` notebooks, with outputs removed, Colab-specific commands commented, and personal Google Drive paths replaced by repository-relative path variables.

## Study region
Bay of Bengal domain: **5–22°N, 80–100°E**, divided into northern, central, and southern sub-basins.

## Main data sources
- NOAA OISST v2.1 daily sea-surface temperature, 1995–2025.
- NOAA CPC Oceanic Niño Index, 1995–2025.
- NOAA PSL Dipole Mode Index, 1995–2025.
- Copernicus Marine physical product for temperature, salinity, and mixed-layer depth, 1995–2024.
- Copernicus Marine biogeochemical product for nitrate, phosphate, silicate, chlorophyll-a, and dissolved oxygen, 2000–2025.

## Recommended GitHub repository name
`Marine-Heat-Waves-Exposure-BoB`

## Installation

```bash
conda env create -f environment.yml
conda activate Marine-Heat-Waves-Exposure-BoB
```

or

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configure paths
By default, scripts use:

- `data/` for inputs and processed data
- `outputs/` for figures and tables

You can override these locations without editing scripts:

```bash
export MHW_DATA_ROOT=/path/to/your/data
export MHW_OUTPUT_ROOT=/path/to/your/outputs
```

On Windows PowerShell:

```powershell
$env:MHW_DATA_ROOT="D:\MHW_Data"
$env:MHW_OUTPUT_ROOT="D:\MHW_Outputs"
```

## Recommended execution order

1. `scripts/01_study_area_map.py`
2. `scripts/02_mhw_detection_main.py`
3. `scripts/03_mhw_indices_and_metrics.py`
4. `scripts/04_spatial_mhw_maps.py`
5. `scripts/05_trend_regime_shift_diagnostics.py`
6. `scripts/06_teleconnection_analysis.py`
7. `scripts/07_event_chronology_severity.py`
8. `scripts/08_bvf_n2_calculation.py`
9. `scripts/09_upper_ocean_persistence_metrics.py`
10. `scripts/10_upper_ocean_persistence_diagnostics.py`
11. `scripts/11_biogeochemical_event_composites.py`
12. `scripts/12_supplementary_figures.py`

## Important note on large data
Do **not** commit large raw NOAA/Copernicus data to GitHub. Put large processed and figure-ready data in Zenodo and cite the Zenodo DOI in the manuscript Open Research section.

## Zenodo release plan
- GitHub release: archive code as software DOI.
- Separate Zenodo dataset: archive processed data, masks, event catalogues, statistics, and figure-ready data.

## License
- Code: MIT License.
- Processed data recommended license: CC BY 4.0 in the Zenodo dataset record.

## Contact
Corresponding author: Md. Amit Hasan, `ocn19011.amit@gmail.com`.


## Official repository for Open Research
GitHub repository: https://github.com/amithasanbd/Marine-Heat-Waves-Exposure-BoB
Zenodo username/account: amithasanbd

## AGU/Earth's Future Open Research status
This repository is prepared for the code/software component of the AGU Open Research requirement. For submission, create:

1. a Zenodo **software DOI** by archiving the GitHub release, and
2. a separate Zenodo **dataset DOI** for processed data, figure-ready data, supplementary tables, and reproducibility outputs.

After both DOI records are created, replace `[https://doi.org/10.5281/zenodo.20477464]` and `[https://doi.org/10.5281/zenodo.20477464]` placeholders in `manuscript/open_research_statement.txt`, `manuscript/data_software_citation.txt`, `CITATION.cff`, and this README.

## Final release checklist
- Repository is public.
- Scripts compile with `python -m py_compile scripts/*.py`.
- Personal Google Drive paths and credentials are absent.
- Large raw NOAA/Copernicus files are not committed to GitHub.
- Processed and figure-ready data are uploaded to Zenodo as a dataset.
- GitHub release is archived in Zenodo as software.
- Both DOI records are cited in the manuscript Reference list.
