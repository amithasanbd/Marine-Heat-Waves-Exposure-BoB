# Step-by-step Open Research submission procedure for the MHW manuscript

Repository: https://github.com/amithasanbd/Marine-Heat-Waves-Exposure-BoB  
Zenodo username/account: amithasanbd  
Manuscript: **An Emerging Marine Heatwave Exposure Regime in the Bay of Bengal: Stratification-Mediated Persistence and Compound Heat-Oxygen Risk**

This folder is prepared as the GitHub + Zenodo Open Research package for AGU/Earth's Future submission. Follow the steps below exactly.

---

## 1. What you need to create

You need two public, citable records:

1. **GitHub repository + Zenodo software DOI** for the code and notebooks.
2. **Separate Zenodo dataset DOI** for processed data, figure-ready data, tables, masks, event catalogues, and reproducibility outputs.

The manuscript Open Research section must cite both records.

---

## 2. Final local folder check before upload

Before uploading this package, open the folder and confirm these files/folders exist:

```text
README.md
LICENSE
CITATION.cff
requirements.txt
environment.yml
.gitignore
config/
docs/
manuscript/
notebooks/
scripts/
src/
data/processed/
data/figure_ready/
data/tables/
outputs/
```

Delete any temporary files if present:

```text
__pycache__/
*.pyc
.ipynb_checkpoints/
credentials.json
token.json
.env
```

---

## 3. Decide what goes to GitHub and what goes to Zenodo

### Upload to GitHub

Upload:

```text
README.md
LICENSE
CITATION.cff
requirements.txt
environment.yml
.gitignore
config/
docs/
manuscript/
notebooks/
scripts/
src/
small sample/figure-ready tables if already included
```

### Upload to Zenodo dataset record

Upload processed and figure-ready data:

```text
data/processed/
data/figure_ready/
data/tables/
outputs/tables/
outputs/figures_main/
outputs/figures_supplementary/
README_data.md or docs/data_dictionary.md
```

### Do not upload to GitHub

Do not commit huge raw files or private credentials:

```text
large raw NOAA OISST files
large raw Copernicus Marine 3D files
Google Drive cache folders
passwords
API keys
tokens
credentials.json
token.json
```

Large processed data should go to Zenodo, not GitHub.

---

## 4. Test the package locally

Open terminal in this folder:

```bash
cd Marine-Heat-Waves-Exposure-BoB
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate Marine-Heat-Waves-Exposure-BoB
```

Or use pip:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

Check Python syntax:

```bash
python -m py_compile scripts/*.py src/*.py
```

Run scripts in this order if the required data files are present:

```bash
python scripts/01_study_area_map.py
python scripts/02_mhw_detection_main.py
python scripts/03_mhw_indices_and_metrics.py
python scripts/04_spatial_mhw_maps.py
python scripts/05_trend_regime_shift_diagnostics.py
python scripts/06_teleconnection_analysis.py
python scripts/07_event_chronology_severity.py
python scripts/08_bvf_n2_calculation.py
python scripts/09_upper_ocean_persistence_metrics.py
python scripts/10_upper_ocean_persistence_diagnostics.py
python scripts/11_biogeochemical_event_composites.py
python scripts/12_supplementary_figures.py
```

If your data are stored outside this repository, set paths without changing scripts:

Windows PowerShell:

```powershell
$env:MHW_DATA_ROOT="D:\MHW_Data"
$env:MHW_OUTPUT_ROOT="D:\MHW_Outputs"
```

macOS/Linux/Git Bash:

```bash
export MHW_DATA_ROOT=/path/to/MHW_Data
export MHW_OUTPUT_ROOT=/path/to/MHW_Outputs
```

---

## 5. Push the package to GitHub

Use your repository:

```text
https://github.com/amithasanbd/Marine-Heat-Waves-Exposure-BoB
```

From the local folder:

```bash
git init
git branch -M main
git remote add origin https://github.com/amithasanbd/Marine-Heat-Waves-Exposure-BoB.git
git add .
git commit -m "Prepare AGU Earths Future Open Research code package"
git push -u origin main
```

If the remote already exists:

```bash
git remote set-url origin https://github.com/amithasanbd/Marine-Heat-Waves-Exposure-BoB.git
git push -u origin main
```

Confirm the repository is public and opens correctly.

---

## 6. Create GitHub release v1.0.0

In GitHub:

1. Open the repository.
2. Click **Releases**.
3. Click **Draft a new release**.
4. Use:

```text
Tag: v1.0.0
Release title: AGU Earth’s Future submission release
```

5. Description:

```text
This release contains the reproducibility code package for the manuscript “An Emerging Marine Heatwave Exposure Regime in the Bay of Bengal: Stratification-Mediated Persistence and Compound Heat-Oxygen Risk.” It includes Python scripts, Jupyter/Colab notebooks, configuration files, documentation, source-data access instructions, Open Research templates, and citation metadata.
```

6. Click **Publish release**.

---

## 7. Connect GitHub to Zenodo

In Zenodo account `amithasanbd`:

1. Log in to Zenodo.
2. Open your profile/account menu.
3. Go to **Linked accounts** and connect GitHub if not already connected.
4. Go to the **GitHub** integration page.
5. Find:

```text
amithasanbd/Marine-Heat-Waves-Exposure-BoB
```

6. Turn archiving **ON**.
7. Zenodo should detect the GitHub release and archive it.
8. Wait until a Zenodo software DOI is created.

Record the software DOI here:

```text
SOFTWARE DOI = https://doi.org/10.5281/zenodo.xxxxxxx
```

---

## 8. Edit the Zenodo software metadata

For the software record, use:

```text
Upload type: Software
Title: Code for: An Emerging Marine Heatwave Exposure Regime in the Bay of Bengal
Version: v1.0.0
License: MIT License
Repository: https://github.com/amithasanbd/Marine-Heat-Waves-Exposure-BoB
```

Creators, in manuscript order:

```text
Md. Amit Hasan
Mohan Kumar Das
Sheikh Fahim Faysal Sowrav
Farjana Akther Koly
Md. Zuhaib Kabir
S.M. Mustafizur Rahman
Muhammad Sajid Anam Hoque
Maisha Salwoa Haque
```

Description:

```text
This software record archives the Python scripts, Jupyter/Google Colab notebooks, configuration files, documentation, and reproducibility workflow used for the manuscript “An Emerging Marine Heatwave Exposure Regime in the Bay of Bengal: Stratification-Mediated Persistence and Compound Heat-Oxygen Risk.” The code supports marine heatwave detection, annual and seasonal metric calculation, spatial hotspot mapping, trend and regime-shift diagnostics, ENSO/IOD teleconnection analysis, upper-ocean stratification diagnostics, biogeochemical event composites, and generation of main and supplementary figures.
```

Keywords:

```text
marine heatwaves; Bay of Bengal; NOAA OISST; Copernicus Marine; upper-ocean stratification; dissolved oxygen; ENSO; Indian Ocean Dipole; climate teleconnections; Earth’s Future
```

---

## 9. Create separate Zenodo dataset record

Create a new upload in Zenodo. This is separate from the software DOI.

Upload type:

```text
Dataset
```

Title:

```text
Processed data for: An Emerging Marine Heatwave Exposure Regime in the Bay of Bengal
```

Recommended uploaded ZIP name:

```text
Marine_Heat_Waves_Exposure_BoB_processed_data_v1_0.zip
```

Include:

```text
data/processed/
data/figure_ready/
data/tables/
outputs/figures_main/
outputs/figures_supplementary/
outputs/tables/
docs/data_dictionary.md
docs/source_data_access.md
```

License:

```text
Creative Commons Attribution 4.0 International (CC BY 4.0)
```

Description:

```text
This dataset contains the processed data, Bay of Bengal domain and sub-basin masks, marine heatwave event catalogue, annual and seasonal marine heatwave metrics, trend and regime-shift outputs, teleconnection statistics, upper-ocean stratification diagnostics, biogeochemical event-composite statistics, figure-ready data, and supplementary tables supporting the manuscript “An Emerging Marine Heatwave Exposure Regime in the Bay of Bengal: Stratification-Mediated Persistence and Compound Heat-Oxygen Risk.”
```

Record the data DOI here:

```text
DATA DOI = https://doi.org/10.5281/zenodo.xxxxxxx
```

---

## 10. Link the two Zenodo records

In the software record, add the data DOI as a related identifier:

```text
Relation: Is supplemented by / Is related to
Identifier: DATA DOI
```

In the dataset record, add the software DOI as a related identifier:

```text
Relation: Is supplemented by / Is related to
Identifier: SOFTWARE DOI
```

Also add the GitHub repository URL to the related identifiers if possible:

```text
https://github.com/amithasanbd/Marine-Heat-Waves-Exposure-BoB
```

---

## 11. Update repository files after DOI creation

Replace DOI placeholders in these files:

```text
README.md
CITATION.cff
manuscript/open_research_statement.txt
manuscript/data_software_citation.txt
manuscript/supplementary_code_inventory.md
docs/reviewer_reproducibility_guide.md
```

Search for placeholders:

```bash
grep -RIn "\[DATA DOI\]\|\[SOFTWARE DOI\]\|zenodo.xxxxxxx" .
```

After replacement:

```bash
git add .
git commit -m "Add Zenodo DOI metadata for AGU Open Research"
git push origin main
```

Create a second release:

```text
v1.0.1
```

Use this final release DOI in the manuscript if Zenodo creates a new version DOI. Prefer the concept DOI if you want a DOI that always points to the latest version, or the version DOI if you want an exact archived release.

---

## 12. Final Open Research text for the manuscript

Use this after replacing DOI placeholders:

```text
Open Research

The source datasets used in this study are publicly available from NOAA OISST v2.1, the NOAA Climate Prediction Center Oceanic Niño Index, the NOAA Physical Sciences Laboratory Dipole Mode Index, and Copernicus Marine Service physical and biogeochemical products. The processed Bay of Bengal domain masks, sub-basin masks, marine heatwave event catalogue, annual and seasonal marine heatwave metrics, trend and regime-shift outputs, teleconnection summary tables, upper-ocean stratification diagnostics, biogeochemical composite statistics, figure-ready data, and supplementary tables supporting this study are available at Zenodo: https://doi.org/[DATA DOI]. The Python scripts and Google Colab/Jupyter notebooks used to reproduce the analyses and figures are available through GitHub at https://github.com/amithasanbd/Marine-Heat-Waves-Exposure-BoB and archived at Zenodo: https://doi.org/[SOFTWARE DOI]. Processed data are released under the Creative Commons Attribution 4.0 International License, and code is released under the MIT License.
```

---

## 13. Add these references to the manuscript

Data citation:

```text
Hasan, M. A., Das, M. K., Sowrav, S. F. F., Koly, F. A., Kabir, M. Z., Rahman, S. M. M., Hoque, M. S. A., & Haque, M. S. (2026). Processed data for “An Emerging Marine Heatwave Exposure Regime in the Bay of Bengal: Stratification-Mediated Persistence and Compound Heat-Oxygen Risk” (Version 1.0) [Dataset]. Zenodo. https://doi.org/[DATA DOI]
```

Software citation:

```text
Hasan, M. A., Das, M. K., Sowrav, S. F. F., Koly, F. A., Kabir, M. Z., Rahman, S. M. M., Hoque, M. S. A., & Haque, M. S. (2026). Code for “An Emerging Marine Heatwave Exposure Regime in the Bay of Bengal: Stratification-Mediated Persistence and Compound Heat-Oxygen Risk” (Version 1.0) [Software]. Zenodo. https://doi.org/[SOFTWARE DOI]
```

---

## 14. AGU submission system text

Data availability field:

```text
Processed data and figure-ready data supporting this manuscript are available at Zenodo: https://doi.org/[DATA DOI].
```

Software availability field:

```text
Python scripts and Jupyter/Google Colab notebooks are available on GitHub at https://github.com/amithasanbd/Marine-Heat-Waves-Exposure-BoB and archived at Zenodo: https://doi.org/[SOFTWARE DOI].
```

Data license:

```text
CC BY 4.0
```

Software license:

```text
MIT License
```

---

## 15. Final checklist before AGU submission

```text
[ ] GitHub repository is public.
[ ] GitHub repository URL is correct: https://github.com/amithasanbd/Marine-Heat-Waves-Exposure-BoB
[ ] README.md explains workflow and DOI availability.
[ ] Scripts are numbered and professionally named.
[ ] Notebooks are output-stripped.
[ ] No __pycache__ or .pyc files remain.
[ ] No private Google Drive paths, credentials, tokens, or passwords are present.
[ ] Zenodo software DOI opens correctly.
[ ] Zenodo data DOI opens correctly.
[ ] Zenodo software record uses MIT License.
[ ] Zenodo data record uses CC BY 4.0.
[ ] Software and data records are linked in Zenodo.
[ ] README.md contains real DOI links.
[ ] CITATION.cff contains real software DOI.
[ ] Open Research section contains real DOI links.
[ ] Reference list includes data citation.
[ ] Reference list includes software citation.
[ ] Supplementary Code Inventory includes GitHub + Zenodo links.
[ ] Manuscript has no [insert DOI], [DATA DOI], [SOFTWARE DOI], or zenodo.xxxxxxx placeholders.
```
