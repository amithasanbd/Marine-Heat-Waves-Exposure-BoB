# Reviewer Reproducibility Guide

1. Download the GitHub/Zenodo software release.
2. Download the paired Zenodo processed-data archive.
3. Extract the processed data into the `data/` folder, or set `MHW_DATA_ROOT` to the extracted dataset path.
4. Create the environment using `environment.yml`.
5. Run the numbered scripts or notebooks in order.
6. Compare generated figures/tables in `outputs/` with the manuscript figures and supplementary materials.

The original Colab shell install commands and Google Drive mount commands were intentionally removed to make the package suitable for local, GitHub, and Zenodo use.
