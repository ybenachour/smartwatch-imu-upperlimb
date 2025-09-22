[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17164809.svg)](https://doi.org/10.5281/zenodo.17164809)
[![Dataset DOI](https://img.shields.io/badge/Mendeley%20Data-10.17632/s86tdtmcc2.1-blue)](https://doi.org/10.17632/s86tdtmcc2.1)

# smartwatch-imu-upperlimb
Reproducible preprocessing &amp; feature-extraction pipeline for a smartwatch IMU upper-limb dataset (20 Hz). Includes segmentation (1.0 s &amp; 2.56 s), time/frequency features, jerk/magnitude, and a 46-feature subset. Pairs with data on Mendeley Data and a Zenodo DOI release.

# Smartwatch IMU for Upper-Limb Movement Analysis

Reproducible preprocessing & feature-extraction pipeline for our **Data in Brief** dataset (supporting the EAAI paper).

## Data
Download data from Mendeley Data and place under `data/` (not tracked in git):
- DOI: **10.17632/XXXXXX.1** (replace once issued)

## Quick start
```bash
# conda (recommended)
conda env create -f env/environment.yml
conda activate upperlimb

# or pip
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r env/requirements.txt

## How to cite

**Software (this release):**  
Benachour, Y. (2025). *ybenachour/smartwatch-imu-upperlimb: HemiPhysioData Feature Engineering Pipeline v1.0.0* (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.17164809

**Dataset:**  
Benachour, Y., Rehman, M., & Flitti, F. (2025). *Smartwatch IMU Dataset for Upper-Limb Movement Analysis: Raw, Processed, and Feature Matrices* (v1.0). Mendeley Data. https://doi.org/10.17632/s86tdtmcc2.1

**Article:**  
Benachour, Y., Rehman, M., & Flitti, F. (2025). Towards improved human arm movement analysis… *Engineering Applications of Artificial Intelligence*, 156, 111194. https://doi.org/10.1016/j.engappai.2025.111194

