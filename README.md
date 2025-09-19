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
