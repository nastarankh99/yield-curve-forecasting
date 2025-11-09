
# Yield Curve Forecasting (FRED • 2003–2025)

This is my end-to-end project to forecast the U.S. Treasury yield curve with
**Nelson–Siegel factors** + **supervised models** (AR/VAR/Elastic Net/Random Forest/Gradient Boosting),
using **only public FRED data**. Monthly frequency, 2003-01 to 2025-06.

# Roadmap:
1) Load my FRED CSV/XLS files and make one clean monthly dataset.
2) Estimate **β₁ (level), β₂ (slope), β₃ (curvature)** and **τ** each month (robust NS fit).
3) Build features (lags of β’s + macro + spreads) to predict **next-month β’s**.
4) Train/tune **supervised models** with a realistic expanding window:
   - Train: 2003–2009  •  Val: 2010–2014  •  Test: 2015–2025
5) Compare models in **factor space** (errors on β’s) and **yield space** (errors on reconstructed yields).
6) Save tidy CSVs + plots for the report.

# Quick start 
```powershell
cd "C:\Users\98910\Desktop\python project github"

# 1) Create/activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install deps
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3) Run the pipeline (my script)
python .\yieldcurve_fc.py

