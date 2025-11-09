# =========================
# Yield Curve Project - Full Pipeline
# Local ingest -> monthly filter -> transforms -> NS factors -> dataset splits
# =========================

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import least_squares

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = r"C:\Users\98910\Desktop\python project github"  # <- your folder
START, END = "2003-01-01", "2025-06-01"

# optional: which series you ideally expect (used only for column ordering/report)
REQUIRED = [
    "GS6","GS1","GS2","GS3","GS5","GS7","GS10","GS20","GS30",
    "FEDFUNDS","TB3MS",
    "CPIAUCSL","PCEPI","T5YIE",
    "INDPRO","UNRATE","PAYEMS",
    "STLFSI2","NFCI",
]

# -------------------------
# HELPERS (local ingest)
# -------------------------
def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception:
            return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")

def _find_date_col(df: pd.DataFrame):
    for c in ["DATE","date","Date","observation_date","TIME","time"]:
        if c in df.columns:
            return c
    first = df.columns[0]
    try:
        pd.to_datetime(df[first])
        return first
    except Exception:
        return None

def _pick_value_cols(df: pd.DataFrame, series_name: str):
    if series_name in df.columns:
        return [series_name]
    numerics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numerics or [c for c in df.columns if c != series_name]

# -------------------------
# 0) INGEST LOCAL FILES -> monthly combined
# -------------------------
print("Reading from:", DATA_DIR)
print("Files seen:", [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.csv','.xlsx','.xls'))])

frames = {}
for fn in os.listdir(DATA_DIR):
    if not fn.lower().endswith((".csv",".xlsx",".xls")):
        continue
    series_name = os.path.splitext(fn)[0]              # filename becomes column name
    path = os.path.join(DATA_DIR, fn)

    try:
        df0 = _read_any(path)
    except Exception as e:
        print(f"[SKIP] {fn}: could not read ({e})")
        continue

    date_col = _find_date_col(df0)
    if date_col is None:
        print(f"[SKIP] {fn}: no date column detected")
        continue

    df0[date_col] = pd.to_datetime(df0[date_col], errors="coerce")
    df0 = df0.dropna(subset=[date_col]).set_index(date_col).sort_index()

    value_cols = _pick_value_cols(df0, series_name)
    for c in value_cols:
        df0[c] = pd.to_numeric(df0[c], errors="coerce")
    if len(value_cols) > 1:
        # pick the column with the fewest NaNs
        best_c = min(value_cols, key=lambda x: df0[x].isna().sum())
        value_cols = [best_c]

    ser = df0[value_cols[0]].rename(series_name)
    # align all to month-end and clip to your window
    ser = ser.resample("M").last()
    ser = ser.loc[(ser.index >= START) & (ser.index <= END)]
    frames[series_name] = ser

if not frames:
    raise SystemExit("No series loaded. Check the folder and filenames/DATE columns.")

combined = pd.concat(frames.values(), axis=1).sort_index()

# order with REQUIRED first (if present)
ordered_cols = [s for s in REQUIRED if s in combined.columns] + \
               [c for c in combined.columns if c not in REQUIRED]
combined = combined[ordered_cols]

found = set(combined.columns)
missing = [s for s in REQUIRED if s not in found]

print("\nLoaded shape:", combined.shape)
print("Date range:", combined.index.min().date(), "→", combined.index.max().date())
print("\nFound series:", sorted(found))
print("Missing (from REQUIRED):", missing)

combined_path = Path(DATA_DIR) / "combined_2003_2025M.csv"
combined.to_csv(combined_path, index=True)
print(f"\nSaved cleaned monthly dataset → {combined_path}")

print("\nPreview (head):")
print(combined.head(6).to_string())
print("\nPreview (tail):")
print(combined.tail(6).to_string())

# -------------------------
# 1) TRANSFORMS (YoY log) + SPREADS
# -------------------------
df = pd.read_csv(combined_path, parse_dates=[0], index_col=0).sort_index()
df = df.apply(pd.to_numeric, errors="coerce")

def yoy_log(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    x = x.where(x > 0)                 # avoid log of non-positive
    return 100.0 * np.log(x).diff(12)  # first 12 months NaN by design

macro = df.copy()

for name in ["CPIAUCSL","PCEPI","INDPRO","PAYEMS"]:
    if name in macro.columns:
        macro[f"{name}_YOY"] = yoy_log(macro[name])

if {"GS2","GS10"}.issubset(macro.columns):
    macro["SPREAD_2s10s"] = macro["GS2"] - macro["GS10"]
if {"TB3MS","GS10"}.issubset(macro.columns):
    macro["SPREAD_3m10y"] = macro["TB3MS"] - macro["GS10"]
if {"GS5","GS30"}.issubset(macro.columns):
    macro["SPREAD_5s30s"] = macro["GS5"] - macro["GS30"]

# accept either STLFSI4 (new) or STLFSI2 (old)—keep whichever exists
macro_cols = [
    "FEDFUNDS","TB3MS","T5YIE","UNRATE",
    "CPIAUCSL_YOY","PCEPI_YOY","INDPRO_YOY","PAYEMS_YOY",
    "NFCI","STLFSI4","STLFSI2",
    "SPREAD_2s10s","SPREAD_3m10y","SPREAD_5s30s"
]
macro_cols = [c for c in macro_cols if c in macro.columns]
macro_out = Path(DATA_DIR) / "macro_transformed.csv"
macro[macro_cols].to_csv(macro_out, index=True)
print("Saved:", macro_out)

# -------------------------
# 2) NELSON–SIEGEL FACTORS
# -------------------------
def ns_loadings(maturity_years: np.ndarray, tau: float) -> np.ndarray:
    m = np.asarray(maturity_years, dtype=float)
    x = m / tau
    x = np.where(x == 0.0, 1e-8, x)
    L1 = np.ones_like(m)
    L2 = (1 - np.exp(-x)) / x
    L3 = L2 - np.exp(-x)
    return np.vstack([L1, L2, L3]).T

def fit_ns_single(maturities, yields, tau_init=2.0):
    y = np.asarray(yields, dtype=float)
    def residuals(theta):
        b1, b2, b3, tau = theta
        if tau <= 1e-4:
            return y * 1e6
        X = ns_loadings(maturities, tau)
        return y - (X @ np.array([b1, b2, b3]))
    X0 = ns_loadings(maturities, tau_init)
    b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    theta0 = np.r_[b0, tau_init]
    res = least_squares(residuals, theta0,
                        bounds=([-np.inf]*3 + [1e-3], [np.inf]*3 + [50.0]))
    return res.x  # b1,b2,b3,tau

# map maturities you actually have in your CSV
maturity_map = {
    "1Y": "GS1", "2Y": "GS2", "3Y": "GS3", "5Y": "GS5",
    "7Y": "GS7", "10Y": "GS10", "20Y": "GS20", "30Y": "GS30",
}
maturity_map = {k:v for k,v in maturity_map.items() if v in df.columns}
tenors = np.array([float(k[:-1]) for k in maturity_map.keys()])
yc_cols = [maturity_map[k] for k in maturity_map.keys()]

rows = []
for dt, row in df[yc_cols].dropna().iterrows():
    b1, b2, b3, tau = fit_ns_single(tenors, row.values, tau_init=2.0)
    rows.append((dt, b1, b2, b3, tau))

ns = pd.DataFrame(rows, columns=["date","beta1","beta2","beta3","tau"]).set_index("date").sort_index()
ns_out = Path(DATA_DIR) / "ns_factors.csv"
ns.to_csv(ns_out, index=True)
print("Saved:", ns_out)

# -------------------------
# 3) DATASET (features + beta lags + targets) & SPLITS
# -------------------------
macroX = pd.read_csv(macro_out, parse_dates=[0], index_col=0).sort_index()

def add_lags(df_in, cols, L=3):
    out = df_in.copy()
    for c in cols:
        for l in range(1, L+1):
            out[f"{c}_L{l}"] = out[c].shift(l)
    return out

betas_with_lags = add_lags(ns[["beta1","beta2","beta3"]], ["beta1","beta2","beta3"], L=3)
targets = ns[["beta1","beta2","beta3"]].shift(-1).rename(
    columns={"beta1":"beta1_t1","beta2":"beta2_t1","beta3":"beta3_t1"}
)

X = pd.concat([macroX, betas_with_lags], axis=1)
dataset = pd.concat([X, targets], axis=1).dropna().copy()

train_end = "2009-12-31"
val_end   = "2014-12-31"

train = dataset.loc[:train_end]
val   = dataset.loc[train_end:].loc[:val_end]
test  = dataset.loc[val_end:]

OUT_DIR = Path(DATA_DIR) / "dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(train.to_csv(OUT_DIR / "train.csv"),
 val.to_csv(OUT_DIR / "val.csv"),
 test.to_csv(OUT_DIR / "test.csv"),
 dataset.to_csv(OUT_DIR / "full_dataset.csv"))

print("Saved dataset splits to:", OUT_DIR)

# -------------------------
# (Optional) AR baseline helper
# Uncomment after installing: pip install statsmodels
# -------------------------
import statsmodels.api as sm
full = pd.read_csv(OUT_DIR / "full_dataset.csv", parse_dates=[0], index_col=0)
betas = full[["beta1","beta2","beta3"]]
def ar_forecast_one(s: pd.Series, start_idx: int, end_idx: int, max_p: int = 12, ic: str = "aic"):
     preds = []
     idx = s.index
     for i in range(start_idx, end_idx):  # expanding window
         train_s = s.iloc[:i+1]
         best, best_ic = None, float("inf")
         for p in range(1, max_p+1):
             res = sm.tsa.ARIMA(train_s, order=(p,0,0)).fit()
             crit = getattr(res, ic)
             if crit < best_ic:
                 best, best_ic = res, crit
         preds.append(best.forecast(1)[0])
     return pd.Series(preds, index=idx[start_idx:end_idx])

# -------------------------
# (Optional) ML starter
# Uncomment after: pip install scikit-learn
# -------------------------
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
full = pd.read_csv(OUT_DIR / "full_dataset.csv", parse_dates=[0], index_col=0)
feat_cols = [c for c in full.columns if not c.endswith("_t1")]
X = full[feat_cols]
y = full[["beta1_t1","beta2_t1","beta3_t1"]]
is_train = (full.index <= "2009-12-31")
is_val   = (full.index > "2009-12-31") & (full.index <= "2014-12-31")
is_test  = (full.index > "2014-12-31")
pipe = Pipeline([
("scaler", StandardScaler(with_mean=True, with_std=True)),
("model", MultiOutputRegressor(ElasticNet(max_iter=20000)))
])
pipe.fit(X[is_train], y[is_train])


