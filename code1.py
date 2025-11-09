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
# -------------------------
# 2) ROBUST NELSON–SIEGEL FACTORS (PATCH)
# -------------------------
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from pathlib import Path

def ns_loadings(maturity_years: np.ndarray, tau: float) -> np.ndarray:
    m = np.asarray(maturity_years, dtype=float)
    x = m / tau
    x = np.where(x == 0.0, 1e-8, x)
    L1 = np.ones_like(m)
    L2 = (1 - np.exp(-x)) / x
    L3 = L2 - np.exp(-x)
    return np.vstack([L1, L2, L3]).T

def fit_ns_single(maturities, yields, tau_bounds=(0.5, 20.0), tau_init=1.5, f_scale=0.5):
    y = np.asarray(yields, dtype=float)

    def residuals(theta):
        b1, b2, b3, tau = theta
        if tau <= 1e-4:
            return y * 1e6
        X = ns_loadings(maturities, tau)
        return y - (X @ np.array([b1, b2, b3]))

    # LS init at given tau
    X0 = ns_loadings(maturities, tau_init)
    b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    theta0 = np.r_[b0, tau_init]

    # robust fit
    res = least_squares(
        residuals,
        theta0,
        bounds=([-50, -200, -200, tau_bounds[0]], [50, 200, 200, tau_bounds[1]]),
        loss="huber",
        f_scale=f_scale,
        max_nfev=5000,
    )
    return res.x  # b1,b2,b3,tau

# ---- choose maturities: require core set present ----
core_cols = ["GS1","GS2","GS5","GS10","GS30"]
yc_available = [c for c in core_cols if c in df.columns]
if len(yc_available) < 5:
    raise SystemExit(f"Need all core maturities {core_cols}, have {yc_available}")

tenors = np.array([float(c[2:]) for c in yc_available])  # "GS10" -> 10.0

rows = []
skipped = []
for dt, row in df[yc_available].iterrows():
    vals = row.values.astype(float)

    # drop months with missing/obviously wrong inputs
    if np.isnan(vals).any():
        skipped.append((dt, "missing core tenor"))
        continue
    if (vals <= -1).any() or (vals > 25).any():  # simple sanity bounds (%)
        skipped.append((dt, f"abnormal yields: {vals}"))
        continue

    # first pass
    b1, b2, b3, tau = fit_ns_single(tenors, vals, tau_bounds=(0.5, 20.0), tau_init=1.5, f_scale=0.5)

    # sanity check factors; re-try with tighter tau if needed
    def insane(b1, b2, b3):
        return (abs(b1) > 50) or (abs(b2) > 50) or (abs(b3) > 50)
    if insane(b1, b2, b3):
        b1, b2, b3, tau = fit_ns_single(tenors, vals, tau_bounds=(0.8, 10.0), tau_init=2.0, f_scale=0.3)

    # if still insane, skip the month
    if insane(b1, b2, b3):
        skipped.append((dt, f"insane betas: {b1:.1f},{b2:.1f},{b3:.1f} (tau={tau:.2f})"))
        continue

    rows.append((dt, b1, b2, b3, tau))

ns = pd.DataFrame(rows, columns=["date","beta1","beta2","beta3","tau"]).set_index("date").sort_index()
ns_out = Path(DATA_DIR) / "ns_factors.csv"
ns.to_csv(ns_out, index=True)
print("Saved:", ns_out)

if skipped:
    print("\nNS skipped months (diagnostics):")
    for dt, why in skipped[:20]:
        print(dt.date(), "->", why)
    if len(skipped) > 20:
        print(f"... and {len(skipped)-20} more")


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
# OPTIONAL: AR baseline helper (keep if you installed statsmodels)
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
# ML: Elastic Net baseline (UNCOMMENTED)
# Requires: pip install scikit-learn
# -------------------------
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load the full assembled dataset (already loaded above as 'full', reuse)
# features = everything that is NOT a target (_t1)
feat_cols = [c for c in full.columns if not c.endswith("_t1")]
X = full[feat_cols]
y = full[["beta1_t1", "beta2_t1", "beta3_t1"]]

# time splits
is_train = (full.index <= "2009-12-31")
is_val   = (full.index > "2009-12-31") & (full.index <= "2014-12-31")
is_test  = (full.index > "2014-12-31")

# simple Elastic Net pipeline (you can grid-search later)
enet = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("model", MultiOutputRegressor(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000)))
])

# fit on TRAIN only
enet.fit(X[is_train], y[is_train])

# predictions
pred_train = pd.DataFrame(enet.predict(X[is_train]),
                          index=X[is_train].index,
                          columns=["beta1_t1_pred","beta2_t1_pred","beta3_t1_pred"])
pred_val   = pd.DataFrame(enet.predict(X[is_val]),
                          index=X[is_val].index,
                          columns=["beta1_t1_pred","beta2_t1_pred","beta3_t1_pred"])
pred_test  = pd.DataFrame(enet.predict(X[is_test]),
                          index=X[is_test].index,
                          columns=["beta1_t1_pred","beta2_t1_pred","beta3_t1_pred"])

# evaluation helpers
def rmse(a, f): 
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(f))**2)))
def mae(a, f):  
    return float(np.nanmean(np.abs(np.asarray(a) - np.asarray(f))))

def score_block(y_true_df, y_pred_df, label):
    r = {}
    for col_t, col_p in zip(["beta1_t1","beta2_t1","beta3_t1"],
                            ["beta1_t1_pred","beta2_t1_pred","beta3_t1_pred"]):
        r[f"{label}_RMSE_{col_t}"] = rmse(y_true_df[col_t], y_pred_df[col_p])
        r[f"{label}_MAE_{col_t}"]  = mae(y_true_df[col_t],  y_pred_df[col_p])
    return r

scores = {}
scores.update(score_block(y[is_train], pred_train, "train"))
scores.update(score_block(y[is_val],   pred_val,   "val"))
scores.update(score_block(y[is_test],  pred_test,  "test"))

# print a compact summary
print("\nElastic Net scores:")
for k in sorted(scores.keys()):
    print(f"{k}: {scores[k]:.4f}")

# save predictions
ART_DIR = Path(DATA_DIR) / "artifacts" / "preds"
ART_DIR.mkdir(parents=True, exist_ok=True)
pred_train.to_csv(ART_DIR / "enet_train_preds.csv")
pred_val.to_csv(ART_DIR / "enet_val_preds.csv")
pred_test.to_csv(ART_DIR / "enet_test_preds.csv")

# also save the scores
pd.Series(scores).to_csv(ART_DIR / "enet_scores.csv")
print("\nSaved predictions & scores to:", ART_DIR)

# ==========================
# ML: Random Forest & Gradient Boosting (supervised)
# Tune on VAL (2010–2014), then refit on TRAIN+VAL and score TEST
# ==========================
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Reuse 'full' from above; if not present, load it
try:
    full
except NameError:
    full = pd.read_csv(OUT_DIR / "full_dataset.csv", parse_dates=[0], index_col=0)

# Features = everything not a target (_t1)
feat_cols = [c for c in full.columns if not c.endswith("_t1")]
X = full[feat_cols]
y = full[["beta1_t1", "beta2_t1", "beta3_t1"]]

# Reuse masks; if not present, create them
try:
    is_train, is_val, is_test
except NameError:
    is_train = (full.index <= "2009-12-31")
    is_val   = (full.index > "2009-12-31") & (full.index <= "2014-12-31")
    is_test  = (full.index > "2014-12-31")

def rmse(a, f): 
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(f))**2)))
def mae(a, f):  
    return float(np.nanmean(np.abs(np.asarray(a) - np.asarray(f))))

def eval_scores(y_true_df, y_pred_df, label):
    out = {}
    for ct, cp in zip(["beta1_t1","beta2_t1","beta3_t1"],
                      ["beta1_t1_pred","beta2_t1_pred","beta3_t1_pred"]):
        out[f"{label}_RMSE_{ct}"] = rmse(y_true_df[ct], y_pred_df[cp])
        out[f"{label}_MAE_{ct}"]  = mae(y_true_df[ct],  y_pred_df[cp])
    return out

def tune_on_val_then_refit(model_name, base_model, param_grid, scale=False):
    """
    Simple manual grid over 'param_grid' on VAL. 
    1) Fit on TRAIN with each param set, score VAL (avg RMSE across betas), pick best.
    2) Refit on TRAIN+VAL with best params.
    3) Predict TRAIN/VAL/TEST, save preds & scores.
    """
    if scale:
        pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                         ("model", MultiOutputRegressor(base_model))])
        param_names = {f"model__estimator__{k}": v for k, v in param_grid.items()}
    else:
        pipe = MultiOutputRegressor(base_model)
        param_names = {f"estimator__{k}": v for k, v in param_grid.items()}

    # Build all combinations
    from itertools import product
    keys = list(param_names.keys())
    grids = list(product(*[param_names[k] for k in keys]))

    best_cfg = None
    best_val = float("inf")

    Xtr, Ytr = X[is_train], y[is_train]
    Xv,  Yv  = X[is_val],   y[is_val]
    Xt,  Yt  = X[is_test],  y[is_test]

    for combo in grids:
        cfg = dict(zip(keys, combo))
        # clone pipeline each time
        import copy
        model = copy.deepcopy(pipe)
        # set params
        model.set_params(**cfg)
        model.fit(Xtr, Ytr)
        Pv = pd.DataFrame(model.predict(Xv), index=Xv.index, 
                          columns=["beta1_t1_pred","beta2_t1_pred","beta3_t1_pred"])
        sc = eval_scores(Yv, Pv, "val")
        avg_val_rmse = np.mean([sc["val_RMSE_beta1_t1"], sc["val_RMSE_beta2_t1"], sc["val_RMSE_beta3_t1"]])
        if avg_val_rmse < best_val:
            best_val = avg_val_rmse
            best_cfg = cfg

    # Refit on TRAIN+VAL with best params
    is_trval = is_train | is_val
    model = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                      ("model", MultiOutputRegressor(base_model))]) if scale else MultiOutputRegressor(base_model)
    model.set_params(**best_cfg)
    model.fit(X[is_trval], y[is_trval])

    # Predict all splits
    Ptrain = pd.DataFrame(model.predict(X[is_train]), index=X[is_train].index,
                          columns=["beta1_t1_pred","beta2_t1_pred","beta3_t1_pred"])
    Pval   = pd.DataFrame(model.predict(X[is_val]),   index=X[is_val].index,
                          columns=["beta1_t1_pred","beta2_t1_pred","beta3_t1_pred"])
    Ptest  = pd.DataFrame(model.predict(X[is_test]),  index=X[is_test].index,
                          columns=["beta1_t1_pred","beta2_t1_pred","beta3_t1_pred"])

    # Scores
    scores = {}
    scores.update(eval_scores(y[is_train], Ptrain, "train"))
    scores.update(eval_scores(y[is_val],   Pval,   "val"))
    scores.update(eval_scores(y[is_test],  Ptest,  "test"))
    # Add best params to scores for reference
    for k, v in best_cfg.items():
        scores[f"bestparam::{k}"] = v

    # Save artifacts
    ART = Path(DATA_DIR) / "artifacts" / "preds"
    ART.mkdir(parents=True, exist_ok=True)
    Ptrain.to_csv(ART / f"{model_name}_train_preds.csv")
    Pval.to_csv(ART / f"{model_name}_val_preds.csv")
    Ptest.to_csv(ART / f"{model_name}_test_preds.csv")
    pd.Series(scores).to_csv(ART / f"{model_name}_scores.csv")
    print(f"[{model_name}] best val avg RMSE: {best_val:.4f} | saved to", ART)

    return model, best_cfg

# --- Random Forest (trees don't need scaling) ---
rf_params = {
    "n_estimators": [500, 800, 1000],
    "max_depth": [3, 4, 5, None],
    "max_features": ["sqrt", "log2", None],
    "min_samples_leaf": [1, 2, 4],
    "random_state": [42],
    "n_jobs": [-1],
}
_ = tune_on_val_then_refit(
    model_name="rf",
    base_model=RandomForestRegressor(),
    param_grid=rf_params,
    scale=False
)

# --- Gradient Boosting (often benefits from mild scaling; we can keep it unscaled too) ---
gb_params = {
    "n_estimators": [300, 600, 900],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "subsample": [0.8, 1.0],
    "random_state": [42],
}
_ = tune_on_val_then_refit(
    model_name="gb",
    base_model=GradientBoostingRegressor(),
    param_grid=gb_params,
    scale=False  # trees don't require scaling; set True if you want standardization in a Pipeline
)



import pandas as pd
from pathlib import Path

DATA_DIR = r"C:\Users\98910\Desktop\python project github"
P = Path(DATA_DIR) / "artifacts" / "preds"

def read_scores(name: str) -> pd.Series | None:
    f = P / f"{name}_scores.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f, header=None, names=["k","v"], dtype={"k":"string"})
    # force keys to string; fill missing
    s["k"] = s["k"].astype("string").fillna("")
    s = s.set_index("k")["v"]
    s.name = name
    return s

models = ["enet","rf","gb","ar","var"]  # include only those you actually ran
series_list = [read_scores(m) for m in models]
series_list = [s for s in series_list if s is not None]

# Combine
scores_tbl = pd.concat(series_list, axis=1)

# Ensure string index (prevents AttributeError)
scores_tbl.index = scores_tbl.index.map(lambda x: "" if pd.isna(x) else str(x))

# Keep only TEST RMSE/MAE rows
mask = scores_tbl.index.str.startswith("test_RMSE_") | scores_tbl.index.str.startswith("test_MAE_")
scores_test = scores_tbl.loc[mask].copy().round(4)

print(scores_test)
out = P / "compare_test_scores_beta.csv"
scores_test.to_csv(out)
print("Saved:", out)

# ============================================================
# FINAL STEPS: Yield-space evaluation + Figures (ENet + RF)
# ============================================================
import pandas as pd, numpy as np, os
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\98910\Desktop\python project github"
BASE = Path(DATA_DIR)
PRED_DIR = BASE / "artifacts" / "preds"
FIG_DIR  = BASE / "artifacts" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def ns_loadings(maturity_years, tau):
    m = np.asarray(maturity_years, float)
    x = m / tau
    x = np.where(x==0.0, 1e-8, x)
    L1 = np.ones_like(m); L2 = (1-np.exp(-x))/x; L3 = L2 - np.exp(-x)
    return np.vstack([L1, L2, L3]).T

def reconstruct_curve(b1, b2, b3, tau, tenors):
    X = ns_loadings(tenors, tau)
    return X @ np.array([b1, b2, b3])

def rmse(a, f): 
    a = np.asarray(a); f = np.asarray(f)
    return float(np.sqrt(np.nanmean((a-f)**2)))

def curve_rmse_for_block(pred_df, tau_next, Y_next, tenors, tenor_cols, label):
    rows_pred, rows_true = [], []
    for dt in pred_df.index:
        if dt not in tau_next.index: 
            continue
        tau = float(tau_next.loc[dt])
        if np.isnan(tau): 
            continue
        b1,b2,b3 = pred_df.loc[dt, ["beta1_t1_pred","beta2_t1_pred","beta3_t1_pred"]]
        yhat = reconstruct_curve(b1,b2,b3,tau,tenors)
        rows_pred.append((dt, *yhat))
        if dt in Y_next.index:
            yt = Y_next.loc[dt, tenor_cols].values
            rows_true.append((dt, *yt))
    cols = ["date"] + [f"y_{t}y" for t in tenors]
    P = pd.DataFrame(rows_pred, columns=cols).set_index("date")
    T = pd.DataFrame(rows_true, columns=cols).set_index("date")
    rmses = {f"{label}_RMSE_{t}y": rmse(T.iloc[:,i], P.iloc[:,i]) for i,t in enumerate(tenors) if i < P.shape[1]}
    return P, T, rmses

# ---------- data ----------
NS   = pd.read_csv(BASE/"ns_factors.csv", parse_dates=[0], index_col=0).sort_index()
COMB = pd.read_csv(BASE/"combined_2003_2025M.csv", parse_dates=[0], index_col=0).sort_index()
FULL = pd.read_csv(BASE/"dataset"/"full_dataset.csv", parse_dates=[0], index_col=0).sort_index()

# splits
is_train = (FULL.index <= "2009-12-31")
is_val   = (FULL.index >  "2009-12-31") & (FULL.index <= "2014-12-31")
is_test  = (FULL.index >  "2014-12-31")

# targets for plotting β on test (shifted to t+1 to match the *_t1 convention)
betas_t1 = NS[["beta1","beta2","beta3"]].shift(-1).rename(
    columns={"beta1":"beta1_t1","beta2":"beta2_t1","beta3":"beta3_t1"}
)

# tenors to evaluate
tenors    = [1,2,5,10,30]
tenor_map = {1:"GS1", 2:"GS2", 5:"GS5", 10:"GS10", 30:"GS30"}
tenor_cols = [tenor_map[t] for t in tenors if tenor_map[t] in COMB.columns]

# actual yields at t+1 aligned to index t
Y_next   = COMB[tenor_cols].shift(-1)
tau_next = NS["tau"].shift(-1)

# load predictions (ENet required; RF optional)
enet = {
    "train": pd.read_csv(PRED_DIR/"enet_train_preds.csv", parse_dates=[0], index_col=0),
    "val":   pd.read_csv(PRED_DIR/"enet_val_preds.csv",   parse_dates=[0], index_col=0),
    "test":  pd.read_csv(PRED_DIR/"enet_test_preds.csv",  parse_dates=[0], index_col=0),
}
rf = {}
if (PRED_DIR/"rf_test_preds.csv").exists():
    rf = {
        "train": pd.read_csv(PRED_DIR/"rf_train_preds.csv", parse_dates=[0], index_col=0),
        "val":   pd.read_csv(PRED_DIR/"rf_val_preds.csv",   parse_dates=[0], index_col=0),
        "test":  pd.read_csv(PRED_DIR/"rf_test_preds.csv",  parse_dates=[0], index_col=0),
    }

# ---------- reconstruct curves + save RMSE tables ----------
curve_rmse = {}

for name, preds in [("enet", enet), ("rf", rf)]:
    if not preds: 
        continue
    for split, mask in [("train", is_train), ("val", is_val), ("test", is_test)]:
        if preds.get(split) is None or preds[split].empty:
            continue
        # align to split index
        idx = FULL.index[mask]
        pred_df = preds[split].loc[preds[split].index.intersection(idx)]

        P, T, r = curve_rmse_for_block(pred_df, tau_next, Y_next, tenors, tenor_cols, f"{name}_{split}")
        # save curves and collect rmse
        P.to_csv(PRED_DIR / f"{name}_curve_{split}.csv")
        curve_rmse.update(r)

# save RMSE summary
pd.Series(curve_rmse).to_csv(PRED_DIR / "curve_rmse_enet_rf.csv")
print("Saved curve RMSE summary →", PRED_DIR / "curve_rmse_enet_rf.csv")

# ============================================================
# FIGURES
# ============================================================

# ------ 1) β time series on TEST (actual vs ENet) ------
enet_test = enet["test"]
common = betas_t1.index.intersection(enet_test.index)
bt = betas_t1.loc[common]
pt = enet_test.loc[common]

for (true_col, pred_col, title) in [
    ("beta1_t1","beta1_t1_pred","Level (β1)"),
    ("beta2_t1","beta2_t1_pred","Slope (β2)"),
    ("beta3_t1","beta3_t1_pred","Curvature (β3)")
]:
    plt.figure()
    plt.plot(bt.index, bt[true_col], label="Actual")
    plt.plot(pt.index, pt[pred_col], label="ENet Pred")
    plt.title(f"Test period — {title}")
    plt.xlabel("Date"); plt.ylabel("Value"); plt.legend()
    plt.tight_layout()
    out = FIG_DIR / f"enet_test_{true_col}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved plot:", out)

# ------ 2) Curve RMSE by tenor on TEST (ENet vs RF) ------
def read_curve_rmse(path):
    s = pd.read_csv(path, header=None, names=["k","v"]).set_index("k")["v"]
    return s

s_curve = read_curve_rmse(PRED_DIR / "curve_rmse_enet_rf.csv")

# Build a small DataFrame for test rows
rows = []
for model in ["enet","rf"]:
    for t in tenors:
        key = f"{model}_test_RMSE_{t}y"
        if key in s_curve.index:
            rows.append((model, f"{t}Y", float(s_curve[key])))
curve_tbl = pd.DataFrame(rows, columns=["model","tenor","rmse"])
if not curve_tbl.empty:
    pivot = curve_tbl.pivot(index="tenor", columns="model", values="rmse").sort_index()
    pivot.plot(kind="bar")
    plt.title("Curve RMSE by tenor — TEST (lower = better)")
    plt.ylabel("RMSE (pp)"); plt.tight_layout()
    out = FIG_DIR / "curve_rmse_test_by_tenor.png"
    plt.savefig(out, dpi=150); plt.close()
    print("Saved plot:", out)

# ------ 3) TEST RMSE by β (from compare_test_scores_beta.csv) ------
cmp_path = PRED_DIR / "compare_test_scores_beta.csv"
if cmp_path.exists():
    tbl = pd.read_csv(cmp_path, index_col=0)
    # keep only RMSE rows; rename for nicer labels
    rmse_rows = [i for i in tbl.index if str(i).startswith("test_RMSE_")]
    if rmse_rows:
        tb = tbl.loc[rmse_rows].copy()
        tb.index = [i.replace("test_RMSE_","").replace("_t1","").upper() for i in tb.index]
        tb.plot(kind="bar")
        plt.title("TEST RMSE by β (lower = better)")
        plt.ylabel("RMSE"); plt.tight_layout()
        out = FIG_DIR / "test_rmse_by_beta.png"
        plt.savefig(out, dpi=150); plt.close()
        print("Saved plot:", out)

# ------ 4) Curve snapshots (actual vs ENet) ------
snapshot_dates = [
    "2020-04-30",  # COVID shock
    "2022-10-31",  # inflation peak-ish
    "2019-06-30"   # calm sample
]
for d in snapshot_dates:
    d = pd.to_datetime(d)
    if d not in enet_test.index: 
        continue
    tau = float(tau_next.loc[d]) if d in tau_next.index else np.nan
    if np.isnan(tau): 
        continue
    b1,b2,b3 = enet_test.loc[d, ["beta1_t1_pred","beta2_t1_pred","beta3_t1_pred"]]
    yhat = reconstruct_curve(b1,b2,b3,tau,tenors)
    # actual next-month yields
    if d not in Y_next.index: 
        continue
    yt = Y_next.loc[d, tenor_cols].values
    # plot
    plt.figure()
    plt.plot(tenors, yt, marker="o", label="Actual (t+1)")
    plt.plot(tenors, yhat, marker="o", label="ENet Pred (t+1)")
    plt.xticks(tenors, [f"{t}Y" for t in tenors])
    plt.title(f"Yield curve snapshot — {d.date()}")
    plt.ylabel("Yield (%)"); plt.legend(); plt.tight_layout()
    out = FIG_DIR / f"curve_snapshot_{d.date()}.png"
    plt.savefig(out, dpi=150); plt.close()
    print("Saved plot:", out)

print("\nAll final artifacts saved under:")
print(" -", PRED_DIR)
print(" -", FIG_DIR)

