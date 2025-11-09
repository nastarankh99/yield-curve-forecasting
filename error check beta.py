import pandas as pd, numpy as np
from pathlib import Path

DATA_DIR = r"C:\Users\98910\Desktop\python project github"
full = pd.read_csv(Path(DATA_DIR)/"dataset"/"full_dataset.csv", parse_dates=[0], index_col=0)

is_test = (full.index > "2014-12-31")
t = full.loc[is_test, ["beta1_t1","beta2_t1","beta3_t1"]]
print(t.abs().sort_values("beta2_t1", ascending=False).head(5))
print(t.abs().sort_values("beta3_t1", ascending=False).head(5))