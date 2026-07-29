import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

panel = pd.read_parquet(PROCESSED_DIR / "features.parquet")

# Each region have different data of date, some exist in 2003, some does not.
# To accurately treat them, we have to find every distinct date value in the panel
sorted_dates = np.sort(panel["date"].unique())


split_index = round(0.75 * len(sorted_dates))
cutoff_date = sorted_dates[split_index - 1]

train_mask = panel.date <= cutoff_date
test_mask = panel.date > cutoff_date

target = panel.target

features = panel.drop(
    columns=[
        "date",
        "region",
        "state",
        "median_price",
        "log_price",
        "target",
        "mean_price",
        "total_value",
    ]
)

target_train = target[train_mask]
target_test = target[test_mask]

feat_train = features[train_mask]
feat_test = features[test_mask]
