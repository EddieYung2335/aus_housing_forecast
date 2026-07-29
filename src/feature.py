import pandas as pd
from pathlib import Path
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["region", "date"]).copy()
    panel["log_price"] = np.log(panel["median_price"])
    panel["log_return"] = panel.groupby("region")["log_price"].diff()

    # Lags = the log value of last month
    # Model will try to understand the relationship between increase/decrease of last month and this month
    # To determine whether there is a relationship between them
    panel["lag1"] = panel.groupby("region")["log_return"].shift(1)
    panel["lag2"] = panel.groupby("region")["log_return"].shift(2)
    panel["lag3"] = panel.groupby("region")["log_return"].shift(3)
    panel["lag4"] = panel.groupby("region")["log_return"].shift(4)

    # Rolling = the average of log_return of past few month
    # In order to get a more stable trend
    # Use lag1 to prevent the model know the actual log_price of the month
    roll3 = panel.groupby("region")["lag1"].rolling(3)
    roll6 = panel.groupby("region")["lag1"].rolling(6)
    roll3_mean = roll3.mean().reset_index(level=0, drop=True)
    roll6_mean = roll6.mean().reset_index(level=0, drop=True)
    roll3_std = roll3.std().reset_index(level=0, drop=True)
    roll6_std = roll6.std().reset_index(level=0, drop=True)

    panel["roll3_mean"] = roll3_mean
    panel["roll3_std"] = roll3_std
    panel["roll6_mean"] = roll6_mean
    panel["roll6_std"] = roll6_std

    # Seasonality - to investigate the 周期性 of buying house
    # e.g. house is more expensive in spring
    months = panel["date"].dt.month
    sin_col = np.sin(2 * np.pi * months / 12)
    cos_col = np.cos(2 * np.pi * months / 12)
    panel["month_sin"] = sin_col
    panel["month_cos"] = cos_col

    # Region Encoding
    # Model cannot understand "Adelaide", "Melbourne", set them to
    # |Adelaide | Melbourne  | Brisbane |
    # --------- | ---------- | -------- |
    # |     1       0           0
    #       0       1           0
    #       0       0           1
    # One column for each region, if the current target is in the region, set 1,
    # 0 otherwise
    dummies = pd.get_dummies(panel["region"], drop_first=False)

    panel = pd.concat([panel, dummies], axis=1)

    panel = panel.dropna(subset=["roll6_mean", "roll6_std"]).reset_index(drop=True)

    print(panel["roll6_mean"])

    print(panel)

    return panel


if __name__ == "__main__":
    print("Reading panel parquet...")
    data = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    print("Building features parquet")
    panel = build_features(data)
    print("Writing to features.parquet")
    panel.to_parquet(PROCESSED_DIR / "features.parquet")

