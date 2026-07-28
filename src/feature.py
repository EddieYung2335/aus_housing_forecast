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
    print(panel)

    return panel


if __name__ == "__main__":
    print("Reading panel parquet...")
    data = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    print("Building features parquet")
    panel = build_features(data)
    print("Writing to features.parquet")
    panel.to_parquet(PROCESSED_DIR / "features.parquet")

