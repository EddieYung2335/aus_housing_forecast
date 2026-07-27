import pandas as pd
from pathlib import Path
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["region", "date"]).copy()

    panel["log_price"] = np.log(panel["median_price"])

    panel["log_return"] = panel.groupby("region")["log_price"].diff()

    print(panel)

    return panel


if __name__ == "__main__":
    # print("Reading panel parquet...")
    data = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    # print("Building features parquet")
    panel = build_features(data)
    # print("Writing to features.parquet")
    panel.to_parquet(PROCESSED_DIR / "features.parquet")

