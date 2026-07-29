import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


panel = pd.read_parquet(PROCESSED_DIR / "features.parquet")


target = panel["target"]

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

print(features)
