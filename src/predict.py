# take in date -> give back log_return for next month -> human_readable price
# input: features.parquet last row per region | model.joblib
# output: table[region, last_date, target_date, last_price, pred_log_return, pred_price, lower, upper]
#
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

metadata_path = MODELS_DIR / "model_metadata.json"


def predict_next_month(panel, model, metadata):
    # Take the last log_return for each region from features.parquet
    # check the last index for each region and store them into []
    latest_index = panel.groupby("region")["date"].idxmax()
    # the last row for each region where target=NaN
    latest = panel.loc[latest_index]
    latest = latest.reset_index(drop=True)

    # Keep the columns that we want only
    cols = model.feature_names_in_
    features_latest = latest[cols]
    missing = set(cols) - set(latest.columns)
    assert not missing, f"missing features: {missing}"

    # Predict on the features of the last column for each region
    pred_log_return = model.predict(features_latest)

    result = pd.DataFrame(
        {
            "region": latest["region"],
            "last_date": latest["date"],
            "target_date": latest["date"] + pd.DateOffset(months=1),
            "last_price": latest["median_price"],
            "pred_log_return": pred_log_return,
            "pred_price": latest["median_price"] * np.exp(pred_log_return),
        }
    )

    mae = metadata["test_mae"]

    result["lower"] = latest["median_price"] * np.exp(pred_log_return - mae)
    result["upper"] = latest["median_price"] * np.exp(pred_log_return + mae)

    result["model_name"] = metadata["name"]

    return result


if __name__ == "__main__":
    model = joblib.load(MODELS_DIR / "model.joblib")
    panel = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"Using model ({metadata['name']}) to predict on the features given... ")
    result = predict_next_month(panel, model, metadata)
    print(f"Prediction finished!\nWriting the predictions result to {PROCESSED_DIR}...")
    result.to_parquet(PROCESSED_DIR / "predictions.parquet")
    print("Done!")
