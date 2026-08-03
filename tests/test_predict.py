import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from predict import predict_next_quarter

FAKE_RETURN = 0.01
FAKE_MAE = 0.02
FEATURE_COLS = ["lag1", "roll3_mean"]


class FakeModel:
    feature_names_in_ = np.array(FEATURE_COLS)

    def predict(self, X):
        return np.full(len(X), FAKE_RETURN)


def make_fake_panel():
    dates = pd.date_range("2024-03-01", periods=8, freq="QS-MAR")
    rows = []
    for region, start_price in [("Alpha", 100.0), ("Beta", 500.0)]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "region": region,
                    "median_price": start_price + i,
                    "lag1": 0.001 * i,
                    "roll3_mean": 0.002 * i,
                }
            )
    return pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)


def test_predict_next_quarter():
    panel = make_fake_panel()
    metadata = {"test_mae": FAKE_MAE, "name": "FakeModel"}

    result = predict_next_quarter(panel, FakeModel(), metadata)

    assert len(result) == panel["region"].nunique(), (
        f"expected {panel['region'].nunique()} rows, got {len(result)}"
    )

    expected_target = result["last_date"] + pd.DateOffset(months=3)
    assert (result["target_date"] == expected_target).all(), (
        f"target_date is not last_date + 1 quarter:\n{result[['last_date', 'target_date']]}"
    )

    assert (result["last_date"] == panel["date"].max()).all()

    assert (result["lower"] < result["pred_price"]).all()
    assert (result["pred_price"] < result["upper"]).all()

    expected_price = result["last_price"] * np.exp(FAKE_RETURN)
    assert np.isclose(result["pred_price"], expected_price).all(), (
        f"expected {expected_price.tolist()}, got {result['pred_price'].tolist()}"
    )

    up = result["upper"] - result["pred_price"]
    down = result["pred_price"] - result["lower"]
    assert (up > down).all(), "bands should be wider on the upside after exp()"


if __name__ == "__main__":
    test_predict_next_quarter()
    print("test_predict_next_quarter OK")
