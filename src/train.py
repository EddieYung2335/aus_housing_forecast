import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor


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

### Baseline Model
baseline_pred = feat_test.log_return


# How wrong the prediction is here...
mae = mean_absolute_error(target_test, baseline_pred)
# rmse punish big misses harder than small one
rmse = np.sqrt(mean_squared_error(target_test, baseline_pred))

print(f"Raw MAE: {mae}\nRaw RMSE: {rmse}\n")

model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
model.fit(feat_train, target_train)
model_pred = model.predict(feat_test)
rf_mae = mean_absolute_error(target_test, model_pred)
rf_rmse = np.sqrt(mean_squared_error(target_test, model_pred))

print(f"RF Metrics MAE: {rf_mae}\nRF Metrics RMSE: {rf_rmse}\n")
