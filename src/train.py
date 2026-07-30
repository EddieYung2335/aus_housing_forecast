import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib
import json
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

model_path = MODELS_DIR / "rf_model.joblib"
metadata_path = MODELS_DIR / "rf_model_metadata.json"
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

################## Baseline Model ########################
baseline_pred = feat_test.log_return

# How wrong the prediction is here...
mae = mean_absolute_error(target_test, baseline_pred)
# rmse punish big misses harder than small one
rmse = np.sqrt(mean_squared_error(target_test, baseline_pred))


################## Random Forest Regressor Prediction ########################
rf_model = RandomForestRegressor(
    n_estimators=500, random_state=42, max_depth=10, min_samples_leaf=10, n_jobs=-1
)
rf_model.fit(feat_train, target_train)
rf_model_pred = rf_model.predict(feat_test)
rf_mae = mean_absolute_error(target_test, rf_model_pred)
rf_rmse = np.sqrt(mean_squared_error(target_test, rf_model_pred))


###################### XGBoost Regressor Prediction #########################
xgb_model = XGBRegressor(
    n_estimators=500, random_state=42, max_depth=5, learning_rate=0.05, n_jobs=-1
)

xgb_model.fit(feat_train, target_train)
xgb_model_pred = xgb_model.predict(feat_test)
xgb_mae = mean_absolute_error(target_test, xgb_model_pred)
xgb_rmse = np.sqrt(mean_squared_error(target_test, xgb_model_pred))


###################### Overfitting Check ########################
rf_train_pred = rf_model.predict(feat_train)
rf_train_mae = mean_absolute_error(target_train, rf_train_pred)
rf_train_rmse = np.sqrt(mean_squared_error(target_train, rf_train_pred))

xgb_train_pred = xgb_model.predict(feat_train)
xgb_train_mae = mean_absolute_error(target_train, xgb_train_pred)
xgb_train_rmse = np.sqrt(mean_squared_error(target_train, xgb_train_pred))


### Quantify Gap
rf_mae_gap = rf_mae / rf_train_mae
rf_rmse_gap = rf_rmse / rf_train_rmse

xgb_mae_gap = xgb_mae / xgb_train_mae
xgb_rmse_gap = xgb_rmse / xgb_train_rmse


### Time Series Cross Validation
# We use TimeSeriesSplit to ensure that every fold respects the temporal order of the data, preventing data leakage from future to past.
tscv = TimeSeriesSplit(n_splits=5)

rf_train_maes = []
rf_test_maes = []
rf_train_rmses = []
rf_test_rmses = []


for date_train_index, date_test_index in tscv.split(sorted_dates):
    train_dates = sorted_dates[date_train_index]
    test_dates = sorted_dates[date_test_index]

    fold_train_mask = panel.date.isin(train_dates)
    fold_test_mask = panel.date.isin(test_dates)

    X_train_fold = features[fold_train_mask]
    y_train_fold = target[fold_train_mask]
    X_test_fold = features[fold_test_mask]
    y_test_fold = target[fold_test_mask]

    rf_fold_model = RandomForestRegressor(
        n_estimators=500, random_state=42, max_depth=10, min_samples_leaf=10, n_jobs=-1
    )
    rf_fold_model.fit(X_train_fold, y_train_fold)

    train_pred_fold = rf_fold_model.predict(X_train_fold)
    test_pred_fold = rf_fold_model.predict(X_test_fold)

    train_mae_fold = mean_absolute_error(y_train_fold, train_pred_fold)
    test_mae_fold = mean_absolute_error(y_test_fold, test_pred_fold)
    train_rmse_fold = np.sqrt(mean_squared_error(y_train_fold, train_pred_fold))
    test_rmse_fold = np.sqrt(mean_squared_error(y_test_fold, test_pred_fold))

    rf_train_maes.append(train_mae_fold)
    rf_test_maes.append(test_mae_fold)
    rf_train_rmses.append(train_rmse_fold)
    rf_test_rmses.append(test_rmse_fold)

rf_avg_train_mae = np.mean(rf_train_maes)
rf_avg_test_mae = np.mean(rf_test_maes)
rf_avg_train_rmse = np.mean(rf_train_rmses)
rf_avg_test_rmse = np.mean(rf_test_rmses)
rf_avg_gap_mae = rf_avg_test_mae / rf_avg_train_mae
rf_avg_gap_rmse = rf_avg_test_rmse / rf_avg_train_rmse

print(f"Average Train MAE: {rf_avg_train_mae}\nAverage Test MAE: {rf_avg_test_mae}\n")
print(f"Average Train RMSE: {rf_avg_train_rmse}\nAverage Test RMSE: {rf_avg_test_rmse}\n")

### GridSearchCV
rf_param_grid = {
    "n_estimators": [50, 100, 200, 500, 700, 1000],
    "max_depth": [2, 3, 4, 5, 10, 15, 20, 25, 30, None],
    "min_samples_leaf": [1, 5, 10, 20, 50],
}

xgb_param_grid = {
    "n_estimators": [50, 100, 200, 500, 700, 1000],
    "max_depth": [2, 3, 4, 5, 10, 15, 20, 25, 30, None],
    "learning_rate": [0.01, 0.05, 0.1],
}


rf_base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
xgb_base_model = XGBRegressor(random_state=42, n_jobs=-1)

train_sorted_dates = np.sort(panel.date[train_mask].unique())
train_date_series = panel.date[train_mask]

custom_cv = []

for date_train_index, date_val_index in tscv.split(train_sorted_dates):
    fold_train_dates = train_sorted_dates[date_train_index]
    fold_val_dates = train_sorted_dates[date_val_index]

    fold_train_pos = train_date_series.isin(fold_train_dates)
    fold_val_pos = train_date_series.isin(fold_val_dates)

    custom_cv.append((fold_train_pos, fold_val_pos))

rf_search = GridSearchCV(
    estimator=rf_base_model,
    param_grid=rf_param_grid,
    cv=custom_cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)

xgb_search = GridSearchCV(
    estimator=xgb_base_model,
    param_grid=xgb_param_grid,
    cv=custom_cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)

rf_search.fit(feat_train, target_train)

xgb_search.fit(feat_train, target_train)

rf_best_params = rf_search.best_params_
rf_best_score = rf_search.best_score_
rf_best_model = rf_search.best_estimator_

xgb_best_params = xgb_search.best_params_
xgb_best_score = xgb_search.best_score_
xgb_best_model = xgb_search.best_estimator_

print(
    f"RF\nBest Params = {rf_best_params}\nBest Score = {rf_best_score}\nBest Model = {rf_best_model}"
)

print(
    f"XGB\nBest Params = {xgb_best_params}\nBest Score = {xgb_best_score}\nBest Model = {xgb_best_model}"
)

### Using the best model now
rf_test_pred_best = rf_best_model.predict(feat_test)
xgb_test_pred_best = xgb_best_model.predict(feat_test)

rf_best_mae = mean_absolute_error(target_test, rf_test_pred_best)
rf_best_rmse = np.sqrt(mean_squared_error(target_test, rf_test_pred_best))

xgb_best_mae = mean_absolute_error(target_test, xgb_test_pred_best)
xgb_best_rmse = np.sqrt(mean_squared_error(target_test, xgb_test_pred_best))


### Best model Overfitting
rf_train_pred_best = rf_best_model.predict(feat_train)
xgb_train_pred_best = xgb_best_model.predict(feat_train)

rf_train_best_mae = mean_absolute_error(target_train, rf_train_pred_best)
rf_train_best_rmse = np.sqrt(mean_squared_error(target_train, rf_train_pred_best))
print(f"Best Train MAE = {rf_train_best_mae}\nBest Train RMSE = {rf_train_best_rmse}\n")

xgb_train_best_mae = mean_absolute_error(target_train, xgb_train_pred_best)
xgb_train_best_rmse = np.sqrt(mean_squared_error(target_train, xgb_train_pred_best))
print(
    f"Best Train MAE = {xgb_train_best_mae}\nBest Train RMSE = {xgb_train_best_rmse}\n"
)

# Gap approx = 1 no overfit
# Gap >> 1 overfit (test error much worse than train)
# Gap < 1 unusual
rf_gap_best_mae = rf_best_mae / rf_train_best_mae
rf_gap_best_rmse = rf_best_rmse / rf_train_best_rmse

xgb_gap_best_mae = xgb_best_mae / xgb_train_best_mae
xgb_gap_best_rmse = xgb_best_rmse / xgb_train_best_rmse


### Model Comparison Table
comparison = pd.DataFrame(
    {
        "Model": [
            "Baseline",
            "Original RF (depth=10)",
            f"Tuned RF (depth={rf_best_params['max_depth']})",
            f"Tuned XGB (depth={xgb_best_params['max_depth']})",
        ],
        "Test MAE": [mae, rf_mae, rf_best_mae, xgb_best_mae],
        "Test RMSE": [rmse, rf_rmse, rf_best_rmse, xgb_best_rmse],
        "Gap (test/train)": [None, rf_mae_gap, rf_gap_best_mae, xgb_gap_best_mae],
    }
)
print("\n" + comparison.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


print(f"Saving the best model into {MODELS_DIR}...")
joblib.dump(rf_best_model, model_path)
print("Done!")

metadata = {
    "best_params": rf_best_params,
    "best_score": rf_best_score,
    "test_mae": rf_best_mae,
    "test_rmse": rf_best_rmse,
    "gap_mae": rf_gap_best_mae,
}
print(f"Saving the metadata into {MODELS_DIR}...")

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print("Done!")
