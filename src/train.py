import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

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


### Random Forest Regressor Prediction
model = RandomForestRegressor(
    n_estimators=500, random_state=42, max_depth=10, min_samples_leaf=10, n_jobs=-1
)
model.fit(feat_train, target_train)
model_pred = model.predict(feat_test)
rf_mae = mean_absolute_error(target_test, model_pred)
rf_rmse = np.sqrt(mean_squared_error(target_test, model_pred))

print(f"RF Metrics MAE: {rf_mae}\nRF Metrics RMSE: {rf_rmse}\n")

### Overfitting Check - Prevent the model from memorizing the data instead of generalizing it
train_pred = model.predict(feat_train)
train_mae = mean_absolute_error(target_train, train_pred)
train_rmse = np.sqrt(mean_squared_error(target_train, train_pred))

print(f"Train Metrics MAE: {train_mae}\nTrain Metrics RMSE: {train_rmse}\n")

### Quantify Gap
mae_gap = rf_mae / train_mae
rmse_gap = rf_rmse / train_rmse
print(f"MAE Gap: {mae_gap}\nRMSE Gap: {rmse_gap}\n")

### Time Series Cross Validation
# We use TimeSeriesSplit to ensure that every fold respects the temporal order of the data, preventing data leakage from future to past.
tscv = TimeSeriesSplit(n_splits=5)

train_maes = []
test_maes = []
train_rmses = []
test_rmses = []


for date_train_index, date_test_index in tscv.split(sorted_dates):
    train_dates = sorted_dates[date_train_index]
    test_dates = sorted_dates[date_test_index]

    fold_train_mask = panel.date.isin(train_dates)
    fold_test_mask = panel.date.isin(test_dates)

    X_train_fold = features[fold_train_mask]
    y_train_fold = target[fold_train_mask]
    X_test_fold = features[fold_test_mask]
    y_test_fold = target[fold_test_mask]

    fold_model = RandomForestRegressor(
        n_estimators=500, random_state=42, max_depth=10, min_samples_leaf=10, n_jobs=-1
    )
    fold_model.fit(X_train_fold, y_train_fold)

    train_pred_fold = fold_model.predict(X_train_fold)
    test_pred_fold = fold_model.predict(X_test_fold)

    train_mae_fold = mean_absolute_error(y_train_fold, train_pred_fold)
    test_mae_fold = mean_absolute_error(y_test_fold, test_pred_fold)
    train_rmse_fold = np.sqrt(mean_squared_error(y_train_fold, train_pred_fold))
    test_rmse_fold = np.sqrt(mean_squared_error(y_test_fold, test_pred_fold))

    train_maes.append(train_mae_fold)
    test_maes.append(test_mae_fold)
    train_rmses.append(train_rmse_fold)
    test_rmses.append(test_rmse_fold)

avg_train_mae = np.mean(train_maes)
avg_test_mae = np.mean(test_maes)
avg_train_rmse = np.mean(train_rmses)
avg_test_rmse = np.mean(test_rmses)
avg_gap_mae = avg_test_mae / avg_train_mae
avg_gap_rmse = avg_test_rmse / avg_train_rmse

print(f"Average Train MAE: {avg_train_mae}\nAverage Test MAE: {avg_test_mae}\n")
print(f"Average Train RMSE: {avg_train_rmse}\nAverage Test RMSE: {avg_test_rmse}\n")

### GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 200, 500, 700, 1000],
    "max_depth": [2, 3, 4, 5, 10, 15, 20, 25, 30, None],
    "min_samples_leaf": [1, 5, 10, 20, 50],
}

base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

train_sorted_dates = np.sort(panel.date[train_mask].unique())
train_date_series = panel.date[train_mask]

custom_cv = []

for date_train_index, date_val_index in tscv.split(train_sorted_dates):
    fold_train_dates = train_sorted_dates[date_train_index]
    fold_val_dates = train_sorted_dates[date_val_index]

    fold_train_pos = train_date_series.isin(fold_train_dates)
    fold_val_pos = train_date_series.isin(fold_val_dates)

    custom_cv.append((fold_train_pos, fold_val_pos))

search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=custom_cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)

search.fit(feat_train, target_train)

best_params = search.best_params_
best_score = search.best_score_
best_model = search.best_estimator_
print(
    f"Best Params = {best_params}\nBest Score = {best_score}\nBest Model = {best_model}"
)
