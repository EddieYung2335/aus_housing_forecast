# aus_housing_forecast

Forecasting Australia housing prices by state/capital using ML (Random Forest, XGBoost) on historical ABS and RBA data.

## Setup

```bash
git clone https://github.com/EddieYung2335/aus_housing_forecast
python3 -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

## Run

Download raw data (ABS dwelling values + median price, RBA cash rate) into `data/raw`:

```bash
python src/ingestion.py
```

Build cleaned quarterly panel (`data/processed/panel.parquet`):

```bash
python src/cleaning.py
```

**Ingestion skips files already cached in `data/raw`. Delete a file there to force re-download.**

Build model features (lags, rolling stats, seasonality, region dummies) into `data/processed/features.parquet`:

```bash
python src/feature.py
```

Train baselines, Random Forest, and XGBoost, tune hyperparameters via `GridSearchCV` with time-series CV folds, and evaluate on a time-based holdout split:

```bash
python src/train.py
```

Saves whichever of the two tuned models has the lower test MAE to `models/model.joblib`, and its parameters/metrics to `models/model_metadata.json`.

Predict next quarter's median price for every region:

```bash
python src/predict.py
```

Takes each region's latest feature row, predicts the next quarter's log return, and converts it back to a price. Feature columns are read from the saved model's `feature_names_in_`, so they always match what the model was trained on. Bands are ±`test_mae` applied in log space — a typical error range, not a confidence interval. Writes `data/processed/predictions.parquet`.

**`models/model.joblib` is gitignored, so run `train.py` before `predict.py` on a fresh clone.**

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

Per region: current price, forecast with its error band, and a 10-year history chart; below that, all regions ranked by forecast change. The dashboard only reads `predictions.parquet` and `features.parquet` — it loads no model and recomputes nothing, so every number on screen traces back to a `predict.py` run.

## Data Source

- ABS Total Value of Dwellings (`total_value_dwellings.xlsx`)
- ABS Median Price of Established House Transfers (`median_price_and_number_of_transfers.xlsx`)
- RBA Cash Rate Target (`cash_rate_target.xlsx`)

Panel merges city-level series to their state `CITY_TO_STATE` map in `src/cleaning.py`, plus cash rate, on `date`.
