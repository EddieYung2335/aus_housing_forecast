# aus_housing_forecast

Forecasting Australia housing prices by state/capital using ML (Random Forest, XGBoost) on historical ABS and RBA data.

## Setup

```bash
git clone https://github.com/EddieYung2335/aus_housing_forecast
cd BrisHouse
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

## Data Source

- ABS Total Value of Dwellings (`total_value_dwellings.xlsx`)
- ABS Median Price of Established House Transfers (`median_price_and_number_of_transfers.xlsx`)
- RBA Cash Rate Target (`cash_rate_target.xlsx`)

Panel merges city-level series to their state `CITY_TO_STATE` map in `src/cleaning.py`, plus cash rate, on `date`.

