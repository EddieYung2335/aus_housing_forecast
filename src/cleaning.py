import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def wanted_series(text: str) -> str | None:
    if "Owned by All Sectors" in text:
        return "total_value"
    if text.startswith("Mean price of residential dwellings"):
        return "mean_price"
    if text.startswith("Median Price of Established House Transfers (Unstratified) "):
        return "median_price"
    return None


def region_from(text: str) -> str:
    parts = [p.strip() for p in text.split(";") if p.strip()]
    return parts[-1]


CITY_TO_STATE = {
    "Sydney": "New South Wales",
    "Rest of NSW": "New South Wales",
    "Melbourne": "Victoria",
    "Rest of Vic.": "Victoria",
    "Brisbane": "Queensland",
    "Rest of Qld.": "Queensland",
    "Adelaide": "South Australia",
    "Rest of SA": "South Australia",
    "Perth": "Western Australia",
    "Rest of WA": "Western Australia",
    "Hobart": "Tasmania",
    "Rest of Tas.": "Tasmania",
    "Darwin": "Northern Territory",
    "Rest of NT": "Northern Territory",
    "Canberra": "Australian Capital Territory",
}


def load_abs_sheet(path: Path, sheet_name: str = "Data1") -> pd.DataFrame:
    """Peel one ABS time-series sheet into long format: [date, region, value]."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    description = raw.iloc[0]

    temp = []
    index = 0
    for text in description:
        wanted = wanted_series(str(text))
        if wanted:
            region = region_from(str(text))
            temp.append((index, wanted, region))
        index += 1

    actual_data = raw.iloc[10:]
    date_col = pd.to_datetime(actual_data.iloc[:, 0])

    frames = []

    for i, l, r in temp:
        values = actual_data.iloc[:, i]
        frame = pd.DataFrame(
            {"date": date_col, "region": r, "series": l, "value": pd.to_numeric(values)}
        )
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    return result


def load_cash_rate(path: Path) -> pd.DataFrame:
    """Peel RBA cash rate sheet into [date, cash_rate], quarterly."""
    raw = pd.read_excel(path, sheet_name="Data", header=None)

    actual_data = raw.iloc[11:]
    date_col = pd.to_datetime(actual_data.iloc[:, 0])
    rate_col = pd.to_numeric(actual_data.iloc[:, 1])

    daily = pd.DataFrame({"date": date_col, "rate": rate_col}).set_index("date")
    quarterly = daily.resample("QE").last().reset_index()
    # ABS sheets date each quarter by the 1st of its final month (eg 2011-03-01);
    # QE resample labels by quarter end (2011-03-31) — snap to match for merge.
    quarterly["date"] = quarterly["date"].values.astype("datetime64[M]")

    return quarterly


def build_panel() -> pd.DataFrame:
    """Combine both ABS series + cash rate into the tidy quarterly panel.

    total_value_dwellings is state-level, median_price is city-level.
    City rows carry the mapped state (CITY_TO_STATE) so state-level series
    can be attached to each city without conflating the two region grains.
    """
    total_value = load_abs_sheet(RAW_DIR / "total_value_dwellings.xlsx")
    median_price = load_abs_sheet(RAW_DIR / "median_price_and_number_of_transfers.xlsx")
    cash_rate = load_cash_rate(RAW_DIR / "cash_rate_target.xlsx")

    state_panel = (
        total_value.pivot(index=["date", "region"], columns="series", values="value")
        .reset_index()
        .rename(columns={"region": "state"})
    )

    city_panel = median_price.pivot(
        index=["date", "region"], columns="series", values="value"
    ).reset_index()

    city_panel["state"] = city_panel["region"].map(CITY_TO_STATE)

    unmapped = city_panel.loc[city_panel["state"].isna(), "region"].unique()

    if len(unmapped):
        raise ValueError(f"unmapped city regions, add to CITY_TO_STATE: {unmapped}")

    panel = city_panel.merge(state_panel, on=["date", "state"], how="left")
    panel = panel.merge(cash_rate, on="date", how="left")

    # Each series starts recording at a different date (eg "Rest of Qld." lacks
    # data before 2003) — leading nulls before a region's first observation are
    # expected. Only an interior gap (null after data has already started) is a bug.
    value_cols = ["median_price", "total_value", "mean_price", "rate"]
    for region, group in panel.groupby("region"):
        group = group.sort_values("date")
        for col in value_cols:
            first_valid = group[col].first_valid_index()
            if first_valid is None:
                continue
            after_start = group.loc[first_valid:, col]
            if after_start.isnull().any():
                raise ValueError(f"interior null gap in {col!r} for region {region!r}")

        dates = group["date"]
        gaps = dates.diff().dropna().dt.days > 100  # quarter ~91 days, allow slack
        if gaps.any():
            raise ValueError(f"non-continuous quarterly index for region {region!r}")

    return panel


if __name__ == "__main__":
    print("Creating panel...")
    panel = build_panel()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PROCESSED_DIR / "panel.parquet")
    print("Successfully created panel.parquet")
