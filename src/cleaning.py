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

    return quarterly
