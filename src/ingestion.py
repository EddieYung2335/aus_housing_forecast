# I should write two functions here
#
# Their job is to fetch the source file, and write them into data/raw

import requests
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
target_directory = project_root / "data" / "raw"

filename_abs_total = "total_value_dwellings.xlsx"
filename_abs_median_price_number_transfers = "total_price_and_number_of_transfers.xlsx"

filename_rba_cash_rate = "cash_rate_target.xlsx"


def download_abs_dwelling_values():
    # The first link https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/total-value-dwellings/mar-quarter-2026/643201.xlsx
    # The second link https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/total-value-dwellings/mar-quarter-2026/643202.xlsx

    total_value = "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/total-value-dwellings/mar-quarter-2026/643201.xlsx"

    median_price_number_transfer = "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/total-value-dwellings/mar-quarter-2026/643202.xlsx"

    full_destination_path_1 = target_directory / filename_abs_total
    target_directory.mkdir(parents=True, exist_ok=True)

    try:
        print("Fetching abs total...")
        response = requests.get(total_value, timeout=15)

        if response.status_code == 200:
            full_destination_path_1.write_bytes(response.content)

        else:
            print(f"Download failed. Status COde: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Network Error: {e}")

    full_destination_path_2 = (
        target_directory / filename_abs_median_price_number_transfers
    )
    target_directory.mkdir(parents=True, exist_ok=True)

    print("successfully download ABS dwelling values")


def download_rba_cash_rate():
    # https://www.rba.gov.au/statistics/tables/xls/f01d.xlsx
    print("successfully download RBA cash rate")


if __name__ == "__main__":
    download_abs_dwelling_values()
    download_rba_cash_rate()
