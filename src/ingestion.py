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

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

target_directory.mkdir(parents=True, exist_ok=True)


def download_abs_dwelling_values():
    total_value = "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/total-value-dwellings/mar-quarter-2026/643201.xlsx"

    median_price_number_transfer = "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/total-value-dwellings/mar-quarter-2026/643202.xlsx"

    full_destination_path_1 = target_directory / filename_abs_total

    if full_destination_path_1.exists():
        print(f"Skipped download: '{filename_abs_total}' already exists in cache.")
    else:
        try:
            print("Fetching abs total...")
            response = requests.get(total_value, headers=headers, timeout=15)

            if response.status_code == 200:
                full_destination_path_1.write_bytes(response.content)
                print("Successfully download Total value of dwellings, all series")

            else:
                print(f"Download failed. Status Code: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"Network Error: {e}")

    full_destination_path_2 = (
        target_directory / filename_abs_median_price_number_transfers
    )

    if full_destination_path_2.exists():
        print(
            f"Skipped download: '{filename_abs_median_price_number_transfers}' already exists in cache."
        )

    else:
        try:
            print("Fetching ABS median price...")
            response = requests.get(
                median_price_number_transfer, headers=headers, timeout=15
            )

            if response.status_code == 200:
                full_destination_path_2.write_bytes(response.content)
                print("Successfully download median price and number of transfers")

            else:
                print(f"Download failed. Status Code: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"Network Error: {e}")


def download_rba_cash_rate():
    # https://www.rba.gov.au/statistics/tables/xls/f01d.xlsx
    print("successfully download RBA cash rate")


if __name__ == "__main__":
    download_abs_dwelling_values()
    download_rba_cash_rate()
