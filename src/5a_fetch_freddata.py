"""
Stage 5a: Pull macro data from FRED for the stress-testing scenario calibration (3.7).

Series pulled:
  DTB1YR        - 1-year Treasury yield (also replaces the placeholder risk-free rate
                  used in Stages 2 and 4)
  UNRATE        - civilian unemployment rate, monthly, back to 1948
  GDPC1         - real GDP, quarterly, back to 1947
  BAMLH0A0HYM2  - ICE BofA US High Yield Option-Adjusted Spread, daily
  BAMLC0A0CM    - ICE BofA US Investment Grade Option-Adjusted Spread, daily
  WILL5000PRFC  - Wilshire 5000 Full Cap Price Index, daily, back to the 1970s.
                  Used (not FRED's own "SP500" series, which only starts 2015) as the
                  broad-market equity benchmark to empirically measure the 2008 and 2020
                  drawdowns for scenario calibration.

Requires a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
Paste it into FRED_API_KEY below before running.

MUST be run somewhere with open internet access to fred.stlouisfed.org (same constraint
as 01_pull_data.py - won't work in a network-restricted sandbox).

Output: data/raw/fred/{series_id}.csv  (date, value)
        data/raw/fred/fred_pull_log.csv
"""
from pathlib import Path

import pandas as pd
import requests

FRED_API_KEY = "USE_YOUR_FRED_API_KEY"  # <-- EDIT THIS before running

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "fred"

SERIES = {
    "DTB1YR": "1-year Treasury yield",
    "UNRATE": "Civilian unemployment rate",
    "GDPC1": "Real GDP",
    "BAMLH0A0HYM2": "US High Yield OAS",
    "BAMLC0A0CM": "US Investment Grade OAS",
    "NASDAQCOM": "Nasdaq Composite Index",
}

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(series_id: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": "1990-01-01",  # covers both 2008 and 2020 with margin
    }
    resp = requests.get(BASE_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    obs = data.get("observations", [])
    df = pd.DataFrame(obs)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    return df


def main():
    if FRED_API_KEY == "PASTE_YOUR_KEY_HERE":
        print("EDIT FRED_API_KEY at the top of this script before running. "
              "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_rows = []

    for series_id, description in SERIES.items():
        print(f"Pulling {series_id} ({description})...")
        try:
            df = fetch_series(series_id)
            df.to_csv(OUT_DIR / f"{series_id}.csv", index=False)
            print(f"  {len(df)} observations, {df['date'].min()} to {df['date'].max()}")
            log_rows.append({"series_id": series_id, "status": "ok", "n_obs": len(df),
                              "start": df["date"].min(), "end": df["date"].max()})
        except Exception as e:
            print(f"  FAILED: {e}")
            log_rows.append({"series_id": series_id, "status": "FAILED", "n_obs": 0,
                              "start": "", "end": ""})

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(OUT_DIR / "fred_pull_log.csv", index=False)
    print(f"\nDone. See data/raw/fred/fred_pull_log.csv for a summary.")


if __name__ == "__main__":
    main()