"""
Stage 4a: Pull General Electric's 2014-2018 annual balance sheet data directly from
SEC EDGAR's XBRL "companyconcept" API for the historical backtest (Section 3.6a).

Why this exists: yfinance's free fundamentals only go back ~4 years, nowhere near far
enough for the pre-2018-downgrade backtest window. SEC's XBRL API returns every value a
company has EVER reported for a given line item (a "concept"/tag), straight from the
filed, audited 10-K - exactly what we need, and it's free with no API key.

This script queries GE's CIK (0000040545) for each XBRL concept we need, keeps only
annual (10-K, fp="FY") data points, and picks the value for each fiscal year-end in
2014-2018. Writes one row per fiscal year to:

    data/raw/fundamentals/GE_backtest_2014_2018.csv

MUST be run somewhere with open internet access to data.sec.gov (same constraint as
01_pull_data.py - won't work in a network-restricted sandbox).

SEC's Fair Access policy requires a descriptive User-Agent header with contact info -
edit CONTACT_EMAIL below before running, or SEC may reject/rate-limit the requests.
"""
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "fundamentals" / "GE_backtest_2014_2018.csv"

GE_CIK = "0000040545"
FISCAL_YEARS = [2014, 2015, 2016, 2017, 2018]

CONTACT_EMAIL = "adishshah2943@gmail.com"  # <-- EDIT THIS before running
HEADERS = {"User-Agent": f"corp-default-risk-project {CONTACT_EMAIL}"}

# Map our field names -> candidate XBRL tags to try, in priority order.
# GE's filings sometimes use different tags across years as accounting standards
# and GE's own reporting structure changed (e.g. GE Capital deconsolidation).
CONCEPT_CANDIDATES = {
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "short_term_debt": ["ShortTermBorrowings", "DebtCurrent", "LongTermDebtCurrent"],
    "long_term_debt": [
        "LongTermDebtNoncurrent", "LongTermDebt", "LongTermBorrowings",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    ],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "ebit": [
        "OperatingIncomeLoss", "OperatingIncomeLossFromContinuingOperations",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "total_revenue": ["Revenues", "SalesRevenueNet"],
    "shares_outstanding": ["CommonStockSharesOutstanding", "CommonStockSharesIssued"],
}


def fetch_concept(cik: str, tag: str) -> dict | None:
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return None
    return resp.json()


def extract_annual_values(concept_json: dict) -> dict:
    """
    Returns {fiscal_year_end_date_str: value} for 10-K annual ("FY") data points only.
    Some concepts are reported multiple times (original + amended filings) - keep the
    latest-filed value for each period end.
    """
    out = {}
    units = concept_json.get("units", {})
    # most concepts are USD; shares outstanding is reported in "shares"
    for unit_key, entries in units.items():
        for e in entries:
            if e.get("form") != "10-K" or e.get("fp") != "FY":
                continue
            end = e.get("end")
            if end is None:
                continue
            if end not in out or e.get("filed", "") > out[end].get("filed", ""):
                out[end] = e
    return out


def get_field_for_years(field: str, tags: list, years: list) -> dict:
    """Try each candidate tag until one returns data; return {year: value}."""
    for tag in tags:
        print(f"  trying {field} <- us-gaap:{tag} ...")
        data = fetch_concept(GE_CIK, tag)
        time.sleep(0.3)  # be polite to SEC's servers
        if data is None:
            continue
        annual = extract_annual_values(data)
        result = {}
        for end_date, entry in annual.items():
            fy = int(end_date[:4])
            # GE's fiscal year end is Dec 31, so end date year == fiscal year
            if fy in years and end_date.endswith("12-31"):
                result[fy] = entry["val"]
        if result:
            print(f"    found {len(result)} year(s) via {tag}: {sorted(result.keys())}")
            return result
    print(f"    WARNING: no data found for {field} under any candidate tag")
    return {}


def main():
    if "your_email" in CONTACT_EMAIL:
        print("EDIT CONTACT_EMAIL at the top of this script before running "
              "(SEC requires a descriptive User-Agent).")
        return

    print(f"Pulling GE (CIK {GE_CIK}) annual XBRL data for {FISCAL_YEARS} from SEC EDGAR...\n")

    all_fields = {}
    for field, tags in CONCEPT_CANDIDATES.items():
        all_fields[field] = get_field_for_years(field, tags, FISCAL_YEARS)

    rows = []
    for fy in FISCAL_YEARS:
        row = {"ticker": "GE", "fiscal_period_end": f"{fy}-12-31"}
        for field in CONCEPT_CANDIDATES:
            row[field] = all_fields[field].get(fy)
        rows.append(row)

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nWrote {len(df)} rows to {OUT_PATH}")
    print(df.to_string(index=False))

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"\nFields with gaps (may need a manual patch from the 10-K PDF directly):")
        print(missing.to_string())


if __name__ == "__main__":
    main()