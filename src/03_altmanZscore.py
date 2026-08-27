"""
Stage 3: Altman (1968) Z-score - independent cross-check against the Merton model.

Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
  X1 = Working Capital / Total Assets            (liquidity)
  X2 = Retained Earnings / Total Assets           (cumulative profitability / age)
  X3 = EBIT / Total Assets                        (operating profitability)
  X4 = Market Value of Equity / Total Liabilities (leverage, market-based)
  X5 = Sales / Total Assets                       (asset turnover / efficiency)

Zones: Z > 2.99 safe | 1.81-2.99 grey zone | Z < 1.81 distress risk

Computed entirely independently of the Merton solver (Stage 2) - same raw fundamentals,
different model logic. The value of this stage is in Section 3.6's cross-validation:
do the two independent models agree on which companies are risky?

Run AFTER src/01_pull_data.py (needs data/raw/fundamentals/). Also reads
data/processed/merton_results.csv (from Stage 2) to compute rank-correlation.

Output:
  data/processed/altman_results.csv   - Z-score, X1-X5, zone per company
  data/processed/cross_validation.csv - Spearman correlations: Altman vs Merton vs tier
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
FUND_DIR = ROOT / "data" / "raw" / "fundamentals"
UNIVERSE_CSV = ROOT / "data" / "company_universe.csv"
MERTON_CSV = ROOT / "data" / "processed" / "merton_results.csv"
OUT_DIR = ROOT / "data" / "processed"

REQUIRED_FIELDS = [
    "total_assets", "total_liabilities", "current_assets", "current_liabilities",
    "retained_earnings", "ebit", "total_revenue", "market_cap",
]


def zscore_zone(z: float) -> str:
    if z > 2.99:
        return "safe"
    elif z >= 1.81:
        return "grey_zone"
    else:
        return "distress_risk"


def compute_altman(ticker: str, company: str) -> dict | None:
    fund = pd.read_csv(FUND_DIR / f"{ticker}.csv").iloc[0]

    if fund[REQUIRED_FIELDS].isna().any():
        missing = [f for f in REQUIRED_FIELDS if pd.isna(fund[f])]
        print(f"  SKIP {ticker}: missing {missing} - patch data/raw/fundamentals/{ticker}.csv")
        return None

    total_assets = fund["total_assets"]
    total_liab = fund["total_liabilities"]
    working_capital = fund["current_assets"] - fund["current_liabilities"]
    retained_earnings = fund["retained_earnings"]
    ebit = fund["ebit"]
    sales = fund["total_revenue"]
    mve = fund["market_cap"]  # market value of equity

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = mve / total_liab
    x5 = sales / total_assets

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    return {
        "ticker": ticker, "company": company,
        "X1_working_capital_ta": x1, "X2_retained_earnings_ta": x2,
        "X3_ebit_ta": x3, "X4_mve_tl": x4, "X5_sales_ta": x5,
        "altman_z": z, "altman_zone": zscore_zone(z),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(UNIVERSE_CSV)

    results = []
    for _, row in universe.iterrows():
        ticker, company = row["ticker"], row["company"]
        res = compute_altman(ticker, company)
        if res is not None:
            results.append(res)

    altman_df = pd.DataFrame(results)
    altman_df = altman_df.merge(
        universe[["ticker", "sector", "cyclicality", "approx_tier"]], on="ticker", how="left"
    )
    altman_df.to_csv(OUT_DIR / "altman_results.csv", index=False)
    print(f"\nAltman Z-scores computed for {len(altman_df)}/{len(universe)} companies "
          f"-> data/processed/altman_results.csv")

    print("\nRanked riskiest -> safest by Altman Z:")
    view = altman_df.sort_values("altman_z")
    print(view[["ticker", "altman_z", "altman_zone", "approx_tier"]].to_string(index=False))

    # --- Cross-validation against Merton (Section 3.6) ---
    if not MERTON_CSV.exists():
        print("\nmerton_results.csv not found - run src/02_merton_solver.py first "
              "for the cross-validation step.")
        return

    merton_df = pd.read_csv(MERTON_CSV)
    merton_kmv = merton_df[merton_df.default_point_convention == "kmv"][
        ["ticker", "distance_to_default", "implied_PD"]
    ]

    merged = altman_df.merge(merton_kmv, on="ticker", how="inner")
    n_common = len(merged)
    n_altman_only = len(altman_df) - n_common
    n_merton_only = len(merton_kmv) - n_common
    if n_altman_only or n_merton_only:
        print(f"\nNote: {n_altman_only} companies have an Altman score but no Merton result "
              f"(or vice versa, {n_merton_only}) - excluded from rank correlation below.")

    # Altman Z is a "higher = safer" score; Merton PD is "higher = riskier".
    # Correlate Altman Z against -PD (or equivalently DD, which is also "higher = safer")
    # so a positive rho means the two models agree on ranking.
    rho_z_dd, p_z_dd = spearmanr(merged["altman_z"], merged["distance_to_default"])
    rho_z_pd, p_z_pd = spearmanr(merged["altman_z"], -merged["implied_PD"])

    # Ordinal credit tier as a third (rough) reference point
    tier_order = {"high_grade": 3, "mid_grade": 2, "speculative": 1}  # higher = safer
    merged["tier_rank"] = merged["approx_tier"].map(tier_order)
    rho_z_tier, p_z_tier = spearmanr(merged["altman_z"], merged["tier_rank"])
    rho_dd_tier, p_dd_tier = spearmanr(merged["distance_to_default"], merged["tier_rank"])

    cross_val = pd.DataFrame([
        {"comparison": "Altman Z vs Merton Distance-to-Default", "n": n_common,
         "spearman_rho": rho_z_dd, "p_value": p_z_dd},
        {"comparison": "Altman Z vs Merton implied PD (sign-flipped)", "n": n_common,
         "spearman_rho": rho_z_pd, "p_value": p_z_pd},
        {"comparison": "Altman Z vs illustrative rating tier", "n": n_common,
         "spearman_rho": rho_z_tier, "p_value": p_z_tier},
        {"comparison": "Merton DD vs illustrative rating tier", "n": n_common,
         "spearman_rho": rho_dd_tier, "p_value": p_dd_tier},
    ])
    cross_val.to_csv(OUT_DIR / "cross_validation.csv", index=False)

    print(f"\nCross-validation (Spearman rank correlation, n={n_common}):")
    print(cross_val.to_string(index=False))

    # Flag the biggest disagreements between the two models - a real finding, not noise.
    merged["altman_rank"] = merged["altman_z"].rank(ascending=True)    # 1 = riskiest (lowest Z)
    merged["merton_rank"] = merged["implied_PD"].rank(ascending=False)  # 1 = riskiest (highest PD)
    merged["rank_gap"] = (merged["altman_rank"] - merged["merton_rank"]).abs()
    disagreements = merged.sort_values("rank_gap", ascending=False).head(5)
    print("\nBiggest ranking disagreements between Altman and Merton (report honestly, per 3.6):")
    print(disagreements[["ticker", "altman_rank", "merton_rank", "rank_gap"]].to_string(index=False))


if __name__ == "__main__":
    main()