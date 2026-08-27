"""
Stage 6: DD/PD term structure (Section 3.4).

The Stage 2 solve fixes T=1 year (the standard DD/EDF convention) and produces a single
point estimate of implied asset value (V) and asset volatility (sigma_V) per company.
This stage shows that PD is NOT just that one fixed-horizon number - it's a curve.

Simplification, flagged explicitly: V and sigma_V are treated as horizon-independent firm
characteristics and held fixed at their Stage-2-solved values; only T is varied in the DD
formula itself:
    DD(T) = [ln(V/D) + (r - 0.5*sigma_V^2)*T] / (sigma_V*sqrt(T))
A fully rigorous term structure would re-solve the 2-equation Merton system separately at
each horizon (since strictly, the system's own T parameter affects the implied V/sigma_V
split too), and would use the Treasury yield of MATCHING maturity at each point (a real
yield curve) rather than one flat risk-free rate. Both are noted here as possible future
refinements rather than implemented, in the interest of a tractable, illustrative term
structure - consistent with how this kind of chart is typically produced in practice.

Companies shown: a representative handful spanning the risk spectrum (not all 26, per the
plan's "a handful of companies" framing) - two speculative-grade, two mid-tier, two
high-grade, chosen from Stage 2/3's actual risk ranking rather than picked arbitrarily.

Output:
  data/processed/term_structure.csv
  output/charts/dd_pd_term_structure.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MERTON_CSV = ROOT / "data" / "processed" / "merton_results.csv"
FRED_DTB1YR = ROOT / "data" / "raw" / "fred" / "DTB1YR.csv"
OUT_DIR = ROOT / "data" / "processed"
CHART_DIR = ROOT / "output" / "charts"

HORIZONS = [0.25, 0.5, 1, 2, 3, 5, 7, 10]  # years

# Representative subset spanning the risk spectrum, chosen from the actual Stage 2/3
# ranking rather than arbitrarily: 2 speculative-grade, 2 mid-tier, 2 high-grade.
SELECTED_TICKERS = ["LUMN", "AAL", "GE", "KHC", "JNJ", "AAPL"]


def get_current_risk_free_rate() -> float:
    if FRED_DTB1YR.exists():
        dtb = pd.read_csv(FRED_DTB1YR)
        return float(dtb.iloc[-1]["value"]) / 100.0
    print("WARNING: FRED DTB1YR not found, falling back to placeholder 3.95%")
    return 0.0395


def distance_to_default(V, D, sigma_V, r, T):
    return (np.log(V / D) + (r - 0.5 * sigma_V ** 2) * T) / (sigma_V * np.sqrt(T))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    merton = pd.read_csv(MERTON_CSV)
    kmv = merton[merton.default_point_convention == "kmv"].set_index("ticker")

    missing = [t for t in SELECTED_TICKERS if t not in kmv.index]
    if missing:
        print(f"WARNING: {missing} not found in merton_results.csv, dropping from selection")
    tickers = [t for t in SELECTED_TICKERS if t in kmv.index]

    r = get_current_risk_free_rate()
    print(f"Using risk-free rate: {r:.2%} (flat across all horizons - see docstring caveat)")

    rows = []
    for ticker in tickers:
        row = kmv.loc[ticker]
        V, D, sigma_V = row["V_implied"], row["D"], row["sigma_V_implied"]
        company = row["company"]
        for T in HORIZONS:
            dd = distance_to_default(V, D, sigma_V, r, T)
            pd_implied = norm.cdf(-dd)
            rows.append({"ticker": ticker, "company": company, "horizon_years": T,
                         "distance_to_default": dd, "implied_PD": pd_implied})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "term_structure.csv", index=False)
    print(f"\nWrote {len(out_df)} rows to data/processed/term_structure.csv\n")

    pivot_pd = out_df.pivot(index="horizon_years", columns="ticker", values="implied_PD")
    print("Implied cumulative PD by horizon:")
    print(pivot_pd.to_string())

    # --- chart ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ticker in tickers:
        sub = out_df[out_df.ticker == ticker]
        ax1.plot(sub["horizon_years"], sub["distance_to_default"], marker="o", label=ticker)
        ax2.plot(sub["horizon_years"], sub["implied_PD"], marker="o", label=ticker)

    ax1.set_xlabel("Horizon (years)")
    ax1.set_ylabel("Distance-to-Default")
    ax1.set_title("DD term structure")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Horizon (years)")
    ax2.set_ylabel("Implied cumulative PD")
    ax2.set_yscale("log")
    ax2.set_title("PD term structure (log scale)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_DIR / "dd_pd_term_structure.png", dpi=130)
    plt.close()
    print("\nChart written to output/charts/dd_pd_term_structure.png")


if __name__ == "__main__":
    main()
    