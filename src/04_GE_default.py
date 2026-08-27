"""
Stage 4: Historical backtest on General Electric (Section 3.6a).

The single highest-value addition to this project per the build plan: run the full
Merton pipeline on GE using only data that would have been available at each point in
time, and check whether Distance-to-Default / implied PD showed visible deterioration
BEFORE the well-documented 2018 credit crisis (dividend cut to $0.01/share and SEC
accounting investigation disclosed Oct 30, 2018; multi-notch downgrades from S&P/Moody's/
Fitch followed through 2018-2019).

Method:
  - Monthly snapshots from Jan 2015 (first date with a full trailing 252-day equity vol
    window) through mid-2019 (~8 months past the event, to show the deterioration play
    out and get confirmed by the actual downgrades).
  - At each snapshot date, the default point D and shares outstanding are taken from
    whichever fiscal year-end (2014-2018) balance sheet was MOST RECENTLY FILED as of
    that date - i.e. a January 2016 snapshot uses FY2015 fundamentals, a January 2017
    snapshot uses FY2016 fundamentals, etc. This respects the "use only information
    available at the time" constraint: GE's FY2018 10-K wasn't filed until Feb 2019, so
    snapshots between the Oct 2018 event and the FY2018 filing still use FY2017 data,
    which is a genuine out-of-sample test - the model doesn't get to "peek" at the crisis
    year's own balance sheet to explain the crisis.
  - Equity volatility: trailing 252-day historical vol as of each snapshot date (same
    convention as Stage 2).
  - Reuses the same KMV iterative volatility-restriction solver as Stage 2.

Simplifications, flagged explicitly (this is a backtest add-on, not a full re-derivation):
  - The default point and shares outstanding are held constant across the trailing
    252-day window used for each snapshot's volatility estimate, rather than varying
    day-by-day with the exact filing calendar. Standard practice for this kind of
    exercise; the balance sheet only updates quarterly/annually anyway.
  - Risk-free rate held at the same flat placeholder used in Stage 2 (see note there).
  - current_assets/current_liabilities aren't available for GE in this window (GE filed
    an unclassified balance sheet due to the size of GE Capital) - Altman Z-score isn't
    computed for the backtest window for this reason; this stage is Merton-only.

Output:
  data/processed/ge_backtest_trajectory.csv
  output/charts/ge_backtest_dd_trajectory.png
"""
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT / "data" / "raw" / "prices" / "GE.csv"
FUND_PATH = ROOT / "data" / "raw" / "fundamentals" / "GE_backtest_2014_2018.csv"
OUT_DIR = ROOT / "data" / "processed"
CHART_DIR = ROOT / "output" / "charts"

RISK_FREE_RATE = 0.0395   # same flat placeholder as Stage 2 - see note there
HORIZON_T = 1.0
VOL_WINDOW_DAYS = 252
MAX_ITER = 8               # capped lower than Stage 2's 25 - this runs many more solves
                            # (one full iterative restriction per monthly snapshot) so we
                            # trade a little precision for tractable runtime. Stage 2 showed
                            # convergence typically happens within 2 iterations anyway.
CONVERGENCE_TOL = 1e-4

EVENT_DATE = pd.Timestamp("2018-10-30")  # dividend cut to $0.01 + SEC investigation disclosed
EVENT_LABEL = "Oct 30 2018: dividend cut to $0.01,\nSEC accounting investigation disclosed"

SNAPSHOT_START = "2015-01-31"
SNAPSHOT_END = "2019-06-30"


# --- same Merton machinery as Stage 2 (kept self-contained per this repo's one-file-per-stage convention) ---

def merton_equations(x, E, D, r, T, sigma_E):
    V, sigma_V = x
    if V <= 0 or sigma_V <= 1e-6:
        return [1e10, 1e10]
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    eq1 = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
    eq2 = norm.cdf(d1) * sigma_V * V - sigma_E * E
    return [eq1, eq2]


def solve_single_day(E, D, r, T, sigma_E, sigma_V_guess):
    V_guess = E + D
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = fsolve(merton_equations, x0=[V_guess, sigma_V_guess],
                      args=(E, D, r, T, sigma_E), full_output=False)
    return float(sol[0]), float(sol[1])


def iterative_volatility_restriction(equity_series, D, r, T, sigma_E_hist,
                                      max_iter=MAX_ITER, tol=CONVERGENCE_TOL):
    sigma_V_guess = sigma_E_hist
    V_series = np.empty_like(equity_series)
    for _ in range(max_iter):
        V_series = np.empty_like(equity_series)
        for i, E_t in enumerate(equity_series):
            V_t, _ = solve_single_day(E_t, D, r, T, sigma_E_hist, sigma_V_guess)
            V_series[i] = V_t
        log_ret = np.diff(np.log(V_series))
        sigma_V_new = float(np.std(log_ret, ddof=1) * np.sqrt(252))
        if abs(sigma_V_new - sigma_V_guess) < tol:
            sigma_V_guess = sigma_V_new
            break
        sigma_V_guess = sigma_V_new
    return V_series, sigma_V_guess


def distance_to_default(V, D, sigma_V, r, T):
    return (np.log(V / D) + (r - 0.5 * sigma_V ** 2) * T) / (sigma_V * np.sqrt(T))


# --- backtest-specific logic ---

def load_fundamentals_by_year() -> dict:
    df = pd.read_csv(FUND_PATH)
    df["fy"] = pd.to_datetime(df["fiscal_period_end"]).dt.year
    out = {}
    for _, row in df.iterrows():
        d_kmv = row["short_term_debt"] + 0.5 * row["long_term_debt"]
        d_naive = row["total_liabilities"]
        out[int(row["fy"])] = {
            "shares_outstanding": row["shares_outstanding"],
            "D_kmv": d_kmv,
            "D_naive": d_naive,
        }
    return out


def fundamentals_as_of(snapshot_date: pd.Timestamp, fund_by_year: dict) -> dict:
    """
    Most recent fiscal-year-end balance sheet actually FILED as of snapshot_date.
    10-Ks are typically filed ~6-8 weeks after fiscal year-end, so we treat a given
    FY's data as available starting March 1 of the following year (conservative: GE's
    actual 10-K filing dates are late Feb, so March 1 is a safe cutoff).
    """
    available_years = [
        fy for fy in fund_by_year
        if pd.Timestamp(f"{fy + 1}-03-01") <= snapshot_date
    ]
    if not available_years:
        # before any fundamentals would have been filed - use earliest year as fallback
        fy = min(fund_by_year.keys())
        print(f"  WARNING: no fundamentals would have been filed yet as of "
              f"{snapshot_date.date()}, falling back to FY{fy}")
        return fund_by_year[fy]
    fy = max(available_years)
    return fund_by_year[fy]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    prices = pd.read_csv(PRICES_PATH)
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.tz_localize(None)
    prices = prices.sort_values("Date").reset_index(drop=True)

    fund_by_year = load_fundamentals_by_year()
    print(f"Fundamentals available for fiscal years: {sorted(fund_by_year.keys())}\n")

    snapshot_dates = pd.date_range(SNAPSHOT_START, SNAPSHOT_END, freq="M")

    results = []
    for snap in snapshot_dates:
        window = prices[prices["Date"] <= snap].tail(VOL_WINDOW_DAYS + 1)
        if len(window) < VOL_WINDOW_DAYS + 1:
            print(f"SKIP {snap.date()}: insufficient trailing price history "
                  f"({len(window)} days)")
            continue

        fund = fundamentals_as_of(snap, fund_by_year)
        shares_out = fund["shares_outstanding"]
        equity_series = window["Adj Close"].to_numpy() * shares_out

        px = window["Adj Close"].to_numpy()
        sigma_E_hist = float(np.std(np.diff(np.log(px)), ddof=1) * np.sqrt(252))

        row = {"snapshot_date": snap, "sigma_E_hist": sigma_E_hist,
               "E_today": equity_series[-1]}

        for convention, D in [("kmv", fund["D_kmv"]), ("naive", fund["D_naive"])]:
            V_series, sigma_V = iterative_volatility_restriction(
                equity_series, D, RISK_FREE_RATE, HORIZON_T, sigma_E_hist
            )
            V_today = V_series[-1]
            dd = distance_to_default(V_today, D, sigma_V, RISK_FREE_RATE, HORIZON_T)
            pd_implied = norm.cdf(-dd)
            row[f"D_{convention}"] = D
            row[f"sigma_V_{convention}"] = sigma_V
            row[f"DD_{convention}"] = dd
            row[f"PD_{convention}"] = pd_implied

        results.append(row)
        print(f"{snap.date()}  DD(kmv)={row['DD_kmv']:.2f}  PD(kmv)={row['PD_kmv']:.4%}  "
              f"DD(naive)={row['DD_naive']:.2f}  PD(naive)={row['PD_naive']:.4%}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_DIR / "ge_backtest_trajectory.csv", index=False)
    print(f"\nWrote {len(out_df)} monthly snapshots to data/processed/ge_backtest_trajectory.csv")

    # --- chart: DD trajectory with event marker ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(out_df["snapshot_date"], out_df["DD_kmv"], marker="o", label="KMV default point")
    ax1.plot(out_df["snapshot_date"], out_df["DD_naive"], marker="o", label="Naive default point",
              alpha=0.6)
    ax1.axvline(EVENT_DATE, color="red", linestyle="--", linewidth=1.5)
    ax1.text(EVENT_DATE, ax1.get_ylim()[1] * 0.95, EVENT_LABEL, color="red", fontsize=8,
             ha="right", va="top")
    ax1.set_ylabel("Distance-to-Default")
    ax1.set_title("General Electric: Merton Distance-to-Default, 2015-2019")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(out_df["snapshot_date"], out_df["PD_kmv"], marker="o", label="KMV default point")
    ax2.plot(out_df["snapshot_date"], out_df["PD_naive"], marker="o", label="Naive default point",
              alpha=0.6)
    ax2.axvline(EVENT_DATE, color="red", linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Implied 1-year PD")
    ax2.set_yscale("log")
    ax2.set_xlabel("Snapshot date")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    plt.savefig(CHART_DIR / "ge_backtest_dd_trajectory.png", dpi=130)
    plt.close()
    print("Chart written to output/charts/ge_backtest_dd_trajectory.png")


if __name__ == "__main__":
    main()