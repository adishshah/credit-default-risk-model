"""
Stage 2: Merton (1974) structural default model.

For each company:
  1. Estimate historical equity volatility (sigma_E) from daily log returns.
  2. Compute the default point D under both KMV-style conventions:
       - naive: D = total_liabilities
       - kmv:   D = short_term_debt + 0.5 * long_term_debt
  3. Solve the Merton system for implied asset value (V) and asset volatility (sigma_V)
     via scipy.optimize.fsolve, given a starting guess sigma_V = sigma_E.
  4. Iterate the KMV volatility restriction: re-solve the system day-by-day using the
     current sigma_V estimate to build an implied daily asset-value series, recompute
     sigma_V as the annualized vol of that series' log returns, and repeat until sigma_V
     converges (or max_iter is hit). Record every iteration's sigma_V for a convergence plot.
  5. Compute Distance-to-Default: DD = [ln(V/D) + (r - 0.5*sigma_V^2)*T] / (sigma_V*sqrt(T))
  6. Compute implied PD = N(-DD)  (the standard Merton/KMV mapping; KMV's proprietary EDF
     recalibrates this against a real default database, which we don't have - this script
     uses the theoretical normal-CDF mapping and is explicit about that).

Run AFTER src/01_pull_data.py. Reads data/raw/{prices,fundamentals}/, writes
data/processed/merton_results.csv and output/charts/vol_convergence_{ticker}.png
for a handful of names.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PRICES_DIR = ROOT / "data" / "raw" / "prices"
FUND_DIR = ROOT / "data" / "raw" / "fundamentals"
UNIVERSE_CSV = ROOT / "data" / "company_universe.csv"
OUT_DIR = ROOT / "data" / "processed"
CHART_DIR = ROOT / "output" / "charts"

# --- assumptions, flagged for later replacement with real pulled data ---
RISK_FREE_RATE = 0.0395   # placeholder: current ~1yr Treasury yield (Aug 2026).
                           # Replace with an actual FRED DTB1YR pull in Stage 5.
HORIZON_T = 1.0            # years, matching standard DD/EDF convention (Section 3.4)
VOL_WINDOW_DAYS = 252      # ~1 trading year of daily returns for historical equity vol
MAX_ITER = 25
CONVERGENCE_TOL = 1e-4      # stop when |sigma_V change| < this


def historical_equity_vol(prices: pd.DataFrame, window: int = VOL_WINDOW_DAYS) -> float:
    """Annualized historical equity volatility from daily log returns."""
    px = prices["Adj Close"].tail(window + 1).to_numpy()
    log_ret = np.diff(np.log(px))
    return float(np.std(log_ret, ddof=1) * np.sqrt(252))


def default_points(fund_row: pd.Series) -> dict:
    naive = fund_row["total_liabilities"]
    kmv = fund_row["short_term_debt"] + 0.5 * fund_row["long_term_debt"]
    return {"naive": naive, "kmv": kmv}


def merton_equations(x, E, D, r, T, sigma_E):
    """
    x = [V, sigma_V]. Returns residuals of the two Merton equations:
      E = V*N(d1) - D*exp(-rT)*N(d2)
      sigma_E * E = N(d1) * sigma_V * V
    """
    V, sigma_V = x
    if V <= 0 or sigma_V <= 1e-6:
        return [1e10, 1e10]  # push solver away from invalid region
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    eq1 = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
    eq2 = norm.cdf(d1) * sigma_V * V - sigma_E * E
    return [eq1, eq2]


def solve_single_day(E, D, r, T, sigma_E, sigma_V_guess):
    """Solve the 2x2 Merton system for one day's (E, sigma_E, D) given a sigma_V guess."""
    V_guess = E + D  # sensible starting point: assets ~ equity + debt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = fsolve(merton_equations, x0=[V_guess, sigma_V_guess],
                      args=(E, D, r, T, sigma_E), full_output=False)
    V, sigma_V = float(sol[0]), float(sol[1])
    return V, sigma_V


def iterative_volatility_restriction(equity_series: np.ndarray, D: float, r: float, T: float,
                                      sigma_E_hist: float, shares_out: float,
                                      max_iter: int = MAX_ITER, tol: float = CONVERGENCE_TOL):
    """
    KMV-style iteration:
      - Start with sigma_V_guess = sigma_E_hist (equity vol as a first approximation).
      - Solve the Merton system for EVERY day in equity_series using the current
        sigma_V_guess, producing a daily implied asset-value series V_t.
      - Recompute sigma_V from the annualized vol of log(V_t) returns.
      - Repeat until sigma_V stops moving (or max_iter reached).

    equity_series: daily market cap (E_t) over the vol window, in dollars.
    Returns: (V_final series, sigma_V_final, convergence_history list of sigma_V per iteration)
    """
    sigma_V_guess = sigma_E_hist
    convergence_history = [sigma_V_guess]
    V_series = np.empty_like(equity_series)

    for iteration in range(max_iter):
        V_series = np.empty_like(equity_series)
        for i, E_t in enumerate(equity_series):
            V_t, _ = solve_single_day(E_t, D, r, T, sigma_E_hist, sigma_V_guess)
            V_series[i] = V_t

        log_ret = np.diff(np.log(V_series))
        sigma_V_new = float(np.std(log_ret, ddof=1) * np.sqrt(252))

        convergence_history.append(sigma_V_new)
        if abs(sigma_V_new - sigma_V_guess) < tol:
            sigma_V_guess = sigma_V_new
            break
        sigma_V_guess = sigma_V_new

    return V_series, sigma_V_guess, convergence_history


def distance_to_default(V, D, sigma_V, r, T):
    dd = (np.log(V / D) + (r - 0.5 * sigma_V ** 2) * T) / (sigma_V * np.sqrt(T))
    return dd


def plot_convergence(ticker: str, history: list, out_path: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(history)), history, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Implied asset volatility (sigma_V)")
    plt.title(f"KMV volatility-restriction convergence: {ticker}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def run_one_company(ticker: str, company: str) -> list:
    prices = pd.read_csv(PRICES_DIR / f"{ticker}.csv")
    fund = pd.read_csv(FUND_DIR / f"{ticker}.csv").iloc[0]

    required_fields = ["total_liabilities", "short_term_debt", "long_term_debt", "shares_outstanding"]
    if fund[required_fields].isna().any():
        missing = [f for f in required_fields if pd.isna(fund[f])]
        print(f"  SKIP {ticker}: missing {missing} - patch data/raw/fundamentals/{ticker}.csv "
              f"from the 10-K before re-running")
        return []

    shares_out = fund["shares_outstanding"]
    prices = prices.tail(VOL_WINDOW_DAYS + 1).reset_index(drop=True)
    equity_series = prices["Adj Close"].to_numpy() * shares_out  # daily market cap proxy

    sigma_E_hist = historical_equity_vol(prices)
    E_today = equity_series[-1]

    dpoints = default_points(fund)
    results = []

    for convention, D in dpoints.items():
        if D <= 0 or np.isnan(D):
            print(f"  SKIP {ticker}/{convention}: non-positive or missing default point")
            continue

        V_series, sigma_V, history = iterative_volatility_restriction(
            equity_series, D, RISK_FREE_RATE, HORIZON_T, sigma_E_hist, shares_out
        )
        V_today = V_series[-1]
        dd = distance_to_default(V_today, D, sigma_V, RISK_FREE_RATE, HORIZON_T)
        pd_implied = norm.cdf(-dd)

        if convention == "kmv":  # save one convergence chart per company, KMV convention
            CHART_DIR.mkdir(parents=True, exist_ok=True)
            plot_convergence(ticker, history, CHART_DIR / f"vol_convergence_{ticker}.png")

        results.append({
            "ticker": ticker, "company": company, "default_point_convention": convention,
            "E_today": E_today, "D": D, "sigma_E_hist": sigma_E_hist,
            "V_implied": V_today, "sigma_V_implied": sigma_V,
            "n_iterations": len(history) - 1, "converged": len(history) - 1 < MAX_ITER,
            "distance_to_default": dd, "implied_PD": pd_implied,
            "risk_free_rate": RISK_FREE_RATE, "horizon_years": HORIZON_T,
        })

    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(UNIVERSE_CSV)

    all_results = []
    for _, row in universe.iterrows():
        ticker, company = row["ticker"], row["company"]
        print(f"Solving Merton system for {ticker} ({company})...")
        try:
            res = run_one_company(ticker, company)
            all_results.extend(res)
        except Exception as e:
            print(f"  ERROR on {ticker}: {e}")

    out_df = pd.DataFrame(all_results)
    out_df.to_csv(OUT_DIR / "merton_results.csv", index=False)
    print(f"\nDone. {len(out_df)} rows written to data/processed/merton_results.csv")
    print(f"Convergence charts written to output/charts/vol_convergence_*.png")

    # quick sanity print: ranked by KMV PD
    kmv_view = out_df[out_df.default_point_convention == "kmv"].sort_values("implied_PD", ascending=False)
    print("\nTop 10 riskiest by Merton implied PD (KMV default point):")
    print(kmv_view[["ticker", "distance_to_default", "implied_PD"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
