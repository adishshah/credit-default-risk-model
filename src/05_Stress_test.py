"""
Stage 5: Macro stress testing (Section 3.7) - simplified, illustrative CCAR/DFAST-style.

Explicitly NOT a full macro-econometric model - stated here and in the memo.

Methodology:
  1. Measure the REAL 2008 and 2020 equity drawdowns (Nasdaq Composite, since FRED
     discontinued Wilshire 5000 in June 2024) alongside the REAL unemployment-rate
     increase and real-GDP decline over the same windows, all from data/raw/fred/.
  2. From those two historical episodes, compute an empirical sensitivity: how much
     equity drawdown historically accompanied each percentage-point of unemployment
     increase, and each percent of GDP decline. Average the two episodes' sensitivities
     per channel (they differ meaningfully - 2008 was a slower, deeper equity decline
     per unit of unemployment/GDP shock than 2020's sharp-but-short COVID crash; this
     divergence is reported honestly rather than papered over).
  3. Define baseline / adverse / severely-adverse scenarios as unemployment-change and
     GDP-decline magnitudes (severely adverse is anchored to the actual 2008 episode;
     adverse is set to half that severity), then run those magnitudes through the
     empirical sensitivity from step 2 to get an aggregate market equity shock per
     scenario - NOT a round number picked by hand.
  4. Differentiate by sector cyclicality (company_universe.csv): cyclical sectors get a
     larger multiplier on the aggregate shock, defensive sectors a smaller one. The
     multipliers themselves (1.35x / 0.65x) are a standard equity-beta stylized fact from
     the finance literature, not derived from this project's own price data - our price
     history doesn't reach back to 2008 for most of the universe, so a genuine sector-beta
     re-estimation isn't possible here. Flagged clearly as an assumption, not a result.
  5. Recompute DD/PD per company per scenario: shock E (today's equity value) by the
     company's scenario-specific equity shock, hold the Stage-2-implied asset volatility
     (sigma_V) fixed, and re-solve the single Merton pricing equation for the new implied
     asset value V under the shocked E. Holding sigma_V fixed rather than re-running the
     full iterative volatility restriction is the simplifying assumption that makes this
     a fast, tractable illustrative stress test rather than a full re-derivation - flagged
     explicitly per the plan's own framing of this stage.

Run AFTER src/02_merton_solver.py and src/05a_pull_fred_data.py.

Output:
  data/processed/stress_test_results.csv
  data/processed/stress_scenario_calibration.csv
  output/charts/stress_test_pd_by_scenario.png
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FRED_DIR = ROOT / "data" / "raw" / "fred"
MERTON_CSV = ROOT / "data" / "processed" / "merton_results.csv"
UNIVERSE_CSV = ROOT / "data" / "company_universe.csv"
OUT_DIR = ROOT / "data" / "processed"
CHART_DIR = ROOT / "output" / "charts"

HORIZON_T = 1.0

# Sector-beta stylized-fact multipliers (see module docstring, point 4) - assumption, not
# derived from this project's own data.
CYCLICAL_MULTIPLIER = 1.35
DEFENSIVE_MULTIPLIER = 0.65


def measure_historical_episode(nasdaq, unrate, gdp, peak_window, trough_window,
                                 u_before_window, u_after_window,
                                 gdp_before_window, gdp_after_window):
    peak = nasdaq[(nasdaq.date >= peak_window[0]) & (nasdaq.date <= peak_window[1])]["value"].max()
    trough = nasdaq[(nasdaq.date >= trough_window[0]) & (nasdaq.date <= trough_window[1])]["value"].min()
    equity_dd = trough / peak - 1

    u_before = unrate[(unrate.date >= u_before_window[0]) & (unrate.date <= u_before_window[1])]["value"].min()
    u_after = unrate[(unrate.date >= u_after_window[0]) & (unrate.date <= u_after_window[1])]["value"].max()
    u_change = u_after - u_before

    g_before = gdp[(gdp.date >= gdp_before_window[0]) & (gdp.date <= gdp_before_window[1])]["value"].max()
    g_after = gdp[(gdp.date >= gdp_after_window[0]) & (gdp.date <= gdp_after_window[1])]["value"].min()
    gdp_decline = g_after / g_before - 1

    return {"equity_drawdown": equity_dd, "unemployment_change_pp": u_change,
            "gdp_decline_pct": gdp_decline}


def load_fred_calibration_data():
    nasdaq = pd.read_csv(FRED_DIR / "NASDAQCOM.csv")
    nasdaq["date"] = pd.to_datetime(nasdaq["date"])
    unrate = pd.read_csv(FRED_DIR / "UNRATE.csv")
    unrate["date"] = pd.to_datetime(unrate["date"])
    gdp = pd.read_csv(FRED_DIR / "GDPC1.csv")
    gdp["date"] = pd.to_datetime(gdp["date"])

    ep_2008 = measure_historical_episode(
        nasdaq, unrate, gdp,
        peak_window=("2007-09-01", "2007-11-15"), trough_window=("2008-11-01", "2009-03-31"),
        u_before_window=("2007-01-01", "2007-12-31"), u_after_window=("2009-01-01", "2010-06-30"),
        gdp_before_window=("2007-01-01", "2008-06-30"), gdp_after_window=("2008-06-30", "2009-12-31"),
    )
    ep_2020 = measure_historical_episode(
        nasdaq, unrate, gdp,
        peak_window=("2020-01-01", "2020-02-20"), trough_window=("2020-03-01", "2020-03-31"),
        u_before_window=("2020-01-01", "2020-02-28"), u_after_window=("2020-03-01", "2020-06-30"),
        gdp_before_window=("2019-06-30", "2020-03-31"), gdp_after_window=("2020-03-31", "2020-09-30"),
    )
    return ep_2008, ep_2020


def compute_sensitivities(ep_2008, ep_2020):
    u_sens_2008 = ep_2008["equity_drawdown"] / ep_2008["unemployment_change_pp"]
    u_sens_2020 = ep_2020["equity_drawdown"] / ep_2020["unemployment_change_pp"]
    u_sens_avg = (u_sens_2008 + u_sens_2020) / 2

    g_sens_2008 = ep_2008["equity_drawdown"] / ep_2008["gdp_decline_pct"]
    g_sens_2020 = ep_2020["equity_drawdown"] / ep_2020["gdp_decline_pct"]
    g_sens_avg = (g_sens_2008 + g_sens_2020) / 2

    return {
        "unemployment_sensitivity": u_sens_avg, "unemployment_sensitivity_2008": u_sens_2008,
        "unemployment_sensitivity_2020": u_sens_2020,
        "gdp_sensitivity": g_sens_avg, "gdp_sensitivity_2008": g_sens_2008,
        "gdp_sensitivity_2020": g_sens_2020,
    }


def define_scenarios(ep_2008):
    """
    Severely adverse anchored to the ACTUAL 2008 magnitude (a common CCAR practice:
    ground the tail scenario in a real historical episode rather than an arbitrary
    number). Adverse set to half that severity. Baseline = no shock.
    """
    return {
        "baseline": {"unemployment_change_pp": 0.0, "gdp_decline_pct": 0.0},
        "adverse": {"unemployment_change_pp": ep_2008["unemployment_change_pp"] / 2,
                    "gdp_decline_pct": ep_2008["gdp_decline_pct"] / 2},
        "severely_adverse": {"unemployment_change_pp": ep_2008["unemployment_change_pp"],
                              "gdp_decline_pct": ep_2008["gdp_decline_pct"]},
    }


def scenario_equity_shock(scenario, sensitivities):
    u_component = sensitivities["unemployment_sensitivity"] * scenario["unemployment_change_pp"]
    g_component = sensitivities["gdp_sensitivity"] * scenario["gdp_decline_pct"]
    return 0.5 * u_component + 0.5 * g_component  # equal-weight blend of both channels


def get_current_risk_free_rate() -> float:
    dtb = pd.read_csv(FRED_DIR / "DTB1YR.csv")
    return float(dtb.iloc[-1]["value"]) / 100.0


# --- Merton re-solve under a shocked equity value, holding sigma_V fixed (see docstring point 5) ---

def merton_price_equation(V, E, D, r, T, sigma_V):
    if V <= 0:
        return 1e10
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    return V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E


def solve_V_given_shocked_E(E_shocked, D, r, T, sigma_V, V_guess):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = fsolve(merton_price_equation, x0=[V_guess], args=(E_shocked, D, r, T, sigma_V))
    return float(sol[0])


def distance_to_default(V, D, sigma_V, r, T):
    return (np.log(V / D) + (r - 0.5 * sigma_V ** 2) * T) / (sigma_V * np.sqrt(T))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Calibrating scenarios to real 2008/2020 data ===\n")
    ep_2008, ep_2020 = load_fred_calibration_data()
    print(f"2008 crisis (measured from FRED): equity drawdown {ep_2008['equity_drawdown']:.1%}, "
          f"unemployment +{ep_2008['unemployment_change_pp']:.1f}pp, "
          f"GDP {ep_2008['gdp_decline_pct']:.1%}")
    print(f"2020 crisis (measured from FRED): equity drawdown {ep_2020['equity_drawdown']:.1%}, "
          f"unemployment +{ep_2020['unemployment_change_pp']:.1f}pp, "
          f"GDP {ep_2020['gdp_decline_pct']:.1%}")

    sens = compute_sensitivities(ep_2008, ep_2020)
    print(f"\nEmpirical sensitivities (equity drawdown per unit of macro shock):")
    print(f"  Unemployment channel: 2008={sens['unemployment_sensitivity_2008']:.3f}/pp, "
          f"2020={sens['unemployment_sensitivity_2020']:.3f}/pp, avg={sens['unemployment_sensitivity']:.3f}/pp")
    print(f"  GDP channel:          2008={sens['gdp_sensitivity_2008']:.2f}x, "
          f"2020={sens['gdp_sensitivity_2020']:.2f}x, avg={sens['gdp_sensitivity']:.2f}x")
    print(f"  NOTE: the two episodes disagree substantially on sensitivity per channel - "
          f"2008 was a slower, deeper decline per unit of macro shock than 2020's sharp "
          f"V-shaped crash. Averaging them is a simplification, reported honestly.")

    scenarios = define_scenarios(ep_2008)
    scenario_shocks = {name: scenario_equity_shock(s, sens) for name, s in scenarios.items()}

    print(f"\nScenario definitions and resulting AGGREGATE market equity shock:")
    calib_rows = []
    for name, s in scenarios.items():
        shock = scenario_shocks[name]
        print(f"  {name:18s}: unemployment +{s['unemployment_change_pp']:.1f}pp, "
              f"GDP {s['gdp_decline_pct']:.1%}  ->  aggregate equity shock {shock:.1%}")
        calib_rows.append({"scenario": name, **s, "aggregate_equity_shock": shock})
    pd.DataFrame(calib_rows).to_csv(OUT_DIR / "stress_scenario_calibration.csv", index=False)

    r_current = get_current_risk_free_rate()
    print(f"\nCurrent 1yr Treasury (FRED DTB1YR, latest obs): {r_current:.2%} "
          f"(replaces the Stage 2/4 placeholder)")

    # --- apply to all companies ---
    merton = pd.read_csv(MERTON_CSV)
    merton_kmv = merton[merton.default_point_convention == "kmv"].copy()
    universe = pd.read_csv(UNIVERSE_CSV)[["ticker", "sector", "cyclicality"]]
    merged = merton_kmv.merge(universe, on="ticker", how="left")

    print(f"\n=== Recomputing DD/PD for {len(merged)} companies under each scenario ===")

    results = []
    for _, row in merged.iterrows():
        ticker = row["ticker"]
        E0, D, sigma_V = row["E_today"], row["D"], row["sigma_V_implied"]
        V0 = row["V_implied"]
        mult = CYCLICAL_MULTIPLIER if row["cyclicality"] == "cyclical" else DEFENSIVE_MULTIPLIER

        for scen_name, agg_shock in scenario_shocks.items():
            company_shock = agg_shock * mult
            E_shocked = E0 * (1 + company_shock)
            V_shocked = solve_V_given_shocked_E(E_shocked, D, r_current, HORIZON_T, sigma_V, V0)
            dd = distance_to_default(V_shocked, D, sigma_V, r_current, HORIZON_T)
            pd_implied = norm.cdf(-dd)

            results.append({
                "ticker": ticker, "company": row["company"], "sector": row["sector"],
                "cyclicality": row["cyclicality"], "scenario": scen_name,
                "aggregate_market_shock": agg_shock, "company_equity_shock": company_shock,
                "E_shocked": E_shocked, "V_shocked": V_shocked,
                "distance_to_default": dd, "implied_PD": pd_implied,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "stress_test_results.csv", index=False)
    print(f"\nWrote {len(results_df)} rows to data/processed/stress_test_results.csv")

    # --- which companies are most stress-sensitive? ---
    pivot = results_df.pivot(index="ticker", columns="scenario", values="implied_PD")
    pivot["PD_increase_severely_adverse"] = pivot["severely_adverse"] - pivot["baseline"]
    most_sensitive = pivot.sort_values("PD_increase_severely_adverse", ascending=False).head(10)
    print("\nMost stress-sensitive companies (largest PD increase, baseline -> severely adverse):")
    print(most_sensitive[["baseline", "adverse", "severely_adverse", "PD_increase_severely_adverse"]]
          .to_string())

    # --- chart ---
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_df = results_df.copy()
    plot_df["scenario"] = pd.Categorical(plot_df["scenario"],
                                          categories=["baseline", "adverse", "severely_adverse"],
                                          ordered=True)
    order = (plot_df[plot_df.scenario == "severely_adverse"]
             .sort_values("implied_PD", ascending=False)["ticker"].tolist())

    for scen, color in zip(["baseline", "adverse", "severely_adverse"],
                            ["#2ca02c", "#ff7f0e", "#d62728"]):
        sub = plot_df[plot_df.scenario == scen].set_index("ticker").loc[order]
        ax.plot(sub.index, sub["implied_PD"], marker="o", label=scen.replace("_", " "), color=color)

    ax.set_yscale("log")
    ax.set_ylabel("Implied 1-year PD (log scale)")
    ax.set_xlabel("Ticker (sorted by severely-adverse PD)")
    ax.set_title("Implied PD by macro scenario, all companies")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "stress_test_pd_by_scenario.png", dpi=130)
    plt.close()
    print("\nChart written to output/charts/stress_test_pd_by_scenario.png")


if __name__ == "__main__":
    main()