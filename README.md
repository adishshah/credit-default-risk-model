# Corporate Default Risk + Macro Stress Test

A structural credit risk model (Merton 1974) estimating Distance-to-Default and
Probability of Default for 26 public companies, cross-checked against Altman Z-scores,
backtested against a real historical credit deterioration (General Electric, 2015-2019),
and extended with a CCAR/DFAST-style macro stress test calibrated to real 2008 and 2020
FRED data. Final output is a risk-committee-style credit memo.

This is the corporate/wholesale-credit counterpart to a retail credit scorecard project —
together they cover both halves of what a credit risk desk does.

**Read the full memo:** `output/memo/Corporate_Default_Risk_Credit_Memo.docx` — executive
summary, ranked risk table, and all findings in one document.

## Status: Complete

- [x] Stage 1: Data pull — 26 companies' equity prices + balance sheet fundamentals
- [x] Stage 2: Merton system solver + KMV iterative volatility restriction
- [x] Stage 3: Altman Z-score + cross-validation (Spearman rho = 0.81 vs. Merton)
- [x] Stage 4: Historical backtest — General Electric, 2015-2019
- [x] Stage 5: Macro stress testing, calibrated to real 2008/2020 FRED data
- [x] Stage 6: DD/PD term structure across multiple horizons
- [x] Stage 7: Credit risk memo (Word doc, executive-summary-first format)

## Key Findings

- **Cross-model validation**: the Merton (market-based) and Altman (accounting-based)
  models agree strongly on relative risk ranking (Spearman rho = 0.81, p < 0.0001) despite
  using entirely different inputs — genuine validation, not circular reasoning.
- **Historical backtest works**: running the model on General Electric using only data
  that would have been available at the time shows Distance-to-Default declining
  continuously from a peak of 20.7 (March 2017) to 4.6 (December 2018) — more than a year
  before GE's October 2018 dividend cut and SEC investigation became public.
- **Stress test calibrated to real data, not round numbers**: scenarios are anchored to
  the actual measured 2008 equity drawdown (-55.6%, from Nasdaq Composite data on FRED,
  since FRED discontinued the Wilshire 5000 series in 2024) and unemployment/GDP shocks,
  not hand-picked magnitudes.
- **A genuine methodology finding, reported honestly**: sector-differentiated stress
  sensitivity is invisible on a naive PD-fold-change metric (masked by the normal
  distribution's tail convexity) but clearly present on the linear Distance-to-Default
  scale — cyclical sectors deteriorate 37% more than defensive sectors, exactly as
  designed. Full writeup in the memo, Section 4.

## Why network access mattered during development

This repo was originally scaffolded in a sandboxed environment that couldn't reach
`query1/2.finance.yahoo.com`, `fred.stlouisfed.org`, or `sec.gov`. All data-pulling
scripts (`01_pull_data.py`, `4a.py`, `5a_fetch_freddata.py`) were run locally by hand,
which is why `data/raw/` already ships with the real pulled data — you don't need to
re-run those pulls unless you want fresh data.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run order

```bash
python src/01_pull_data.py            # equity prices + balance sheet fundamentals -> data/raw/
python src/02_merton_solver.py        # Merton DD/PD, both default-point conventions
python src/03_altmanZscore.py         # Altman Z-score + cross-validation vs. Merton
python src/4a.py                      # GE 2014-2018 fundamentals from SEC EDGAR (backtest input)
python src/04_GE_default.py           # GE historical backtest, monthly DD/PD trajectory
python src/5a_fetch_freddata.py       # macro data from FRED (edit in your API key first)
python src/05_Stress_test.py          # macro stress test, calibrated to 2008/2020 data
python src/5b_sector.py               # sector-cyclicality deep dive (DD reduction vs PD fold-change)
python src/06_term_structure.py       # DD/PD across multiple horizons
```

The memo itself (`output/memo/Corporate_Default_Risk_Credit_Memo.docx`) was assembled
from these scripts' outputs — see the memo's own appendices for full methodology.

## Company universe

See `data/company_universe.csv` — 26 companies spanning:

| Sector | Tickers |
|---|---|
| Technology | MSFT, AAPL |
| Healthcare | JNJ, PFE |
| Consumer Staples | PG, KO, WMT, KHC |
| Consumer Discretionary | HD, MCD, F, CCL, RCL |
| Communications | VZ, T, CHTR, LUMN |
| Energy | XOM, OXY |
| Industrials | CAT, GE, AAL |
| Utilities | NEE, DUK |
| Materials | NUE, FCX |

**Backtest name: General Electric (GE)**, downgraded multiple notches by S&P/Moody's/Fitch
across 2018-2019 (accounting irregularities, GE Capital insurance liabilities, ballooning
leverage), cut its dividend to $0.01/share in October 2018. 2014-2018 fundamentals for the
backtest were pulled directly from SEC EDGAR's XBRL company-facts API (`src/4a.py`), since
yfinance's free tier only goes back ~4 years.

Rating tiers (`approx_tier` column) are illustrative placeholders for company selection
diversity, not sourced from a ratings API.

## Repo layout
