# Corporate Default Risk + Macro Stress Test

A structural credit risk model (Merton 1974) estimating Distance-to-Default and
Probability of Default for 25 public companies, cross-checked against Altman Z-scores,
backtested against a real historical credit deterioration, and extended with a
CCAR/DFAST-style macro stress test.

This is the corporate/wholesale-credit counterpart to a retail credit scorecard project —
together they cover both halves of what a credit risk desk does.

## Status

- [x] Project scaffold + company universe defined
- [x] Stage 1 script written: equity + balance sheet data pull (`src/01_pull_data.py`)
- [ ] Stage 1 executed (**must be run on a machine with open internet access** — see below)
- [ ] Stage 2: Merton system solver + KMV iterative volatility restriction
- [ ] Stage 3: Altman Z-score + cross-validation (rank correlation, FRED OAS proxy)
- [ ] Stage 4: Historical backtest (GE, 2018-2019 downgrade cascade)
- [ ] Stage 5: Macro stress testing (baseline / adverse / severely adverse)
- [ ] Stage 6: DD/PD term structure
- [ ] Stage 7: Credit risk memo

## Why network access matters here

This repo was scaffolded in a sandboxed environment that cannot reach
`query1/2.finance.yahoo.com`, `fred.stlouisfed.org`, or `sec.gov`. **Every data-pulling
script must be run on your own machine.** The modeling scripts (Merton solver, Altman,
stress test, memo) only need the CSVs those pulls produce, so once the data lands in
`data/raw/`, all downstream work can happen anywhere, including back here.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run order

```bash
python src/01_pull_data.py        # equity prices + balance sheet fundamentals -> data/raw/
# (Stage 2 onward added as we build them)
```

## Company universe

See `data/company_universe.csv` — 25 companies spanning:

| Sector | Tickers |
|---|---|
| Technology | MSFT, AAPL |
| Healthcare | JNJ, PFE |
| Consumer Staples | PG, KO, WMT, KHC |
| Consumer Discretionary | HD, MCD, F, CCL, RCL |
| Communications | VZ, T, CHTR, DISH |
| Energy | XOM, OXY |
| Industrials | CAT, GE, AAL |
| Utilities | NEE, DUK |
| Materials | NUE, FCX |

**Backtest name: General Electric (GE)**, flagged `is_backtest_name=True`. GE was
downgraded multiple notches by S&P/Moody's/Fitch across 2018-2019 (accounting
irregularities, GE Capital insurance liabilities, ballooning leverage), cut its dividend
to $0.01/share in October 2018, and its CDS spreads and equity vol both blew out well
before the credit-rating downgrades were finalized. Unlike bankrupt/delisted names
(J.C. Penney, original Hertz listing), GE has an unbroken trading history, so `yfinance`
returns clean data spanning the full pre-event window. Swap `is_backtest_name` to a
different ticker in `company_universe.csv` if you'd rather use an actual default —
just make sure the replacement's price history survives past the event date in
`yfinance` (delisted names often don't).

Rating tiers (`approx_tier` column) are illustrative placeholders for company selection
diversity, not sourced from a ratings API — replace with actual S&P/Moody's ratings
where available for the ratings cross-check in Section 3.6.

## Repo layout

```
data/
  company_universe.csv        # the 25 companies + sector/cyclicality/tier metadata
  raw/
    prices/{ticker}.csv       # daily OHLCV from yfinance
    fundamentals/{ticker}.csv # latest annual balance sheet + income statement line items
    pull_log.csv              # success/failure log from Stage 1
  processed/                  # cleaned, model-ready datasets (built in later stages)
src/
  01_pull_data.py             # Stage 1
notebooks/                    # exploratory / hand-verification work (Week 2 Day 1-3)
output/
  charts/                     # convergence plots, backtest chart, term structure, stress chart
  memo/                       # final credit risk memo
docs/                         # reading notes, methodology references
```

## References

- Hull, *Options, Futures, and Other Derivatives* — Black-Scholes + credit risk chapters
- Moody's KMV EDF methodology (default point convention, iterative volatility restriction)
- Altman, E. (1968) — "Financial Ratios, Discriminant Analysis and the Prediction of
  Corporate Bankruptcy"
- Federal Reserve CCAR/DFAST methodology
- FRED series: `DTB1YR` (1yr Treasury), `BAMLH0A0HYM2` (US HY OAS), `BAMLC0A0CM` (US IG OAS)
