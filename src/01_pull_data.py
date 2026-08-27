"""
Stage 1: Data acquisition.

Pulls, for every company in data/company_universe.csv:
  - Daily adjusted close price history (for equity volatility + market cap time series)
  - Key balance sheet line items needed for the Merton default point and Altman Z-score:
        total_assets, total_liabilities, current_assets, current_liabilities,
        short_term_debt, long_term_debt, retained_earnings, ebit, total_revenue,
        shares_outstanding
  - Current market cap (E = equity value input to Merton)

Notes:
  - yfinance's fundamentals only go back ~4 years via the free API. That's fine for the
    cross-sectional part of the project (Sections 3.1-3.7), but the backtest name (GE,
    flagged is_backtest_name=True in company_universe.csv) needs financials from
    2015-2018, well before the free API's window. For that name, pull whatever yfinance
    gives you first, then go to SEC EDGAR (sec.gov/edgar, company facts API or the 10-K
    filings directly) for the older annual figures and hardcode/patch them into
    data/raw/GE_balance_sheet_manual.csv - see the stub this script creates for GE.
  - This script must be run somewhere with open internet access to
    query.finance.yahoo.com. It will NOT run inside a network-restricted sandbox.

Output:
  data/raw/prices/{ticker}.csv       - daily OHLCV + adj close
  data/raw/fundamentals/{ticker}.csv - one row per available fiscal period
  data/raw/pull_log.csv              - success/failure log for every ticker
"""
import sys
import time
import logging
from pathlib import Path
import random 

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install -r requirements.txt --break-system-packages")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_CSV = ROOT / "data" / "company_universe.csv"
RAW_DIR = ROOT / "data" / "raw"
PRICES_DIR = RAW_DIR / "prices"
FUND_DIR = RAW_DIR / "fundamentals"

# how far back to pull daily prices. 2y is enough for a 1y historical-vol window with
# a buffer; the backtest name additionally needs a long pre-event window (handled below).
PRICE_LOOKBACK = "2y"
BACKTEST_PRICE_START = "2014-01-01"  # GE: capture years before the 2018 downgrade cascade

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("data_pull")


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV)
    required = {"ticker", "company", "sector", "cyclicality", "approx_tier", "is_backtest_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"company_universe.csv missing columns: {missing}")
    return df

def with_retry(fn, *args, retries=3, base_delay=8, **kwargs):
    last_err = RuntimeError(f"with_retry called with retries={retries}, no attempts made")
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait = base_delay * (2 ** attempt) + random.uniform(0, 3)
                log.warning(f"  rate limited, backing off {wait:.0f}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise
    raise last_err

def pull_prices(ticker: str, is_backtest: bool) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    if is_backtest:
        hist = tk.history(start=BACKTEST_PRICE_START, auto_adjust=False)
    else:
        hist = tk.history(period=PRICE_LOOKBACK, auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"No price history returned for {ticker}")
    hist = hist.reset_index()
    hist["ticker"] = ticker
    return hist


def safe_date_str(col) -> str:
    """
    Column labels from yfinance's statement DataFrames come back as different types
    depending on version/ticker (pandas.Timestamp, numpy.datetime64, plain str, and
    occasionally NaT). Route everything through pd.to_datetime rather than relying on
    hasattr(col, 'date') - that check passes for some types but not others (e.g.
    datetime.date has no .date() method, numpy.datetime64 has none either) and was
    causing the "not defined type" error.
    """
    ts = pd.to_datetime(col, errors="coerce")
    if pd.isna(ts):
        return str(col)  # fall back to raw repr rather than crashing
    return ts.strftime("%Y-%m-%d")

def pull_fundamentals(ticker: str) -> dict:
    """
    Pull the balance sheet / income statement fields needed for:
      - Merton default point (short-term debt, long-term debt, total liabilities)
      - Altman Z-score (working capital, retained earnings, EBIT, total assets,
        market value of equity, total liabilities, sales)
    yfinance exposes annual + quarterly statements as DataFrames indexed by line item.
    Field label availability varies by ticker/filer, so we probe a list of known aliases
    for each concept and take the first match.
    """
    tk = yf.Ticker(ticker)
    bs = tk.balance_sheet          # annual balance sheet, columns = fiscal period end dates
    fin = tk.financials            # annual income statement
    info = tk.info or {}

    if bs is None or bs.empty:
        raise RuntimeError(f"No balance sheet returned for {ticker}")

    latest_col = bs.columns[0]     # most recent fiscal year

    def first_match(df, aliases):
        if df is None or df.empty:
            return None
        for alias in aliases:
            if alias in df.index:
                val = df.loc[alias, latest_col]
                if pd.notna(val):
                    return float(val)
        return None

    total_assets = first_match(bs, ["Total Assets"])
    total_liab = first_match(bs, ["Total Liabilities Net Minority Interest", "Total Liab"])
    current_assets = first_match(bs, ["Current Assets", "Total Current Assets"])
    current_liab = first_match(bs, ["Current Liabilities", "Total Current Liabilities"])
    st_debt = first_match(bs, ["Current Debt", "Short Long Term Debt", "Short Term Debt"])
    lt_debt = first_match(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
    retained_earnings = first_match(bs, ["Retained Earnings"])

    ebit = None
    total_revenue = None
    if fin is not None and not fin.empty and latest_col in fin.columns:
        ebit = first_match(fin, ["EBIT", "Operating Income"])
        total_revenue = first_match(fin, ["Total Revenue"])

    shares_out = info.get("sharesOutstanding")
    market_cap = info.get("marketCap")

    return {
        "ticker": ticker,
        "fiscal_period_end": safe_date_str(latest_col),
        "total_assets": total_assets,
        "total_liabilities": total_liab,
        "current_assets": current_assets,
        "current_liabilities": current_liab,
        "short_term_debt": st_debt,
        "long_term_debt": lt_debt,
        "retained_earnings": retained_earnings,
        "ebit": ebit,
        "total_revenue": total_revenue,
        "shares_outstanding": shares_out,
        "market_cap": market_cap,
    }


def main():
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    FUND_DIR.mkdir(parents=True, exist_ok=True)

    universe = load_universe()
    log_rows = []

    for _, row in universe.iterrows():
        ticker = row["ticker"]
        is_backtest = bool(row["is_backtest_name"])
        log.info(f"Pulling {ticker} ({row['company']})...")

        # --- NEW: skip if already pulled ---
        price_file = PRICES_DIR / f"{ticker}.csv"
        fund_file = FUND_DIR / f"{ticker}.csv"
        if price_file.exists() and fund_file.exists():
            log.info(f"  {ticker} already pulled, skipping (delete the CSVs to force a re-pull)")
            log_rows.append({
                "ticker": ticker, "price_status": "skipped_exists", "price_error": "",
                "fundamentals_status": "skipped_exists", "fundamentals_error": "",
            })
            continue
        # --- end new block ---

        # prices
        try:
            prices = with_retry(pull_prices, ticker, is_backtest)
            prices.to_csv(PRICES_DIR / f"{ticker}.csv", index=False)
            price_status, price_err = "ok", ""
        except Exception as e:
            price_status, price_err = "FAILED", str(e)
            log.error(f"  price pull failed for {ticker}: {e}")

        # fundamentals
        try:
            fund = with_retry(pull_fundamentals, ticker)
            pd.DataFrame([fund]).to_csv(FUND_DIR / f"{ticker}.csv", index=False)
            fund_status, fund_err = "ok", ""
            n_missing = sum(v is None for k, v in fund.items() if k not in ("ticker", "fiscal_period_end"))
            if n_missing:
                log.warning(f"  {ticker}: {n_missing} fundamental field(s) missing - "
                            f"cross-check against SEC EDGAR before modeling")
        except Exception as e:
            fund_status, fund_err = "FAILED", str(e)
            log.error(f"  fundamentals pull failed for {ticker}: {e}")

        log_rows.append({
            "ticker": ticker, "price_status": price_status, "price_error": price_err,
            "fundamentals_status": fund_status, "fundamentals_error": fund_err,
        })
        time.sleep(random.uniform(3, 6))  # longer, randomized gap between tickers

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(RAW_DIR / "pull_log.csv", index=False)

    n_fail = ((log_df.price_status == "FAILED") | (log_df.fundamentals_status == "FAILED")).sum()
    log.info(f"Done. {len(log_df) - n_fail}/{len(log_df)} tickers fully pulled. "
             f"See data/raw/pull_log.csv for details.")

    if n_fail:
        log.warning("Some tickers failed - re-run, or check ticker validity / rate limits.")

if __name__ == "__main__":
    main()
