"""Download market data via yfinance."""

from pathlib import Path

import pandas as pd
import yfinance as yf

from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import RAW_DIR, write_parquet

logger = get_logger(__name__)

MARKET_DIR = RAW_DIR / "market"
START_DATE = "2009-01-01"
END_DATE = "2020-12-31"


def fetch_ticker(
    ticker: str, start: str = START_DATE, end: str = END_DATE
) -> pd.DataFrame | None:
    """Fetch daily OHLCV for a single ticker. Returns None on failure."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        df["ticker"] = ticker
        return df
    except Exception:
        logger.warning("Failed to download %s", ticker, exc_info=True)
        return None


def download_all(
    tickers: list[str],
    output_dir: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Download market data for all tickers.

    Returns (succeeded, failed) ticker lists.
    """
    out = output_dir or MARKET_DIR
    out.mkdir(parents=True, exist_ok=True)

    succeeded, failed = [], []

    for ticker in tickers:
        outfile = out / f"{ticker}.parquet"
        if outfile.exists():
            logger.info("Skipping %s (already exists)", ticker)
            succeeded.append(ticker)
            continue

        df = fetch_ticker(ticker)
        if df is not None and not df.empty:
            write_parquet(df, outfile)
            succeeded.append(ticker)
            logger.info("Downloaded %s (%d rows)", ticker, len(df))
        else:
            failed.append(ticker)

    # Log failed tickers
    if failed:
        failed_path = out / "failed_tickers.txt"
        failed_path.write_text("\n".join(failed) + "\n")
        logger.warning("Failed tickers (%d): %s", len(failed), failed_path)

    logger.info("Done: %d succeeded, %d failed", len(succeeded), len(failed))
    return succeeded, failed
