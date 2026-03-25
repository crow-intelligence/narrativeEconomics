"""Step 4: Merge news + metaphor annotations + market data into a single dataset."""

import pandas as pd
from market_metaphors.analysis.kindleberger import label_dataframe
from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import (
    PROCESSED_DIR,
    RAW_DIR,
    read_parquet,
    write_parquet,
)

logger = get_logger(__name__)


def _normalize_yfinance_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance parquet with multi-level column names.

    yfinance saves columns as ('Close', 'AAPL'), ('High', 'AAPL') etc.
    This flattens them to just 'Close', 'High', etc.
    """
    # Flatten column names: ('Close', 'AAPL') -> 'Close'
    new_cols = {}
    for col in df.columns:
        if isinstance(col, str) and col.startswith("("):
            # String repr of tuple from parquet
            import ast

            parsed = ast.literal_eval(col)
            new_cols[col] = parsed[0] if parsed[0] else col
        elif isinstance(col, tuple):
            new_cols[col] = col[0] if col[0] else col
        else:
            new_cols[col] = col
    df = df.rename(columns=new_cols)

    # Handle date — yfinance stores it as the index or as 'Date'
    if "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"], utc=True).dt.normalize()
    elif df.index.name == "Date":
        df["date"] = pd.to_datetime(df.index, utc=True).normalize()
        df = df.reset_index(drop=True)
    else:
        # No date info available (index lost during save)
        return None

    return df


def main():
    news_path = PROCESSED_DIR / "news_clean.parquet"
    metaphors_path = PROCESSED_DIR / "metaphors.parquet"
    output_path = PROCESSED_DIR / "merged.parquet"

    # Load news + metaphors
    news = read_parquet(news_path)
    metaphors = read_parquet(metaphors_path)

    assert len(news) == len(metaphors), (
        f"Row count mismatch: news={len(news)}, " f"metaphors={len(metaphors)}"
    )
    merged = pd.concat([news, metaphors], axis=1)
    merged["date"] = pd.to_datetime(merged["date"], utc=True).dt.normalize()
    logger.info("Merged news + metaphors: %d rows", len(merged))

    # Load and join S&P 500 data
    sp500_path = RAW_DIR / "market" / "^GSPC.parquet"
    sp500_raw = None
    if sp500_path.exists():
        sp500_raw = _normalize_yfinance_df(read_parquet(sp500_path), "^GSPC")
    if sp500_raw is not None:
        sp500 = (
            sp500_raw[["date", "Close"]]
            .rename(columns={"Close": "sp500_close"})
            .drop_duplicates("date")
            .sort_values("date")
        )
        sp500["sp500_return"] = sp500["sp500_close"].pct_change()
        merged = merged.merge(sp500, on="date", how="left")
        logger.info("Joined S&P 500 data")

    # Per-ticker stock data: load one at a time to avoid OOM
    tickers = merged["stock"].dropna().unique()
    close_records = []
    loaded = 0
    for ticker in tickers:
        path = RAW_DIR / "market" / f"{ticker}.parquet"
        if not path.exists():
            continue
        df = _normalize_yfinance_df(read_parquet(path), ticker)
        if df is None or "Close" not in df.columns:
            continue

        ticker_data = df[["date", "Close"]].copy()
        ticker_data["stock"] = ticker
        ticker_data["daily_return"] = ticker_data["Close"].pct_change()
        ticker_data = ticker_data.rename(columns={"Close": "close"})
        close_records.append(ticker_data)
        loaded += 1

    if close_records:
        stocks = pd.concat(close_records, ignore_index=True)
        merged = merged.merge(stocks, on=["date", "stock"], how="left")
        logger.info("Joined per-ticker data for %d tickers", loaded)

    # Add Kindleberger phase labels
    merged = label_dataframe(merged)

    write_parquet(merged, output_path)
    logger.info(
        "Saved merged dataset: %d rows to %s",
        len(merged),
        output_path,
    )


if __name__ == "__main__":
    main()
