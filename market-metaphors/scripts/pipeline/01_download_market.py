"""Step 1: Download market data for all tickers in the news dataset + S&P 500."""

from market_metaphors.ingest.market import download_all
from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import PROCESSED_DIR, read_parquet

logger = get_logger(__name__)


def main():
    news_path = PROCESSED_DIR / "news_clean.parquet"
    if not news_path.exists():
        logger.error("news_clean.parquet not found — run 02_clean_news.py first")
        return

    df = read_parquet(news_path)
    tickers = df["stock"].dropna().unique().tolist()
    logger.info("Found %d unique tickers in news data", len(tickers))

    # Always include S&P 500
    if "^GSPC" not in tickers:
        tickers.append("^GSPC")

    succeeded, failed = download_all(tickers)
    logger.info(
        "Market download complete: %d succeeded, %d failed",
        len(succeeded),
        len(failed),
    )


if __name__ == "__main__":
    main()
