"""Load and clean the Kaggle stock news dataset."""

from pathlib import Path

import pandas as pd

from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import RAW_DIR, write_parquet

logger = get_logger(__name__)

RAW_NEWS_DIR = RAW_DIR / "news"


def load_raw_news(filename: str = "analyst_ratings_processed.csv") -> pd.DataFrame:
    """Load the raw Kaggle CSV."""
    path = RAW_NEWS_DIR / filename
    logger.info("Loading raw news from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows", len(df))
    return df


def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and deduplicate the news dataset.

    Steps:
      1. Parse dates with UTC normalization (mixed TZ in source).
      2. Drop rows with unparseable dates or null headlines.
      3. Normalize ticker symbols to uppercase.
      4. Deduplicate on (headline, date, stock).
    """
    initial = len(df)

    # Parse dates — source has mixed timezones, normalize to UTC
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    # Drop rows where date didn't parse or headline is missing
    df = df.dropna(subset=["title", "date"])
    logger.info(
        "Dropped %d rows with null/unparseable headline or date",
        initial - len(df),
    )

    # Normalize tickers
    if "stock" in df.columns:
        df["stock"] = df["stock"].str.upper().str.strip()

    # Deduplicate
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["title", "date", "stock"])
    logger.info("Removed %d duplicate rows", before_dedup - len(df))

    # Rename title -> headline for consistency
    df = df.rename(columns={"title": "headline"})

    # Drop the unnamed index column if present
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Clean dataset: %d rows", len(df))
    return df


def run(output_path: Path | None = None) -> pd.DataFrame:
    """Full news cleaning pipeline."""
    df = load_raw_news()
    df = clean_news(df)
    out = output_path or (RAW_DIR.parent / "processed" / "news_clean.parquet")
    write_parquet(df, out)
    logger.info("Saved cleaned news to %s", out)
    return df
