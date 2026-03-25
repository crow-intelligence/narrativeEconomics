"""Aggregate token-level metaphor annotations to headline and daily levels."""

import pandas as pd

from market_metaphors.utils.logging import get_logger

logger = get_logger(__name__)


def aggregate_daily(
    metaphors_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Aggregate metaphor metrics by date.

    Input columns expected: date, metaphor_ratio, metaphor_token_count, dominant_domain.
    """
    daily = (
        metaphors_df.groupby(date_col)
        .agg(
            mean_metaphor_ratio=("metaphor_ratio", "mean"),
            median_metaphor_ratio=("metaphor_ratio", "median"),
            total_metaphor_tokens=("metaphor_token_count", "sum"),
            headline_count=("metaphor_ratio", "count"),
        )
        .reset_index()
    )
    return daily


def aggregate_daily_by_domain(
    metaphors_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Count domain occurrences per day.

    Expects a 'domains' column containing lists of domain strings.
    Returns a date × domain pivot table of counts.
    """
    # Explode domains lists into individual rows
    exploded = metaphors_df[[date_col, "domains"]].explode("domains")
    exploded = exploded.dropna(subset=["domains"])
    exploded = exploded[exploded["domains"] != "UNCLASSIFIED"]

    if exploded.empty:
        logger.warning("No classified domains found")
        return pd.DataFrame()

    pivot = (
        exploded.groupby([date_col, "domains"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    return pivot


def aggregate_daily_by_ticker(
    metaphors_df: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "stock",
) -> pd.DataFrame:
    """Aggregate metaphor metrics by date and ticker."""
    daily = (
        metaphors_df.groupby([date_col, ticker_col])
        .agg(
            mean_metaphor_ratio=("metaphor_ratio", "mean"),
            total_metaphor_tokens=("metaphor_token_count", "sum"),
            headline_count=("metaphor_ratio", "count"),
        )
        .reset_index()
    )
    return daily
