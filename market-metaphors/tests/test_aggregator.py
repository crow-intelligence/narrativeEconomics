"""Tests for the metaphor aggregator module."""

import pandas as pd
from market_metaphors.metaphor.aggregator import (
    aggregate_daily,
    aggregate_daily_by_domain,
    aggregate_daily_by_ticker,
)


def _make_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-03-01", "2020-03-01", "2020-03-02", "2020-03-02"]
            ),
            "stock": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "metaphor_ratio": [0.2, 0.3, 0.1, 0.4],
            "metaphor_token_count": [2, 3, 1, 4],
            "dominant_domain": ["ASCENT", "HEAT", "ASCENT", "DARKNESS_DEATH"],
            "domains": [
                ["ASCENT", "HEAT"],
                ["HEAT"],
                ["ASCENT"],
                ["DARKNESS_DEATH", "WEIGHT_PRESSURE"],
            ],
        }
    )


class TestAggregateDaily:
    def test_basic_aggregation(self):
        df = _make_sample_df()
        result = aggregate_daily(df)
        assert len(result) == 2  # 2 unique dates
        assert "mean_metaphor_ratio" in result.columns
        assert "headline_count" in result.columns

    def test_mean_values(self):
        df = _make_sample_df()
        result = aggregate_daily(df)
        day1 = result[result["date"] == "2020-03-01"]
        assert day1["mean_metaphor_ratio"].iloc[0] == 0.25  # (0.2 + 0.3) / 2
        assert day1["headline_count"].iloc[0] == 2


class TestAggregateDailyByDomain:
    def test_domain_pivot(self):
        df = _make_sample_df()
        result = aggregate_daily_by_domain(df)
        assert not result.empty
        assert "date" in result.columns
        # Should have domain columns
        assert any(col in result.columns for col in ["ASCENT", "HEAT"])

    def test_unclassified_excluded(self):
        df = _make_sample_df()
        result = aggregate_daily_by_domain(df)
        assert "UNCLASSIFIED" not in result.columns


class TestAggregateDailyByTicker:
    def test_ticker_aggregation(self):
        df = _make_sample_df()
        result = aggregate_daily_by_ticker(df)
        assert len(result) == 4  # 2 dates × 2 tickers
        assert "stock" in result.columns
