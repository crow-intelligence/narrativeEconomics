"""Metaphor intensity × price/return correlation analysis."""

import pandas as pd
from scipy import stats

from market_metaphors.utils.logging import get_logger

logger = get_logger(__name__)


def compute_correlation(
    df: pd.DataFrame,
    metaphor_col: str = "mean_metaphor_ratio",
    return_col: str = "sp500_return",
) -> dict:
    """Compute Pearson and Spearman correlations between metaphor intensity
    and returns."""
    clean = df[[metaphor_col, return_col]].dropna()
    if len(clean) < 10:
        logger.warning("Too few data points (%d) for correlation", len(clean))
        return {
            "pearson_r": None,
            "pearson_p": None,
            "spearman_r": None,
            "spearman_p": None,
        }

    pearson_r, pearson_p = stats.pearsonr(clean[metaphor_col], clean[return_col])
    spearman_r, spearman_p = stats.spearmanr(clean[metaphor_col], clean[return_col])

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def rolling_correlation(
    df: pd.DataFrame,
    metaphor_col: str = "mean_metaphor_ratio",
    return_col: str = "sp500_return",
    window: int = 30,
) -> pd.Series:
    """Compute rolling Pearson correlation between metaphor intensity and returns."""
    return df[metaphor_col].rolling(window).corr(df[return_col])


def cross_correlation(
    metaphor_series: pd.Series,
    return_series: pd.Series,
    max_lag: int = 21,
) -> pd.DataFrame:
    """Compute cross-correlation at various lags.

    Positive lag = metaphor leads returns.
    Negative lag = returns lead metaphor.
    """
    results = []
    m = metaphor_series.dropna()
    r = return_series.dropna()
    # Align on common index
    common = m.index.intersection(r.index)
    m, r = m.loc[common], r.loc[common]

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            corr = (
                m.iloc[:-lag]
                .reset_index(drop=True)
                .corr(r.iloc[lag:].reset_index(drop=True))
            )
        elif lag < 0:
            corr = (
                m.iloc[-lag:]
                .reset_index(drop=True)
                .corr(r.iloc[:lag].reset_index(drop=True))
            )
        else:
            corr = m.corr(r)
        results.append({"lag": lag, "correlation": corr})

    return pd.DataFrame(results)
