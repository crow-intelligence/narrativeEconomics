"""Analysis: Granger causality — do metaphors lead or follow markets?

Tests:
  - Granger causality between metaphor_ratio and S&P 500 returns (both directions)
  - Cross-correlation function (CCF) to find lead/lag structure
  - Event study: metaphor domain shifts around phase transitions
"""

import matplotlib.pyplot as plt
import pandas as pd
from market_metaphors.analysis.correlations import cross_correlation
from market_metaphors.analysis.kindleberger import CRISIS_EVENTS
from market_metaphors.metaphor.aggregator import aggregate_daily
from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import OUTPUTS_DIR, PROCESSED_DIR, read_parquet
from statsmodels.tsa.stattools import grangercausalitytests

logger = get_logger(__name__)

LAGS = [1, 3, 5, 10, 21]
CAUSALITY_DIR = OUTPUTS_DIR / "causality"


def granger_test(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_lag: int = 21,
) -> pd.DataFrame:
    """Run Granger causality test: does x Granger-cause y?

    Returns a DataFrame with F-test p-values per lag.
    """
    clean = df[[x_col, y_col]].dropna()
    if len(clean) < max_lag + 10:
        logger.warning("Too few observations (%d) for Granger test", len(clean))
        return pd.DataFrame()

    results = grangercausalitytests(
        clean[[y_col, x_col]], maxlag=max_lag, verbose=False
    )

    records = []
    for lag in range(1, max_lag + 1):
        f_test = results[lag][0]["ssr_ftest"]
        records.append(
            {
                "lag": lag,
                "f_stat": f_test[0],
                "p_value": f_test[1],
                "df_denom": f_test[2],
                "df_num": f_test[3],
            }
        )
    return pd.DataFrame(records)


def run_full_sample_tests(daily: pd.DataFrame) -> None:
    """Run Granger tests on the full 2009-2020 sample."""
    CAUSALITY_DIR.mkdir(parents=True, exist_ok=True)

    # Metaphor → Returns
    logger.info("Testing: metaphor_ratio -> sp500_return")
    m2r = granger_test(daily, "mean_metaphor_ratio", "sp500_return")
    if not m2r.empty:
        m2r.to_csv(CAUSALITY_DIR / "granger_metaphor_to_returns.csv", index=False)
        sig = m2r[m2r["p_value"] < 0.05]
        logger.info("Significant lags (metaphor→returns): %s", sig["lag"].tolist())

    # Returns → Metaphor
    logger.info("Testing: sp500_return -> metaphor_ratio")
    r2m = granger_test(daily, "sp500_return", "mean_metaphor_ratio")
    if not r2m.empty:
        r2m.to_csv(CAUSALITY_DIR / "granger_returns_to_metaphor.csv", index=False)
        sig = r2m[r2m["p_value"] < 0.05]
        logger.info("Significant lags (returns→metaphor): %s", sig["lag"].tolist())


def run_cross_correlation(daily: pd.DataFrame) -> None:
    """Compute and plot cross-correlation between metaphor intensity and returns."""
    ccf = cross_correlation(
        daily.set_index("date")["mean_metaphor_ratio"],
        daily.set_index("date")["sp500_return"],
        max_lag=21,
    )

    ccf.to_csv(CAUSALITY_DIR / "cross_correlation.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ccf["lag"], ccf["correlation"], color="steelblue", width=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag (positive = metaphor leads)")
    ax.set_ylabel("Correlation")
    ax.set_title("Cross-Correlation: Metaphor Intensity vs S&P 500 Returns")
    fig.tight_layout()
    fig.savefig(CAUSALITY_DIR / "cross_correlation.png", dpi=150)
    plt.close(fig)
    logger.info(
        "Peak correlation at lag %d",
        ccf.loc[ccf["correlation"].abs().idxmax(), "lag"],
    )


def run_event_study(merged: pd.DataFrame) -> None:
    """Event study: metaphor ratio around phase transitions (T-5 to T+5)."""
    records = []

    for event_name, phases in CRISIS_EVENTS.items():
        # Look at each phase boundary as a transition point
        sorted_phases = sorted(phases.items(), key=lambda x: x[1][0])
        for i in range(len(sorted_phases) - 1):
            _, (_, end_current) = sorted_phases[i]
            next_phase, (start_next, _) = sorted_phases[i + 1]

            transition = pd.Timestamp(start_next, tz="UTC")
            window_start = transition - pd.Timedelta(days=5)
            window_end = transition + pd.Timedelta(days=5)

            mask = (merged["date"] >= window_start) & (merged["date"] <= window_end)
            window_data = merged[mask]

            if window_data.empty:
                continue

            for _, row in window_data.iterrows():
                days_from_transition = (row["date"] - transition).days
                records.append(
                    {
                        "event": event_name,
                        "transition_to": next_phase,
                        "days_from_transition": days_from_transition,
                        "metaphor_ratio": row.get("metaphor_ratio"),
                        "dominant_domain": row.get("dominant_domain"),
                    }
                )

    if records:
        event_df = pd.DataFrame(records)
        event_df.to_csv(CAUSALITY_DIR / "event_study.csv", index=False)

        # Average metaphor ratio around transitions
        avg = event_df.groupby("days_from_transition")["metaphor_ratio"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(avg.index, avg.values, marker="o", color="steelblue")
        ax.axvline(0, color="red", linestyle="--", label="Phase transition")
        ax.set_xlabel("Days from phase transition")
        ax.set_ylabel("Mean metaphor ratio")
        ax.set_title("Event Study: Metaphor Intensity Around Phase Transitions")
        ax.legend()
        fig.tight_layout()
        fig.savefig(CAUSALITY_DIR / "event_study.png", dpi=150)
        plt.close(fig)


def main():
    merged = read_parquet(PROCESSED_DIR / "merged.parquet")
    merged["date"] = pd.to_datetime(merged["date"])

    # Aggregate to daily level
    daily = aggregate_daily(merged)
    daily["date"] = pd.to_datetime(daily["date"])

    # Add S&P 500 returns
    sp500 = merged[["date", "sp500_return"]].drop_duplicates("date").sort_values("date")
    daily = daily.merge(sp500, on="date", how="left").sort_values("date")

    CAUSALITY_DIR.mkdir(parents=True, exist_ok=True)

    run_full_sample_tests(daily)
    run_cross_correlation(daily)
    run_event_study(merged)

    logger.info("All causality analyses complete. Results in %s", CAUSALITY_DIR)


if __name__ == "__main__":
    main()
