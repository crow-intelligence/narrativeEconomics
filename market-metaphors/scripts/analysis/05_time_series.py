"""Analysis: Daily metaphor prevalence over time with S&P 500 overlay.

Generates a dual-axis time series plot with crisis event windows marked.
"""

import matplotlib.pyplot as plt
import pandas as pd
from market_metaphors.analysis.kindleberger import CRISIS_EVENTS, get_event_window
from market_metaphors.metaphor.aggregator import aggregate_daily
from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import OUTPUTS_DIR, PROCESSED_DIR, read_parquet

logger = get_logger(__name__)

FIGURE_DIR = OUTPUTS_DIR / "figures"

# Colors for crisis event bands
EVENT_COLORS = {
    "eu_debt_2011": "#ff9999",
    "china_2015": "#ffcc99",
    "brexit_2016": "#99ccff",
    "rate_hike_2018": "#cc99ff",
    "covid_2020": "#ff99cc",
}


def main():
    merged = read_parquet(PROCESSED_DIR / "merged.parquet")
    merged["date"] = pd.to_datetime(merged["date"])

    # Aggregate daily
    daily = aggregate_daily(merged)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    # Get S&P 500 close per day
    sp500 = merged[["date", "sp500_close"]].drop_duplicates("date").sort_values("date")

    # Smooth metaphor ratio with 7-day rolling mean
    daily["metaphor_ratio_7d"] = daily["mean_metaphor_ratio"].rolling(7).mean()

    # Plot
    fig, ax1 = plt.subplots(figsize=(16, 6))

    # Metaphor ratio
    ax1.plot(
        daily["date"],
        daily["metaphor_ratio_7d"],
        color="tab:blue",
        linewidth=0.8,
        label="Metaphor ratio (7d MA)",
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Mean metaphor ratio", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # S&P 500 on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(
        sp500["date"],
        sp500["sp500_close"],
        color="tab:red",
        linewidth=0.8,
        alpha=0.7,
        label="S&P 500",
    )
    ax2.set_ylabel("S&P 500 Close", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Mark crisis windows
    for event, color in EVENT_COLORS.items():
        if event in CRISIS_EVENTS:
            start, end = get_event_window(event)
            ax1.axvspan(
                pd.Timestamp(start, tz="UTC"),
                pd.Timestamp(end, tz="UTC"),
                alpha=0.2,
                color=color,
                label=event.replace("_", " ").title(),
            )

    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax1.set_title("Metaphor Prevalence in Financial Headlines vs S&P 500 (2009–2020)")
    fig.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "metaphor_timeseries.png"
    fig.savefig(out, dpi=150)
    logger.info("Saved figure to %s", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
