"""Analysis: Shiller-style narrative contagion — metaphor spread velocity.

Measures:
  - Rolling 7-day and 30-day prevalence of top-N keyword-matched metaphor words
  - Spread velocity: time from baseline to peak prevalence
"""

import ast
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from market_metaphors.analysis.kindleberger import (
    CRISIS_EVENTS,
    get_event_window,
)
from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import (
    OUTPUTS_DIR,
    PROCESSED_DIR,
    read_parquet,
)

logger = get_logger(__name__)


def compute_word_prevalence(df: pd.DataFrame, word: str, window: int = 7) -> pd.Series:
    """Compute rolling prevalence of a keyword across headlines."""
    has_word = df["keyword_matches_list"].apply(
        lambda words: word in words if isinstance(words, list) else False
    )
    daily = (
        pd.DataFrame({"date": df["date"], "has_word": has_word})
        .groupby("date")["has_word"]
        .mean()
    )
    return daily.rolling(window, min_periods=1).mean()


def analyze_crisis(merged: pd.DataFrame, event_name: str, top_n: int = 10) -> None:
    """Analyze narrative contagion for a single crisis event."""
    start, end = get_event_window(event_name)

    context_start = pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=60)
    context_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=60)

    mask = (merged["date"] >= context_start) & (merged["date"] <= context_end)
    crisis_df = merged[mask].copy()

    if crisis_df.empty:
        logger.warning("No data for %s", event_name)
        return

    # Parse keyword matches
    crisis_df["keyword_matches_list"] = crisis_df["keyword_matches"].apply(
        lambda x: ast.literal_eval(x)
        if isinstance(x, str)
        else (x if isinstance(x, list) else [])
    )

    # Find top keyword matches during the crisis
    all_words = []
    for words in crisis_df["keyword_matches_list"]:
        if isinstance(words, list):
            all_words.extend(words)
    top_words = [w for w, _ in Counter(all_words).most_common(top_n)]

    if not top_words:
        logger.warning("No keyword matches found for %s", event_name)
        return

    out_dir = OUTPUTS_DIR / "contagion" / event_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot prevalence curves
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for window, ax in zip([7, 30], axes):
        for word in top_words[:5]:
            prevalence = compute_word_prevalence(crisis_df, word, window=window)
            ax.plot(
                prevalence.index,
                prevalence.values,
                label=word,
                linewidth=1.2,
            )

        ax.axvspan(
            pd.Timestamp(start, tz="UTC"),
            pd.Timestamp(end, tz="UTC"),
            alpha=0.15,
            color="red",
        )
        ax.set_title(
            f"{window}-day rolling prevalence — "
            f"{event_name.replace('_', ' ').title()}"
        )
        ax.set_ylabel("Prevalence")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "prevalence_curves.png", dpi=150)
    plt.close(fig)

    # Spread velocity
    velocity_records = []
    for word in top_words:
        prev = compute_word_prevalence(crisis_df, word, window=7)
        if prev.empty or prev.max() == 0:
            continue
        baseline = prev.quantile(0.1)
        peak_date = prev.idxmax()
        above_baseline = prev[prev > baseline]
        if not above_baseline.empty:
            first_rise = above_baseline.index[0]
            days_to_peak = (peak_date - first_rise).days
            velocity_records.append(
                {
                    "word": word,
                    "baseline": baseline,
                    "peak": prev.max(),
                    "peak_date": peak_date,
                    "days_to_peak": days_to_peak,
                }
            )

    if velocity_records:
        vel_df = pd.DataFrame(velocity_records)
        vel_df.to_csv(out_dir / "spread_velocity.csv", index=False)
        logger.info(
            "Spread velocity for %s:\n%s",
            event_name,
            vel_df.to_string(),
        )


def main():
    merged = read_parquet(PROCESSED_DIR / "merged.parquet")
    merged["date"] = pd.to_datetime(merged["date"])

    for event_name in CRISIS_EVENTS:
        logger.info("Analyzing contagion for %s", event_name)
        analyze_crisis(merged, event_name)


if __name__ == "__main__":
    main()
