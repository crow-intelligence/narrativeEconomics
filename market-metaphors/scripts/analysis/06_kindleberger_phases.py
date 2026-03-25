"""Analysis: Metaphor distributions per Kindleberger phase for each crisis event.

For each crisis:
  - Bar charts of top keyword-matched metaphor words per phase
  - Domain distribution comparison across phases
"""

import ast
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from market_metaphors.analysis.kindleberger import CRISIS_EVENTS
from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import (
    OUTPUTS_DIR,
    PROCESSED_DIR,
    read_parquet,
)

logger = get_logger(__name__)


def parse_list_col(series: pd.Series) -> pd.Series:
    """Parse string-encoded list columns."""
    return series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


def plot_top_words_per_phase(event_df: pd.DataFrame, event_name: str, out_dir):
    """Bar chart of top keyword-matched words per phase."""
    event_df = event_df.copy()
    event_df["keyword_matches"] = parse_list_col(event_df["keyword_matches"])

    phases = event_df["phase"].unique()
    phases = sorted([p for p in phases if p != "calm"])

    if not phases:
        logger.warning("No crisis phases found for %s", event_name)
        return

    fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 5))
    if len(phases) == 1:
        axes = [axes]

    for ax, phase in zip(axes, phases):
        phase_data = event_df[event_df["phase"] == phase]
        all_words = []
        for words in phase_data["keyword_matches"]:
            if isinstance(words, list):
                all_words.extend(words)

        top = Counter(all_words).most_common(15)
        if top:
            words, counts = zip(*top)
            ax.barh(range(len(words)), counts, color="steelblue")
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.invert_yaxis()
        ax.set_title(f"{phase.title()}")
        ax.set_xlabel("Count")

    fig.suptitle(
        "Top Metaphor Keywords by Phase — " f"{event_name.replace('_', ' ').title()}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "top_words_per_phase.png", dpi=150)
    plt.close(fig)


def plot_domain_distribution(event_df: pd.DataFrame, event_name: str, out_dir):
    """Stacked bar chart of domain distributions across phases."""
    event_df = event_df.copy()
    event_df["keyword_domains"] = parse_list_col(event_df["keyword_domains"])

    exploded = event_df[["phase", "keyword_domains"]].explode("keyword_domains")
    exploded = exploded.rename(columns={"keyword_domains": "domain"})
    exploded = exploded[
        (exploded["domain"].notna())
        & (exploded["domain"] != "UNCLASSIFIED")
        & (exploded["phase"] != "calm")
    ]

    if exploded.empty:
        logger.warning("No domain data for %s", event_name)
        return

    pivot = exploded.groupby(["phase", "domain"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title(
        "Domain Distribution by Phase — " f"{event_name.replace('_', ' ').title()}"
    )
    ax.set_xlabel("Phase")
    ax.set_ylabel("Count")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "domain_distribution.png", dpi=150)
    plt.close(fig)


def main():
    merged = read_parquet(PROCESSED_DIR / "merged.parquet")
    merged["date"] = pd.to_datetime(merged["date"])

    for event_name in CRISIS_EVENTS:
        logger.info("Analyzing %s", event_name)
        event_df = merged[merged["crisis_event"] == event_name]

        if event_df.empty:
            logger.warning("No data for %s", event_name)
            continue

        out_dir = OUTPUTS_DIR / "figures" / event_name
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_top_words_per_phase(event_df, event_name, out_dir)
        plot_domain_distribution(event_df, event_name, out_dir)
        logger.info("Saved figures for %s", event_name)


if __name__ == "__main__":
    main()
