"""Comprehensive visualizations of keyword-based metaphor analysis results.

Generates:
  1. Domain heatmap across Kindleberger phases (all crises combined)
  2. Per-crisis domain shift radar/polar charts
  3. Metaphor intensity + S&P 500 with domain stacking
  4. Crisis comparison: normalized domain proportions
  5. COVID deep-dive: daily domain timeline
  6. Contagion word cloud style: top words per phase
"""

import ast
from collections import Counter

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

FIGURE_DIR = OUTPUTS_DIR / "figures"

# Consistent domain colors
DOMAIN_COLORS = {
    "ASCENT": "#2ecc71",
    "HEAT": "#e74c3c",
    "LIGHT": "#f39c12",
    "STRUCTURAL_FAILURE": "#7f8c8d",
    "WATER_FLOOD": "#3498db",
    "WEIGHT_PRESSURE": "#9b59b6",
    "DARKNESS_DEATH": "#2c3e50",
    "MOVEMENT_DISPLACEMENT": "#e67e22",
}

DOMAIN_ORDER = [
    "ASCENT",
    "HEAT",
    "LIGHT",
    "MOVEMENT_DISPLACEMENT",
    "STRUCTURAL_FAILURE",
    "WATER_FLOOD",
    "WEIGHT_PRESSURE",
    "DARKNESS_DEATH",
]

PHASE_ORDER = [
    "displacement",
    "boom",
    "distress",
    "revulsion",
]

CRISIS_LABELS = {
    "eu_debt_2011": "EU Debt 2011",
    "china_2015": "China 2015",
    "brexit_2016": "Brexit 2016",
    "rate_hike_2018": "Rate Hike 2018",
    "covid_2020": "COVID 2020",
}


def parse_list_col(series):
    return series.apply(
        lambda x: (
            ast.literal_eval(x)
            if isinstance(x, str)
            else (x if isinstance(x, list) else [])
        )
    )


def load_data():
    df = read_parquet(PROCESSED_DIR / "merged.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["keyword_matches"] = parse_list_col(df["keyword_matches"])
    df["keyword_domains"] = parse_list_col(df["keyword_domains"])
    return df


# ── Figure 1: Domain × Phase Heatmap (all crises) ─────────────


def fig1_domain_phase_heatmap(df):
    """Heatmap of domain frequency across Kindleberger phases."""
    crisis_df = df[df["phase"] != "calm"].copy()
    exploded = crisis_df[["phase", "keyword_domains"]].explode("keyword_domains")
    exploded = exploded[exploded["keyword_domains"].notna()]
    exploded = exploded.rename(columns={"keyword_domains": "domain"})

    # Normalize by phase size to get rate
    phase_sizes = crisis_df.groupby("phase").size()
    pivot = (
        exploded.groupby(["phase", "domain"]).size().unstack(fill_value=0).astype(float)
    )
    # Rate per 1000 headlines
    for phase in pivot.index:
        pivot.loc[phase] = pivot.loc[phase] / phase_sizes[phase] * 1000

    # Reorder
    phases = [p for p in PHASE_ORDER if p in pivot.index]
    domains = [d for d in DOMAIN_ORDER if d in pivot.columns]
    pivot = pivot.loc[phases, domains]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Matches per 1,000 headlines"},
    )
    ax.set_title(
        "Metaphor Domain Density Across Kindleberger Phases\n(all 5 crises combined)",
        fontsize=14,
    )
    ax.set_ylabel("Phase")
    ax.set_xlabel("Metaphor Domain")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "domain_phase_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved domain_phase_heatmap.png")


# ── Figure 2: Per-crisis domain proportion comparison ──────────


def fig2_crisis_domain_comparison(df):
    """Grouped bar chart: domain proportions per crisis per phase."""
    crisis_df = df[(df["phase"] != "calm") & (df["crisis_event"].notna())].copy()
    exploded = crisis_df[["crisis_event", "phase", "keyword_domains"]].explode(
        "keyword_domains"
    )
    exploded = exploded[exploded["keyword_domains"].notna()]
    exploded = exploded.rename(columns={"keyword_domains": "domain"})

    # Focus on boom/euphoria vs distress/revulsion
    boom_phases = ["displacement", "boom"]
    crash_phases = ["distress", "revulsion"]

    records = []
    for event in CRISIS_EVENTS:
        for regime, phases in [
            ("Boom", boom_phases),
            ("Crash", crash_phases),
        ]:
            subset = exploded[
                (exploded["crisis_event"] == event) & (exploded["phase"].isin(phases))
            ]
            total = len(subset) if len(subset) > 0 else 1
            for domain in DOMAIN_ORDER:
                count = (subset["domain"] == domain).sum()
                records.append(
                    {
                        "crisis": CRISIS_LABELS.get(event, event),
                        "regime": regime,
                        "domain": domain,
                        "proportion": count / total,
                    }
                )

    rdf = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, regime in zip(axes, ["Boom", "Crash"]):
        subset = rdf[rdf["regime"] == regime].pivot(
            index="crisis", columns="domain", values="proportion"
        )
        domains = [d for d in DOMAIN_ORDER if d in subset.columns]
        subset = subset[domains]
        subset.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[DOMAIN_COLORS[d] for d in domains],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title(
            f"{regime} Phases (displacement + boom)"
            if regime == "Boom"
            else " (distress + revulsion)",
            fontsize=12,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Proportion of domain mentions")
        ax.legend(
            bbox_to_anchor=(1.0, 1.0),
            fontsize=7,
            title="Domain",
        )
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "Metaphor Domain Mix: Boom vs Crash Phases Across Crises",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "crisis_domain_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info("Saved crisis_domain_comparison.png")


# ── Figure 3: Time series with domain stacking ────────────────


def fig3_domain_timeseries(df):
    """Stacked area: daily domain counts + S&P 500 overlay."""
    df_with_domains = df[df["keyword_count"] > 0].copy()
    exploded = df_with_domains[["date", "keyword_domains"]].explode("keyword_domains")
    exploded = exploded[exploded["keyword_domains"].notna()]
    exploded = exploded.rename(columns={"keyword_domains": "domain"})

    # Weekly aggregation for smoother plot
    exploded["week"] = exploded["date"].dt.to_period("W").dt.start_time
    weekly = exploded.groupby(["week", "domain"]).size().unstack(fill_value=0)
    domains = [d for d in DOMAIN_ORDER if d in weekly.columns]
    weekly = weekly[domains]

    # S&P 500 weekly
    sp500 = (
        df[["date", "sp500_close"]]
        .drop_duplicates("date")
        .dropna()
        .set_index("date")
        .resample("W")
        .last()
    )

    fig, ax1 = plt.subplots(figsize=(16, 7))

    # Stacked area
    weekly.plot.area(
        ax=ax1,
        color=[DOMAIN_COLORS[d] for d in domains],
        alpha=0.7,
        linewidth=0,
    )
    ax1.set_ylabel("Weekly domain keyword count", fontsize=11)
    ax1.set_xlabel("")

    # S&P 500 overlay
    ax2 = ax1.twinx()
    ax2.plot(
        sp500.index,
        sp500["sp500_close"],
        color="black",
        linewidth=1.5,
        alpha=0.8,
        label="S&P 500",
    )
    ax2.set_ylabel("S&P 500 Close", fontsize=11)

    # Crisis bands
    crisis_colors = [
        "#ff999966",
        "#ffcc9966",
        "#99ccff66",
        "#cc99ff66",
        "#ff99cc66",
    ]
    for (event, _), color in zip(CRISIS_EVENTS.items(), crisis_colors):
        start, end = get_event_window(event)
        ax1.axvspan(
            pd.Timestamp(start, tz="UTC"),
            pd.Timestamp(end, tz="UTC"),
            alpha=0.15,
            color=color,
        )
        mid = (
            pd.Timestamp(start, tz="UTC")
            + (pd.Timestamp(end, tz="UTC") - pd.Timestamp(start, tz="UTC")) / 2
        )
        ax1.text(
            mid,
            ax1.get_ylim()[1] * 0.95,
            CRISIS_LABELS.get(event, event),
            ha="center",
            fontsize=7,
            style="italic",
        )

    ax1.legend(
        loc="upper left",
        fontsize=7,
        title="Domain",
        ncol=2,
    )
    ax2.legend(loc="upper right", fontsize=9)

    ax1.set_title(
        "Weekly Metaphor Domain Frequency vs S&P 500 (2009–2020)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "domain_timeseries_stacked.png",
        dpi=150,
    )
    plt.close(fig)
    logger.info("Saved domain_timeseries_stacked.png")


# ── Figure 4: COVID deep-dive daily timeline ──────────────────


def fig4_covid_deepdive(df):
    """COVID 2020: daily domain timeline with phase boundaries."""
    start = pd.Timestamp("2019-12-01", tz="UTC")
    end = pd.Timestamp("2020-05-31", tz="UTC")
    covid = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    exploded = covid[["date", "keyword_domains"]].explode("keyword_domains")
    exploded = exploded[exploded["keyword_domains"].notna()]
    exploded = exploded.rename(columns={"keyword_domains": "domain"})

    daily = exploded.groupby(["date", "domain"]).size().unstack(fill_value=0)
    # 7-day rolling
    daily = daily.rolling(7, min_periods=1).mean()
    domains = [d for d in DOMAIN_ORDER if d in daily.columns]
    daily = daily[domains]

    fig, ax = plt.subplots(figsize=(14, 6))
    daily.plot.area(
        ax=ax,
        color=[DOMAIN_COLORS[d] for d in domains],
        alpha=0.7,
        linewidth=0,
    )

    # Phase boundaries
    phases = CRISIS_EVENTS["covid_2020"]
    phase_colors = {
        "displacement": "#3498db",
        "distress": "#e67e22",
        "revulsion": "#e74c3c",
    }
    for phase, (ps, pe) in phases.items():
        ax.axvspan(
            pd.Timestamp(ps, tz="UTC"),
            pd.Timestamp(pe, tz="UTC"),
            alpha=0.1,
            color=phase_colors.get(phase, "gray"),
        )
        mid = (
            pd.Timestamp(ps, tz="UTC")
            + (pd.Timestamp(pe, tz="UTC") - pd.Timestamp(ps, tz="UTC")) / 2
        )
        ax.text(
            mid,
            ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1,
            phase.upper(),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=phase_colors.get(phase, "gray"),
        )

    ax.set_title(
        "COVID-19 Crisis: Daily Metaphor Domain Evolution (7d MA)",
        fontsize=14,
    )
    ax.set_ylabel("Domain mentions (7-day avg)")
    ax.set_xlabel("")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "covid_deepdive.png", dpi=150)
    plt.close(fig)
    logger.info("Saved covid_deepdive.png")


# ── Figure 5: Top words per phase — all crises combined ───────


def fig5_top_words_all_crises(df):
    """Top 10 metaphor keywords per phase (all crises combined)."""
    crisis_df = df[df["phase"] != "calm"].copy()

    phases = [p for p in PHASE_ORDER if p in crisis_df["phase"].unique()]
    fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 6))
    if len(phases) == 1:
        axes = [axes]

    for ax, phase in zip(axes, phases):
        phase_data = crisis_df[crisis_df["phase"] == phase]
        all_words = []
        for words in phase_data["keyword_matches"]:
            if isinstance(words, list):
                all_words.extend(words)

        top = Counter(all_words).most_common(12)
        if not top:
            continue

        words, counts = zip(*top)
        # Color by domain
        from market_metaphors.metaphor.domain_lexicon import (
            classify_word,
        )

        colors = [DOMAIN_COLORS.get(classify_word(w), "#bdc3c7") for w in words]

        ax.barh(range(len(words)), counts, color=colors)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(phase.upper(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Count")

    fig.suptitle(
        "Top Metaphor Keywords by Kindleberger Phase (all 5 crises, colored by domain)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "top_words_all_phases.png", dpi=150)
    plt.close(fig)
    logger.info("Saved top_words_all_phases.png")


# ── Figure 6: Granger results visualization ───────────────────


def fig6_granger_summary(df):
    """Visualize Granger causality p-values for both directions."""
    causality_dir = OUTPUTS_DIR / "causality"
    m2r_path = causality_dir / "granger_metaphor_to_returns.csv"
    r2m_path = causality_dir / "granger_returns_to_metaphor.csv"

    if not m2r_path.exists() or not r2m_path.exists():
        logger.warning("Granger results not found, skipping")
        return

    m2r = pd.read_csv(m2r_path)
    r2m = pd.read_csv(r2m_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, data, title, color in [
        (axes[0], m2r, "Metaphor → Returns", "#3498db"),
        (axes[1], r2m, "Returns → Metaphor", "#e74c3c"),
    ]:
        ax.bar(
            data["lag"],
            -np.log10(data["p_value"]),
            color=color,
            alpha=0.7,
            edgecolor="white",
        )
        ax.axhline(
            -np.log10(0.05),
            color="red",
            linestyle="--",
            linewidth=1,
            label="p = 0.05",
        )
        ax.axhline(
            -np.log10(0.01),
            color="darkred",
            linestyle=":",
            linewidth=1,
            label="p = 0.01",
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Lag (trading days)")
        ax.set_ylabel("-log₁₀(p-value)")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Granger Causality: Do Metaphors Lead or Follow Markets?",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "granger_summary.png", dpi=150)
    plt.close(fig)
    logger.info("Saved granger_summary.png")


# ── Figure 7: Metaphor ratio distribution by phase ────────────


def fig7_metaphor_ratio_by_phase(df):
    """Violin plot of model metaphor_ratio across phases."""
    crisis_df = df[df["phase"] != "calm"].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    phase_data = []
    for phase in PHASE_ORDER:
        subset = crisis_df[crisis_df["phase"] == phase]["metaphor_ratio"]
        if not subset.empty:
            phase_data.append(subset.values)

    present_phases = [p for p in PHASE_ORDER if p in crisis_df["phase"].unique()]
    ax.violinplot(
        phase_data,
        positions=range(len(present_phases)),
        showmeans=True,
        showmedians=True,
    )
    ax.set_xticks(range(len(present_phases)))
    ax.set_xticklabels([p.upper() for p in present_phases], fontsize=11)
    ax.set_ylabel("Metaphor Ratio (model-detected)")
    ax.set_title(
        "Distribution of Metaphor Intensity by Kindleberger Phase",
        fontsize=14,
    )

    # Add means as text
    for i, data in enumerate(phase_data):
        ax.text(
            i,
            np.mean(data) + 0.01,
            f"μ={np.mean(data):.3f}",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "metaphor_ratio_by_phase.png", dpi=150)
    plt.close(fig)
    logger.info("Saved metaphor_ratio_by_phase.png")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    logger.info("Loaded %d rows", len(df))

    fig1_domain_phase_heatmap(df)
    fig2_crisis_domain_comparison(df)
    fig3_domain_timeseries(df)
    fig4_covid_deepdive(df)
    fig5_top_words_all_crises(df)
    fig6_granger_summary(df)
    fig7_metaphor_ratio_by_phase(df)

    logger.info("All visualizations saved to %s", FIGURE_DIR)


if __name__ == "__main__":
    main()
