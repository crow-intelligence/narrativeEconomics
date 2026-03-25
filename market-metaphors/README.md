# Market Metaphors: Kindleberger Phases × Shiller Narrative Economics

Detecting and tracing the evolution of **metaphorical language** in financial news headlines across the five phases of a financial panic (Kindleberger's *Manias, Panics and Crashes*), connecting to Shiller's *Narrative Economics* thesis that viral economic narratives drive real market behavior.

This is an **exploratory research project** — we build a reproducible pipeline, generate data, and see what emerges.

## Setup

```bash
# Clone and enter the project
cd market-metaphors

# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Data Preparation

1. Download the Kaggle dataset (see Data Sources below) and place CSV files in `data/raw/news/`
2. Run the pipeline:

```bash
uv run python scripts/pipeline/02_clean_news.py     # Clean headlines
uv run python scripts/pipeline/01_download_market.py # Fetch market data
uv run python scripts/pipeline/03_run_metaphor.py    # Run metaphor detection
uv run python scripts/pipeline/04_build_merged.py    # Merge all data
```

3. Run analysis scripts:

```bash
uv run python scripts/analysis/05_time_series.py        # Time series plot
uv run python scripts/analysis/06_kindleberger_phases.py # Phase analysis
uv run python scripts/analysis/07_narrative_contagion.py # Contagion analysis
uv run python scripts/analysis/08_granger_causality.py   # Lead/lag analysis
```

## Data Sources

| Source | URL | License | Notes |
|---|---|---|---|
| Kaggle Stock News | https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests | See Kaggle page | ~2M rows, 2009–2020. Place CSVs in `data/raw/news/` |
| S&P 500 + Stock Prices | Yahoo Finance via `yfinance` | Public market data | Downloaded programmatically by `01_download_market.py` |
| Metaphor Detection Model | https://huggingface.co/lwachowiak/Metaphor-Detection-XLMR | Apache 2.0 | XLM-RoBERTa base, token-level, F1=0.76 |
| Large Model (optional, GPU) | https://huggingface.co/HiTZ/xlm-roberta-large-metaphor-detection-en | Apache 2.0 | XLM-RoBERTa large, higher quality |

## Model Citation

```bibtex
@inproceedings{wachowiak-etal-2022-parameter,
    title = "Does It Capture {STEL}? A Modular, Similarity-based Linguistic Style Evaluation Framework",
    author = "Wachowiak, Lennart and Gromann, Dagmar",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022",
}

@inproceedings{sanchez-bayona-agerri-2022-leveraging,
    title = "Leveraging a New {S}panish Corpus for Multilingual and Crosslingual Metaphor Detection",
    author = "Sanchez-Bayona, Elisa and Agerri, Rodrigo",
    booktitle = "Proceedings of the 26th Conference on Computational Natural Language Learning (CoNLL)",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

## Pipeline

```
┌─────────────────┐    ┌──────────────────┐
│  Kaggle CSV      │    │  yfinance API    │
│  (raw news)      │    │  (market data)   │
└───────┬─────────┘    └───────┬──────────┘
        │                       │
        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│ 02_clean_news   │    │ 01_download_mkt  │
│ → news_clean    │    │ → ticker.parquet │
└───────┬─────────┘    └───────┬──────────┘
        │                       │
        ▼                       │
┌─────────────────┐             │
│ 03_run_metaphor │             │
│ XLM-R + lexicon │             │
│ → metaphors.pq  │             │
└───────┬─────────┘             │
        │                       │
        ▼                       ▼
┌─────────────────────────────────────────┐
│           04_build_merged               │
│  news + metaphors + market + phases     │
│           → merged.parquet              │
└───────────────────┬─────────────────────┘
                    │
        ┌───────────┼───────────┬──────────────┐
        ▼           ▼           ▼              ▼
   05_time     06_kindle   07_contagion   08_granger
   series      berger      (Shiller)      causality
```

## Sampling Strategy

The full cleaned dataset contains ~1.4M headlines across ~6,200 tickers. Running the
XLM-RoBERTa metaphor detection model on CPU is slow (~10.7 headlines/sec), so we evaluated
several sampling strategies to make overnight batch runs feasible:

| Strategy | Headlines | Est. CPU hours | Notes |
|---|---|---|---|
| **Full dataset** | 1,395,031 | ~36h | All tickers, all headlines |
| **Tickers with market data only** | 891,669 | ~23h | Drop ~2,600 delisted/failed tickers |
| **A) Top 500 tickers, 3/ticker/day** | 419,703 | ~10.9h | Good per-ticker + daily coverage (~108/day avg) |
| **B) 100 random headlines/day** | 285,341 | ~7.4h | Composite index view, loses ticker granularity |
| **B) 50 random headlines/day** | 151,928 | ~3.9h | Safe overnight, decent daily signal |
| **1 headline/ticker/day** | 532,185 | ~13.8h | Aggressive dedup, risks losing nuance |
| **Crisis windows ±60 days only** | 457,072 | ~11.9h | Drops calm periods entirely |

**Recommended for full analysis**: Strategy A (top 500 tickers, 3 headlines/ticker/day) —
preserves both per-ticker and daily aggregate signal with manageable runtime.

**Current run**: Strategy B with 50 headlines/day, focused on S&P 500 composite index
analysis. This trades per-ticker granularity for faster iteration (~4 hours on CPU).

## Crisis Events Analyzed

| Event | Period | Phases Covered |
|---|---|---|
| 2011 EU Debt Crisis | May 2010 – Sep 2011 | Displacement → Revulsion |
| 2015 China Slowdown | Jun 2015 – Sep 2015 | Displacement → Revulsion |
| 2016 Brexit Vote | Feb 2016 – Jun 2016 | Displacement → Revulsion |
| 2018 Rate Hike Selloff | Jan 2018 – Dec 2018 | Displacement → Revulsion |
| 2020 COVID Crash | Jan 2020 – Mar 2020 | Displacement → Revulsion |

## Known Limitations

- **Headline-only analysis**: Article bodies are not available in the Kaggle dataset; metaphor detection operates only on short headline text, which limits metaphor density and context.
- **Publication time unknown**: The dataset does not distinguish pre-market vs. post-market publication times, which affects lead/lag interpretation in Granger causality tests.
- **Domain lexicon is hand-curated**: The metaphor domain taxonomy is manually constructed and may miss relevant metaphorical expressions or mis-categorize edge cases.
- **Metaphor model trained on general text**: The XLM-RoBERTa model is trained on the VU Amsterdam Metaphor Corpus (general English), not financial text — domain-specific metaphors may be under-detected.
- **Class imbalance**: Most tokens are non-metaphoric; low metaphor density per headline means signal may be noisy at daily aggregation level.
- **Granger causality ≠ true causality**: Statistical lead/lag relationships do not establish causal mechanisms.
- **No source credibility weighting**: All headlines are treated equally regardless of source prominence or reach.
