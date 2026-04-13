"""Generate an HTML report describing all collected datasets.

Produces a self-contained HTML file with embedded charts showing:
- Text length distributions
- Temporal coverage (articles/docs per year)
- Basic statistics (row counts, avg/median lengths)

Usage:
    uv run python scripts/generate_dataset_report.py
"""

import base64
import csv
import glob
import json
import sys
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import OUTPUTS_DIR, RAW_DIR

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", palette="muted")

logger = get_logger(__name__)

csv.field_size_limit(sys.maxsize)


def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_length_hist(lengths: pd.Series, title: str) -> str:
    """Create a text length histogram, return as base64 img."""
    fig, ax = plt.subplots(figsize=(8, 3))
    clipped = lengths.clip(upper=lengths.quantile(0.95))
    ax.hist(clipped, bins=50, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Text length (chars)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    return fig_to_base64(fig)


def make_timeline(dates: pd.Series, title: str) -> str:
    """Create a yearly article count bar chart, return as base64 img."""
    years = dates.dt.year.dropna()
    if years.empty:
        return ""
    counts = years.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3))
    counts.plot.bar(ax=ax, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.xticks(rotation=45)
    return fig_to_base64(fig)


def stats_table(lengths: pd.Series) -> dict:
    return {
        "count": int(len(lengths)),
        "mean_chars": int(lengths.mean()) if len(lengths) else 0,
        "median_chars": int(lengths.median()) if len(lengths) else 0,
        "min_chars": int(lengths.min()) if len(lengths) else 0,
        "max_chars": int(lengths.max()) if len(lengths) else 0,
        "p25": int(lengths.quantile(0.25)) if len(lengths) else 0,
        "p75": int(lengths.quantile(0.75)) if len(lengths) else 0,
    }


# ── Dataset loaders ─────────────────────────────────────────────────
# Each returns (description_md, df_with_text_and_date)


def load_fomc_minutes():
    path = RAW_DIR / "fed" / "fomc-minutes.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    df = pd.DataFrame(rows)
    df["text_len"] = df["Text"].str.len()
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    desc = """**FOMC Meeting Minutes** contain the full deliberation
records of the Federal Reserve's rate-setting body. These are among the
most carefully worded documents in economics — every phrase is
negotiated, and language shifts ("considerable period" → "patient" →
"data-dependent") signal policy changes that move global markets.
For Kindleberger analysis, the minutes reveal how the Fed narratively
frames risk during each crisis phase. Each row is one meeting's
complete minutes (~10–40K chars). Covers the Greenspan, Bernanke,
Yellen, and Powell eras. Source: Kaggle (ganghyeoklee).
`data/raw/fed/fomc-minutes.csv`"""
    return desc, df.rename(columns={"Text": "text"}), "text_len"


def load_fomc_paragraphs():
    path = RAW_DIR / "fed" / "fomc_documents.csv"
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= 10000:
                break
    df = pd.DataFrame(rows)
    df["text_len"] = df["text"].astype(str).str.len()
    df["date"] = pd.to_datetime(df["meeting_date"], errors="coerce")
    kinds = df["document_kind"].value_counts().to_dict()
    kinds_str = ", ".join(f"{k}: {v}" for k, v in kinds.items())
    desc = f"""**FOMC Paragraph-Level Documents** provide a broader set of
Fed publications beyond just the minutes. This includes the Beige Book
(anecdotal regional economic summaries — rich in everyday metaphors like
"overheating", "cooling", "soft landing"), greenbooks (staff economic
forecasts), bluebooks (policy options), press conference transcripts,
and policy statements. 327 MB total. Document types (first 10K rows):
{kinds_str}. Note: individual records are continuous text blocks without
paragraph breaks. Source: Kaggle (edwardbickerton).
`data/raw/fed/fomc_documents.csv`"""
    return desc, df, "text_len"


def load_cbs_speeches():
    path = RAW_DIR / "speeches" / "cbs_speeches.csv"
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= 20000:
                break
    df = pd.DataFrame(rows)
    df["text_len"] = df["text"].astype(str).str.len()
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    banks = (
        df["CentralBank"].value_counts().head(10).to_dict()
    )
    banks_str = ", ".join(f"{k}: {v}" for k, v in banks.items())
    desc = f"""**CBS Central Bank Speeches** is the single most
comprehensive central bank speech dataset available. It covers 35,487
speeches from 131 central banks worldwide (1986–2023). Speeches are
public-facing communications where central bankers explain policy to
broader audiences — they use more accessible, metaphor-laden language
than formal policy documents. This enables cross-country comparison of
narrative framing during shared crises (2008 GFC, 2011 EU debt, 2020
COVID). 498 MB. Top banks (first 20K): {banks_str}. Source:
cbspeeches.com. `data/raw/speeches/cbs_speeches.csv`"""
    return desc, df, "text_len"


def load_bis_speeches():
    path = RAW_DIR / "speeches" / "bis" / "speeches.csv"
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= 20000:
                break
    df = pd.DataFrame(rows)
    df["text_len"] = df["text"].astype(str).str.len()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    desc = """**BIS Speeches Archive** complements the CBS dataset with
speeches curated by the Bank for International Settlements. BIS
focuses on senior central bankers (governors, deputy governors) and
extends coverage to 2025, filling the gap after CBS stops at 2023.
The BIS is the "central bank of central banks", so its curation
reflects what the international monetary community considers
significant. 366 MB. Source: bis.org.
`data/raw/speeches/bis/speeches.csv`"""
    return desc, df, "text_len"


def load_ecb_fed():
    path = (
        RAW_DIR / "speeches" / "ecb_fed" / "data"
        / "train-00000-of-00001.parquet"
    )
    df = pd.read_parquet(path)
    df["text_len"] = df["clean_text"].astype(str).str.len()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    countries = df["country"].value_counts().to_dict()
    countries_str = ", ".join(
        f"{k}: {v}" for k, v in countries.items()
    )
    desc = f"""**ECB-FED Speeches** is a curated subset focused on the
two most influential central banks. What makes it unique is the
OCR-extracted text column (via Mistral API) — many older ECB speeches
were only available as scanned PDFs, and this dataset recovered their
full text. Useful for comparing the ECB's "price stability" narrative
with the Fed's "dual mandate" framing. 4,987 speeches (1996–2025),
140 MB. Countries: {countries_str}. Source: HuggingFace
(istat-ai/ECB-FED-speeches). `data/raw/speeches/ecb_fed/`"""
    return desc, df.rename(columns={"clean_text": "text"}), "text_len"


def load_rba():
    df = pd.read_parquet(RAW_DIR / "rba" / "minutes.parquet")
    ok = df[df["status"] == "ok"].copy()
    ok["text_len"] = ok["text"].str.len()
    ok["date"] = pd.to_datetime(ok["date"], errors="coerce")
    desc = f"""**RBA Board Minutes** record the Reserve Bank of
Australia's monetary policy decisions. Australia's economy is
commodity-driven and interest-rate-sensitive, producing distinctive
metaphorical language around mining cycles, housing markets, and
the "terms of trade". Scraped from rba.gov.au. {len(ok)} minutes
successfully extracted (2015–2026), {len(df) - len(ok)} failed.
`data/raw/rba/minutes.parquet`"""
    return desc, ok, "text_len"


def load_boc():
    df = pd.read_parquet(RAW_DIR / "boc" / "deliberations.parquet")
    ok = df[df["status"] == "ok"].copy()
    ok["text_len"] = ok["text"].str.len()
    ok["date"] = pd.to_datetime(ok["date"], errors="coerce")
    desc = f"""**BoC Deliberations** are the Bank of Canada's summaries
of Governing Council discussions. This is a new publication program
(started Jan 2025), so coverage is limited to {len(ok)} documents.
Canada's economy is tightly coupled with US markets and commodity
prices, making these useful for cross-border narrative comparison.
`data/raw/boc/deliberations.parquet`"""
    return desc, ok, "text_len"


def load_huffpost():
    dfs = []
    for path in sorted(glob.glob(str(RAW_DIR / "huffpost" / "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            dfs.append(pd.DataFrame(data))
    df = pd.concat(dfs, ignore_index=True)
    df["text_len"] = df["short_description"].astype(str).str.len()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    desc = f"""**HuffPost Sentiment** provides {len(df):,} articles with
both a headline and an editorial lead (`short_description` field,
2012–2022). The lead is where editors "sell" the story to readers —
it's typically the most metaphor-dense sentence in any article. This
makes it particularly valuable for studying how editorial framing
choices vary across market conditions. 95 MB. Source: Kaggle.
`data/raw/huffpost/`"""
    return (
        desc,
        df.rename(columns={"short_description": "text"}),
        "text_len",
    )


def load_articles():
    dfs = []
    for path in sorted(
        glob.glob(str(RAW_DIR / "articles" / "*.parquet"))
    ):
        dfs.append(pd.read_parquet(path))
    if not dfs:
        return None, None, None
    df = pd.concat(dfs, ignore_index=True)
    ok = df[df["status"] == "ok"].copy()
    ok["text_len"] = (
        ok["lead"].astype(str).str.len()
        + ok["body"].astype(str).str.len()
    )
    ok["date"] = pd.to_datetime(ok.get("fetch_date", ""), errors="coerce")
    total = len(df)
    n_ok = len(ok)
    by_domain = (
        df["domain"].value_counts().head(5).to_dict()
    )
    domain_str = ", ".join(f"{k}: {v}" for k, v in by_domain.items())
    status = df["status"].value_counts().to_dict()
    status_str = ", ".join(f"{k}: {v}" for k, v in status.items())
    desc = f"""**Kaggle Stock News (scraped)** — Our primary news corpus.
We scraped full article text from URLs in the Kaggle stock news dataset
using trafilatura, splitting each article into lead (first paragraph)
and body. The dataset spans 16 financial news domains (2009–2020),
covering the post-GFC recovery, the China scare, Brexit, and COVID.
{total:,} URLs attempted, {n_ok:,} successful. Top domains:
{domain_str}. Status breakdown: {status_str}. `data/raw/articles/`"""
    return desc, ok, "text_len"


def load_phrasebank():
    df = pd.read_parquet(RAW_DIR / "phrasebank" / "phrasebank.parquet")
    df = df[df["text"].str.len() >= 30]
    df["text_len"] = df["text"].str.len()
    labels = df["label"].value_counts().to_dict()
    labels_str = ", ".join(f"{k}: {v}" for k, v in labels.items())
    desc = f"""**Financial PhraseBank** is the standard academic benchmark
for financial sentiment analysis. {len(df):,} sentences were hand-picked
from financial news by domain experts and annotated for sentiment. Each
sentence was selected because it expresses an opinion about a company
or market — exactly the type of text where metaphorical framing is most
prevalent. Useful for validating our metaphor detector against an
established gold standard. Labels: {labels_str}. Source: HuggingFace
(takala/financial_phrasebank). `data/raw/phrasebank/`"""
    return desc, df, "text_len"


def load_fiqa():
    df = pd.read_parquet(RAW_DIR / "fiqa" / "fiqa.parquet")
    df = df.dropna(subset=["doc"])
    df["text"] = df["doc"].astype(str)
    df["text_len"] = df["text"].str.len()
    configs = df["config"].value_counts().to_dict()
    configs_str = ", ".join(f"{k}: {v}" for k, v in configs.items())
    desc = f"""**FiQA** (Financial Opinion Mining and Question Answering)
is a dataset designed specifically for extracting opinions from
financial text. It contains posts from financial forums and news
with aspect-level sentiment annotations — not just "positive/negative"
but targeted at specific financial aspects (e.g. "bullish on revenue
but bearish on margins"). {len(df):,} entries. Configs: {configs_str}.
Source: HuggingFace (vibrantlabsai/fiqa). `data/raw/fiqa/`"""
    return desc, df, "text_len"


def load_twitter():
    df = pd.read_parquet(
        RAW_DIR / "twitter_finance" / "twitter_finance.parquet",
    )
    df["text_len"] = df["text"].str.len()
    labels = df["label"].value_counts().to_dict()
    labels_str = ", ".join(f"{k}: {v}" for k, v in labels.items())
    desc = f"""**Twitter Financial Sentiment** captures retail investor
narrative on social media. Tweets are short, opinion-heavy, and use
informal metaphorical language ("to the moon", "diamond hands", "bears
in shambles", "blood in the streets"). {len(df):,} tweets with
sentiment labels. Labels: {labels_str}. Source: HuggingFace
(zeroshot/twitter-financial-news-sentiment).
`data/raw/twitter_finance/`"""
    return desc, df, "text_len"


def load_reddit():
    df = pd.read_parquet(
        RAW_DIR / "reddit_finance" / "reddit_finance_sample.parquet",
    )
    df["text_len"] = df["text"].astype(str).str.len()
    df["date"] = pd.to_datetime(df.get("created", ""), errors="coerce")
    subs = df["subreddit"].value_counts().to_dict()
    subs_str = ", ".join(f"r/{k}: {v}" for k, v in subs.items())
    desc = f"""**Reddit Finance** provides retail investor opinion text
from 14 finance-related subreddits. r/wallstreetbets is particularly
rich in colorful metaphorical language during market events (2021 GME
saga, COVID crash). Unlike professional financial writing, Reddit posts
are unfiltered opinion — the raw narrative that Shiller argues drives
market behavior. {len(df):,} sampled posts. Subreddits: {subs_str}.
Source: Kaggle (leukipp/reddit-finance-data). Sample of 20K.
`data/raw/reddit_finance/`"""
    return desc, df, "text_len"


def load_earnings():
    df = pd.read_parquet(
        RAW_DIR / "earnings_calls" / "strux_sample.parquet",
    )
    df["text"] = df["full_text"]
    df["text_len"] = df["text"].str.len()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    desc = f"""**Strux Earnings Calls** contain management's prepared
remarks and analyst Q&A from quarterly earnings calls. This is where
CEOs and CFOs narrate their company's trajectory using rich metaphorical
language — "headwinds", "tailwinds", "runway", "bridge financing",
"war chest", "inflection point". These are opinion-grade texts from
corporate insiders, making them ideal for detecting how management
framing shifts across Kindleberger phases. {len(df):,} transcripts
from {df['ticker'].nunique()} tickers
({df['date'].min().strftime('%Y-%m')} to
{df['date'].max().strftime('%Y-%m')}). Sample of 1,000 from 11,950
total. Source: HuggingFace (BUILDERlym/STRUX-Transcripts).
`data/raw/earnings_calls/`"""
    return desc, df, "text_len"


def load_bloomberg():
    df = pd.read_parquet(
        RAW_DIR / "multisource" / "bloomberg_reuters.parquet",
    )
    df["text_len"] = df["text"].str.len()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_long = (df["text_len"] >= 500).sum()
    n_short = (df["text_len"] < 500).sum()
    desc = f"""**Bloomberg/Reuters** (multisource subset) represents
premium financial journalism from two of the world's most influential
wire services. Covers 2006–2013, including the Global Financial Crisis
— the definitive Kindleberger cycle of our era. <strong>Important
composition note:</strong> of {len(df):,} entries, only <strong>{n_long}
are actual articles</strong> (≥500 chars, avg ~3,500 chars) while
<strong>{n_short} are headlines or short summaries</strong> (<500 chars,
avg ~110 chars). The full-text articles are the valuable portion for
metaphor analysis. 10K sample from 580K total rows. Source: HuggingFace
(financial-news-multisource/bloomberg_reuters).
`data/raw/multisource/bloomberg_reuters.parquet`"""
    return desc, df, "text_len"


def load_all_the_news():
    df = pd.read_parquet(
        RAW_DIR / "multisource" / "all_the_news_2.parquet",
    )
    df["text_len"] = df["text"].str.len()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    desc = f"""**All The News 2** (multisource subset) is a broad
collection of full-length news articles from multiple publishers.
Unlike the Bloomberg/Reuters subset, nearly all entries here contain
substantial article text (avg {df['text_len'].mean():.0f} chars),
making it one of our richest sources for full-text analysis. The
diversity of publishers provides a range of editorial perspectives
and writing styles. {len(df):,} articles, 10K sample. Source:
HuggingFace (financial-news-multisource/all_the_news_2).
`data/raw/multisource/all_the_news_2.parquet`"""
    return desc, df, "text_len"


def load_reddit_sp500():
    df = pd.read_parquet(
        RAW_DIR / "multisource" / "reddit_finance_sp500.parquet",
    )
    df["text_len"] = df["text"].str.len()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    desc = f"""**Reddit Finance SP500** (multisource subset) contains
Reddit posts specifically discussing S&P 500 stocks (2008–2025). This
complements our Kaggle Reddit sample with a different time range and
ticker focus. Posts include both discussion-length selftext and shorter
title-only entries. Average length {df['text_len'].mean():.0f} chars.
{len(df):,} posts, 10K sample. Source: HuggingFace
(financial-news-multisource/reddit_finance_sp500).
`data/raw/multisource/reddit_finance_sp500.parquet`"""
    return desc, df, "text_len"


# ── Report generation ───────────────────────────────────────────────

ALL_DATASETS = [
    ("FOMC Minutes", load_fomc_minutes),
    ("FOMC Paragraph-Level", load_fomc_paragraphs),
    ("CBS Speeches", load_cbs_speeches),
    ("BIS Speeches", load_bis_speeches),
    ("ECB-FED Speeches", load_ecb_fed),
    ("RBA Board Minutes", load_rba),
    ("BoC Deliberations", load_boc),
    ("HuffPost Sentiment", load_huffpost),
    ("Kaggle Stock News", load_articles),
    ("Financial PhraseBank", load_phrasebank),
    ("FiQA", load_fiqa),
    ("Twitter Financial", load_twitter),
    ("Reddit Finance", load_reddit),
    ("Earnings Calls", load_earnings),
    ("Bloomberg/Reuters", load_bloomberg),
    ("All The News 2", load_all_the_news),
    ("Reddit SP500", load_reddit_sp500),
]


def generate_html() -> str:
    sections = []

    for name, loader in ALL_DATASETS:
        logger.info("Processing %s...", name)
        try:
            desc, df, len_col = loader()
        except Exception:
            logger.exception("Failed: %s", name)
            continue
        if df is None or df.empty:
            continue

        lengths = df[len_col].dropna()
        st = stats_table(lengths)

        # Charts
        hist_img = make_length_hist(lengths, f"{name} — Text Length")
        timeline_img = ""
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce").dropna()
            if len(dates) > 10:
                date_range = (
                    f"{dates.min().strftime('%Y-%m-%d')} to "
                    f"{dates.max().strftime('%Y-%m-%d')}"
                )
                timeline_img = make_timeline(
                    dates, f"{name} — Documents per Year",
                )
            else:
                date_range = "insufficient date data"
        else:
            date_range = "no date column"

        section = f"""
<div class="dataset">
<h2>{name}</h2>
<p>{desc}</p>
<table>
<tr><th>Rows</th><td>{st['count']:,}</td>
    <th>Mean</th><td>{st['mean_chars']:,} chars</td>
    <th>Median</th><td>{st['median_chars']:,} chars</td></tr>
<tr><th>Min</th><td>{st['min_chars']:,} chars</td>
    <th>P25</th><td>{st['p25']:,} chars</td>
    <th>P75</th><td>{st['p75']:,} chars</td></tr>
<tr><th>Max</th><td>{st['max_chars']:,} chars</td>
    <th colspan="2">Date range</th><td>{date_range}</td></tr>
</table>
<div class="charts">
<img src="data:image/png;base64,{hist_img}" />
"""
        if timeline_img:
            section += (
                f'<img src="data:image/png;base64,{timeline_img}" />'
            )
        section += "</div></div>"
        sections.append(section)

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Dataset Report — Narrative Economics</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1100px;
       margin: 40px auto; padding: 0 20px; color: #333;
       background: #fafafa; }}
h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
h2 {{ color: #2c3e50; margin-top: 40px;
      border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
.dataset {{ background: white; padding: 20px; margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
table {{ border-collapse: collapse; margin: 15px 0; }}
th, td {{ padding: 6px 14px; text-align: left;
          border: 1px solid #ddd; }}
th {{ background: #f5f5f5; font-weight: 600; }}
.charts {{ display: flex; flex-wrap: wrap; gap: 10px;
           margin-top: 15px; }}
.charts img {{ max-width: 48%; height: auto; }}
p {{ line-height: 1.6; }}
</style>
</head><body>
<h1>Dataset Report — Narrative Economics</h1>
<p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}.
All datasets collected for Kindleberger phase × Shiller narrative
economics analysis.</p>
{body}
</body></html>"""


def main() -> None:
    html = generate_html()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / "dataset_report.html"
    path.write_text(html, encoding="utf-8")
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
