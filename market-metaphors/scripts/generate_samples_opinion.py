"""Generate sample review spreadsheet for opinion/editorial datasets.

Produces an Excel file (one sheet per dataset + summary) with 100 samples
from each dataset collected on 2026-04-11.

Usage:
    uv run python scripts/generate_samples_opinion.py
"""

import random

import pandas as pd

from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import OUTPUTS_DIR, RAW_DIR

logger = get_logger(__name__)

SAMPLE_SIZE = 100
MIN_LEN = 50
MAX_TEXT_FOR_EXCEL = 5000

random.seed(42)

BOILERPLATE = [
    "operator instructions",
    "forward-looking statements",
    "safe harbor",
    "this call is being recorded",
    "thank you for standing by",
    "copyright",
    "all rights reserved",
]


def is_boilerplate(text: str) -> bool:
    lower = text.lower()
    return any(pat in lower for pat in BOILERPLATE)


def split_paragraphs(text: str) -> list[str]:
    """Split long text into paragraphs."""
    if not text or not isinstance(text, str):
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) <= 2 and any(len(p) > 3000 for p in paras):
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    # Split on ". " for very long blocks with no newlines
    if len(paras) <= 2 and any(len(p) > 3000 for p in paras):
        new_paras = []
        for p in paras:
            if len(p) > 3000:
                sentences = p.split(". ")
                chunk = ""
                for s in sentences:
                    chunk += s + ". "
                    if len(chunk) > 500:
                        new_paras.append(chunk.strip())
                        chunk = ""
                if chunk.strip():
                    new_paras.append(chunk.strip())
            else:
                new_paras.append(p)
        paras = new_paras
    return [
        p[:MAX_TEXT_FOR_EXCEL]
        for p in paras
        if len(p) >= MIN_LEN and not is_boilerplate(p)
    ]


def sample_items(items: list[dict], n: int = SAMPLE_SIZE) -> list[dict]:
    """Sample n items, trying to get diversity."""
    if len(items) <= n:
        return items
    return random.sample(items, n)


# ── Loaders ─────────────────────────────────────────────────────────


def load_phrasebank() -> list[dict]:
    """Sentences — already atomic."""
    df = pd.read_parquet(RAW_DIR / "phrasebank" / "phrasebank.parquet")
    # Filter out the header rows that leaked in
    df = df[df["text"].str.len() >= MIN_LEN]
    items = []
    for _, row in df.iterrows():
        items.append({
            "text": row["text"],
            "source": "Financial PhraseBank",
            "label": row.get("label", ""),
            "date": "",
            "notes": f"Agreement: {row.get('agreement', '')}",
        })
    return items


def load_fiqa() -> list[dict]:
    """Opinion text from the `doc` column."""
    df = pd.read_parquet(RAW_DIR / "fiqa" / "fiqa.parquet")
    df = df.dropna(subset=["doc"])
    df = df[df["doc"].astype(str).str.len() >= MIN_LEN]
    items = []
    for _, row in df.iterrows():
        items.append({
            "text": str(row["doc"])[:MAX_TEXT_FOR_EXCEL],
            "source": "FiQA",
            "label": "",
            "date": "",
            "notes": f"Config: {row.get('config', '')}",
        })
    return items


def load_twitter() -> list[dict]:
    """Tweets — already atomic."""
    df = pd.read_parquet(
        RAW_DIR / "twitter_finance" / "twitter_finance.parquet",
    )
    items = []
    for _, row in df.iterrows():
        items.append({
            "text": row["text"],
            "source": "Twitter Financial",
            "label": str(row.get("label", "")),
            "date": "",
            "notes": "",
        })
    return items


def load_reddit() -> list[dict]:
    """Reddit posts — title + selftext."""
    df = pd.read_parquet(
        RAW_DIR / "reddit_finance" / "reddit_finance_sample.parquet",
    )
    items = []
    for _, row in df.iterrows():
        text = str(row.get("text", ""))
        if len(text) < MIN_LEN:
            continue
        items.append({
            "text": text[:MAX_TEXT_FOR_EXCEL],
            "source": "Reddit Finance",
            "label": "",
            "date": str(row.get("created", "")),
            "notes": f"r/{row.get('subreddit', '')}",
        })
    return items


def load_earnings_calls() -> list[dict]:
    """Paragraphs from prepared remarks."""
    df = pd.read_parquet(
        RAW_DIR / "earnings_calls" / "strux_sample.parquet",
    )
    items = []
    for _, row in df.iterrows():
        paras = split_paragraphs(row.get("prepared_remarks", ""))
        # Skip first 3 paragraphs (operator intro, safe harbor)
        paras = paras[3:]
        for para in paras:
            items.append({
                "text": para,
                "source": "Strux Earnings Calls",
                "label": "",
                "date": str(row.get("date", "")),
                "notes": f"Ticker: {row.get('ticker', '')}",
            })
    return items


def load_bloomberg_reuters() -> list[dict]:
    """Articles — prefer longer entries (actual articles over headlines)."""
    df = pd.read_parquet(
        RAW_DIR / "multisource" / "bloomberg_reuters.parquet",
    )
    # Strongly prefer actual articles (>500 chars) over short headlines
    df["text_len"] = df["text"].str.len()
    long = df[df["text_len"] >= 500].copy()
    short = df[
        (df["text_len"] >= MIN_LEN) & (df["text_len"] < 500)
    ].copy()
    # Take mostly long articles, pad with short if needed
    if len(long) >= SAMPLE_SIZE:
        source_df = long
    else:
        source_df = pd.concat([long, short.head(SAMPLE_SIZE - len(long))])

    items = []
    for _, row in source_df.iterrows():
        items.append({
            "text": row["text"][:MAX_TEXT_FOR_EXCEL],
            "source": "Bloomberg/Reuters",
            "label": "",
            "date": str(row.get("date", "")),
            "notes": f"{row['text_len']} chars",
        })
    return items


def load_all_the_news() -> list[dict]:
    """Full articles — split into paragraphs."""
    df = pd.read_parquet(
        RAW_DIR / "multisource" / "all_the_news_2.parquet",
    )
    items = []
    for _, row in df.sample(min(500, len(df)), random_state=42).iterrows():
        paras = split_paragraphs(row.get("text", ""))
        # Skip first paragraph (often a lede that's just a headline rewrite)
        for para in paras[1:]:
            items.append({
                "text": para,
                "source": "All The News 2",
                "label": "",
                "date": str(row.get("date", "")),
                "notes": "",
            })
    return items


def load_reddit_sp500() -> list[dict]:
    """Reddit posts about S&P 500."""
    df = pd.read_parquet(
        RAW_DIR / "multisource" / "reddit_finance_sp500.parquet",
    )
    df = df[df["text"].str.len() >= MIN_LEN]
    items = []
    for _, row in df.iterrows():
        items.append({
            "text": row["text"][:MAX_TEXT_FOR_EXCEL],
            "source": "Reddit SP500 (multisource)",
            "label": "",
            "date": str(row.get("date", "")),
            "notes": "",
        })
    return items


# ── Main ────────────────────────────────────────────────────────────

DATASETS = [
    ("PhraseBank", load_phrasebank),
    ("FiQA", load_fiqa),
    ("Twitter Financial", load_twitter),
    ("Reddit Finance", load_reddit),
    ("Earnings Calls", load_earnings_calls),
    ("Bloomberg Reuters", load_bloomberg_reuters),
    ("All The News 2", load_all_the_news),
    ("Reddit SP500", load_reddit_sp500),
]


def main() -> None:
    per_dataset: dict[str, pd.DataFrame] = {}

    for name, loader in DATASETS:
        logger.info("Loading %s...", name)
        try:
            items = loader()
        except Exception:
            logger.exception("Failed to load %s", name)
            continue

        if not items:
            logger.warning("No data for %s", name)
            continue

        sampled = sample_items(items, SAMPLE_SIZE)
        logger.info("  %d samples from %s", len(sampled), name)
        per_dataset[name] = pd.DataFrame(sampled)

    # Write Excel
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_path = OUTPUTS_DIR / "sample_review_opinion.xlsx"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Summary sheet
        summary = []
        for name, df in per_dataset.items():
            summary.append({
                "Dataset": name,
                "Samples": len(df),
                "Avg text length (chars)": int(
                    df["text"].str.len().mean()
                ),
                "Source": df["source"].iloc[0] if len(df) > 0 else "",
            })
        pd.DataFrame(summary).to_excel(
            writer, sheet_name="Summary", index=False,
        )

        for name, df in per_dataset.items():
            sheet = name[:31]
            df_xl = df.copy()
            df_xl["text"] = df_xl["text"].str[:MAX_TEXT_FOR_EXCEL]
            df_xl.to_excel(writer, sheet_name=sheet, index=False)

    logger.info("Wrote %s (%d sheets)", xlsx_path, len(per_dataset) + 1)


if __name__ == "__main__":
    main()
