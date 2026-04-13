"""Stream and sample from Financial-news-multisource subsets.

Downloads 10K rows from each selected subset to avoid the full 21.4 GB.

Usage:
    uv run python scripts/pipeline/00b_download_multisource.py
    uv run python scripts/pipeline/00b_download_multisource.py \
        --subset bloomberg_reuters
"""

import argparse

import pandas as pd
from datasets import load_dataset

from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import RAW_DIR

logger = get_logger(__name__)

SAMPLE_SIZE = 10_000
MIN_TEXT_LEN = 100
OUT_DIR = RAW_DIR / "multisource"

SUBSETS = [
    "bloomberg_reuters",
    "all_the_news_2",
    "reddit_finance_sp500",
]


def stream_subset(subset: str) -> None:
    """Stream a subset and save a filtered sample."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{subset}.parquet"
    if path.exists():
        logger.info("[%s] Already exists, skipping", subset)
        return

    logger.info("[%s] Streaming from HuggingFace...", subset)
    ds = load_dataset(
        "Brianferrell787/financial-news-multisource",
        data_files=f"data/{subset}/*.parquet",
        split="train",
        streaming=True,
    )

    rows = []
    skipped = 0
    for item in ds:
        text = item.get("text", "")
        if not text or len(text) < MIN_TEXT_LEN:
            skipped += 1
            continue
        rows.append({
            "date": item.get("date", ""),
            "text": text[:10000],  # cap length for storage
            "extra_fields": item.get("extra_fields", ""),
            "source_subset": subset,
        })
        if len(rows) >= SAMPLE_SIZE:
            break

    if not rows:
        logger.warning("[%s] No valid rows found", subset)
        return

    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    logger.info(
        "[%s] Saved %d rows to %s (skipped %d short/empty)",
        subset, len(df), path, skipped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream multisource subsets",
    )
    parser.add_argument(
        "--subset", choices=SUBSETS,
        help="Download specific subset (default: all)",
    )
    args = parser.parse_args()

    targets = [args.subset] if args.subset else SUBSETS
    for subset in targets:
        stream_subset(subset)


if __name__ == "__main__":
    main()
