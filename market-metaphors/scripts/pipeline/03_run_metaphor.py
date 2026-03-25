"""Step 3: Batch metaphor detection + domain labeling on cleaned headlines.

Supports checkpointing: saves progress every CHECKPOINT_INTERVAL rows.
On restart, skips already-processed rows.
"""

import ast
import time

import pandas as pd
from market_metaphors.metaphor.detector import MetaphorDetector
from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import (
    PROCESSED_DIR,
    read_parquet,
    write_parquet,
)

logger = get_logger(__name__)

CHECKPOINT_INTERVAL = 5_000
LOG_INTERVAL = 100  # Log every N batches
BATCH_SIZE_CPU = 32
BATCH_SIZE_GPU = 128


def main():
    news_path = PROCESSED_DIR / "news_clean.parquet"
    output_path = PROCESSED_DIR / "metaphors.parquet"

    if not news_path.exists():
        logger.error("news_clean.parquet not found — run 02_clean_news.py first")
        return

    news = read_parquet(news_path)
    headlines = news["headline"].tolist()
    total = len(headlines)
    logger.info("Processing %d headlines", total)

    # Check for existing checkpoint
    start_idx = 0
    existing_results = []
    if output_path.exists():
        existing = read_parquet(output_path)
        start_idx = len(existing)
        existing_results = existing.to_dict("records")
        logger.info("Resuming from row %d", start_idx)

    if start_idx >= total:
        logger.info("All rows already processed")
        return

    # Initialize detector
    import torch

    detector = MetaphorDetector()
    batch_size = BATCH_SIZE_GPU if torch.cuda.is_available() else BATCH_SIZE_CPU

    results = existing_results
    t0 = time.time()
    batch_count = 0

    for i in range(start_idx, total, batch_size):
        batch = headlines[i : i + batch_size]
        batch_results = detector.detect_batch(batch, batch_size=batch_size)

        # Convert list fields to strings for Parquet storage
        for r in batch_results:
            r["metaphor_words"] = str(r["metaphor_words"])
            r["domains"] = str(r["domains"])
        results.extend(batch_results)
        batch_count += 1

        # Progress logging
        if batch_count % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            processed = len(results) - len(existing_results)
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total - len(results)) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d / %d (%.1f%%) | " "%.0f headlines/sec | " "ETA: %.0f min",
                len(results),
                total,
                100 * len(results) / total,
                rate,
                remaining / 60,
            )

        # Checkpoint
        if len(results) % CHECKPOINT_INTERVAL < batch_size:
            logger.info("Checkpoint at row %d / %d", len(results), total)
            df = pd.DataFrame(results)
            write_parquet(df, output_path)

    # Final save
    df = pd.DataFrame(results)
    write_parquet(df, output_path)
    elapsed = time.time() - t0
    logger.info(
        "Saved %d metaphor annotations to %s (%.1f min)",
        len(df),
        output_path,
        elapsed / 60,
    )


def parse_list_column(series: pd.Series) -> pd.Series:
    """Parse string-encoded list columns back to actual lists."""
    return series.apply(ast.literal_eval)


if __name__ == "__main__":
    main()
