"""Step 0: Fetch full article text for all unique URLs in the Kaggle dataset.

Reads raw_analyst_ratings.csv and raw_partner_headlines.csv, deduplicates URLs,
and fetches article text using trafilatura. Results are saved per-domain as
parquet files for crash recovery — rerunning skips already-fetched URLs.

Usage:
    uv run python scripts/pipeline/00_fetch_articles.py
    uv run python scripts/pipeline/00_fetch_articles.py --delay 2.0
    uv run python scripts/pipeline/00_fetch_articles.py --limit 100  # test run
"""

import argparse
import csv
import dataclasses
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from urllib.parse import urlparse

import pandas as pd

from market_metaphors.ingest.articles import (
    FetchResult,
    fetch_domain_batch,
    normalize_url,
)
from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import RAW_DIR

logger = get_logger(__name__)

ARTICLES_DIR = RAW_DIR / "articles"
RAW_NEWS_DIR = RAW_DIR / "news"
SOURCE_FILES = ["raw_analyst_ratings.csv", "raw_partner_headlines.csv"]


def extract_domain(url: str) -> str:
    """Extract and normalize the domain from a URL."""
    return urlparse(url).netloc.replace("www.", "")


def load_all_urls() -> dict[str, list[str]]:
    """Load and deduplicate URLs from all source CSVs, grouped by domain."""
    seen: set[str] = set()
    by_domain: dict[str, list[str]] = defaultdict(list)

    for filename in SOURCE_FILES:
        path = RAW_NEWS_DIR / filename
        logger.info("Reading URLs from %s", path)
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 3 and row[2].startswith("http"):
                    normalized = normalize_url(row[2])
                    if normalized not in seen:
                        seen.add(normalized)
                        domain = extract_domain(normalized)
                        by_domain[domain].append(normalized)

    total = sum(len(v) for v in by_domain.values())
    logger.info(
        "Loaded %d unique URLs across %d domains", total, len(by_domain)
    )
    for domain, urls in sorted(by_domain.items(), key=lambda x: -len(x[1])):
        logger.info("  %s: %d URLs", domain, len(urls))

    return dict(by_domain)


def load_already_fetched(domain: str) -> set[str]:
    """Load URLs already fetched for a domain from its checkpoint parquet."""
    path = ARTICLES_DIR / f"{domain}.parquet"
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["url"])
    already = set(df["url"].tolist())
    logger.info("[%s] %d URLs already fetched, skipping", domain, len(already))
    return already


def save_checkpoint(domain: str, results: list[FetchResult]) -> None:
    """Append results to the domain's parquet file."""
    if not results:
        return

    new_df = pd.DataFrame([dataclasses.asdict(r) for r in results])
    new_df["fetch_date"] = datetime.now(UTC).isoformat()

    path = ARTICLES_DIR / f"{domain}.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
        combined = new_df

    combined.to_parquet(path, index=False)


def process_domain(
    domain: str,
    urls: list[str],
    delay: float,
    seen_hashes: set[str],
    hash_lock: threading.Lock,
    checkpoint_every: int = 100,
) -> dict[str, int]:
    """Fetch all URLs for one domain with incremental checkpointing."""
    already_fetched = load_already_fetched(domain)
    remaining = [u for u in urls if u not in already_fetched]

    if not remaining:
        logger.info("[%s] All URLs already fetched, nothing to do", domain)
        return {"total": 0, "ok": 0, "skipped": 0}

    logger.info(
        "[%s] %d remaining out of %d total",
        domain, len(remaining), len(urls),
    )

    batch_buffer: list[FetchResult] = []
    stats = {"total": 0, "ok": 0, "skipped": 0}

    def on_result(result: FetchResult) -> None:
        # Thread-safe content hash dedup
        if result.status == "ok" and result.content_hash:
            with hash_lock:
                if result.content_hash in seen_hashes:
                    # Mutate to mark as dupe — already appended to results list
                    result.status = "duplicate_content"
                    result.lead = ""
                    result.body = ""
                else:
                    seen_hashes.add(result.content_hash)

        stats["total"] += 1
        if result.status == "ok":
            stats["ok"] += 1
        else:
            stats["skipped"] += 1

        batch_buffer.append(result)
        if len(batch_buffer) >= checkpoint_every:
            save_checkpoint(domain, batch_buffer)
            batch_buffer.clear()

    fetch_domain_batch(
        remaining, domain, delay=delay,
        seen_hashes=set(),  # dedup handled in on_result via shared set
        result_callback=on_result,
    )

    # Flush remaining
    save_checkpoint(domain, batch_buffer)
    batch_buffer.clear()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch full article text")
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Seconds between requests per domain (default: 1.5)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max URLs per domain (0 = all, useful for testing)",
    )
    parser.add_argument(
        "--domains", nargs="*", default=None,
        help="Only fetch these domains (default: all)",
    )
    args = parser.parse_args()

    url_map = load_all_urls()

    if args.domains:
        url_map = {d: v for d, v in url_map.items() if d in args.domains}

    if args.limit > 0:
        url_map = {d: v[: args.limit] for d, v in url_map.items()}

    # Shared state for cross-domain content dedup
    seen_hashes: set[str] = set()
    hash_lock = threading.Lock()

    # Load hashes from existing checkpoints
    for domain in url_map:
        path = ARTICLES_DIR / f"{domain}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["content_hash", "status"])
            ok_hashes = df.loc[df["status"] == "ok", "content_hash"].tolist()
            seen_hashes.update(h for h in ok_hashes if h)

    if seen_hashes:
        logger.info(
            "Loaded %d content hashes from existing checkpoints",
            len(seen_hashes),
        )

    start = time.time()
    all_stats: dict[str, dict[str, int]] = {}

    # One thread per domain — each thread does sequential fetches with delay
    with ThreadPoolExecutor(max_workers=len(url_map)) as pool:
        futures = {
            pool.submit(
                process_domain, domain, urls, args.delay,
                seen_hashes, hash_lock,
            ): domain
            for domain, urls in url_map.items()
        }
        for future in as_completed(futures):
            domain = futures[future]
            try:
                all_stats[domain] = future.result()
            except Exception:
                logger.exception("[%s] Domain worker failed", domain)
                all_stats[domain] = {"total": 0, "ok": 0, "skipped": 0, "error": 1}

    elapsed = time.time() - start
    total_ok = sum(s.get("ok", 0) for s in all_stats.values())
    total_skip = sum(s.get("skipped", 0) for s in all_stats.values())

    logger.info("=" * 60)
    logger.info(
        "Done in %.1f hours. %d articles fetched, %d skipped.",
        elapsed / 3600, total_ok, total_skip,
    )
    for domain, s in sorted(all_stats.items()):
        logger.info(
            "  %s: ok=%d skipped=%d",
            domain, s.get("ok", 0), s.get("skipped", 0),
        )


if __name__ == "__main__":
    main()
