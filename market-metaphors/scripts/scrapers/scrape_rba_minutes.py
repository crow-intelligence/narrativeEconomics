"""Scrape RBA Board Minutes (2006--2025) from rba.gov.au.

Discovers meeting URLs from yearly index pages, then fetches each minutes page.
Saves incrementally to parquet for crash recovery.

Usage:
    uv run python scripts/scrapers/scrape_rba_minutes.py
    uv run python scripts/scrapers/scrape_rba_minutes.py --delay 3.0
"""

import argparse
import re
import time

import pandas as pd
import requests
import trafilatura
from tqdm import tqdm

from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import RAW_DIR

logger = get_logger(__name__)

BASE_URL = "https://www.rba.gov.au"
INDEX_URL = BASE_URL + "/monetary-policy/rba-board-minutes/{year}/"
OUTPUT_PATH = RAW_DIR / "rba" / "minutes.parquet"

# RBA Board ran from Oct 2006 to Feb 2025 (replaced by Monetary Policy Board)
YEARS = list(range(2006, 2027))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (research project; narrative economics)"
        " academic-scraper/1.0"
    ),
}


def discover_minutes_urls() -> list[dict]:
    """Scrape yearly index pages to find all individual minutes URLs."""
    entries = []
    for year in YEARS:
        url = INDEX_URL.format(year=year)
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            logger.debug("No index page for %d (status %d)", year, resp.status_code)
            continue

        # Find links matching the minutes URL pattern
        pattern = (
            rf"/monetary-policy/rba-board-minutes/"
            rf"{year}/\d{{4}}-\d{{2}}-\d{{2}}\.html"
        )
        found = re.findall(pattern, resp.text)
        for path in sorted(set(found)):
            full_url = BASE_URL + path
            # Extract date from URL
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path)
            date_str = date_match.group(1) if date_match else ""
            entries.append({"url": full_url, "date": date_str})

        logger.info("[RBA] %d: found %d minutes", year, len(found))
        time.sleep(1)

    logger.info("[RBA] Total: %d minutes URLs discovered", len(entries))
    return entries


def load_already_fetched() -> set[str]:
    """Load URLs already fetched from checkpoint."""
    if not OUTPUT_PATH.exists():
        return set()
    df = pd.read_parquet(OUTPUT_PATH, columns=["url"])
    return set(df["url"].tolist())


def save_checkpoint(results: list[dict]) -> None:
    """Append results to the parquet checkpoint."""
    if not results:
        return
    new_df = pd.DataFrame(results)
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined = new_df
    combined.to_parquet(OUTPUT_PATH, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape RBA Board Minutes")
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds between requests (default: 2.0)",
    )
    args = parser.parse_args()

    entries = discover_minutes_urls()
    already = load_already_fetched()
    remaining = [e for e in entries if e["url"] not in already]

    if not remaining:
        logger.info("All %d minutes already fetched", len(entries))
        return

    logger.info(
        "%d remaining out of %d total (%d already fetched)",
        len(remaining), len(entries), len(already),
    )

    batch: list[dict] = []
    ok_count = 0
    skip_count = 0

    pbar = tqdm(remaining, desc="RBA minutes", unit="doc", postfix={"ok": 0, "skip": 0})
    for i, entry in enumerate(pbar):
        try:
            downloaded = trafilatura.fetch_url(entry["url"])
        except Exception:
            logger.debug("Failed to fetch %s", entry["url"])
            batch.append({
                "url": entry["url"], "date": entry["date"],
                "text": "", "status": "http_error",
            })
            skip_count += 1
            pbar.set_postfix(ok=ok_count, skip=skip_count)
            continue

        text = trafilatura.extract(downloaded) if downloaded else None

        if text and len(text) >= 300:
            batch.append({
                "url": entry["url"], "date": entry["date"],
                "text": text, "status": "ok",
            })
            ok_count += 1
        else:
            batch.append({
                "url": entry["url"], "date": entry["date"],
                "text": "", "status": "too_short" if text else "extraction_failed",
            })
            skip_count += 1

        pbar.set_postfix(ok=ok_count, skip=skip_count)

        # Checkpoint every 10 docs
        if len(batch) >= 10:
            save_checkpoint(batch)
            batch.clear()

        if i < len(remaining) - 1:
            time.sleep(args.delay)

    save_checkpoint(batch)
    batch.clear()

    logger.info("Done: %d ok, %d skipped", ok_count, skip_count)


if __name__ == "__main__":
    main()
