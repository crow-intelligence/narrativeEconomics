"""Scrape Bank of Canada Governing Council Deliberation summaries.

Only ~10 documents exist (program started Jan 2025). Fetches from the
BoC publications listing page, extracts text with trafilatura.

Usage:
    uv run python scripts/scrapers/scrape_boc_deliberations.py
"""

import re
import time

import pandas as pd
import requests
import trafilatura
from tqdm import tqdm

from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import RAW_DIR

logger = get_logger(__name__)

LISTING_URL = (
    "https://www.bankofcanada.ca"
    "/publications/summary-governing-council-deliberations/"
)
OUTPUT_PATH = RAW_DIR / "boc" / "deliberations.parquet"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (research project; narrative economics)"
        " academic-scraper/1.0"
    ),
}


def discover_urls() -> list[dict]:
    """Scrape the listing page for individual deliberation URLs."""
    resp = requests.get(LISTING_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    # Match deliberation URLs
    pattern = (
        r"https://www\.bankofcanada\.ca/\d{4}/\d{2}/"
        r"summary-[\w-]*deliberations[\w-]*/"
    )
    urls = sorted(set(re.findall(pattern, resp.text)))

    entries = []
    for url in urls:
        # Extract the meeting date from the URL slug
        date_match = re.search(
            r"(?:of-|date-)(\w+-\d{1,2}-\d{4})", url
        )
        date_str = date_match.group(1) if date_match else ""
        entries.append({"url": url, "date": date_str})

    logger.info("[BoC] Found %d deliberation URLs", len(entries))
    return entries


def main() -> None:
    entries = discover_urls()
    if not entries:
        logger.error("No deliberation URLs found")
        return

    results: list[dict] = []
    ok_count = 0

    pbar = tqdm(entries, desc="BoC deliberations", unit="doc")
    for i, entry in enumerate(pbar):
        downloaded = trafilatura.fetch_url(entry["url"])
        text = trafilatura.extract(downloaded) if downloaded else None

        if text and len(text) >= 300:
            results.append({
                "url": entry["url"],
                "date": entry["date"],
                "text": text,
                "status": "ok",
            })
            ok_count += 1
        else:
            results.append({
                "url": entry["url"],
                "date": entry["date"],
                "text": "",
                "status": "extraction_failed",
            })

        pbar.set_postfix(ok=ok_count)
        if i < len(entries) - 1:
            time.sleep(2)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(OUTPUT_PATH, index=False)
    logger.info("Done: %d/%d ok, saved to %s", ok_count, len(entries), OUTPUT_PATH)


if __name__ == "__main__":
    main()
