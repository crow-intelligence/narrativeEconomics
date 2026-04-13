"""Fetch full article text from URLs using trafilatura."""

import hashlib
import re
import time
from dataclasses import dataclass

import trafilatura
from tqdm import tqdm

from market_metaphors.utils.logging import get_logger

logger = get_logger(__name__)

MIN_TEXT_LENGTH = 300

ERROR_PAGE_PATTERNS = re.compile(
    r"(?i)"
    r"(?:page\s+not\s+found"
    r"|404\s+error"
    r"|access\s+denied"
    r"|403\s+forbidden"
    r"|sign\s+in\s+to\s+(?:read|continue|view)"
    r"|subscribe\s+to\s+(?:read|continue|view)"
    r"|log\s+in\s+to\s+(?:read|continue|view)"
    r"|this\s+(?:page|article)\s+is\s+(?:no\s+longer\s+)?available"
    r"|you\s+(?:need|must)\s+(?:a\s+)?(?:premium|pro)\s+(?:account|membership|subscription)"
    r"|content\s+is\s+(?:only\s+)?available\s+to\s+(?:premium|pro|registered)"
    r"|we\s+can(?:'t|not)\s+find\s+(?:the|this)\s+page"
    r")"
)


@dataclass
class FetchResult:
    url: str
    domain: str
    lead: str
    body: str
    status: str
    content_hash: str


def normalize_url(url: str) -> str:
    """Strip query params and trailing slashes for dedup."""
    url = url.split("?")[0].split("#")[0]
    return url.rstrip("/")


def content_hash(text: str) -> str:
    """SHA-256 hex digest of the text for duplicate detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_lead_body(text: str) -> tuple[str, str]:
    """Split article text into lead (first paragraph) and body (rest).

    Uses double-newline as paragraph separator.
    """
    parts = text.strip().split("\n\n", 1)
    lead = parts[0].strip()
    body = parts[1].strip() if len(parts) > 1 else ""
    return lead, body


def validate_text(text: str | None) -> str | None:
    """Check extracted text against sanity rules.

    Returns None if text is valid, or a reason string if it should be skipped.
    """
    if text is None:
        return "extraction_failed"
    if len(text) < MIN_TEXT_LENGTH:
        return "too_short"
    if ERROR_PAGE_PATTERNS.search(text):
        return "error_page"
    return None


def fetch_article(url: str, domain: str) -> FetchResult:
    """Download and extract article text from a single URL."""
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception:
        logger.debug("HTTP error fetching %s", url)
        return FetchResult(
            url=url, domain=domain, lead="", body="",
            status="http_error", content_hash="",
        )

    if downloaded is None:
        return FetchResult(
            url=url, domain=domain, lead="", body="",
            status="http_error", content_hash="",
        )

    text = trafilatura.extract(downloaded)
    skip_reason = validate_text(text)
    if skip_reason:
        return FetchResult(
            url=url, domain=domain, lead="", body="",
            status=skip_reason, content_hash="",
        )

    lead, body = split_lead_body(text)
    return FetchResult(
        url=url, domain=domain, lead=lead, body=body,
        status="ok", content_hash=content_hash(text),
    )


def fetch_domain_batch(
    urls: list[str],
    domain: str,
    delay: float = 1.5,
    seen_hashes: set[str] | None = None,
    result_callback=None,
) -> list[FetchResult]:
    """Fetch all URLs for a single domain sequentially with rate limiting.

    Args:
        urls: URLs to fetch.
        domain: Domain name for logging.
        delay: Seconds to wait between requests.
        seen_hashes: Set of content hashes already seen (for cross-URL dedup).
        result_callback: Called with each FetchResult for incremental saving.

    Returns:
        List of all FetchResults.
    """
    if seen_hashes is None:
        seen_hashes = set()

    results: list[FetchResult] = []
    ok_count = 0
    skip_count = 0

    pbar = tqdm(
        urls, desc=domain, unit="url",
        postfix={"ok": 0, "skip": 0},
        position=None, leave=True,
    )
    for i, url in enumerate(pbar):
        result = fetch_article(url, domain)

        # Check content-level dedup
        if result.status == "ok" and result.content_hash in seen_hashes:
            result = FetchResult(
                url=url, domain=domain, lead="", body="",
                status="duplicate_content", content_hash=result.content_hash,
            )

        if result.status == "ok":
            seen_hashes.add(result.content_hash)
            ok_count += 1
        else:
            skip_count += 1

        pbar.set_postfix(ok=ok_count, skip=skip_count)

        results.append(result)
        if result_callback is not None:
            result_callback(result)

        # Rate limit (skip delay after last URL)
        if i < len(urls) - 1:
            time.sleep(delay)

    logger.info(
        "[%s] Finished: %d total, %d ok, %d skipped",
        domain, len(urls), ok_count, skip_count,
    )
    return results
