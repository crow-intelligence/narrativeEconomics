"""Generate a sample review spreadsheet from all collected datasets.

Produces CSV + Excel (one sheet per dataset) with 15 diverse paragraph-level
samples from each dataset, plus metadata for the reviewer.

Usage:
    uv run python scripts/generate_samples.py
"""

import csv
import glob
import json
import random
import sys

import pandas as pd

from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import OUTPUTS_DIR, RAW_DIR

logger = get_logger(__name__)

SAMPLE_SIZE = 100
MIN_PARA_LEN = 80  # skip very short paragraphs
MAX_PARA_LEN = 3000  # cap paragraph length for readability

random.seed(42)

# Patterns that indicate boilerplate, not economic content
BOILERPLATE_PATTERNS = [
    "prefatory note",
    "the attached document represents",
    "comprehensive digitization process",
    "optimal character recognition",
    "freedom of information act",
    "scanned images were deskewed",
    "present at this meeting",
    "voted for this action",
    "voted against this action",
    "secretary's note",
    "notation vote",
    "approved unanimously",
    "meeting adjourned",
    "attended the meeting",
    "the meeting was called to order",
    "this electronic document",
    "quality assurance process",
    "redaction process",
    "copyright",
    "all rights reserved",
    "terms and conditions",
    "disclaimer",
]


def is_boilerplate(text: str) -> bool:
    """Check if a paragraph is boilerplate/metadata rather than content."""
    lower = text.lower()
    return any(pat in lower for pat in BOILERPLATE_PATTERNS)


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, filtering out short/empty/boilerplate."""
    if not text or not isinstance(text, str):
        return []
    # Try double-newline first
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    # If that produced very few long chunks, try single newline
    if len(paras) <= 3 and any(len(p) > 5000 for p in paras):
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    return [
        p[:MAX_PARA_LEN]
        for p in paras
        if len(p) >= MIN_PARA_LEN and not is_boilerplate(p)
    ]


def sample_paragraphs_from_docs(
    docs: list[dict],
    n: int = SAMPLE_SIZE,
    skip_first_n: int = 0,
) -> list[dict]:
    """Sample n paragraphs from a list of documents, max 2 per document.

    Args:
        skip_first_n: Skip the first N paragraphs of each document
            (useful for skipping preambles/attendance lists).
    """
    all_paras = []
    for doc in docs:
        paras = split_paragraphs(doc.get("text", ""))
        paras = paras[skip_first_n:]
        for i, para in enumerate(paras):
            all_paras.append({
                "source_doc": doc.get("source_doc", ""),
                "date": doc.get("date", ""),
                "author": doc.get("author", ""),
                "text": para,
                "position": f"{i + skip_first_n + 1}/{len(paras) + skip_first_n}",
                "notes": doc.get("notes", ""),
            })

    if not all_paras:
        return []

    # Try to pick from different documents
    by_doc: dict[str, list[dict]] = {}
    for p in all_paras:
        key = p["source_doc"]
        by_doc.setdefault(key, []).append(p)

    selected = []
    doc_keys = list(by_doc.keys())
    random.shuffle(doc_keys)

    # First pass: 1 per doc
    for key in doc_keys:
        if len(selected) >= n:
            break
        selected.append(random.choice(by_doc[key]))

    # Second pass: fill up if needed
    if len(selected) < n:
        remaining = [p for p in all_paras if p not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected[:n]


# ── Dataset loaders ─────────────────────────────────────────────────


def load_fomc_minutes() -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    path = RAW_DIR / "fed" / "fomc-minutes.csv"
    docs = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            docs.append({
                "text": row.get("Text", ""),
                "source_doc": f"FOMC Minutes {row.get('Date', '')}",
                "date": row.get("Date", ""),
                "author": row.get("Chair", ""),
                "notes": f"Chair: {row.get('Chair', '')}",
            })
    return docs


def load_fomc_paragraphs() -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    path = RAW_DIR / "fed" / "fomc_documents.csv"
    # These docs have no newlines — they're single text blocks.
    # Extract a ~2000 char chunk from the middle of each document.
    docs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reservoir: list[dict] = []
        for i, row in enumerate(reader):
            text = row.get("text", "")
            if not text or len(text) < 1000:
                continue
            # Take a chunk from 30-60% into the document (past preamble)
            start = len(text) // 3
            chunk = text[start : start + 2000]
            # Try to start/end at sentence boundaries
            first_dot = chunk.find(". ")
            if first_dot > 0:
                chunk = chunk[first_dot + 2 :]
            last_dot = chunk.rfind(". ")
            if last_dot > 0:
                chunk = chunk[: last_dot + 1]
            if len(chunk) < MIN_PARA_LEN:
                continue
            entry = {
                "text": chunk,
                "source_doc": row.get("url", ""),
                "date": row.get("meeting_date", ""),
                "author": "",
                "notes": row.get("document_kind", ""),
            }
            if len(reservoir) < 200:
                reservoir.append(entry)
            else:
                j = random.randint(0, i)
                if j < 200:
                    reservoir[j] = entry
        docs = reservoir
    return docs


def load_cbs_speeches() -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    path = RAW_DIR / "speeches" / "cbs_speeches.csv"
    docs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Reservoir sample to avoid reading all 498 MB into memory
        reservoir: list[dict] = []
        for i, row in enumerate(reader):
            entry = {
                "text": row.get("text", ""),
                "source_doc": row.get("Title", ""),
                "date": row.get("Date", ""),
                "author": row.get("Authorname", ""),
                "notes": (
                    f"{row.get('CentralBank', '')}"
                    f", {row.get('Country', '')}"
                ),
            }
            if i < 200:
                reservoir.append(entry)
            else:
                j = random.randint(0, i)
                if j < 200:
                    reservoir[j] = entry
        docs = reservoir
    return docs


def load_bis_speeches() -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    path = RAW_DIR / "speeches" / "bis" / "speeches.csv"
    docs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reservoir: list[dict] = []
        for i, row in enumerate(reader):
            entry = {
                "text": row.get("text", ""),
                "source_doc": row.get("title", ""),
                "date": row.get("date", ""),
                "author": row.get("author", ""),
                "notes": "BIS archive",
            }
            if i < 200:
                reservoir.append(entry)
            else:
                j = random.randint(0, i)
                if j < 200:
                    reservoir[j] = entry
        docs = reservoir
    return docs


def load_ecb_fed_speeches() -> list[dict]:
    path = RAW_DIR / "speeches" / "ecb_fed" / "data"
    parquet = list(path.glob("*.parquet"))[0]
    df = pd.read_parquet(parquet)
    sampled = df.sample(min(200, len(df)), random_state=42)
    docs = []
    for _, row in sampled.iterrows():
        docs.append({
            "text": row.get("clean_text", row.get("text", "")),
            "source_doc": row.get("title", ""),
            "date": str(row.get("date", "")),
            "author": row.get("author", ""),
            "notes": f"Country: {row.get('country', '')}",
        })
    return docs


def load_rba_minutes() -> list[dict]:
    path = RAW_DIR / "rba" / "minutes.parquet"
    df = pd.read_parquet(path)
    df = df[df["status"] == "ok"]
    docs = []
    for _, row in df.iterrows():
        docs.append({
            "text": row.get("text", ""),
            "source_doc": row.get("url", ""),
            "date": row.get("date", ""),
            "author": "",
            "notes": "RBA Board Minutes",
        })
    return docs


def load_boc_deliberations() -> list[dict]:
    path = RAW_DIR / "boc" / "deliberations.parquet"
    df = pd.read_parquet(path)
    df = df[df["status"] == "ok"]
    docs = []
    for _, row in df.iterrows():
        docs.append({
            "text": row.get("text", ""),
            "source_doc": row.get("url", ""),
            "date": row.get("date", ""),
            "author": "",
            "notes": "BoC Governing Council Deliberations",
        })
    return docs


def load_huffpost() -> list[dict]:
    docs = []
    for path in sorted(glob.glob(str(RAW_DIR / "huffpost" / "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                lead = item.get("short_description", "")
                if lead and len(lead) >= MIN_PARA_LEN:
                    docs.append({
                        "text": lead,
                        "source_doc": item.get("headline", ""),
                        "date": item.get("date", ""),
                        "author": item.get("authors", ""),
                        "notes": (
                            f"Category: {item.get('category', '')}"
                        ),
                    })
    return docs


def load_kaggle_articles() -> list[dict]:
    docs = []
    for path in sorted(
        glob.glob(str(RAW_DIR / "articles" / "*.parquet"))
    ):
        df = pd.read_parquet(path)
        ok = df[df["status"] == "ok"]
        if ok.empty:
            continue
        sampled = ok.sample(min(30, len(ok)), random_state=42)
        for _, row in sampled.iterrows():
            lead = row.get("lead", "")
            if lead and len(lead) >= MIN_PARA_LEN:
                docs.append({
                    "text": lead,
                    "source_doc": row.get("url", ""),
                    "date": "",
                    "author": "",
                    "notes": f"Domain: {row.get('domain', '')}",
                })
    return docs


# ── Main ────────────────────────────────────────────────────────────

# (name, loader, skip_first_n) — skip_first_n drops preamble paragraphs
DATASETS = [
    ("FOMC Minutes", load_fomc_minutes, 5),
    ("FOMC Paragraph-Level", load_fomc_paragraphs, 3),
    ("CBS Speeches", load_cbs_speeches, 1),
    ("BIS Speeches", load_bis_speeches, 1),
    ("ECB-FED Speeches", load_ecb_fed_speeches, 1),
    ("RBA Board Minutes", load_rba_minutes, 3),
    ("BoC Deliberations", load_boc_deliberations, 3),
    ("HuffPost Sentiment", load_huffpost, 0),
    ("Kaggle Stock News", load_kaggle_articles, 0),
]


def main() -> None:
    all_samples: list[dict] = []
    per_dataset: dict[str, pd.DataFrame] = {}

    for name, loader, skip_n in DATASETS:
        logger.info("Loading %s...", name)
        try:
            docs = loader()
        except Exception:
            logger.exception("Failed to load %s", name)
            continue

        if not docs:
            logger.warning("No data for %s", name)
            continue

        # These datasets are already at paragraph/chunk level
        already_chunked = (
            "HuffPost Sentiment", "Kaggle Stock News",
            "FOMC Paragraph-Level",
        )
        if name in already_chunked:
            sampled = random.sample(docs, min(SAMPLE_SIZE, len(docs)))
            samples = []
            for s in sampled:
                samples.append({
                    "dataset": name,
                    "source_doc": s["source_doc"],
                    "date": s["date"],
                    "author": s["author"],
                    "text": s["text"],
                    "position": "lead",
                    "notes": s["notes"],
                })
        else:
            raw_samples = sample_paragraphs_from_docs(
                docs, SAMPLE_SIZE, skip_first_n=skip_n,
            )
            samples = [{"dataset": name, **s} for s in raw_samples]

        if not samples:
            logger.warning("  0 samples from %s, skipping", name)
            continue
        logger.info("  %d samples from %s", len(samples), name)
        all_samples.extend(samples)
        per_dataset[name] = pd.DataFrame(samples)

    # Write CSV
    out_dir = OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "sample_review.csv"
    df_all = pd.DataFrame(all_samples)
    df_all.to_csv(csv_path, index=False)
    logger.info("Wrote %s (%d rows)", csv_path, len(df_all))

    # Write Excel with one sheet per dataset
    xlsx_path = out_dir / "sample_review.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Summary sheet
        summary_rows = []
        for name, df in per_dataset.items():
            summary_rows.append({
                "Dataset": name,
                "Samples": len(df),
                "Avg text length": int(
                    df["text"].str.len().mean()
                ),
            })
        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name="Summary", index=False,
        )

        # Per-dataset sheets (truncate text for Excel 32K cell limit)
        for name, df in per_dataset.items():
            sheet = name[:31]
            df_xl = df.copy()
            df_xl["text"] = df_xl["text"].str[:5000]
            df_xl.to_excel(
                writer, sheet_name=sheet, index=False,
            )

    logger.info("Wrote %s (%d sheets)", xlsx_path, len(per_dataset) + 1)


if __name__ == "__main__":
    main()
