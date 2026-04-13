"""Download opinion/editorial datasets from HuggingFace and Kaggle.

Tier A: Full download of small datasets.
Tier B: Sampled download of larger datasets.

Usage:
    uv run python scripts/pipeline/00a_download_opinion_datasets.py
    uv run python scripts/pipeline/00a_download_opinion_datasets.py --dataset phrasebank
"""

import argparse
import zipfile

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from market_metaphors.utils.logging import get_logger
from market_metaphors.utils.storage import RAW_DIR

logger = get_logger(__name__)

SAMPLE_SIZE = 20_000  # max rows for large datasets


def download_phrasebank() -> None:
    """Financial PhraseBank — 4,840 annotated financial sentences."""
    out = RAW_DIR / "phrasebank"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phrasebank.parquet"
    if path.exists():
        logger.info("PhraseBank already exists, skipping")
        return

    logger.info("Downloading Financial PhraseBank...")
    zip_path = hf_hub_download(
        "takala/financial_phrasebank",
        "data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            # Filename encodes the agreement level
            agreement = name.split("Sentences_")[-1].replace(".txt", "")
            data = zf.read(name).decode("latin-1")
            for line in data.strip().split("\n"):
                if "@" in line:
                    text, label = line.rsplit("@", 1)
                    rows.append({
                        "text": text.strip(),
                        "label": label.strip(),
                        "agreement": agreement,
                    })

    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)


def download_fiqa() -> None:
    """FiQA — financial opinion mining dataset."""
    out = RAW_DIR / "fiqa"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fiqa.parquet"
    if path.exists():
        logger.info("FiQA already exists, skipping")
        return

    logger.info("Downloading FiQA...")
    dfs = []
    for config in ["main", "corpus"]:
        ds = load_dataset("vibrantlabsai/fiqa", config)
        for split_name in ds:
            df = ds[split_name].to_pandas()
            df["split"] = split_name
            df["config"] = config
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(path, index=False)
    logger.info(
        "Saved %d rows to %s (columns: %s)",
        len(combined), path, list(combined.columns),
    )


def download_twitter_finance() -> None:
    """Twitter/X Financial News Sentiment."""
    out = RAW_DIR / "twitter_finance"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "twitter_finance.parquet"
    if path.exists():
        logger.info("Twitter Finance already exists, skipping")
        return

    logger.info("Downloading Twitter Financial Sentiment...")
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    dfs = []
    for split_name in ds:
        df = ds[split_name].to_pandas()
        df["split"] = split_name
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(path, index=False)
    logger.info(
        "Saved %d rows to %s (columns: %s)",
        len(combined), path, list(combined.columns),
    )


def download_reddit_finance() -> None:
    """Reddit Finance — sampled posts from r/wallstreetbets etc."""
    import shutil
    import subprocess

    out = RAW_DIR / "reddit_finance"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "reddit_finance_sample.parquet"
    if path.exists() and path.stat().st_size > 1000:
        logger.info("Reddit Finance already exists, skipping")
        return

    logger.info("Downloading Reddit Finance from Kaggle...")
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "leukipp/reddit-finance-data",
            "--unzip", "-p", str(out),
        ],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        logger.error("Kaggle download failed: %s", result.stderr)
        return

    # Find CSVs recursively (data is in subreddit subdirectories)
    csv_files = list(out.rglob("*.csv"))
    if not csv_files:
        logger.error("No CSV files found after download")
        return

    dfs = []
    for csv_file in csv_files:
        subreddit = csv_file.parent.name
        logger.info("Reading %s/%s...", subreddit, csv_file.name)
        try:
            df = pd.read_csv(
                csv_file, nrows=5000,
                on_bad_lines="skip",
            )
        except Exception:
            logger.warning("Failed to read %s, skipping", csv_file)
            continue

        df["subreddit"] = subreddit

        # Use selftext if available and non-empty, otherwise title
        if "selftext" in df.columns:
            df["selftext"] = df["selftext"].astype(str)
            has_text = (
                (df["selftext"].str.len() >= 100)
                & (df["selftext"] != "nan")
                & (df["selftext"] != "[removed]")
                & (df["selftext"] != "[deleted]")
            )
            df_text = df[has_text].copy()
            df_text["text"] = df_text["selftext"]
        else:
            df_text = pd.DataFrame()

        # Also grab title-only posts (always have opinion content)
        if "title" in df.columns:
            df_titles = df[~df.index.isin(df_text.index)].copy()
            df_titles = df_titles[
                df_titles["title"].astype(str).str.len() >= 20
            ]
            df_titles["text"] = df_titles["title"].astype(str)
            df_text = pd.concat([df_text, df_titles])

        if not df_text.empty:
            keep_cols = [
                "subreddit", "title", "text", "score",
                "num_comments", "created",
            ]
            keep_cols = [c for c in keep_cols if c in df_text.columns]
            dfs.append(df_text[keep_cols])

    if not dfs:
        logger.error("No text data found in Reddit files")
        return

    combined = pd.concat(dfs, ignore_index=True)
    if len(combined) > SAMPLE_SIZE:
        combined = combined.sample(SAMPLE_SIZE, random_state=42)
    combined.to_parquet(path, index=False)
    logger.info("Saved %d rows to %s", len(combined), path)

    # Clean up raw files to save space
    for subdir in out.iterdir():
        if subdir.is_dir():
            shutil.rmtree(subdir)
            logger.info("Removed raw %s/", subdir.name)


DATASETS = {
    "phrasebank": download_phrasebank,
    "fiqa": download_fiqa,
    "twitter": download_twitter_finance,
    "reddit": download_reddit_finance,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download opinion datasets",
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS.keys()),
        help="Download a specific dataset (default: all)",
    )
    args = parser.parse_args()

    if args.dataset:
        DATASETS[args.dataset]()
    else:
        for name, func in DATASETS.items():
            logger.info("=== %s ===", name)
            func()


if __name__ == "__main__":
    main()
