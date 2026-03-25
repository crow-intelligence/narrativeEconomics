from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
