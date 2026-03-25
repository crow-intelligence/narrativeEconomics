"""Step 2: Clean and deduplicate the Kaggle news dataset."""

from market_metaphors.ingest.news import run
from market_metaphors.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Starting news cleaning pipeline")
    df = run()
    logger.info("Done — %d clean headlines", len(df))


if __name__ == "__main__":
    main()
