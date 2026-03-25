"""Metaphor detection using XLM-RoBERTa token classification.

Stage 1: Token-level metaphor detection using lwachowiak/Metaphor-Detection-XLMR
Stage 2: Domain labeling using the hand-curated lexicon.
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from market_metaphors.metaphor.domain_lexicon import classify_words, dominant_domain
from market_metaphors.utils.logging import get_logger

logger = get_logger(__name__)

BASE_MODEL = "lwachowiak/Metaphor-Detection-XLMR"
LARGE_MODEL = "HiTZ/xlm-roberta-large-metaphor-detection-en"


class MetaphorDetector:
    """Wraps an XLM-RoBERTa metaphor detection model for batch inference."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or BASE_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading model %s on %s", self.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded")

    def detect_batch(self, headlines: list[str], batch_size: int = 32) -> list[dict]:
        """Run metaphor detection on a batch of headlines.

        Returns a list of dicts, one per headline:
          - metaphor_words: list[str] — flagged metaphoric words
          - metaphor_token_count: int
          - metaphor_ratio: float
          - domains: list[str] — domain labels for each metaphor word
          - dominant_domain: str
        """
        results = []
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i : i + batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        return results

    def _process_batch(self, headlines: list[str]) -> list[dict]:
        """Process a single batch of headlines."""
        encodings = self.tokenizer(
            headlines,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = encodings.pop("offset_mapping")
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            logits = self.model(**encodings).logits
        predictions = torch.argmax(logits, dim=-1).cpu()

        results = []
        for idx, headline in enumerate(headlines):
            metaphor_words = self._extract_metaphor_words(
                headline,
                predictions[idx],
                offset_mapping[idx],
                encodings["attention_mask"][idx].cpu(),
            )
            total_words = len(headline.split())
            domains = classify_words(metaphor_words)
            results.append(
                {
                    "metaphor_words": metaphor_words,
                    "metaphor_token_count": len(metaphor_words),
                    "metaphor_ratio": len(metaphor_words) / max(total_words, 1),
                    "domains": domains,
                    "dominant_domain": dominant_domain(domains),
                }
            )
        return results

    @staticmethod
    def _extract_metaphor_words(
        text: str,
        predictions: torch.Tensor,
        offsets: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[str]:
        """Extract full metaphor words by merging subword tokens.

        XLM-RoBERTa tokenizes words into subword pieces (e.g.
        "raising" → ["▁rais", "ing"]). When any subword in a word
        is labeled metaphoric, we reconstruct the full word from
        the whitespace-delimited original text.
        """
        # Collect character-level spans labeled as metaphoric
        metaphoric_chars: set[int] = set()
        for token_idx in range(len(predictions)):
            if attention_mask[token_idx] == 0:
                continue
            start, end = offsets[token_idx].tolist()
            if start == 0 and end == 0:
                # Special token ([CLS], [SEP], [PAD])
                continue
            if predictions[token_idx] == 1:
                for c in range(start, end):
                    metaphoric_chars.add(c)

        if not metaphoric_chars:
            return []

        # Walk through whitespace-delimited words in the original
        # text; if any character in a word was flagged, include it
        metaphor_words = []
        pos = 0
        for word in text.split():
            # Find where this word sits in the original text
            word_start = text.find(word, pos)
            if word_start == -1:
                continue
            word_end = word_start + len(word)
            pos = word_end

            word_chars = set(range(word_start, word_end))
            if word_chars & metaphoric_chars:
                cleaned = word.strip(".,;:!?\"'()[]{}").lower()
                if cleaned:
                    metaphor_words.append(cleaned)

        return metaphor_words
