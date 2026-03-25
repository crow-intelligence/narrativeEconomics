"""Tests for the metaphor detector module."""

import pytest
from market_metaphors.metaphor.domain_lexicon import (
    DOMAIN_LEXICON,
    WORD_TO_DOMAIN,
    classify_word,
    classify_words,
    dominant_domain,
)


class TestDomainLexicon:
    def test_all_domains_have_words(self):
        for domain, words in DOMAIN_LEXICON.items():
            assert len(words) > 0, f"Domain {domain} has no words"

    def test_word_to_domain_consistency(self):
        for domain, words in DOMAIN_LEXICON.items():
            for word in words:
                assert WORD_TO_DOMAIN[word] == domain

    def test_classify_known_word(self):
        assert classify_word("rocket") == "ASCENT"
        assert classify_word("flood") == "WATER_FLOOD"
        assert classify_word("crash") == "DARKNESS_DEATH"

    def test_classify_unknown_word(self):
        assert classify_word("xyzzy") == "UNCLASSIFIED"

    def test_classify_case_insensitive(self):
        assert classify_word("ROCKET") == "ASCENT"
        assert classify_word("Flood") == "WATER_FLOOD"

    def test_classify_words_batch(self):
        result = classify_words(["rocket", "flood", "unknown"])
        assert result == ["ASCENT", "WATER_FLOOD", "UNCLASSIFIED"]

    def test_dominant_domain_single(self):
        assert dominant_domain(["ASCENT"]) == "ASCENT"

    def test_dominant_domain_multiple(self):
        assert dominant_domain(["ASCENT", "ASCENT", "HEAT"]) == "ASCENT"

    def test_dominant_domain_all_unclassified(self):
        assert dominant_domain(["UNCLASSIFIED", "UNCLASSIFIED"]) == "UNCLASSIFIED"

    def test_dominant_domain_empty(self):
        assert dominant_domain([]) == "UNCLASSIFIED"


class TestDetector:
    """Integration tests that require the model to be downloaded.

    These are marked as slow and skipped by default.
    Run with: pytest -m slow
    """

    @pytest.fixture
    def detector(self):
        try:
            from market_metaphors.metaphor.detector import MetaphorDetector

            return MetaphorDetector()
        except Exception:
            pytest.skip("Model not available")

    @pytest.mark.slow
    def test_detect_single_headline(self, detector):
        results = detector.detect_batch(["Stocks surge as market rockets higher"])
        assert len(results) == 1
        result = results[0]
        assert "metaphor_words" in result
        assert "metaphor_ratio" in result
        assert "domains" in result
        assert isinstance(result["metaphor_words"], list)
        assert result["metaphor_ratio"] >= 0

    @pytest.mark.slow
    def test_detect_batch(self, detector):
        headlines = [
            "Market crashes amid global panic",
            "Tech stocks climb to new heights",
            "Company reports quarterly earnings",
        ]
        results = detector.detect_batch(headlines)
        assert len(results) == 3
        for r in results:
            assert "metaphor_words" in r
            assert "metaphor_token_count" in r
