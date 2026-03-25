"""Hand-curated domain lexicon for metaphor classification.

Source domain = the concrete image (e.g. FLOOD).
Target domain = always the financial/economic concept being described.

Extend this dict to add new domains or words.
"""

import re
from collections import Counter

DOMAIN_LEXICON: dict[str, list[str]] = {
    "ASCENT": [
        "rocket",
        "surge",
        "climb",
        "soar",
        "lift",
        "rally",
        "jump",
        "leap",
        "rise",
        "skyrocket",
        "mount",
        "elevate",
        "boom",
    ],
    "HEAT": [
        "fuel",
        "ignite",
        "hot",
        "fire",
        "burn",
        "blaze",
        "overheat",
        "spark",
        "flame",
        "sizzle",
        "scorching",
        "heated",
        "melt",
    ],
    "LIGHT": [
        "bright",
        "clear",
        "shine",
        "dawn",
        "glow",
        "illuminate",
        "radiant",
        "sunny",
        "brilliant",
        "beacon",
        "spotlight",
    ],
    "STRUCTURAL_FAILURE": [
        "crack",
        "fracture",
        "collapse",
        "crumble",
        "break",
        "shatter",
        "rupture",
        "erode",
        "decay",
        "deteriorate",
        "crumbling",
    ],
    "WATER_FLOOD": [
        "flood",
        "wave",
        "tide",
        "drown",
        "flow",
        "deluge",
        "tsunami",
        "overflow",
        "pour",
        "stream",
        "cascade",
        "submerge",
        "ripple",
    ],
    "WEIGHT_PRESSURE": [
        "crush",
        "drag",
        "burden",
        "sink",
        "weigh",
        "press",
        "squeeze",
        "heavy",
        "anchor",
        "gravity",
        "plunge",
        "plummet",
        "tumble",
    ],
    "DARKNESS_DEATH": [
        "bloodbath",
        "freeze",
        "dead",
        "crash",
        "doom",
        "grim",
        "dark",
        "bleak",
        "death",
        "kill",
        "bury",
        "toxic",
        "contagion",
        "plague",
    ],
    "MOVEMENT_DISPLACEMENT": [
        "shift",
        "turn",
        "pivot",
        "disrupt",
        "shake",
        "transform",
        "revolution",
        "upheaval",
        "transition",
        "reset",
        "overturn",
    ],
}

# Build reverse lookup: word -> domain
WORD_TO_DOMAIN: dict[str, str] = {}
for domain, words in DOMAIN_LEXICON.items():
    for word in words:
        WORD_TO_DOMAIN[word] = domain


def classify_word(word: str) -> str:
    """Return the domain label for a word, or 'UNCLASSIFIED'."""
    return WORD_TO_DOMAIN.get(word.lower(), "UNCLASSIFIED")


def classify_words(words: list[str]) -> list[str]:
    """Return domain labels for a list of words."""
    return [classify_word(w) for w in words]


def dominant_domain(domains: list[str]) -> str:
    """Return the most frequent domain from a list, excluding UNCLASSIFIED."""
    filtered = [d for d in domains if d != "UNCLASSIFIED"]
    if not filtered:
        return "UNCLASSIFIED"
    return Counter(filtered).most_common(1)[0][0]


# Pre-compiled regex: match any lexicon word as a whole word (case-insensitive)
_ALL_WORDS = sorted(WORD_TO_DOMAIN.keys(), key=len, reverse=True)
_LEXICON_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _ALL_WORDS) + r")\b",
    re.IGNORECASE,
)


def scan_headline(headline: str) -> dict:
    """Fast keyword scan of a headline against the domain lexicon.

    Returns:
        dict with keys:
          - keyword_matches: list[str] — matched words
          - keyword_domains: list[str] — domain per match
          - keyword_domain_set: list[str] — unique domains found
          - keyword_dominant_domain: str
          - keyword_count: int
    """
    matches = _LEXICON_PATTERN.findall(headline)
    matches_lower = [m.lower() for m in matches]
    domains = [WORD_TO_DOMAIN[m] for m in matches_lower]
    unique_domains = list(dict.fromkeys(d for d in domains))

    return {
        "keyword_matches": matches_lower,
        "keyword_domains": domains,
        "keyword_domain_set": unique_domains,
        "keyword_dominant_domain": dominant_domain(domains),
        "keyword_count": len(matches),
    }
