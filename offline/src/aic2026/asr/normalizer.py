"""Vietnamese text normalisation and loanword alias expansion.

Produces ``transcript_normalized`` from ``transcript_raw`` for BM25/lexical
search indexing so that both Vietnamese phonetic forms and English
orthographic forms can be matched.
"""

from __future__ import annotations

import re
import unicodedata

# ──────────────────────────────────────────────────────────────────────
# Loanword alias table
#
# Each entry maps one or more PhoWhisper phonetic outputs (Vietnamese
# spellings of English loan words) to the canonical English spelling.
# The alias is *appended* to the text (not replaced) so both forms
# are searchable.
# ──────────────────────────────────────────────────────────────────────

_LOANWORD_ALIASES: list[tuple[list[str], str]] = [
    # Social media & tech brands
    (["phây búc", "phây buc", "phaybook", "phay búc"], "facebook"),
    (["ai phôn", "ai phon", "iphone"], "iphone"),
    (["iu tuýp", "iu tuyp", "youtup", "du túp", "du tup"], "youtube"),
    (["gúc gồ", "guc go", "gút gồ", "gut go"], "google"),
    (["in xta gram", "in-xta-gram", "instagam"], "instagram"),
    (["ti vi", "tivi"], "tv"),
    (["ti-vi"], "tv"),
    (["ô-en-lai", "on-lai", "online"], "online"),
    (["oep xai", "oep-xai", "uep xai", "web-sai"], "website"),
    # Tech terms
    (["ây ai", "ay ai", "ây-ai"], "ai"),
    (["xì-mát phôn", "xì mát phôn", "sì-mát-phôn"], "smartphone"),
    (["com-piu-tơ", "com piu tơ"], "computer"),
    (["lép tốp", "lép top", "lap top"], "laptop"),
    (["ép pi ai", "ép-pi-ai"], "api"),
    # Common English words in Vietnamese context
    (["ô kê", "o kê", "ô-kê"], "ok"),
    (["xô-sơ", "xô sơ", "sốt xo"], "social"),
    (["nho", "niu"], "new"),
    (["sốp", "sop"], "shop"),
]


def normalize_transcript(raw: str) -> str:
    """Normalise a raw PhoWhisper transcript for search indexing.

    Processing steps:

    1. Unicode NFC normalisation (canonical composition).
    2. Lowercase the entire string.
    3. Strip punctuation (keep Vietnamese diacritics and digits intact).
    4. Collapse whitespace.
    5. Append loanword aliases for any matched phonetic forms.

    Parameters
    ----------
    raw:
        The ``transcript_raw`` string from PhoWhisper.

    Returns
    -------
    str
        The normalised transcript with appended aliases.
    """
    if not raw or not raw.strip():
        return ""

    # 1. NFC normalisation
    text = unicodedata.normalize("NFC", raw)

    # 2. Lowercase
    text = text.lower()

    # 3. Strip punctuation but keep Vietnamese diacritics.
    #    We keep: letters (including Vietnamese combining marks), digits,
    #    spaces, hyphens.
    text = re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)

    # 4. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Append loanword aliases
    aliases_found: list[str] = []
    for phonetic_variants, english_alias in _LOANWORD_ALIASES:
        for variant in phonetic_variants:
            if variant in text:
                if english_alias not in aliases_found:
                    aliases_found.append(english_alias)
                break  # one match per alias group is sufficient

    if aliases_found:
        text = text + " " + " ".join(aliases_found)

    return text
