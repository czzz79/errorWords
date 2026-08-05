"""English token-to-ARPAbet resolution for the CMU pronunciation pipeline.

The resolver is deliberately target-blind: it receives only one parsed token
and never accesses a GT confusion string.  It returns an auditable source for
every successful pronunciation or a structured unresolved reason.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


ARPABET_PHONE = re.compile(r"^[A-Z]{1,3}[012]?$")

LETTER_PHONEMES: dict[str, list[str]] = {
    "A": ["EY1"], "B": ["B", "IY1"], "C": ["S", "IY1"],
    "D": ["D", "IY1"], "E": ["IY1"], "F": ["EH1", "F"],
    "G": ["JH", "IY1"], "H": ["EY1", "CH"], "I": ["AY1"],
    "J": ["JH", "EY1"], "K": ["K", "EY1"], "L": ["EH1", "L"],
    "M": ["EH1", "M"], "N": ["EH1", "N"], "O": ["OW1"],
    "P": ["P", "IY1"], "Q": ["K", "Y", "UW1"], "R": ["AA1", "R"],
    "S": ["EH1", "S"], "T": ["T", "IY1"], "U": ["Y", "UW1"],
    "V": ["V", "IY1"], "W": ["D", "AH1", "B", "AH0", "L", "Y", "UW0"],
    "X": ["EH1", "K", "S"], "Y": ["W", "AY1"], "Z": ["Z", "IY1"],
}

DIGIT_PHONEMES: dict[str, list[str]] = {
    "0": ["Z", "IH1", "R", "OW0"],
    "1": ["W", "AH1", "N"],
    "2": ["T", "UW1"],
    "3": ["TH", "R", "IY1"],
    "4": ["F", "AO1", "R"],
    "5": ["F", "AY1", "V"],
    "6": ["S", "IH1", "K", "S"],
    "7": ["S", "EH1", "V", "AH0", "N"],
    "8": ["EY1", "T"],
    "9": ["N", "AY1", "N"],
}


def load_overrides(path: Path | None) -> dict[str, list[str]]:
    """Load a case-insensitive token override table.

    The JSON format is ``{"Nexent": ["N", "EH1", ...]}``.  Empty files
    and a missing optional file both mean that no manual overrides are active.
    """
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"English pronunciation overrides must be an object: {path}")
    result: dict[str, list[str]] = {}
    for raw_token, raw_phones in payload.items():
        token = str(raw_token).strip()
        if not token or not isinstance(raw_phones, list):
            raise ValueError(f"invalid pronunciation override in {path}: {raw_token!r}")
        phones = _validate_phones(raw_phones, owner=f"override {token!r}")
        result[token.casefold()] = phones
    return result


def resolve_token(
    text: str,
    token_type: str,
    *,
    overrides: dict[str, list[str]],
) -> dict[str, Any]:
    """Resolve one parsed token without consulting any expected confusion."""
    normalized = text.strip()
    if not normalized:
        return _unresolved(text, "empty_token")
    override = overrides.get(normalized.casefold())
    if override is not None:
        return {"status": "ready", "phones": list(override), "source": "manual_override"}
    if token_type == "acronym":
        return _letters(normalized)
    if token_type == "number":
        phones = DIGIT_PHONEMES.get(normalized)
        if phones is None:
            return _unresolved(normalized, "unsupported_number")
        return {"status": "ready", "phones": list(phones), "source": "number_normalizer"}

    cmu = _cmudict(normalized)
    if cmu is not None:
        return {"status": "ready", "phones": cmu, "source": "cmudict"}
    g2p = _g2p_en(normalized)
    if g2p is not None:
        return {"status": "ready", "phones": g2p, "source": "g2p_en"}
    return _unresolved(normalized, "g2p_failed")


def letter_subunits(text: str) -> list[dict[str, Any]]:
    result = []
    for letter in text.upper():
        phones = LETTER_PHONEMES.get(letter)
        if phones is None:
            raise ValueError(f"unsupported acronym letter: {letter!r}")
        result.append({"text": letter, "phones": list(phones)})
    return result


def _letters(text: str) -> dict[str, Any]:
    if not text.isalpha() or not text.isupper():
        return _unresolved(text, "invalid_acronym")
    try:
        units = letter_subunits(text)
    except ValueError:
        return _unresolved(text, "unsupported_acronym_letter")
    phones = [phone for unit in units for phone in unit["phones"]]
    return {"status": "ready", "phones": phones, "source": "letter_names", "subunits": units}


@lru_cache(maxsize=4096)
def _cmudict(text: str) -> list[str] | None:
    try:
        import cmudict
    except ImportError:
        return None
    entries = cmudict.dict().get(text.casefold())
    return _validate_phones(entries[0], owner=f"CMUdict {text!r}") if entries else None


@lru_cache(maxsize=1)
def _g2p_model() -> Any | None:
    try:
        from g2p_en import G2p
    except ImportError:
        return None
    try:
        return G2p()
    except (LookupError, OSError):
        return None


def _g2p_en(text: str) -> list[str] | None:
    model = _g2p_model()
    if model is None:
        return None
    try:
        raw = model(text)
    except (LookupError, OSError, ValueError):
        return None
    phones = [str(value).upper() for value in raw if ARPABET_PHONE.fullmatch(str(value).upper())]
    if not phones:
        return None
    return _validate_phones(phones, owner=f"g2p_en {text!r}")


def _validate_phones(values: Any, *, owner: str) -> list[str]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{owner} has no ARPAbet phones")
    phones = [str(value).upper().strip() for value in values]
    if any(not ARPABET_PHONE.fullmatch(phone) for phone in phones):
        raise ValueError(f"{owner} contains unsupported ARPAbet phones: {values!r}")
    return phones


def _unresolved(text: str, reason: str) -> dict[str, Any]:
    return {"status": "unresolved", "text": text, "reason": reason}
