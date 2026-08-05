"""Target-blind English and acronym text variants for TTS.

Every variant is derived solely from ``canonical_text``.  GT confusions remain
metadata for later evaluation, but never influence candidate selection, text,
or the stable variant identifier.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable


_CAMEL_BOUNDARY_1 = re.compile(r"(?<=[a-z])(?=[A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_ALNUM_BOUNDARY = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
_LATIN_TOKEN = re.compile(r"[A-Za-z]+")
_UPPER_RUN = re.compile(r"[A-Z]{2,}")
_SEPARATORS = re.compile(r"[\s_-]+")


def build_english_reading_variants(
    entries: Iterable[Any], *, settings: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build bounded English reading texts without consulting GT targets."""
    entries = list(entries)
    limit = int(settings.get("max_variants_per_term", 6))
    if limit < 1:
        raise ValueError("pronunciation.max_variants_per_term must be positive")
    rows: list[dict[str, Any]] = []
    terms_with_variants = 0
    skipped = 0
    for entry in entries:
        canonical = str(entry.canonical_text).strip()
        if not _LATIN_TOKEN.search(canonical):
            skipped += 1
            continue
        emitted: set[str] = set()
        count = 0
        for kind, tts_text in _candidates(canonical):
            normalized = _normalize_text(tts_text)
            if not normalized or normalized == _normalize_text(canonical) or normalized in emitted:
                continue
            emitted.add(normalized)
            rows.append(_row(entry, kind=kind, tts_text=tts_text))
            count += 1
            if count >= limit:
                break
        terms_with_variants += bool(count)
    return rows, {
        "term_count": len(entries),
        "variant_count": len(rows),
        "terms_with_variants": terms_with_variants,
        "skipped_without_latin": skipped,
        "max_variants_per_term": limit,
        "generator": "canonical_only_english_reading",
    }


def _candidates(value: str) -> list[tuple[str, str]]:
    split = _split_camel_and_alnum(value)
    candidates = [
        ("camel_alnum_split", split),
        ("separator_collapsed", _SEPARATORS.sub("", value)),
        ("all_letters_spelled", _spell_all_latin(value)),
        ("uppercase_runs_spelled", _spell_upper_runs(value)),
    ]
    tokens = [token for token in split.split() if _LATIN_TOKEN.fullmatch(token)]
    if len(tokens) >= 2:
        candidates.extend([
            ("camel_prefix_spelled", _spell_token_at(split, 0)),
            ("camel_suffix_spelled", _spell_token_at(split, len(tokens) - 1)),
        ])
    return candidates


def _split_camel_and_alnum(value: str) -> str:
    value = _CAMEL_BOUNDARY_2.sub(" ", value)
    value = _CAMEL_BOUNDARY_1.sub(" ", value)
    return " ".join(_ALNUM_BOUNDARY.sub(" ", value).split())


def _spell_all_latin(value: str) -> str:
    return _LATIN_TOKEN.sub(lambda match: " ".join(match.group(0).upper()), value)


def _spell_upper_runs(value: str) -> str:
    return _UPPER_RUN.sub(lambda match: " ".join(match.group(0)), _split_camel_and_alnum(value))


def _spell_token_at(value: str, index: int) -> str:
    tokens = list(_LATIN_TOKEN.finditer(value))
    if not tokens or index < 0 or index >= len(tokens):
        return value
    token = tokens[index]
    return value[: token.start()] + " ".join(token.group(0).upper()) + value[token.end() :]


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _row(entry: Any, *, kind: str, tts_text: str) -> dict[str, Any]:
    canonical = str(entry.canonical_text)
    variant_id = hashlib.sha256(
        json.dumps([canonical, "english_word_acronym", kind, tts_text], ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "id": entry.sample_id, "text": canonical, "canonical_text": canonical,
        "tts_text": tts_text, "source_text": canonical,
        "text_source": "english_reading_variant", "pronunciation_processed": True,
        "pronunciation_rule": f"en.{kind}", "pronunciation_variant_id": variant_id,
        "pronunciation_instruction": None, "target_confusions": list(entry.known_confusions),
        "confusion_category": "english_word_acronym",
        "pronunciation_delta": {"type": kind, "origin": "canonical_only", "derived_from": "canonical_text"},
        "variant_kind": kind, "language": "English",
        "tags": ["gt", f"source-lines:{','.join(map(str, entry.source_lines))}",
                 "pronunciation:english-reading", "generator:canonical-only", f"variant-kind:{kind}"],
    }
