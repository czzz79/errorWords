from __future__ import annotations

from error_words_tts.confusion.cli import GtEntry
from error_words_tts.pronunciation.english_reading import build_english_reading_variants


def _project(rows: list[dict[str, object]]) -> list[tuple[str, str, str, str]]:
    return sorted((str(row["canonical_text"]), str(row["variant_kind"]), str(row["tts_text"]), str(row["pronunciation_variant_id"])) for row in rows)


def test_english_reading_variants_are_canonical_only_and_textual() -> None:
    first = [GtEntry("ideahub", "IdeaHub", ["ID Hub"], [1]), GtEntry("asr", "ASR", ["AIS"], [2])]
    second = [GtEntry("ideahub", "IdeaHub", ["unrelated"], [1]), GtEntry("asr", "ASR", ["other"], [2])]
    rows, summary = build_english_reading_variants(first, settings={"max_variants_per_term": 6})
    changed, _ = build_english_reading_variants(second, settings={"max_variants_per_term": 6})
    ideahub = [row for row in rows if row["canonical_text"] == "IdeaHub"]
    assert {row["tts_text"] for row in ideahub} >= {"Idea Hub", "I D E A H U B", "I D E A Hub", "Idea H U B"}
    assert any(row["tts_text"] == "A S R" for row in rows)
    assert all(row["pronunciation_instruction"] is None for row in rows)
    assert _project(rows) == _project(changed)
    assert summary["terms_with_variants"] == 2
