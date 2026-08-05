from __future__ import annotations

import json
from pathlib import Path

import pytest

from error_words_tts.pronunciation.generator import (
    PronunciationError,
    PronunciationSample,
    generate_variants,
    load_rules,
)
from error_words_tts.pronunciation.cli import run_config


ROOT = Path(__file__).resolve().parents[1]
RULES = load_rules(
    [
        ROOT / "src" / "error_words_tts" / "pronunciation" / "rules" / "english.json",
        ROOT / "src" / "error_words_tts" / "pronunciation" / "rules" / "chinese.json",
    ]
)


def _sample(**overrides) -> PronunciationSample:
    payload = {
        "id": "ideahub",
        "canonical_text": "IdeaHub",
        "text": "idea hub",
        "language": "English",
        "tags": ["term"],
        "pronunciation": {"tokens": ["idea", "hub"]},
    }
    payload.update(overrides)
    return PronunciationSample.from_mapping(payload, 0)


def _rows_for(rule_id: str, rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["rule"]["rule_id"] == rule_id]


def test_ideahub_generates_traceable_arpabet_variants() -> None:
    rows, counts = generate_variants(_sample(), RULES)

    h_deleted = _rows_for("en.h_deletion", rows)
    devoiced = _rows_for("en.final_stop_devoicing", rows)
    final_deleted = _rows_for("en.final_consonant_deletion", rows)
    spelled = _rows_for("en.letter_spelling", rows)
    syllable_deleted = _rows_for("en.unstressed_syllable_deletion", rows)

    assert rows[0]["base_pronunciation"]["tokens"] == [
        {"text": "idea", "phonemes": ["AY0", "D", "IY1", "AH0"]},
        {"text": "hub", "phonemes": ["HH", "AH1", "B"]},
    ]
    assert h_deleted[0]["variant_pronunciation"]["tokens"][1]["phonemes"] == ["AH1", "B"]
    assert h_deleted[0]["display_text"] == "idea ub"
    assert devoiced[0]["variant_pronunciation"]["tokens"][1]["phonemes"][-1] == "P"
    assert final_deleted[0]["variant_pronunciation"]["tokens"][1]["phonemes"] == ["HH", "AH1"]
    assert any(row["display_text"] == "I D E A hub" for row in spelled)
    assert len(syllable_deleted) == 2
    assert counts["en.h_deletion"] == 1
    assert all(len(row["variant_id"]) == 16 for row in rows)


def test_single_rules_are_deduplicated_and_inapplicable_pairs_are_absent() -> None:
    rows, _ = generate_variants(_sample(), RULES)
    serialized = [
        json.dumps(row["variant_pronunciation"], ensure_ascii=False, sort_keys=True)
        for row in rows
    ]

    assert len(serialized) == len(set(serialized))
    assert sum(value.endswith('"P"]}]}') for value in serialized) <= 1
    substitutions = _rows_for("en.consonant_substitution", rows)
    changed_from = {
        item
        for row in substitutions
        for item in row["rule"]["applied_change"]["from"]
    }
    assert "R" not in changed_from
    assert "L" not in changed_from
    assert all(row["rule"]["applied_change"]["type"] != "composed" for row in rows)


def test_chinese_rules_cover_initial_final_tone_and_syllable_deletion() -> None:
    sample = _sample(
        id="neiqian",
        canonical_text="内嵌",
        text="内嵌",
        language="Chinese",
        pronunciation={"alphabet": "pinyin-tone3", "syllables": ["nei4", "qian4"]},
    )

    rows, counts = generate_variants(sample, RULES)

    assert any(row["display_text"].startswith("lei4 ") for row in _rows_for("zh.initial_n_l", rows))
    assert any("qiang4" in row["display_text"] for row in _rows_for("zh.nasal_final_confusion", rows))
    assert any("qia4" in row["display_text"] for row in _rows_for("zh.nasal_final_weakening", rows))
    assert counts["zh.tone_neutralization"] == 2
    assert {row["display_text"] for row in _rows_for("zh.syllable_deletion", rows)} == {"内", "嵌"}

    retroflex_rows, _ = generate_variants(
        _sample(
            id="zhiwei",
            canonical_text="志伟",
            text="志伟",
            language="Chinese",
            pronunciation={"alphabet": "pinyin-tone3", "syllables": ["zhi4", "wei3"]},
        ),
        RULES,
    )
    assert any(row["display_text"].startswith("zi4 ") for row in _rows_for("zh.retroflex_flattening", retroflex_rows))


def test_two_rule_composition_can_combine_nasal_final_and_tone_change() -> None:
    sample = _sample(
        id="neiqian",
        canonical_text="内嵌",
        text="内嵌",
        language="Chinese",
        pronunciation={"alphabet": "pinyin-tone3", "syllables": ["nei4", "qian4"]},
    )

    rows, counts = generate_variants(sample, RULES, max_rule_count=2)
    target = ["nei4", "qiang2"]
    matches = [
        row
        for row in rows
        if [token["phonemes"][0] for token in row["variant_pronunciation"]["tokens"]]
        == target
    ]

    assert len(matches) == 1
    assert matches[0]["rule"]["rule_id"] == "composed.2"
    assert [step["rule_id"] for step in matches[0]["rule_chain"]] == [
        "zh.nasal_final_confusion",
        "zh.tone_substitution",
    ]
    assert matches[0]["tts_renderability"] == "phoneme_required"
    assert counts["composed.2"] > 0
    assert all(len(row["rule_chain"]) <= 2 for row in rows)


def test_unknown_english_and_ambiguous_chinese_require_overrides() -> None:
    with pytest.raises(PronunciationError, match="override_tokens"):
        generate_variants(
            _sample(
                id="unknown",
                canonical_text="Zzzxq",
                text="zzzxq",
                pronunciation={"tokens": ["zzzxq"]},
            ),
            RULES,
        )

    with pytest.raises(PronunciationError, match="pronunciation.syllables"):
        generate_variants(
            _sample(id="polyphone", canonical_text="重", text="重", language="Chinese", pronunciation={}),
            RULES,
        )


def test_pronunciation_overrides_support_oov_and_polyphonic_terms() -> None:
    english = _sample(
        id="oov",
        canonical_text="Zzzxq",
        text="zzzxq",
        pronunciation={
            "alphabet": "arpabet",
            "override_tokens": [{"text": "zzzxq", "phonemes": ["Z", "IY1"]}],
        },
    )
    chinese = _sample(
        id="polyphone",
        canonical_text="重",
        text="重",
        language="Chinese",
        pronunciation={"alphabet": "pinyin-tone3", "syllables": ["chong2"]},
    )

    english_rows, _ = generate_variants(english, RULES)
    chinese_rows, _ = generate_variants(chinese, RULES)

    assert all(row["alphabet"] == "arpabet" for row in english_rows)
    assert all(row["alphabet"] == "pinyin-tone3" for row in chinese_rows)


def test_python_config_writes_jsonl_and_summary_without_audio(tmp_path) -> None:
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(
        json.dumps(
            [
                {
                    "id": "ideahub",
                    "canonical_text": "IdeaHub",
                    "text": "idea hub",
                    "language": "English",
                    "pronunciation": {"tokens": ["idea", "hub"]},
                }
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "out" / "pronunciation-variants.jsonl"
    summary_path = tmp_path / "out" / "summary.json"
    config_path.write_text(
        json.dumps(
            {
                "samples": str(samples_path),
                "rule_files": [
                    str(
                        ROOT
                        / "src"
                        / "error_words_tts"
                        / "pronunciation"
                        / "rules"
                        / "english.json"
                    )
                ],
                "enabled_rules": ["en.h_deletion"],
                "output": str(output_path),
                "summary": str(summary_path),
            }
        ),
        encoding="utf-8",
    )

    assert run_config(config_path) == 0
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert rows[0]["rule"]["rule_id"] == "en.h_deletion"
    assert summary["variant_count"] == 1
    assert not list(tmp_path.rglob("*.wav"))
