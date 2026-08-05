from __future__ import annotations

import json
from pathlib import Path

from error_words_tts.asr_cli import _base_result
from error_words_tts.confusion.cli import GtEntry, parse_gt_file
from error_words_tts.pronunciation.gt_directed import (
    CATEGORY_FILES,
    PREPROCESSABLE_CATEGORIES,
    add_homophone_metadata,
    build_gt_directed_variants,
    classify_entries,
    classify_pair,
    write_classification_outputs,
)
from error_words_tts.tts.cli import _job_description, _make_request
from error_words_tts.tts.models import EngineRunConfig, InstructionPreset, TaggedVoice, TermSample


ROOT = Path(__file__).resolve().parents[1]


def test_gt3_classification_is_complete_and_examples_are_stable(tmp_path: Path) -> None:
    entries = parse_gt_file(ROOT / "gt3.txt")
    rows = classify_entries(entries)

    assert len(entries) == 951
    assert len(rows) == 1202
    assert len({row["pair_id"] for row in rows}) == 1202
    assert classify_pair("IdeaHub", "Idea Hub")["category"] == "normalization_alias"
    assert classify_pair("鹭兴", "陆星")["category"] == "zh_same_pinyin"
    assert classify_pair("周清", "周星")["category"] == "zh_single_syllable"
    assert classify_pair("CloudRSE", "CloudRCE")["category"] == "english_word_acronym"
    assert classify_pair("Nexent", "南盛的")["category"] == "cross_language_transliteration"
    assert classify_pair("TR6", "TR六")["category"] == "number_reading_mixed"

    summary = write_classification_outputs(entries, tmp_path)
    assert summary["pair_count"] == 1202
    assert sum(summary["category_counts"].values()) == 1202
    assert len((tmp_path / "classification.jsonl").read_text(encoding="utf-8").splitlines()) == 1202
    preprocessable_lines = (tmp_path / "preprocessable.txt").read_text(encoding="utf-8").splitlines()
    assert len(preprocessable_lines) == summary["preprocessable_pair_count"]
    for filename in CATEGORY_FILES.values():
        path = tmp_path / filename
        assert path.is_file()
        if path.stat().st_size:
            assert parse_gt_file(path)


def test_gt_directed_variants_are_explainable_and_bounded() -> None:
    entries = [
        GtEntry("zhouqing", "周清", ["周星"], [1]),
        GtEntry("cloud", "CloudRSE", ["CloudRCE"], [2]),
        GtEntry("nexent", "Nexent", ["南盛的"], [3]),
        GtEntry("tr6", "TR6", ["TR六"], [4]),
        GtEntry("luxing", "鹭兴", ["陆星"], [5]),
    ]
    classified = classify_entries(entries)
    variants = build_gt_directed_variants(entries, classified)

    chinese = [row for row in variants if row["id"] == "zhouqing"]
    assert len(chinese) == 2
    assert {row["tts_text"] for row in chinese} == {"周清"}
    assert all(row["target_confusions"] == ["周星"] for row in chinese)
    assert {row["pronunciation_delta"]["type"] for row in chinese} == {"initial_change"}
    assert all(row["pronunciation_instruction"] for row in chinese)

    cloud = [row for row in variants if row["id"] == "cloud"]
    nexent = [row for row in variants if row["id"] == "nexent"]
    tr6 = [row for row in variants if row["id"] == "tr6"]
    assert 1 <= len(cloud) <= 4
    assert 1 <= len(nexent) <= 4
    assert 1 <= len(tr6) <= 3
    assert all(row["tts_text"].casefold() != "cloudrce".casefold() for row in cloud)
    assert all(row["tts_text"] != "南盛的" for row in nexent)
    assert all(row["tts_text"].casefold() != "TR六".casefold() for row in tr6)
    assert any("六" in row["tts_text"] for row in tr6)
    assert {row["variant_kind"] for row in tr6} <= {
        "number_chinese_digits",
        "number_chinese_yao",
        "number_chinese_cardinal",
    }
    assert all(row["pronunciation_delta"]["derived_from"] == "canonical_text" for row in cloud + nexent + tr6)

    canonical_rows = [{"id": entry.sample_id} for entry in entries]
    add_homophone_metadata(canonical_rows, classified)
    homophone = next(row for row in canonical_rows if row["id"] == "luxing")
    assert homophone["target_confusions"] == ["陆星"]
    assert homophone["variant_kind"] == "canonical_homophone_baseline"


def test_directed_metadata_and_instruction_survive_tts_and_asr_manifest() -> None:
    sample = TermSample.from_mapping(
        {
            "id": "zhouqing",
            "text": "周清",
            "canonical_text": "周清",
            "tts_text": "周清",
            "source_text": "周清",
            "pronunciation_processed": True,
            "pronunciation_rule": "gt.initial_transition",
            "pronunciation_variant_id": "variant-1",
            "pronunciation_instruction": "只弱化第二个音节。",
            "target_confusions": ["周星"],
            "confusion_category": "zh_single_syllable",
            "pronunciation_delta": {"type": "initial_change", "index": 1},
            "variant_kind": "initial_transition",
        },
        0,
    )
    instruction = InstructionPreset(
        "neutral", parameters={"instruct": "自然、清晰地朗读。"}
    )
    voice = TaggedVoice("reference_voice", "reference")
    config = EngineRunConfig("cosyvoice3", {}, (voice,), (instruction,))

    request = _make_request(sample, instruction, voice, "cosyvoice3")
    tts_row = _job_description(sample, instruction, voice, config)
    tts_row.update({"audio_path": "sample.wav", "status": "generated"})
    asr_row = _base_result(tts_row)

    assert request.text == "周清"
    assert request.instruction == "自然、清晰地朗读。 只弱化第二个音节。"
    assert tts_row["text"] == "周清"
    assert tts_row["tts_text"] == "周清"
    assert tts_row["target_confusions"] == ["周星"]
    assert tts_row["instruction"]["parameters"]["effective_instruct"] == request.instruction
    for field in (
        "target_confusions",
        "confusion_category",
        "pronunciation_delta",
        "variant_kind",
        "pronunciation_instruction",
    ):
        assert asr_row[field] == tts_row[field]


def test_preprocessable_categories_match_public_category_names() -> None:
    assert PREPROCESSABLE_CATEGORIES == {
        "number_reading_mixed",
        "cross_language_transliteration",
        "english_word_acronym",
        "zh_same_pinyin",
        "zh_single_syllable",
    }
