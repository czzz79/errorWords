from __future__ import annotations

import json

import pytest

from error_words_tts.confusion.cli import (
    clean_transcript,
    parse_gt_file,
    write_confusion_outputs,
)


def test_parse_gt_file_uses_first_item_and_preserves_pipe_aliases(tmp_path) -> None:
    path = tmp_path / "gt.txt"
    path.write_text(
        "IdeaHub|ID Hub|Idea hot|IDEAHub\n"
        "志伟|周也\n"
        "志伟|市委|周也\n"
        "Idea Hub|ID Hub|Idea hot\n"
        "# comment\n",
        encoding="utf-8",
    )

    entries = parse_gt_file(path)

    assert [entry.canonical_text for entry in entries] == ["IdeaHub", "志伟", "Idea Hub"]
    assert entries[0].known_confusions == ["ID Hub", "Idea hot"]
    assert entries[1].known_confusions == ["周也", "市委"]
    assert entries[1].source_lines == [2, 3]
    assert entries[2].known_confusions == ["ID Hub", "Idea hot"]


def test_parse_gt_file_rejects_empty_input(tmp_path) -> None:
    path = tmp_path / "gt.txt"
    path.write_text("\n# no terms\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no terms"):
        parse_gt_file(path)


def test_write_confusion_outputs_ranks_counts_and_excludes_gt(tmp_path) -> None:
    input_path = tmp_path / "gt.txt"
    input_path.write_text("陈赵|陈照|陈兆\nIdeaHub|Idea Hub\n", encoding="utf-8")
    entries = parse_gt_file(input_path)
    chen_id, idea_id = (entry.sample_id for entry in entries)
    results = [
        _asr_row(chen_id, "陈，照。"),
        _asr_row(chen_id, "陈照"),
        _asr_row(chen_id, "陈赵。"),
        _asr_row(chen_id, "陈兆。"),
        _asr_row(idea_id, "IdeaHub."),
        _asr_row(idea_id, "Idea Hub."),
        _asr_row(idea_id, "idea hub"),
    ]
    output_txt = tmp_path / "confusion-words.txt"
    summary_path = tmp_path / "summary.json"

    write_confusion_outputs(
        entries,
        results,
        output_txt=output_txt,
        summary_path=summary_path,
        include_known_confusions=False,
        engine_name="fake-tts",
    )

    assert output_txt.read_text(encoding="utf-8").splitlines() == [
        "陈赵|陈照|陈兆",
        "IdeaHub|Idea Hub",
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["generated_confusion_count"] == 3
    assert summary["matched_known_confusion_count"] == 3
    assert summary["known_coverage"] == 1.0
    assert summary["terms"][0]["generated_confusions"][0] == {
        "text": "陈照",
        "count": 2,
    }


def test_clean_transcript_removes_punctuation_but_keeps_word_spacing() -> None:
    assert clean_transcript("  I D E A, Hub. ") == "I D E A Hub"
    assert clean_transcript("陈，照。") == "陈照"


def _asr_row(sample_id: str, text: str) -> dict:
    return {
        "sample_id": sample_id,
        "asr": {"status": "success", "text": text},
    }
