from __future__ import annotations

import json

from error_words_tts.confusion.dictionary_postprocess import process


def test_process_removes_single_characters_and_tags_remaining_candidates(tmp_path) -> None:
    input_path = tmp_path / "confusion-words.txt"
    tagged_output_path = tmp_path / "confusion-words-dict-tagged.txt"
    removed_output_path = tmp_path / "confusion-words-dict-removed-single-character.txt"
    detail_path = tmp_path / "confusion-words-dict.review.jsonl"
    stopword_path = tmp_path / "stopwords.txt"
    wusong_path = tmp_path / "wusong.dict.yaml"
    english_path = tmp_path / "english.txt"
    jieba_path = tmp_path / "jieba-dict.txt"

    input_path.write_text(
        "森伢|森|怎样|生涯|也说|是这|森牙|when|Q W E N\n",
        encoding="utf-8",
    )
    stopword_path.write_text("怎样\n", encoding="utf-8")
    wusong_path.write_text(
        "生涯 sheng ya 135505\n也说 ye shuo 117910\n低频 di pin 4999\n",
        encoding="utf-8",
    )
    english_path.write_text("when\n", encoding="utf-8")
    jieba_path.write_text(
        "是 796991 v\n这 261791 r\n森 742 n\n牙 2746 n\n",
        encoding="utf-8",
    )

    process(
        input_path=input_path,
        tagged_output_path=tagged_output_path,
        removed_output_path=removed_output_path,
        detail_path=detail_path,
        stopword_path=stopword_path,
        wusong_path=wusong_path,
        english_path=english_path,
        min_wusong_weight=5000,
        jieba_path=jieba_path,
        min_jieba_character_frequency=200000,
    )

    assert tagged_output_path.read_text(encoding="utf-8").splitlines() == [
        "#TERM_IMPORT_V2",
        "森伢|怎样|HIGH|生涯|HIGH|也说|HIGH|是这|HIGH|森牙|LOW|when|HIGH|Q W E N|LOW",
    ]
    assert removed_output_path.read_text(encoding="utf-8").splitlines() == [
        "森伢|森",
    ]
    record = json.loads(detail_path.read_text(encoding="utf-8"))
    assert record["removed_single_characters"] == ["森"]
    assert record["tagged_candidates"] == [
        {
            "word": "怎样",
            "frequency": "HIGH",
            "dictionary_matched": True,
            "character_combination_matched": False,
        },
        {
            "word": "生涯",
            "frequency": "HIGH",
            "dictionary_matched": True,
            "character_combination_matched": False,
        },
        {
            "word": "也说",
            "frequency": "HIGH",
            "dictionary_matched": True,
            "character_combination_matched": False,
        },
        {
            "word": "是这",
            "frequency": "HIGH",
            "dictionary_matched": False,
            "character_combination_matched": True,
        },
        {
            "word": "森牙",
            "frequency": "LOW",
            "dictionary_matched": False,
            "character_combination_matched": False,
        },
        {
            "word": "when",
            "frequency": "HIGH",
            "dictionary_matched": True,
            "character_combination_matched": False,
        },
        {
            "word": "Q W E N",
            "frequency": "LOW",
            "dictionary_matched": False,
            "character_combination_matched": False,
        },
    ]
