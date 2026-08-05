from __future__ import annotations

from pathlib import Path

from error_words_tts.confusion.cli import GtEntry
from error_words_tts.pronunciation.fuzzy_pinyin import (
    build_fuzzy_pinyin_variants,
    load_fuzzy_rules,
)


ROOT = Path(__file__).resolve().parents[1]


def test_public_rule_file_has_directional_online_provenance() -> None:
    rules = load_fuzzy_rules(ROOT / "resources" / "chinese-fuzzy-pinyin-rules.csv")
    keys = {rule.key: rule for rule in rules}
    assert keys[("initial", "z", "zh")].origin == "ibus_libpinyin"
    assert keys[("final", "in", "ing")].origin == "ibus_libpinyin"
    assert keys[("final", "ing", "i")].origin == "manual_spec"
    assert keys[("initial", "j", "q")].level == 3


def test_fuzzy_preprocess_generates_tone_exact_carrier_text(tmp_path: Path) -> None:
    rime = tmp_path / "carrier.dict.yaml"
    rime.write_text(
        "# word pinyin weight\n露西 lu xi 10000\n主网图 zhu wang tu 10000\n没有眼镜 mei you yan jing 10000\n",
        encoding="utf-8",
    )
    high_frequency = tmp_path / "high-frequency.txt"
    high_frequency.write_text("", encoding="utf-8")
    jieba = tmp_path / "jieba.txt"
    jieba.write_text("", encoding="utf-8")
    entries = [
        GtEntry("luxing", "鹭兴", ["露西"], [1]),
        GtEntry("zuwangtu", "组网图", ["主网图"], [2]),
        GtEntry("meiyouyanjin", "没有演进", ["没有眼镜"], [3]),
    ]
    rows, summary = build_fuzzy_pinyin_variants(
        entries,
        root=tmp_path,
        settings={
            "rules_path": str(ROOT / "resources" / "chinese-fuzzy-pinyin-rules.csv"),
            "carrier_rime_dictionary": str(rime),
            "carrier_high_frequency_words": str(high_frequency),
            "carrier_jieba_dictionary": str(jieba),
        },
    )

    assert summary["variant_count"] >= 3
    found = {(row["canonical_text"], row["tts_text"]) for row in rows}
    assert ("鹭兴", "露西") in found
    assert ("组网图", "主网图") in found
    assert ("没有演进", "没有眼镜") in found
    for row in rows:
        delta = row["pronunciation_delta"]
        assert row["text"] == row["canonical_text"]
        assert row["tts_text"] == delta["carrier_term"]
        assert len(delta["original_pinyin"]) == len(delta["perturbed_pinyin"])
        assert sum(
            source != target
            for source, target in zip(delta["original_pinyin"], delta["perturbed_pinyin"])
        ) == 1


def test_fuzzy_preprocess_derives_missing_component_relation_from_gt(tmp_path: Path) -> None:
    rime = tmp_path / "carrier.dict.yaml"
    # 棕勇 is a carrier for zong1 yong3, distinct from the GT confusion 宗勇.
    rime.write_text("# word pinyin weight\n棕勇 zong yong 10000\n", encoding="utf-8")
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    entries = [GtEntry("zongyou", "宗友", ["宗勇"], [1])]

    rows, summary = build_fuzzy_pinyin_variants(
        entries,
        root=tmp_path,
        settings={
            "rules_path": str(ROOT / "resources" / "chinese-fuzzy-pinyin-rules.csv"),
            "carrier_rime_dictionary": str(rime),
            "carrier_high_frequency_words": str(empty),
            "carrier_jieba_dictionary": str(empty),
            "derive_gt_component_rules": True,
        },
    )

    derived = [row for row in rows if row["pronunciation_rule"] == "zh.fuzzy_final_ou_to_ong"]
    assert derived
    assert derived[0]["tts_text"] == "棕勇"
    assert derived[0]["pronunciation_delta"]["origin"] == "gt_target_component"
    assert summary["derived_gt_component_rule_count"] > 0


def test_fuzzy_preprocess_can_use_gt_target_tone_without_using_gt_text(tmp_path: Path) -> None:
    rime = tmp_path / "carrier.dict.yaml"
    # 老腕 is a distinct carrier for lao3 wan4; the GT target is 老万.
    rime.write_text("# word pinyin weight\n老腕 lao wan 10000\n", encoding="utf-8")
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    entries = [GtEntry("laowang", "老王", ["老万"], [1])]

    rows, _ = build_fuzzy_pinyin_variants(
        entries,
        root=tmp_path,
        settings={
            "rules_path": str(ROOT / "resources" / "chinese-fuzzy-pinyin-rules.csv"),
            "carrier_rime_dictionary": str(rime),
            "carrier_high_frequency_words": str(empty),
            "carrier_jieba_dictionary": str(empty),
            "derive_gt_component_rules": True,
            "use_gt_target_tones": True,
            "exclude_known_confusion_texts": True,
        },
    )

    derived = [
        row for row in rows
        if row["pronunciation_delta"]["origin"].removesuffix("+gt")
        == "gt_target_component_tone"
    ]
    assert [(row["tts_text"], row["pronunciation_delta"]["perturbed_pinyin"]) for row in derived] == [
        ("老腕", ["lao3", "wan4"])
    ]
    assert all(row["tts_text"] not in row["target_confusions"] for row in derived)


def test_fuzzy_preprocess_derives_one_syllable_insert_as_carrier(tmp_path: Path) -> None:
    rime = tmp_path / "carrier.dict.yaml"
    rime.write_text("# word pinyin weight\n送帅帅 song shuai shuai 10000\n", encoding="utf-8")
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    entries = [GtEntry("songshuai", "宋帅", ["宋帅帅"], [1])]

    rows, _ = build_fuzzy_pinyin_variants(
        entries,
        root=tmp_path,
        settings={
            "rules_path": str(ROOT / "resources" / "chinese-fuzzy-pinyin-rules.csv"),
            "carrier_rime_dictionary": str(rime),
            "carrier_high_frequency_words": str(empty),
            "carrier_jieba_dictionary": str(empty),
            "derive_gt_syllable_edits": True,
            "use_gt_target_tones": True,
            "exclude_known_confusion_texts": True,
        },
    )

    edits = [row for row in rows if row["pronunciation_delta"]["type"] == "syllable_insert"]
    assert [(row["tts_text"], row["pronunciation_delta"]["perturbed_pinyin"]) for row in edits] == [
        ("送帅帅", ["song4", "shuai4", "shuai4"])
    ]
    assert edits[0]["tts_text"] not in edits[0]["target_confusions"]
