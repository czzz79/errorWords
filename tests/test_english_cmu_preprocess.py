from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from error_words_tts.pronunciation.english_cmu import (
    build_english_cmu_samples,
    build_structure,
    generate_variants,
    load_rules,
    parse_term,
)


ROOT = Path(__file__).resolve().parents[1]
RULES = load_rules(ROOT / "src/error_words_tts/pronunciation/rules/english-cmu.json")


def _tokens(text: str) -> list[dict]:
    return [node for node in parse_term(text) if node["kind"] == "token"]


def test_structural_parser_preserves_camel_hyphen_and_alnum_boundaries() -> None:
    idea_ui = parse_term("IdeaUI")
    assert [node["text"] for node in _tokens("IdeaUI")] == ["Idea", "UI"]
    assert idea_ui[1]["boundary_type"] == "camel_case"
    assert _tokens("VLLM-Mindspore")[0]["token_type"] == "acronym"
    assert [node["text"] for node in _tokens("E2E")] == ["E", "2", "E"]
    assert [node["text"] for node in _tokens("DDR5")] == ["DDR", "5"]
    assert [node["text"] for node in _tokens("ES2Pro")] == ["ES", "2", "Pro"]


def test_resolution_uses_acronym_letters_and_manual_override() -> None:
    tmg = build_structure("TMG", overrides={})
    token = _tokens("TMG")[0]
    assert tmg["status"] == "ready"
    assert tmg["nodes"][0]["pronunciation_source"] == "letter_names"
    assert token["letters"] == ["T", "M", "G"]

    override = build_structure("Nexent", overrides={"nexent": ["N", "EH1", "K", "S", "AH0", "N", "T"]})
    assert override["nodes"][0]["pronunciation_source"] == "manual_override"
    assert override["nodes"][0]["phones"] == ["N", "EH1", "K", "S", "AH0", "N", "T"]


def test_rules_are_directional_and_cluster_aware() -> None:
    box = build_structure("Box", overrides={})
    box_variants = generate_variants(box, RULES, include_boundary_variants=False)
    cluster = next(item for item in box_variants if item["rule"] and item["rule"]["rule_id"] == "en.final_cluster_delete_penultimate")
    assert cluster["structure"]["nodes"][0]["phones"] == ["B", "AA1", "S"]

    code = build_structure("Code", overrides={})
    code_variants = generate_variants(code, RULES, include_boundary_variants=False)
    devoiced = next(item for item in code_variants if item["rule"] and item["rule"]["rule_id"] == "en.final_stop_devoicing.d_to_t")
    assert devoiced["structure"]["nodes"][0]["phones"][-1] == "T"

    voice = build_structure("Voice", overrides={})
    voice_variants = generate_variants(voice, RULES, include_boundary_variants=False)
    v_to_w = next(item for item in voice_variants if item["rule"] and item["rule"]["rule_id"] == "en.consonant.v_to_w")
    assert v_to_w["structure"]["nodes"][0]["phones"][0] == "W"


def test_samples_keep_baseline_metadata_and_never_copy_gt_text(tmp_path) -> None:
    entries = [SimpleNamespace(
        sample_id="box", canonical_text="Box", known_confusions=["boss"], source_lines=[1]
    )]
    rows, summary = build_english_cmu_samples(
        entries,
        root=ROOT,
        output_dir=tmp_path,
        settings={"max_variants_per_term": 40},
    )
    baseline = next(row for row in rows if row["variant_kind"] == "baseline")
    assert baseline["phoneme_text"] == "[B][AA1][K][S]"
    assert baseline["tts_text"] == "Box"
    assert all(row["tts_text"] != "boss" for row in rows)
    assert summary["baseline_count"] == 1
    assert (tmp_path / "pronunciation/structured-pronunciations.jsonl").is_file()
    assert (tmp_path / "pronunciation/unresolved.jsonl").is_file()


def test_baseline_only_keeps_the_standard_cmu_variant(tmp_path) -> None:
    entries = [SimpleNamespace(
        sample_id="ideahub", canonical_text="IdeaHub", known_confusions=[], source_lines=[1]
    )]
    rows, summary = build_english_cmu_samples(
        entries, root=ROOT, output_dir=tmp_path, settings={"baseline_only": True}
    )
    assert len(rows) == 1
    assert rows[0]["variant_kind"] == "baseline"
    assert rows[0]["phoneme_text"] == "[AY0][D][IY1][AH0] [HH][AH1][B]"
    assert summary["baseline_only"] is True


def test_boundary_modes_can_connect_camel_case_without_changing_phones(tmp_path) -> None:
    entries = [SimpleNamespace(
        sample_id="ideahub", canonical_text="IdeaHub", known_confusions=[], source_lines=[1]
    )]
    rows, _ = build_english_cmu_samples(
        entries, root=ROOT, output_dir=tmp_path,
        settings={"baseline_only": True, "boundary_modes": {"camel_case": "connected"}},
    )
    assert rows[0]["phoneme_text"] == "[AY0][D][IY1][AH0][HH][AH1][B]"
    boundary = rows[0]["pronunciation_structure"]["nodes"][1]
    assert boundary["mode"] == "connected"
