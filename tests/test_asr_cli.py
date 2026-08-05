from __future__ import annotations

import json

from error_words_tts import asr_cli


def test_encode_multipart_contains_fields_and_audio(tmp_path) -> None:
    audio_path = tmp_path / "term.wav"
    audio_path.write_bytes(b"RIFF-audio")

    body, content_type = asr_cli._encode_multipart(
        {"model": "qwen3-asr", "response_format": "json"}, audio_path
    )

    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="model"' in body
    assert b"qwen3-asr" in body
    assert b'filename="term.wav"' in body
    assert b"RIFF-audio" in body


def test_compare_text_records_exact_and_compact_matches() -> None:
    comparison = asr_cli._compare_text("IdeaHub", "idea hub.")

    assert comparison["exact_match"] is False
    assert comparison["compact_match"] is True


def test_asr_result_preserves_pronunciation_metadata() -> None:
    row = {
        "sample_id": "ideahub",
        "text": "IdeaHub",
        "tts_text": "idea ub",
        "canonical_text": "IdeaHub",
        "pronunciation_variant_id": "variant-h-drop",
        "pronunciation_rule": {"rule_id": "en.h_deletion"},
        "tts_renderability": "text_approximation",
        "render_method": "display_text",
        "status": "generated",
    }

    result = asr_cli._base_result(row)

    assert result["expected_text"] == "IdeaHub"
    assert result["tts_text"] == "idea ub"
    assert result["pronunciation_variant_id"] == "variant-h-drop"
    assert result["pronunciation_rule"] == {"rule_id": "en.h_deletion"}


def test_manifest_reuses_transcription_for_duplicate_audio(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "term.wav"
    audio_path.write_bytes(b"audio")
    rows = [
        {
            "sample_id": "term",
            "text": "ideahub",
            "engine": "azure",
            "audio_path": str(audio_path),
            "status": "generated",
            "instruction": {"name": "default"},
        },
        {
            "sample_id": "term",
            "text": "ideahub",
            "engine": "azure",
            "audio_path": str(audio_path),
            "status": "cached",
            "instruction": {"name": "tag-only"},
        },
    ]
    calls = []

    def fake_transcribe(audio_path, **kwargs):
        calls.append((audio_path, kwargs))
        return {"status": "success", "text": "idea hub", "service_url": kwargs["url"]}

    monkeypatch.setattr(asr_cli, "_transcribe_audio", fake_transcribe)
    output_path = tmp_path / "asr-results.jsonl"

    exit_code = asr_cli._transcribe_manifest(
        rows,
        manifest_path=tmp_path / "manifest.jsonl",
        output_path=output_path,
        url="http://127.0.0.1:8756/v1/audio/transcriptions",
        model="qwen3-asr",
        language=None,
        language_from_manifest=False,
        prompt=None,
        timeout_seconds=1,
        continue_on_error=True,
    )

    results = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 0
    assert len(calls) == 1
    assert results[0]["asr"]["status"] == "success"
    assert results[1]["asr"]["status"] == "reused"
    assert results[0]["comparison"]["compact_match"] is True
