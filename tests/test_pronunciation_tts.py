from __future__ import annotations

import json
from pathlib import Path

from error_words_tts.tts.models import SpeechSynthesisResult
from error_words_tts.pronunciation.tts_cli import run_config


class FakeQwenEngine:
    name = "qwen3-tts"

    def __init__(self) -> None:
        self.requests = []

    def synthesize(self, request, output_path):
        self.requests.append(request)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF-fake")
        return SpeechSynthesisResult(
            audio_path=str(output_path),
            engine=self.name,
            sample_rate=16_000,
            duration_ms=100,
            metadata={"mode": "voice_design", "speaker": None},
        )


def _write_experiment(tmp_path):
    variants_path = tmp_path / "variants.jsonl"
    common = {
        "sample_id": "ideahub",
        "canonical_text": "IdeaHub",
        "language": "English",
        "sample_tags": ["term"],
        "base_pronunciation": {"alphabet": "arpabet"},
        "variant_pronunciation": {"alphabet": "arpabet"},
        "rule": {"rule_id": "en.h_deletion"},
    }
    variants = [
        {
            **common,
            "variant_id": "variant-text",
            "display_text": "idea ub",
            "tts_renderability": "text_approximation",
        },
        {
            **common,
            "variant_id": "variant-phoneme",
            "display_text": "idea hub",
            "tts_renderability": "phoneme_required",
        },
    ]
    variants_path.write_text(
        "".join(json.dumps(row) + "\n" for row in variants), encoding="utf-8"
    )
    engine_path = tmp_path / "qwen.json"
    engine_path.write_text(
        json.dumps(
            {
                "engine": "qwen3-tts",
                "mode": "voice_design",
                "model": "fake-model",
                "instructions": [
                    {"name": "female", "instruct": "A natural female voice."},
                    {"name": "male", "instruct": "A natural male voice."},
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "tts.json"
    config_path.write_text(
        json.dumps(
            {
                "variants": str(variants_path),
                "engine_config": str(engine_path),
                "output_dir": str(tmp_path / "output"),
                "renderabilities": ["text_approximation"],
                "instruction_suffix": "Do not correct unusual pronunciation.",
            }
        ),
        encoding="utf-8",
    )
    return config_path, tmp_path / "output"


def test_pronunciation_tts_filters_phoneme_only_and_preserves_metadata(tmp_path) -> None:
    config_path, output_dir = _write_experiment(tmp_path)
    engine = FakeQwenEngine()

    assert run_config(config_path, engine=engine) == 0

    rows = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(engine.requests) == 2
    assert len(rows) == 2
    assert {request.text for request in engine.requests} == {"idea ub"}
    assert all(request.speaker is None for request in engine.requests)
    assert all("Do not correct unusual pronunciation." in request.instruction for request in engine.requests)
    assert {row["instruction"]["name"] for row in rows} == {"female", "male"}
    assert all(row["text"] == "IdeaHub" for row in rows)
    assert all(row["tts_text"] == "idea ub" for row in rows)
    assert all(row["pronunciation_variant_id"] == "variant-text" for row in rows)
    assert all(row["pronunciation_rule"]["rule_id"] == "en.h_deletion" for row in rows)
    assert all(row["status"] == "generated" for row in rows)
    assert all((output_dir / "audio" / "qwen3-tts") in Path(row["audio_path"]).parents for row in rows)


def test_pronunciation_tts_reuses_existing_audio(tmp_path) -> None:
    config_path, output_dir = _write_experiment(tmp_path)
    first_engine = FakeQwenEngine()
    second_engine = FakeQwenEngine()

    assert run_config(config_path, engine=first_engine) == 0
    assert run_config(config_path, engine=second_engine) == 0

    rows = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(first_engine.requests) == 2
    assert second_engine.requests == []
    assert all(row["status"] == "cached" for row in rows)


def test_pronunciation_tts_dry_run_does_not_create_output(tmp_path) -> None:
    config_path, output_dir = _write_experiment(tmp_path)

    assert run_config(config_path, dry_run=True) == 0
    assert not output_dir.exists()
