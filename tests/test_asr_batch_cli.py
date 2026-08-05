from __future__ import annotations

import json

from error_words_tts import asr_batch_cli


def test_batch_config_transcribes_each_directory(tmp_path, monkeypatch) -> None:
    first = tmp_path / "cosyvoice"
    second = tmp_path / "qwen"
    first.mkdir()
    second.mkdir()
    row = {
        "sample_id": "ideahub",
        "text": "ideahub",
        "audio_path": "audio.wav",
        "status": "generated",
    }
    for directory in (first, second):
        (directory / "manifest.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )

    config_path = tmp_path / "asr_inputs.json"
    config_path.write_text(
        json.dumps(
            {
                "audio_directories": [str(first), str(second)],
                "wait_seconds": 1,
                "output_name": "results.jsonl",
            }
        ),
        encoding="utf-8",
    )
    waited = []
    calls = []
    monkeypatch.setattr(
        asr_batch_cli,
        "_wait_for_service",
        lambda url, seconds: waited.append((url, seconds)),
    )

    def fake_transcribe(rows, **kwargs):
        calls.append((rows, kwargs))
        return 0

    monkeypatch.setattr(asr_batch_cli, "_transcribe_manifest", fake_transcribe)

    exit_code = asr_batch_cli.run_config(config_path)

    assert exit_code == 0
    assert waited == [(asr_batch_cli.DEFAULT_SERVICE_URL, 1.0)]
    assert [call[1]["manifest_path"] for call in calls] == [
        first / "manifest.jsonl",
        second / "manifest.jsonl",
    ]
    assert [call[1]["output_path"] for call in calls] == [
        first / "results.jsonl",
        second / "results.jsonl",
    ]


def test_batch_config_requires_audio_directories(tmp_path) -> None:
    config_path = tmp_path / "asr_inputs.json"
    config_path.write_text("{}", encoding="utf-8")

    try:
        asr_batch_cli.run_config(config_path)
    except ValueError as exc:
        assert "audio_directories" in str(exc)
    else:
        raise AssertionError("missing audio_directories should fail")


def test_batch_config_can_write_results_outside_tts_directories(tmp_path, monkeypatch) -> None:
    tts_directory = tmp_path / "tts" / "experiment"
    tts_directory.mkdir(parents=True)
    (tts_directory / "manifest.jsonl").write_text(
        json.dumps({"text": "term", "audio_path": "audio.wav", "status": "generated"})
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "audio_directories": ["tts/experiment"],
                "output_directory": "asr",
                "output_name": "results.jsonl",
                "wait_seconds": 0,
            }
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        asr_batch_cli,
        "_transcribe_manifest",
        lambda rows, **kwargs: calls.append(kwargs) or 0,
    )

    assert asr_batch_cli.run_config(config_path) == 0
    assert calls[0]["manifest_path"] == tts_directory / "manifest.jsonl"
    assert calls[0]["output_path"] == tmp_path / "asr" / "experiment" / "results.jsonl"
    assert calls[0]["output_path"].parent.is_dir()
