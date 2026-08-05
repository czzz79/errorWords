from __future__ import annotations

import json
import wave

import numpy as np

from error_words_tts.augmentation.cli import run_config
from error_words_tts.tts.audio import write_normalized_wav


def test_low_cost_augmentation_is_deterministic_and_normalized(tmp_path) -> None:
    source_path = tmp_path / "source.wav"
    time = np.arange(16_000, dtype=np.float32) / 16_000
    write_normalized_wav(np.sin(2 * np.pi * 440 * time) * 0.2, 16_000, source_path)
    manifest_path = tmp_path / "tts" / "manifest.jsonl"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "sample_id": "term",
                "text": "术语",
                "audio_path": str(source_path),
                "status": "generated",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "augmentation.json"
    output_dir = tmp_path / "augmented"
    config_path.write_text(
        json.dumps(
            {
                "input_manifest": str(manifest_path),
                "output_dir": str(output_dir),
                "seed": 7,
                "perturbations": [
                    {"name": "slow", "type": "speed", "factor": 0.9},
                    {"name": "fast", "type": "speed", "factor": 1.1},
                    {"name": "noise", "type": "white_noise", "snr_db": 18},
                    {"name": "lowpass", "type": "lowpass", "cutoff_hz": 3200},
                    {"name": "clip", "type": "hard_clip", "threshold": 0.35},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert run_config(config_path) == 0
    first_rows = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first_bytes = {
        row["augmentation"]["name"]: open(row["audio_path"], "rb").read()
        for row in first_rows
    }
    assert len(first_rows) == 5
    assert {row["status"] for row in first_rows} == {"generated"}
    assert next(row for row in first_rows if row["augmentation"]["name"] == "slow")[
        "duration_ms"
    ] > 1000
    assert next(row for row in first_rows if row["augmentation"]["name"] == "fast")[
        "duration_ms"
    ] < 1000
    for row in first_rows:
        with wave.open(row["audio_path"], "rb") as wav_file:
            assert wav_file.getframerate() == 16_000
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2

    assert run_config(config_path) == 0
    second_rows = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["status"] for row in second_rows} == {"cached"}
    assert next(row for row in second_rows if row["augmentation"]["name"] == "slow")[
        "duration_ms"
    ] > 1000
    assert {
        row["augmentation"]["name"]: open(row["audio_path"], "rb").read()
        for row in second_rows
    } == first_bytes
