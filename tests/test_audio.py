from __future__ import annotations

import wave

import numpy as np

from error_words_tts.tts.audio import write_normalized_wav


def test_write_normalized_wav_resamples_to_gateway_contract(tmp_path) -> None:
    source_rate = 8_000
    samples = np.zeros(source_rate, dtype=np.float32)
    output_path = tmp_path / "sample.wav"

    sample_rate, duration_ms = write_normalized_wav(samples, source_rate, output_path)

    assert sample_rate == 16_000
    assert duration_ms == 1000
    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 16_000


def test_write_normalized_wav_accepts_channel_first_input(tmp_path) -> None:
    output_path = tmp_path / "stereo.wav"
    values = np.ones((2, 160), dtype=np.float32) * 0.25

    write_normalized_wav(values, 16_000, output_path)

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() == 160
