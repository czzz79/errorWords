from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np


TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


def write_normalized_wav(
    samples: Any,
    source_sample_rate: int,
    output_path: str | Path,
) -> tuple[int, int]:
    """Write float or integer samples as mono 16 kHz PCM16 WAV.

    Returns ``(sample_rate, duration_ms)``. The conversion intentionally uses
    a small linear resampler so the tool does not require ffmpeg or libsndfile.
    """

    if source_sample_rate <= 0:
        raise ValueError("source_sample_rate must be positive")
    values = _to_mono_float32(samples)
    if values.size == 0:
        raise ValueError("TTS returned an empty waveform")
    if source_sample_rate != TARGET_SAMPLE_RATE:
        values = _resample_linear(values, source_sample_rate, TARGET_SAMPLE_RATE)
    values = np.clip(values, -1.0, 1.0)
    pcm16 = np.rint(values * 32767.0).astype("<i2", copy=False)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as wav_file:
        wav_file.setnchannels(TARGET_CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(TARGET_SAMPLE_RATE)
        wav_file.writeframes(pcm16.tobytes())
    duration_ms = round(len(pcm16) * 1000 / TARGET_SAMPLE_RATE)
    return TARGET_SAMPLE_RATE, duration_ms


def write_raw_pcm16_wav(
    pcm_bytes: bytes,
    output_path: str | Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[int, int]:
    """Write raw little-endian mono PCM16 bytes as a normalized WAV."""

    if len(pcm_bytes) % 2:
        raise ValueError("PCM16 payload has an odd number of bytes")
    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    return write_normalized_wav(samples, sample_rate, output_path)


def _to_mono_float32(samples: Any) -> np.ndarray:
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    values = np.asarray(samples)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim > 2:
        raise ValueError(f"unsupported waveform shape: {values.shape}")
    if values.ndim == 2:
        if values.shape[0] <= 2:
            values = values.mean(axis=0)
        else:
            values = values.mean(axis=1)
    values = values.astype(np.float32, copy=False).reshape(-1)
    if np.issubdtype(np.asarray(samples).dtype, np.integer):
        info = np.iinfo(np.asarray(samples).dtype)
        scale = max(abs(info.min), info.max)
        values = values / scale
    elif np.max(np.abs(values), initial=0.0) > 1.0:
        values = values / 32768.0
    return values


def _resample_linear(values: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    target_length = max(1, round(len(values) * target_rate / source_rate))
    source_positions = np.arange(len(values), dtype=np.float64)
    target_positions = np.linspace(0, len(values) - 1, target_length)
    return np.interp(target_positions, source_positions, values).astype(np.float32)
