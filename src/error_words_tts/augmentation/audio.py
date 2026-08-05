from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

from ..tts.audio import TARGET_SAMPLE_RATE


def read_pcm16_mono_wav(path: str | Path) -> tuple[np.ndarray, int]:
    source = Path(path)
    with wave.open(str(source), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError(f"augmentation input must be PCM16 WAV: {source}")
    if channels <= 0:
        raise ValueError(f"augmentation input has invalid channel count: {source}")
    values = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    if values.size == 0:
        raise ValueError(f"augmentation input is empty: {source}")
    return values, sample_rate


def apply_perturbation(
    values: np.ndarray,
    sample_rate: int,
    specification: dict[str, Any],
    *,
    seed: int,
) -> np.ndarray:
    kind = str(specification["type"])
    if kind == "chain":
        result = values
        steps = specification["steps"]
        for index, step in enumerate(steps):
            # Derive a stable but distinct RNG stream for each stage.  This
            # prevents a noise stage later in the chain from reusing the same
            # random sequence as an earlier one.
            step_seed = seed ^ ((index + 1) * 0x9E3779B1)
            result = apply_perturbation(result, sample_rate, step, seed=step_seed)
        return result
    if kind == "speed":
        factor = float(specification["factor"])
        target_length = max(1, round(len(values) / factor))
        source_positions = np.arange(len(values), dtype=np.float64)
        target_positions = np.linspace(0, len(values) - 1, target_length)
        return np.interp(target_positions, source_positions, values).astype(np.float32)
    if kind == "white_noise":
        snr_db = float(specification["snr_db"])
        signal_rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
        if signal_rms == 0:
            signal_rms = 1e-4
        noise_rms = signal_rms / math.pow(10.0, snr_db / 20.0)
        noise = np.random.default_rng(seed).normal(0.0, noise_rms, len(values))
        return (values + noise.astype(np.float32)).astype(np.float32)
    if kind == "lowpass":
        cutoff_hz = float(specification["cutoff_hz"])
        window_size = max(2, round(sample_rate / cutoff_hz))
        kernel = np.ones(window_size, dtype=np.float32) / window_size
        return np.convolve(values, kernel, mode="same").astype(np.float32)
    if kind == "hard_clip":
        threshold = float(specification["threshold"])
        return (np.clip(values, -threshold, threshold) / threshold).astype(np.float32)
    raise ValueError(f"unsupported perturbation type: {kind}")


def validate_perturbation(specification: Any, index: int) -> dict[str, Any]:
    if not isinstance(specification, dict):
        raise ValueError(f"perturbation {index} must be a JSON object")
    name = str(specification.get("name", "")).strip()
    kind = str(specification.get("type", "")).strip()
    if not name or not all(character.isalnum() or character in "._-" for character in name):
        raise ValueError(f"perturbation {index} has invalid name: {name!r}")
    if kind == "chain":
        steps = specification.get("steps")
        if not isinstance(steps, list) or len(steps) < 2:
            raise ValueError(f"chain perturbation {index} requires at least two steps")
        validated_steps = []
        for step_index, step in enumerate(steps):
            validated = validate_perturbation(step, step_index)
            if validated["type"] == "chain":
                raise ValueError(f"chain perturbation {index} cannot contain another chain")
            validated_steps.append(validated)
        result = dict(specification)
        result["steps"] = validated_steps
        return result
    if kind == "speed":
        factor = float(specification.get("factor", 0))
        if not 0.5 <= factor <= 1.5:
            raise ValueError(f"speed factor must be between 0.5 and 1.5: {factor}")
    elif kind == "white_noise":
        snr_db = float(specification.get("snr_db", -1))
        if not 0 <= snr_db <= 60:
            raise ValueError(f"snr_db must be between 0 and 60: {snr_db}")
    elif kind == "lowpass":
        cutoff_hz = float(specification.get("cutoff_hz", 0))
        if not 300 <= cutoff_hz < TARGET_SAMPLE_RATE / 2:
            raise ValueError(f"lowpass cutoff_hz is out of range: {cutoff_hz}")
    elif kind == "hard_clip":
        threshold = float(specification.get("threshold", 0))
        if not 0.05 <= threshold <= 1.0:
            raise ValueError(f"hard_clip threshold must be between 0.05 and 1.0: {threshold}")
    else:
        raise ValueError(f"unsupported perturbation type: {kind or '<empty>'}")
    return dict(specification)
