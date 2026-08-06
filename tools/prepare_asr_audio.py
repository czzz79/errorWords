"""Prepare ASR audio locally without importing or calling minutes-agent.

Input must be the realtime contract used by the Python ASR service:
16 kHz, mono, uncompressed PCM16 WAV. The script writes the VAD-selected
audio ranges as ordinary WAV files and never contacts an ASR endpoint.

Example:
    python tools/prepare_asr_audio.py input.wav --output-dir prepared
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

LOGGER = logging.getLogger("prepare_asr_audio")
SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
CHANNELS = 1


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    channels: int
    bits: int
    frames: int
    duration_ms: int
    pcm_sha256: str


@dataclass(frozen=True)
class AudioRange:
    start_ms: int
    end_ms: int


def read_input(path: Path) -> tuple[bytes, AudioInfo]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            compression = wav_file.getcomptype()
            pcm = wav_file.readframes(frames)
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"not a readable WAV file: {path}") from exc

    if compression != "NONE":
        raise ValueError("input WAV must be uncompressed PCM")
    if (sample_rate, channels, sample_width) != (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH):
        raise ValueError(
            "input must be 16kHz mono PCM16 WAV "
            f"(got rate={sample_rate}, channels={channels}, bits={sample_width * 8})"
        )
    if len(pcm) % SAMPLE_WIDTH:
        raise ValueError("input contains an incomplete PCM16 sample")

    info = AudioInfo(
        sample_rate=sample_rate,
        channels=channels,
        bits=sample_width * 8,
        frames=frames,
        duration_ms=round(frames * 1000 / sample_rate),
        pcm_sha256=_sha256(pcm),
    )
    return pcm, info


def find_speech_ranges(
    pcm: bytes,
    *,
    threshold: float,
    frame_ms: int,
    padding_ms: int,
    silence_finalize_ms: int,
    min_speech_ms: int,
    merge_gap_ms: int,
) -> list[AudioRange]:
    frame_bytes = round(SAMPLE_RATE * frame_ms / 1000) * SAMPLE_WIDTH
    active_start: int | None = None
    active_end: int | None = None
    silent_ms = 0
    ranges: list[AudioRange] = []

    for offset in range(0, len(pcm), frame_bytes):
        frame = pcm[offset:offset + frame_bytes]
        if len(frame) < SAMPLE_WIDTH:
            continue
        start_ms = round(offset / SAMPLE_WIDTH * 1000 / SAMPLE_RATE)
        end_ms = round((offset + len(frame)) / SAMPLE_WIDTH * 1000 / SAMPLE_RATE)
        is_speech = _energy_probability(frame) >= threshold
        if is_speech:
            if active_start is None:
                active_start = start_ms
            active_end = end_ms
            silent_ms = 0
            continue
        if active_start is None:
            continue
        silent_ms += end_ms - start_ms
        if silent_ms >= silence_finalize_ms:
            ranges.append(AudioRange(active_start, active_end or end_ms))
            active_start = None
            active_end = None
            silent_ms = 0

    if active_start is not None:
        ranges.append(AudioRange(active_start, active_end or round(len(pcm) / 2 * 1000 / SAMPLE_RATE)))

    duration_ms = round(len(pcm) / SAMPLE_WIDTH * 1000 / SAMPLE_RATE)
    padded = [
        AudioRange(max(0, item.start_ms - padding_ms), min(duration_ms, item.end_ms + padding_ms))
        for item in ranges
        if item.end_ms - item.start_ms >= min_speech_ms
    ]
    return _merge_ranges(padded, merge_gap_ms)


def write_segments(
    pcm: bytes,
    info: AudioInfo,
    output_dir: Path,
    ranges: list[AudioRange],
    *,
    reason: str,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    segments: list[dict[str, object]] = []
    for index, item in enumerate(ranges):
        start = round(item.start_ms * SAMPLE_RATE / 1000) * SAMPLE_WIDTH
        end = min(len(pcm), round(item.end_ms * SAMPLE_RATE / 1000) * SAMPLE_WIDTH)
        segment_pcm = pcm[start:end]
        path = output_dir / f"segment-{index:04d}.wav"
        _write_wav(path, segment_pcm, info.sample_rate)
        segments.append(
            {
                "index": index,
                "start_ms": item.start_ms,
                "end_ms": item.end_ms,
                "duration_ms": item.end_ms - item.start_ms,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path.read_bytes()),
                "reason": reason,
            }
        )
    return segments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="16kHz mono PCM16 WAV")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for output WAV files")
    parser.add_argument("--no-vad", action="store_true", help="write the complete input as one segment")
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--padding-ms", type=int, default=200)
    parser.add_argument("--silence-finalize-ms", type=int, default=600)
    parser.add_argument("--min-speech-ms", type=int, default=250)
    parser.add_argument("--merge-gap-ms", type=int, default=300)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    input_path = args.input.resolve()
    pcm, info = read_input(input_path)
    if args.no_vad:
        ranges = [AudioRange(0, info.duration_ms)]
        reason = "full_audio"
    else:
        ranges = find_speech_ranges(
            pcm,
            threshold=args.threshold,
            frame_ms=args.frame_ms,
            padding_ms=args.padding_ms,
            silence_finalize_ms=args.silence_finalize_ms,
            min_speech_ms=args.min_speech_ms,
            merge_gap_ms=args.merge_gap_ms,
        )
        if not ranges:
            LOGGER.warning("VAD found no speech; falling back to the complete input")
            ranges = [AudioRange(0, info.duration_ms)]
        reason = "vad"

    output_dir = args.output_dir.resolve()
    segments = write_segments(pcm, info, output_dir, ranges, reason=reason)
    manifest = {
        "schema_version": 1,
        "input": {"file": str(input_path), **asdict(info)},
        "processing": {
            "contract": "16kHz mono PCM16 WAV",
            "vad_enabled": not args.no_vad,
            "speaker_diarization": "not_run",
            "audio_transform": "PCM ranges copied and wrapped in WAV; no resampling or gain change",
        },
        "segments": segments,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("wrote %d WAV segment(s) to %s", len(segments), output_dir)
    return 0


def _energy_probability(pcm: bytes) -> float:
    samples = len(pcm) // SAMPLE_WIDTH
    if samples == 0:
        return 0.0
    energy = sum(abs(int.from_bytes(pcm[index:index + 2], "little", signed=True)) for index in range(0, len(pcm), 2))
    return min(1.0, energy / (samples * 32768 * 0.2))


def _merge_ranges(ranges: list[AudioRange], merge_gap_ms: int) -> list[AudioRange]:
    merged: list[AudioRange] = []
    for item in ranges:
        if not merged or item.start_ms - merged[-1].end_ms > merge_gap_ms:
            merged.append(item)
        else:
            previous = merged[-1]
            merged[-1] = AudioRange(previous.start_ms, max(previous.end_ms, item.end_ms))
    return merged


def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
