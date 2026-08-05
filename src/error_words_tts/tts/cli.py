from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from html import escape as xml_escape
from pathlib import Path
from typing import Any

from .engines import (
    AzureSpeechEngine,
    CosyVoice3Engine,
    CosyVoice3ApiEngine,
    Qwen3TtsEngine,
    TtsEngine,
    TtsEngineError,
    cosyvoice_instruction_text,
)
from .models import (
    EngineRunConfig,
    InstructionPreset,
    SpeechSynthesisRequest,
    TaggedVoice,
    TermSample,
)


_NAME_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
_RESERVED_CONFIG_KEYS = {"engine", "speakers", "voices", "voice", "instructions"}
_INSTRUCTION_PARAMETERS = {
    "qwen3-tts": {"instruct"},
    "cosyvoice3": {"instruct"},
    "cosyvoice3-api": {"instruct"},
    "azure": {"ssml_template"},
}


def main() -> int:
    args = _build_parser().parse_args()
    samples = _load_samples(Path(args.samples))
    config = _load_engine_config(Path(args.engine_config))
    engine = _build_engine(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        return _print_dry_run(samples, config)
    return _run_generation(
        samples,
        config,
        engine,
        output_dir / "manifest.jsonl",
        continue_on_error=args.continue_on_error,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TTS samples from one sample file and one engine config")
    parser.add_argument("--samples", required=True, help="JSON array containing text samples")
    parser.add_argument("--engine-config", required=True, help="Engine-native JSON configuration")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _load_samples(path: Path) -> list[TermSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("samples must be a JSON array")
    samples = [TermSample.from_mapping(item, index) for index, item in enumerate(payload)]
    if not samples:
        raise ValueError("samples file is empty")
    return samples


def _load_engine_config(path: Path) -> EngineRunConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("engine config must be a JSON object")
    engine = str(payload.get("engine", "")).strip().lower()
    if engine not in _INSTRUCTION_PARAMETERS:
        raise ValueError(f"unsupported engine in config: {engine or '<empty>'}")

    voices = _load_voices(payload, engine)
    instructions = _load_instruction_presets(payload.get("instructions"), engine)
    mode = str(payload.get("mode", "custom_voice")).strip().lower()
    if engine == "qwen3-tts" and mode == "voice_design":
        missing = [preset.name for preset in instructions if not preset.parameters.get("instruct")]
        if missing:
            raise ValueError(
                "voice_design instructions require non-empty instruct fields: " + ", ".join(missing)
            )
    settings = {key: value for key, value in payload.items() if key not in _RESERVED_CONFIG_KEYS}
    return EngineRunConfig(
        engine=engine,
        settings=settings,
        voices=voices,
        instructions=instructions,
    )


def _load_voices(payload: dict[str, Any], engine: str) -> tuple[TaggedVoice, ...]:
    if engine == "qwen3-tts":
        mode = str(payload.get("mode", "custom_voice")).strip().lower()
        if mode == "voice_design":
            if "speakers" in payload:
                raise ValueError("qwen3-tts voice_design config must not contain speakers")
            return (TaggedVoice(kind="voice_design", name="instruction-defined"),)
        if mode != "custom_voice":
            raise ValueError(f"unsupported qwen3-tts mode: {mode}")
        kind, raw_voices = "speaker", payload.get("speakers")
    elif engine == "azure":
        kind, raw_voices = "voice", payload.get("voices")
    else:
        kind, raw_voices = "reference_voice", [payload.get("voice", {"name": "reference"})]
    if not isinstance(raw_voices, list) or not raw_voices:
        raise ValueError(f"{engine} config must contain a non-empty {kind} list")
    voices = tuple(_load_tagged_voice(item, kind, index) for index, item in enumerate(raw_voices))
    return voices


def _load_tagged_voice(value: Any, kind: str, index: int) -> TaggedVoice:
    if not isinstance(value, dict):
        raise ValueError(f"{kind} at index {index} must be a JSON object")
    name = str(value.get("name", "")).strip()
    if not name:
        raise ValueError(f"{kind} at index {index} has empty name")
    return TaggedVoice(kind=kind, name=name, tags=_load_tags(value.get("tags", []), f"{kind} {name}"))


def _load_instruction_presets(value: Any, engine: str) -> tuple[InstructionPreset, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{engine} config must contain a non-empty instructions list")
    allowed = _INSTRUCTION_PARAMETERS[engine]
    presets: list[InstructionPreset] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"instruction at index {index} must be a JSON object")
        name = str(item.get("name", "")).strip()
        if not _NAME_PATTERN.fullmatch(name):
            raise ValueError(f"invalid instruction name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate instruction name: {name}")
        names.add(name)
        parameters = {key: item[key] for key in allowed if key in item and item[key] is not None}
        unknown = sorted(set(item).difference({"name", "tags", *allowed}))
        if unknown:
            raise ValueError(f"unsupported {engine} instruction fields: {', '.join(unknown)}")
        presets.append(
            InstructionPreset(
                name=name,
                tags=_load_tags(item.get("tags", []), f"instruction {name}"),
                parameters=parameters,
            )
        )
    return tuple(presets)


def _load_tags(value: Any, owner: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{owner} tags must be a JSON array")
    return tuple(str(tag).strip() for tag in value if str(tag).strip())


def _build_engine(config: EngineRunConfig) -> TtsEngine:
    settings = config.settings
    if config.engine == "qwen3-tts":
        return Qwen3TtsEngine(
            model=_optional_string(settings.get("model")),
            device_map=_optional_string(settings.get("device")),
            dtype=str(settings.get("dtype", "bfloat16")),
            mode=str(settings.get("mode", "custom_voice")),
        )
    if config.engine == "cosyvoice3":
        return CosyVoice3Engine(
            model_dir=_optional_string(settings.get("model_dir")),
            repo_dir=_optional_string(settings.get("repo_dir")),
            prompt_wav=_optional_string(settings.get("prompt_wav")),
            prompt_text=_optional_string(settings.get("prompt_text")),
            phoneme_input=settings.get("phoneme_input"),
        )
    if config.engine == "cosyvoice3-api":
        return CosyVoice3ApiEngine(
            api_url=_optional_string(settings.get("api_url")),
            model=_optional_string(settings.get("model")),
            api_protocol=str(settings.get("api_protocol", "official")),
            voice_id=_optional_string(settings.get("voice_id")),
            api_key=_optional_string(settings.get("api_key")),
            api_key_header=str(settings.get("api_key_header", "Authorization")),
            api_key_prefix=str(settings.get("api_key_prefix", "Bearer ")),
            prompt_wav=_optional_string(settings.get("prompt_wav")),
            prompt_text=_optional_string(settings.get("prompt_text")),
            sample_rate=int(settings.get("sample_rate", 24_000)),
            timeout_seconds=float(settings.get("timeout_seconds", 180.0)),
        )
    return AzureSpeechEngine(
        key=_optional_string(settings.get("key")),
        region=_optional_string(settings.get("region")),
    )


def _print_dry_run(samples: list[TermSample], config: EngineRunConfig) -> int:
    for sample in samples:
        for instruction in config.instructions:
            for voice in config.voices:
                print(json.dumps(_job_description(sample, instruction, voice, config), ensure_ascii=False))
    return 0


def _run_generation(
    samples: list[TermSample],
    config: EngineRunConfig,
    engine: TtsEngine,
    manifest_path: Path,
    continue_on_error: bool,
) -> int:
    jobs = [
        (sample, instruction, voice)
        for sample in samples
        for instruction in config.instructions
        for voice in config.voices
    ]
    total_jobs = len(jobs)
    workers = int(config.settings.get("workers", 1))
    if workers < 1:
        raise ValueError("TTS workers must be at least 1")
    if workers > 1 and config.engine not in {"cosyvoice3", "cosyvoice3-api"}:
        raise ValueError(
            "parallel TTS workers are currently supported only for cosyvoice3 and cosyvoice3-api"
        )
    batch_size = int(config.settings.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("TTS batch_size must be at least 1")
    if batch_size > 1:
        if config.engine != "qwen3-tts" or not hasattr(engine, "synthesize_batch"):
            raise ValueError("batch_size > 1 is currently supported only for qwen3-tts")
        return _run_generation_batched(
            samples,
            config,
            engine,
            manifest_path,
            continue_on_error,
            batch_size,
        )
    completed_jobs = 0
    run_started = time.perf_counter()
    status_counts: dict[str, int] = {}
    had_error = False
    stop_after_error = False
    print(f"TTS generation starting: jobs={total_jobs} workers={workers}", flush=True)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        if workers == 1:
            row_jobs = (
                (_generate_one(sample, instruction, voice, config, engine, manifest_path.parent),
                 sample, instruction, voice)
                for sample, instruction, voice in jobs
            )
            for row, sample, instruction, voice in row_jobs:
                completed_jobs, row_error = _record_generation_result(
                    manifest,
                    row=row,
                    sample=sample,
                    instruction=instruction,
                    voice=voice,
                    completed_jobs=completed_jobs,
                    total_jobs=total_jobs,
                    run_started=run_started,
                    status_counts=status_counts,
                )
                had_error = had_error or row_error
                if row_error and not continue_on_error:
                    stop_after_error = True
                    break
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cosyvoice") as executor:
                futures: dict[Future[dict[str, Any]], tuple[TermSample, InstructionPreset, TaggedVoice]] = {
                    executor.submit(
                        _generate_one,
                        sample,
                        instruction,
                        voice,
                        config,
                        engine,
                        manifest_path.parent,
                    ): (sample, instruction, voice)
                    for sample, instruction, voice in jobs
                }
                for future in as_completed(futures):
                    sample, instruction, voice = futures[future]
                    row = future.result()
                    completed_jobs, row_error = _record_generation_result(
                        manifest,
                        row=row,
                        sample=sample,
                        instruction=instruction,
                        voice=voice,
                        completed_jobs=completed_jobs,
                        total_jobs=total_jobs,
                        run_started=run_started,
                        status_counts=status_counts,
                    )
                    had_error = had_error or row_error
                    if row_error and not continue_on_error:
                        stop_after_error = True
                        for pending in futures:
                            pending.cancel()
                        break
    elapsed = time.perf_counter() - run_started
    print(
        f"TTS complete: jobs={completed_jobs} elapsed={_format_duration(elapsed)} "
        f"status={json.dumps(status_counts, ensure_ascii=False)} output={manifest_path}",
        flush=True,
    )
    return 1 if had_error or stop_after_error else 0


def _run_generation_batched(
    samples: list[TermSample],
    config: EngineRunConfig,
    engine: TtsEngine,
    manifest_path: Path,
    continue_on_error: bool,
    batch_size: int,
) -> int:
    jobs_by_group: dict[tuple[str, str], list[tuple[TermSample, InstructionPreset, TaggedVoice]]] = {}
    for sample in samples:
        for instruction in config.instructions:
            for voice in config.voices:
                jobs_by_group.setdefault((instruction.name, voice.name), []).append(
                    (sample, instruction, voice)
                )
    total_jobs = sum(len(jobs) for jobs in jobs_by_group.values())
    completed_jobs = 0
    run_started = time.perf_counter()
    status_counts: dict[str, int] = {}
    had_error = False
    print(
        f"TTS generation starting: jobs={total_jobs} workers=1 batch_size={batch_size}",
        flush=True,
    )
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for grouped_jobs in jobs_by_group.values():
            for offset in range(0, len(grouped_jobs), batch_size):
                batch_jobs = grouped_jobs[offset : offset + batch_size]
                rows: list[dict[str, Any]] = []
                requests: list[SpeechSynthesisRequest] = []
                output_paths: list[Path] = []
                fresh_indexes: list[int] = []
                for index, (sample, instruction, voice) in enumerate(batch_jobs):
                    row, output_path = _prepare_generation_row(
                        sample, instruction, voice, config, engine, manifest_path.parent
                    )
                    rows.append(row)
                    if output_path.exists():
                        row["status"] = "cached"
                    else:
                        fresh_indexes.append(index)
                        requests.append(_make_request(sample, instruction, voice, config.engine))
                        output_paths.append(output_path)

                if requests:
                    started = time.perf_counter()
                    try:
                        results = engine.synthesize_batch(requests, output_paths)  # type: ignore[attr-defined]
                        elapsed_ms = round((time.perf_counter() - started) * 1000)
                        for row_index, result in zip(fresh_indexes, results):
                            rows[row_index].update(
                                {
                                    "status": "generated",
                                    "sample_rate": result.sample_rate,
                                    "duration_ms": result.duration_ms,
                                    "metadata": result.metadata,
                                    "elapsed_ms": elapsed_ms,
                                }
                            )
                    except TtsEngineError as exc:
                        for row_index in fresh_indexes:
                            rows[row_index].update({"status": "error", "error": str(exc)})

                for row, (sample, instruction, voice) in zip(rows, batch_jobs):
                    completed_jobs, row_error = _record_generation_result(
                        manifest,
                        row=row,
                        sample=sample,
                        instruction=instruction,
                        voice=voice,
                        completed_jobs=completed_jobs,
                        total_jobs=total_jobs,
                        run_started=run_started,
                        status_counts=status_counts,
                    )
                    had_error = had_error or row_error
                    if row_error and not continue_on_error:
                        return 1
    elapsed = time.perf_counter() - run_started
    print(
        f"TTS complete: jobs={completed_jobs} elapsed={_format_duration(elapsed)} "
        f"status={json.dumps(status_counts, ensure_ascii=False)} output={manifest_path}",
        flush=True,
    )
    return 1 if had_error else 0


def _prepare_generation_row(
    sample: TermSample,
    instruction: InstructionPreset,
    voice: TaggedVoice,
    config: EngineRunConfig,
    engine: TtsEngine,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    cache_payload = {
        "text": sample.text,
        "language": sample.language,
        "engine": config.engine,
        "engine_settings": _cache_settings(config.settings),
        "voice": voice.name,
        "instruction_parameters": instruction.parameters,
        "pronunciation_instruction": sample.pronunciation_instruction,
        "phoneme_text": sample.phoneme_text,
    }
    key = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:12]
    voice_file_part = instruction.name if voice.kind == "voice_design" else voice.name
    output_path = output_dir / "audio" / engine.name / f"{sample.sample_id}__{voice_file_part}__{key}.wav"
    row = _job_description(sample, instruction, voice, config)
    row.update({"audio_path": str(output_path), "status": "generated"})
    return row, output_path


def _record_generation_result(
    manifest: Any,
    *,
    row: dict[str, Any],
    sample: TermSample,
    instruction: InstructionPreset,
    voice: TaggedVoice,
    completed_jobs: int,
    total_jobs: int,
    run_started: float,
    status_counts: dict[str, int],
) -> tuple[int, bool]:
    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest.flush()
    completed_jobs += 1
    status = str(row.get("status", "unknown"))
    status_counts[status] = status_counts.get(status, 0) + 1
    _print_generation_progress(
        completed=completed_jobs,
        total=total_jobs,
        started=run_started,
        sample=sample,
        instruction=instruction,
        voice=voice,
        status=status,
    )
    row_error = row.get("status") == "error"
    if row_error:
        print(
            f"ERROR {sample.sample_id}/{instruction.name}/{voice.name}: {row.get('error')}",
            file=sys.stderr,
        )
    return completed_jobs, row_error


def _print_generation_progress(
    *,
    completed: int,
    total: int,
    started: float,
    sample: TermSample,
    instruction: InstructionPreset,
    voice: TaggedVoice,
    status: str,
) -> None:
    elapsed = max(0.0, time.perf_counter() - started)
    remaining = elapsed / completed * (total - completed) if completed else 0.0
    percentage = completed * 100 / total if total else 100.0
    message = (
        f"TTS [{completed}/{total} {percentage:5.1f}%] "
        f"term={sample.text!r} instruction={instruction.name} voice={voice.name} "
        f"status={status} elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(remaining)}"
    )
    encoding = sys.stdout.encoding or "utf-8"
    safe = message.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe, flush=True)


def _format_duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds_part:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds_part:02d}s"
    return f"{seconds_part:d}s"


def _generate_one(
    sample: TermSample,
    instruction: InstructionPreset,
    voice: TaggedVoice,
    config: EngineRunConfig,
    engine: TtsEngine,
    output_dir: Path,
) -> dict[str, Any]:
    request = _make_request(sample, instruction, voice, config.engine)
    cache_payload = {
        "text": sample.text,
        "language": sample.language,
        "engine": config.engine,
        "engine_settings": _cache_settings(config.settings),
        "voice": voice.name,
        "instruction_parameters": instruction.parameters,
        "pronunciation_instruction": sample.pronunciation_instruction,
        # The same canonical term may have multiple phoneme variants.  Include
        # the explicit phoneme input in the cache identity so those variants
        # never share an audio file or an ASR request by accident.
        "phoneme_text": sample.phoneme_text,
    }
    key = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:12]
    voice_file_part = instruction.name if voice.kind == "voice_design" else voice.name
    output_path = output_dir / "audio" / engine.name / f"{sample.sample_id}__{voice_file_part}__{key}.wav"
    row = _job_description(sample, instruction, voice, config)
    row.update({"audio_path": str(output_path), "status": "generated"})
    if output_path.exists():
        row["status"] = "cached"
        return row
    started = time.perf_counter()
    try:
        result = engine.synthesize(request, output_path)
    except TtsEngineError as exc:
        row.update({"status": "error", "error": str(exc)})
        return row
    row.update(
        {
            "sample_rate": result.sample_rate,
            "duration_ms": result.duration_ms,
            "metadata": result.metadata,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
    )
    return row


def _make_request(
    sample: TermSample,
    instruction: InstructionPreset,
    voice: TaggedVoice,
    engine: str,
) -> SpeechSynthesisRequest:
    if engine == "azure":
        template = instruction.parameters.get("ssml_template")
        ssml = None
        if template is not None:
            ssml = str(template).format(
                text=xml_escape(sample.text),
                voice=xml_escape(voice.name, quote=True),
                locale=xml_escape(_voice_locale(voice.name), quote=True),
            )
        return SpeechSynthesisRequest(
            text=sample.text,
            language=sample.language,
            speaker=voice.name,
            ssml=ssml,
        )
    return SpeechSynthesisRequest(
        text=sample.text,
        language=sample.language,
        speaker=None if voice.kind == "voice_design" else voice.name,
        instruction=_combined_instruction(sample, instruction),
        phoneme_text=sample.phoneme_text if engine == "cosyvoice3" else None,
    )


def _job_description(
    sample: TermSample,
    instruction: InstructionPreset,
    voice: TaggedVoice,
    config: EngineRunConfig,
) -> dict[str, Any]:
    combined_instruction = _combined_instruction(sample, instruction)
    instruction_text = _effective_instruction_text(combined_instruction, config.engine)
    instruction_parameters = dict(instruction.parameters)
    if sample.pronunciation_instruction:
        instruction_parameters.update(
            {
                "pronunciation_instruction": sample.pronunciation_instruction,
                "effective_instruct": combined_instruction,
            }
        )
    row = {
        "sample_id": sample.sample_id,
        "text": sample.canonical_text or sample.source_text or sample.text,
        "source_text": sample.source_text or sample.text,
        "tts_text": sample.text,
        "input_mode": "phoneme" if sample.phoneme_text else "text",
        "text_source": sample.text_source,
        "pronunciation_processed": sample.pronunciation_processed,
        "pronunciation_rule": sample.pronunciation_rule,
        "pronunciation_variant_id": sample.pronunciation_variant_id,
        "language": sample.language,
        "sample_tags": list(sample.tags),
        "engine": config.engine,
        "voice": (
            None
            if voice.kind == "voice_design"
            else {"kind": voice.kind, "name": voice.name, "tags": list(voice.tags)}
        ),
        "instruction": {
            "name": instruction.name,
            "tags": list(instruction.tags),
            "parameters": instruction_parameters,
        },
        "tts_instruction_group": instruction.name,
        "tts_instruction_text": instruction_text,
        "prompt_version": _optional_string(config.settings.get("prompt_version")),
        "augmentation": {"name": "none", "parameters": {}},
        "source_audio_path": None,
        "source_tts_status": None,
    }
    for field, value in (
        ("target_confusions", list(sample.target_confusions)),
        ("confusion_category", sample.confusion_category),
        ("pronunciation_delta", sample.pronunciation_delta),
        ("variant_kind", sample.variant_kind),
        ("pronunciation_instruction", sample.pronunciation_instruction),
        ("phoneme_text", sample.phoneme_text),
        ("pronunciation_structure", sample.pronunciation_structure),
        ("base_pronunciation", sample.base_pronunciation),
        ("variant_pronunciation", sample.variant_pronunciation),
    ):
        if value not in (None, [], ()):
            row[field] = value
    return row


def _effective_instruction_text(
    instruction: str | None,
    engine: str,
) -> str | None:
    raw = _optional_string(instruction)
    if not raw:
        return None
    if engine in {"cosyvoice3", "cosyvoice3-api"}:
        return cosyvoice_instruction_text(raw)
    return raw


def _combined_instruction(
    sample: TermSample,
    instruction: InstructionPreset,
) -> str | None:
    base = _optional_string(instruction.parameters.get("instruct"))
    suffix = _optional_string(sample.pronunciation_instruction)
    return " ".join(value for value in (base, suffix) if value) or None


def _cache_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in settings.items()
        if key not in {"key", "region", "api_key", "workers"}
    }


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _voice_locale(voice_name: str) -> str:
    parts = voice_name.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else "en-US"


if __name__ == "__main__":
    raise SystemExit(main())
