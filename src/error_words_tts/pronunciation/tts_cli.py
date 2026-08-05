from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from ..tts.cli import _build_engine, _load_engine_config
from ..tts.engines import TtsEngine, TtsEngineError
from ..tts.models import EngineRunConfig, SpeechSynthesisRequest
from .generator import PronunciationError


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render text-approximated pronunciation variants with Qwen3-TTS"
    )
    parser.add_argument(
        "--config",
        default="src/error_words_tts/pronunciation/configs/ideahub-qwen-tts.json",
        help="Pronunciation TTS config",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without loading TTS")
    args = parser.parse_args()
    return run_config(Path(args.config), dry_run=args.dry_run)


def run_config(
    config_path: Path,
    *,
    dry_run: bool = False,
    engine: TtsEngine | None = None,
) -> int:
    config = _load_config(config_path)
    root = _project_root(config_path)
    variants_path = _resolve_path(root, config["variants"])
    engine_config_path = _resolve_path(root, config["engine_config"])
    output_dir = _resolve_path(root, config["output_dir"])
    variants = _load_variants(variants_path)
    engine_config = _load_engine_config(engine_config_path)
    _validate_engine_config(engine_config)

    allowed = set(config.get("renderabilities", ["text_approximation"]))
    selected = [row for row in variants if row.get("tts_renderability") in allowed]
    if not selected:
        raise PronunciationError(
            f"no pronunciation variants match renderabilities: {', '.join(sorted(allowed))}"
        )
    guard = str(config.get("instruction_suffix", "")).strip()
    jobs = list(_build_jobs(selected, engine_config, guard))
    if dry_run:
        for job in jobs:
            print(json.dumps(job, ensure_ascii=False))
        print(f"Pronunciation TTS dry-run: variants={len(selected)} jobs={len(jobs)}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    tts_engine = engine or _build_engine(engine_config)
    return _run_jobs(
        jobs,
        engine_config=engine_config,
        engine=tts_engine,
        output_dir=output_dir,
        continue_on_error=bool(config.get("continue_on_error", True)),
        variants_path=variants_path,
    )


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PronunciationError("pronunciation TTS config must be a JSON object")
    for field in ("variants", "engine_config", "output_dir"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise PronunciationError(f"pronunciation TTS config needs non-empty {field}")
    renderabilities = payload.get("renderabilities", ["text_approximation"])
    if not isinstance(renderabilities, list) or not renderabilities:
        raise PronunciationError("renderabilities must be a non-empty JSON array")
    unknown = set(renderabilities).difference({"text_approximation"})
    if unknown:
        raise PronunciationError(f"unsupported renderabilities: {', '.join(sorted(unknown))}")
    return payload


def _load_variants(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise PronunciationError(f"{path}:{line_number} must be a JSON object")
            for field in (
                "variant_id",
                "sample_id",
                "canonical_text",
                "display_text",
                "language",
                "tts_renderability",
                "rule",
            ):
                if field not in row:
                    raise PronunciationError(f"{path}:{line_number} is missing {field}")
            rows.append(row)
    if not rows:
        raise PronunciationError(f"pronunciation variants file is empty: {path}")
    return rows


def _validate_engine_config(config: EngineRunConfig) -> None:
    if config.engine != "qwen3-tts":
        raise PronunciationError("pronunciation TTS currently supports only qwen3-tts")
    if str(config.settings.get("mode", "")).strip().lower() != "voice_design":
        raise PronunciationError("pronunciation TTS requires qwen3-tts mode=voice_design")


def _build_jobs(
    variants: list[dict[str, Any]],
    engine_config: EngineRunConfig,
    instruction_suffix: str,
):
    for variant in variants:
        for profile in engine_config.instructions:
            base_instruction = str(profile.parameters.get("instruct", "")).strip()
            effective = " ".join(part for part in (base_instruction, instruction_suffix) if part)
            yield {
                "sample_id": str(variant["sample_id"]),
                "text": str(variant["canonical_text"]),
                "canonical_text": str(variant["canonical_text"]),
                "tts_text": str(variant["display_text"]),
                "language": str(variant["language"]),
                "sample_tags": list(variant.get("sample_tags", [])),
                "engine": engine_config.engine,
                "voice": None,
                "instruction": {
                    "name": profile.name,
                    "tags": list(profile.tags),
                    "parameters": {
                        "base_instruct": base_instruction,
                        "pronunciation_guard": instruction_suffix or None,
                        "effective_instruct": effective,
                    },
                },
                "pronunciation_variant_id": str(variant["variant_id"]),
                "pronunciation_rule": variant["rule"],
                "base_pronunciation": variant.get("base_pronunciation"),
                "variant_pronunciation": variant.get("variant_pronunciation"),
                "tts_renderability": variant["tts_renderability"],
                "render_method": "display_text",
            }


def _run_jobs(
    jobs: list[dict[str, Any]],
    *,
    engine_config: EngineRunConfig,
    engine: TtsEngine,
    output_dir: Path,
    continue_on_error: bool,
    variants_path: Path,
) -> int:
    manifest_path = output_dir / "manifest.jsonl"
    had_error = False
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for job in jobs:
            row = _render_one(job, engine_config, engine, output_dir)
            row["pronunciation_variants_path"] = str(variants_path.resolve())
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest.flush()
            if row["status"] == "error":
                had_error = True
                print(
                    f"ERROR {job['pronunciation_variant_id']}/{job['instruction']['name']}: "
                    f"{row['error']}",
                    file=sys.stderr,
                )
                if not continue_on_error:
                    return 1
    print(f"Pronunciation TTS complete: jobs={len(jobs)} output={manifest_path}")
    return 1 if had_error else 0


def _render_one(
    job: dict[str, Any],
    engine_config: EngineRunConfig,
    engine: TtsEngine,
    output_dir: Path,
) -> dict[str, Any]:
    effective_instruction = job["instruction"]["parameters"]["effective_instruct"]
    cache_payload = {
        "tts_text": job["tts_text"],
        "language": job["language"],
        "engine": engine_config.engine,
        "engine_settings": engine_config.settings,
        "effective_instruction": effective_instruction,
    }
    key = hashlib.sha256(
        json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    sample_name = _safe_file_part(job["sample_id"])
    profile_name = _safe_file_part(job["instruction"]["name"])
    audio_path = output_dir / "audio" / engine.name / f"{sample_name}__{profile_name}__{key}.wav"
    row = dict(job)
    row.update({"audio_path": str(audio_path.resolve()), "status": "generated"})
    if audio_path.exists():
        row["status"] = "cached"
        return row

    request = SpeechSynthesisRequest(
        text=job["tts_text"],
        language=job["language"],
        speaker=None,
        instruction=effective_instruction,
    )
    started = time.perf_counter()
    try:
        result = engine.synthesize(request, audio_path)
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


def _safe_file_part(value: Any) -> str:
    cleaned = _SAFE_NAME.sub("-", str(value)).strip("-.")
    return cleaned or "item"


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for parent in resolved.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return resolved.parent


if __name__ == "__main__":
    raise SystemExit(main())
