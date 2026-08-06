from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .asr_cli import _load_jsonl, _transcribe_manifest, _wait_for_service


DEFAULT_SERVICE_URL = "http://127.0.0.1:8756/v1/audio/transcriptions"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe TTS output directories listed in one JSON config"
    )
    parser.add_argument(
        "--config",
        default="asr_configs/ideahub-baselines.json",
        help="Batch ASR JSON config",
    )
    args = parser.parse_args()
    return run_config(Path(args.config))


def run_config(config_path: Path) -> int:
    config = _load_config(config_path)
    root = _project_root(config_path)
    url = str(config.get("url", DEFAULT_SERVICE_URL)).strip()
    backend = str(config.get("backend", "local_wsl")).strip() or "local_wsl"
    if backend not in {"local_wsl", "openai_http"}:
        raise ValueError(f"unsupported ASR backend: {backend}")
    model = str(config.get("model", "qwen3-asr")).strip()
    output_name = str(config.get("output_name", "asr-results.jsonl")).strip()
    raw_output_directory = _optional_string(config.get("output_directory"))
    output_directory = (
        _resolve_path(root, raw_output_directory) if raw_output_directory else None
    )
    language = _optional_string(config.get("language"))
    prompt = _optional_string(config.get("prompt"))
    api_key = _optional_string(config.get("api_key"))
    language_from_manifest = bool(config.get("language_from_manifest", False))
    timeout_seconds = float(config.get("timeout_seconds", 180.0))
    wait_seconds = float(config.get("wait_seconds", 5.0))
    continue_on_error = bool(config.get("continue_on_error", True))
    workers = int(config.get("workers", 1))
    if workers < 1:
        raise ValueError("ASR workers must be at least 1")

    if not url:
        raise ValueError("ASR config url must not be empty")
    if not model:
        raise ValueError("ASR config model must not be empty")
    if not output_name or Path(output_name).name != output_name:
        raise ValueError("ASR config output_name must be a file name, not a path")

    if wait_seconds > 0:
        try:
            _wait_for_service(url, wait_seconds)
        except TimeoutError as exc:
            print(f"ASR service error: {exc}", file=sys.stderr)
            return 1

    had_error = False
    inputs = config["audio_directories"]
    for index, raw_directory in enumerate(inputs, start=1):
        directory = _resolve_path(root, raw_directory)
        manifest_path = directory / "manifest.jsonl"
        output_path = (
            output_directory / directory.name / output_name
            if output_directory is not None
            else directory / output_name
        )
        print(f"[{index}/{len(inputs)}] ASR directory: {directory}")

        if not directory.is_dir():
            print(f"ASR input directory does not exist: {directory}", file=sys.stderr)
            had_error = True
            if not continue_on_error:
                return 1
            continue
        if not manifest_path.is_file():
            print(f"TTS manifest does not exist: {manifest_path}", file=sys.stderr)
            had_error = True
            if not continue_on_error:
                return 1
            continue

        try:
            rows = _load_jsonl(manifest_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            exit_code = _transcribe_manifest(
                rows,
                manifest_path=manifest_path,
                output_path=output_path,
                url=url,
                model=model,
                language=language,
                language_from_manifest=language_from_manifest,
                prompt=prompt,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                continue_on_error=continue_on_error,
                workers=workers,
                backend=backend,
                no_proxy=bool(config.get("no_proxy", False)),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"ASR directory failed: {directory}: {exc}", file=sys.stderr)
            exit_code = 1

        if exit_code != 0:
            had_error = True
            if not continue_on_error:
                return 1

    return 1 if had_error else 0


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ASR batch config must be a JSON object")
    inputs = payload.get("audio_directories")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("ASR batch config must contain a non-empty audio_directories list")
    if any(not isinstance(item, str) or not item.strip() for item in inputs):
        raise ValueError("every audio_directories item must be a non-empty string")
    payload["audio_directories"] = [item.strip() for item in inputs]
    return payload


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for parent in resolved.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return resolved.parent


if __name__ == "__main__":
    raise SystemExit(main())
