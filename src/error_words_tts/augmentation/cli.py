from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from ..tts.audio import write_normalized_wav
from .audio import apply_perturbation, read_pcm16_mono_wav, validate_perturbation


def main() -> int:
    parser = argparse.ArgumentParser(description="Create low-cost perturbations from a TTS manifest")
    parser.add_argument(
        "--config",
        default="src/error_words_tts/augmentation/configs/chinese-confusion-pairs-low-cost.json",
    )
    args = parser.parse_args()
    return run_config(Path(args.config))


def run_config(config_path: Path) -> int:
    config = _load_config(config_path)
    root = _project_root(config_path)
    input_manifest = _resolve_path(root, config["input_manifest"])
    output_dir = _resolve_path(root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_dir / "manifest.jsonl"
    source_rows = _load_jsonl(input_manifest)
    perturbations = [
        validate_perturbation(value, index)
        for index, value in enumerate(config["perturbations"])
    ]
    names = [str(value["name"]) for value in perturbations]
    if len(names) != len(set(names)):
        raise ValueError("perturbation names must be unique")
    base_seed = int(config.get("seed", 0))
    continue_on_error = bool(config.get("continue_on_error", True))
    generated_count = 0
    error_count = 0
    total_jobs = len(source_rows) * len(perturbations)
    completed_jobs = 0
    progress_interval = max(1, total_jobs // 100)
    run_started = time.perf_counter()
    with output_manifest.open("w", encoding="utf-8") as output:
        for source_row in source_rows:
            source_path, path_error = _resolve_audio_path(
                source_row.get("audio_path"), input_manifest, root
            )
            for specification in perturbations:
                row = _base_row(source_row, source_path, specification)
                row["source_manifest_path"] = str(input_manifest.resolve())
                if source_row.get("status") not in {"generated", "cached"}:
                    row.update(
                        {
                            "status": "skipped",
                            "error": f"source TTS status is {source_row.get('status', 'missing')}",
                        }
                    )
                elif path_error:
                    row.update({"status": "error", "error": path_error})
                else:
                    try:
                        assert source_path is not None
                        row.update(
                            _render_one(
                                source_path,
                                output_dir,
                                specification,
                                base_seed=base_seed,
                            )
                        )
                    except (OSError, ValueError) as exc:
                        row.update({"status": "error", "error": str(exc)})
                if row["status"] in {"generated", "cached"}:
                    generated_count += 1
                elif row["status"] == "error":
                    error_count += 1
                    print(
                        f"Augmentation error {source_row.get('sample_id')}/{specification['name']}: "
                        f"{row['error']}",
                        file=sys.stderr,
                    )
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                completed_jobs += 1
                if (
                    completed_jobs == 1
                    or completed_jobs == total_jobs
                    or completed_jobs % progress_interval == 0
                ):
                    _print_progress(
                        completed=completed_jobs,
                        total=total_jobs,
                        started=run_started,
                        sample_id=source_row.get("sample_id"),
                        perturbation=str(specification["name"]),
                        status=str(row.get("status", "unknown")),
                    )
                if row["status"] == "error" and not continue_on_error:
                    return 1
    print(
        f"Augmentation complete: source_rows={len(source_rows)} "
        f"perturbations={len(perturbations)} audio={generated_count} "
        f"errors={error_count} output={output_manifest}"
    )
    return 1 if error_count else 0


def _print_progress(
    *,
    completed: int,
    total: int,
    started: float,
    sample_id: Any,
    perturbation: str,
    status: str,
) -> None:
    elapsed = max(0.0, time.perf_counter() - started)
    remaining = elapsed / completed * (total - completed) if completed else 0.0
    percentage = completed * 100 / total if total else 100.0
    print(
        f"Augment [{completed}/{total} {percentage:5.1f}%] "
        f"sample={sample_id} perturbation={perturbation} status={status} "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)}",
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds_part:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds_part:02d}s"
    return f"{seconds_part:d}s"


def _render_one(
    source_path: Path,
    output_dir: Path,
    specification: dict[str, Any],
    *,
    base_seed: int,
) -> dict[str, Any]:
    source_stat = source_path.stat()
    cache_payload = {
        "source": str(source_path.resolve()),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "specification": specification,
        "base_seed": base_seed,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    name = str(specification["name"])
    output_path = output_dir / "audio" / f"{source_path.stem}__{name}__{cache_key}.wav"
    if output_path.exists():
        cached_values, cached_rate = read_pcm16_mono_wav(output_path)
        return {
            "audio_path": str(output_path.resolve()),
            "status": "cached",
            "sample_rate": cached_rate,
            "duration_ms": round(len(cached_values) * 1000 / cached_rate),
        }
    values, sample_rate = read_pcm16_mono_wav(source_path)
    seed = base_seed ^ int(cache_key[:8], 16)
    perturbed = apply_perturbation(values, sample_rate, specification, seed=seed)
    target_rate, duration_ms = write_normalized_wav(perturbed, sample_rate, output_path)
    return {
        "audio_path": str(output_path.resolve()),
        "status": "generated",
        "sample_rate": target_rate,
        "duration_ms": duration_ms,
    }


def _base_row(
    source_row: dict[str, Any],
    source_path: Path | None,
    specification: dict[str, Any],
) -> dict[str, Any]:
    row = dict(source_row)
    row.update(
        {
            "source_audio_path": str(source_path.resolve()) if source_path is not None else None,
            "source_tts_status": source_row.get("status"),
            "augmentation": {
                "name": specification["name"],
                "type": specification["type"],
                "parameters": {
                    key: value
                    for key, value in specification.items()
                    if key not in {"name", "type"}
                },
            },
            "status": "pending",
        }
    )
    row.pop("error", None)
    return row


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("augmentation config must be a JSON object")
    for field in ("input_manifest", "output_dir"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"augmentation config requires non-empty {field}")
    perturbations = payload.get("perturbations")
    if not isinstance(perturbations, list) or not perturbations:
        raise ValueError("augmentation config requires a non-empty perturbations list")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"input manifest is empty: {path}")
    return rows


def _resolve_audio_path(
    value: Any,
    manifest_path: Path,
    root: Path,
) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, "source row has no audio_path"
    path = Path(value)
    candidates = [path] if path.is_absolute() else [root / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), None
    return path, f"source audio file does not exist: {path}"


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
