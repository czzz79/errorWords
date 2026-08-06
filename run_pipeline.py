"""Config driven ErrorWords experiment pipeline.

One JSON file controls the input terms and the optional pronunciation, TTS,
augmentation, ASR, report and dictionary stages.  It is deliberately an
orchestrator: model specific TTS settings remain in the existing engine JSON
files and the reusable implementation remains in ``src/error_words_tts``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from error_words_tts.asr_cli import _load_jsonl, _transcribe_manifest
from error_words_tts.augmentation.cli import run_config as run_augmentation
from error_words_tts.augmentation.audio import validate_perturbation
from error_words_tts.confusion.cli import (
    _run_dictionary_postprocess,
    _write_samples,
    clean_transcript,
    parse_gt_file,
    write_confusion_outputs,
)
from tools.prepare_asr_audio import AudioRange, find_speech_ranges, read_input, write_segments

STAGES = (
    "pronunciation",
    "tts",
    "augmentation",
    "asr_preprocess",
    "asr",
    "report",
    "dictionary_postprocess",
)
READY = {"generated", "cached"}


@dataclass
class Service:
    process: subprocess.Popen[Any]
    log: Any
    host: str = "127.0.0.1"
    port: int = 8756


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="experiment JSON")
    parser.add_argument("--dry-run", action="store_true", help="validate/expand only; never start models")
    parser.add_argument("--stages", help="comma-separated overrides, e.g. tts,augmentation,asr_preprocess,asr,report")
    args = parser.parse_args()
    try:
        run_pipeline(args.config, dry_run=args.dry_run, stage_override=args.stages)
    except Exception as exc:
        print(f"Pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def run_pipeline(config_path: Path, *, dry_run: bool = False, stage_override: str | None = None) -> None:
    config = _read_json(_path(config_path, ROOT))
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a JSON object")
    output = _required_path(config, "output_dir")
    input_txt = _required_path(config, "input_txt")
    entries = parse_gt_file(input_txt)
    stages = _stage_switches(config, stage_override)
    _validate_config(config, stages)
    _print_plan(config, entries, output, stages)
    if dry_run:
        _validate_reuse(config, output, stages)
        return

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "experiment.json", config)
    readme = config.get("output_readme")
    if isinstance(readme, str) and readme.strip():
        (output / "README.md").write_text(readme.strip() + "\n", encoding="utf-8")
    samples = output / "samples.json"
    if stages["pronunciation"]:
        pronunciation = _mapping(config.get("pronunciation", {}), "pronunciation")
        summary = _write_samples(entries, samples, str(config.get("sample_language", "Auto")), root=ROOT,
                                 output_dir=output, settings={"enabled": True, **pronunciation})
        if summary:
            print(f"Pronunciation variants: {summary['variant_count']}")
    elif stages["tts"]:
        _require_file(samples, "pronunciation disabled: reusable samples.json")

    tts_manifest = _augmentation_input_manifest(config, output)
    if stages["tts"]:
        _require_file(samples, "TTS samples")
        tts_manifest = _run_tts(config, samples, output)
    elif stages["augmentation"] or (stages["asr"] and not _asr_manifest_sources(config)):
        _require_manifest(tts_manifest, "TTS disabled: reusable tts/manifest.jsonl")

    asr_manifest = output / "asr-input-manifest.jsonl"
    if stages["augmentation"]:
        asr_manifest = _run_augmentation(config, tts_manifest, output)
    elif stages["asr"] or stages["asr_preprocess"]:
        supplied = _asr_manifest_sources(config)
        if supplied:
            asr = _mapping(config.get("asr"), "asr")
            _merge_manifests(
                supplied,
                asr_manifest,
                filter_text=_optional(asr.get("filter_text")),
                ready_only=bool(asr.get("ready_only", False)),
                expected_rows=_optional_int(asr.get("expected_input_rows"), "asr.expected_input_rows"),
            )
        elif stages["tts"]:
            # A clean TTS run is already a valid ASR input manifest.  Keep the
            # old reusable asr-input-manifest path for experiments that start
            # after augmentation, but do not require a redundant copy here.
            _require_manifest(tts_manifest, "TTS output")
            asr_manifest = tts_manifest
        else:
            _require_manifest(asr_manifest, "augmentation disabled: reusable asr-input-manifest.jsonl")

    if stages["asr_preprocess"]:
        asr_manifest = _run_asr_preprocess(config, asr_manifest, output)

    result_paths: list[dict[str, Any]] = []
    if stages["asr"]:
        result_paths = _run_asr(config, asr_manifest, output)
    elif stages["report"] or stages["dictionary_postprocess"]:
        result_paths = _existing_results(output)
        if not result_paths:
            raise FileNotFoundError("ASR disabled: no reusable results/*/run-*.jsonl")

    report: dict[str, Any] | None = None
    if stages["report"]:
        report = _build_report(entries, config, result_paths)
        report_dir = output / "report"
        report_dir.mkdir(exist_ok=True)
        _write_json(report_dir / "summary.json", report)
        _write_csv(report_dir / "summary.csv", _report_rows(report))
        _write_csv(report_dir / "group-breakdown.csv", report["group_breakdown"])
        _write_csv(report_dir / "confusion-term-hits.csv", report["confusion_term_hits"])
        _write_csv(report_dir / "ground-truth-confusion-hits.csv", report["ground_truth_confusion"]["hits"])
        _write_csv(report_dir / "ground-truth-confusion-misses.csv", report["ground_truth_confusion"]["misses"])
        cmu_summary = report.get("cmu_variant_summary", {})
        if cmu_summary:
            _write_csv(report_dir / "cmu-rule-effectiveness.csv", cmu_summary["rule_effectiveness"])
        print(f"Report: {report_dir / 'summary.json'}")
    if stages["dictionary_postprocess"]:
        # Dictionary processing consumes the normal confusion output. Build it
        # from all runs, so repeated stochastic outputs are retained.
        all_rows = [row for item in result_paths for row in _load_jsonl(item["path"])]
        words = output / "confusion-words.txt"
        write_confusion_outputs(entries, all_rows, output_txt=words, summary_path=output / "confusion-summary.json",
                                include_known_confusions=bool(config.get("include_known_confusions", False)), engine_name="pipeline")
        _run_dictionary_postprocess(root=ROOT, output_dir=output, input_path=words,
                                    settings=_mapping(config.get("dictionary_postprocess", {}), "dictionary_postprocess"))


def _run_tts(config: dict[str, Any], samples: Path, output: Path) -> Path:
    tts = _mapping(config.get("tts"), "tts")
    runs = tts.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("tts.runs must be a non-empty array when TTS is enabled")
    manifests: list[Path] = []
    for index, item in enumerate(runs):
        item = _mapping(item, f"tts.runs[{index}]")
        name = _safe_name(str(item.get("name", f"run-{index + 1}")))
        engine = _path(item.get("engine_config"), ROOT)
        _require_file(engine, f"TTS engine config for {name}")
        python_value = item.get("python")
        python = _path(python_value, ROOT) if python_value is not None else Path(sys.executable)
        _require_file(python, f"TTS Python for {name}")
        target = output / "tts" / name
        target.mkdir(parents=True, exist_ok=True)
        command = [str(python), "-m", "error_words_tts.tts.cli", "--samples", str(samples),
                   "--engine-config", str(engine), "--output-dir", str(target), "--continue-on-error"]
        _run(command)
        manifest = target / "manifest.jsonl"
        _require_manifest(manifest, f"TTS {name}")
        manifests.append(manifest)
    combined = output / "tts" / "manifest.jsonl"
    _merge_manifests(manifests, combined)
    return combined


def _run_augmentation(config: dict[str, Any], tts_manifest: Path, output: Path) -> Path:
    settings = _mapping(config.get("augmentation"), "augmentation")
    perturbations = settings.get("perturbations", [])
    include_original = bool(settings.get("include_original_audio", True))
    if not perturbations and not include_original:
        raise ValueError("augmentation needs perturbations or include_original_audio=true")
    manifests = [tts_manifest] if include_original else []
    if perturbations:
        target = output / "augmentation"
        augmentation_config = target / "config.json"
        target.mkdir(parents=True, exist_ok=True)
        _write_json(augmentation_config, {"input_manifest": str(tts_manifest), "output_dir": str(target),
                                          "seed": int(settings.get("seed", 20260717)),
                                          "continue_on_error": bool(settings.get("continue_on_error", True)),
                                          "perturbations": perturbations})
        code = run_augmentation(augmentation_config)
        manifest = target / "manifest.jsonl"
        _require_manifest(manifest, "augmentation")
        if code:
            print("WARNING: augmentation completed with row errors; usable rows are retained")
        manifests.append(manifest)
    merged = output / "asr-input-manifest.jsonl"
    _merge_manifests(manifests, merged)
    return merged


def _run_asr_preprocess(config: dict[str, Any], input_manifest: Path, output: Path) -> Path:
    """Prepare ASR audio without contacting an ASR service.

    Each generated/cached input row is validated as PCM16/16kHz/mono, then
    optionally split by the lightweight energy VAD in ``tools``.  The output
    manifest is a normal ASR manifest, with one row per segment and all source
    metadata retained.  Rows that cannot be prepared remain in the manifest as
    skipped/error rows so the ASR result count still matches the input count.
    """
    _require_manifest(input_manifest, "ASR preprocess input")
    settings = _mapping(config.get("asr_preprocess", {}), "asr_preprocess")
    enabled = bool(settings.get("enabled", True))
    if not enabled:
        # The stage switch is the primary control.  An explicitly disabled
        # block is useful when sharing a config, so preserve the input exactly.
        return input_manifest

    use_vad = bool(settings.get("use_vad", True))
    threshold = float(settings.get("threshold", 0.02))
    frame_ms = int(settings.get("frame_ms", 20))
    padding_ms = int(settings.get("padding_ms", 200))
    silence_finalize_ms = int(settings.get("silence_finalize_ms", 600))
    min_speech_ms = int(settings.get("min_speech_ms", 250))
    merge_gap_ms = int(settings.get("merge_gap_ms", 300))
    if not 0 <= threshold <= 1:
        raise ValueError("asr_preprocess.threshold must be between 0 and 1")
    for name, value in {
        "frame_ms": frame_ms,
        "padding_ms": padding_ms,
        "silence_finalize_ms": silence_finalize_ms,
        "min_speech_ms": min_speech_ms,
        "merge_gap_ms": merge_gap_ms,
    }.items():
        if value < 0 or (name == "frame_ms" and value == 0):
            raise ValueError(f"asr_preprocess.{name} must be positive/non-negative")

    target = output / "asr-preprocess"
    audio_dir = target / "audio"
    target.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    summary = {
        "schema_version": 1,
        "input_manifest": str(input_manifest),
        "processing": {
            "contract": "16kHz mono PCM16 WAV",
            "vad_enabled": use_vad,
            "speaker_diarization": "not_run",
            "audio_transform": "PCM ranges copied and wrapped in WAV; no resampling or gain change",
            "parameters": {
                "threshold": threshold,
                "frame_ms": frame_ms,
                "padding_ms": padding_ms,
                "silence_finalize_ms": silence_finalize_ms,
                "min_speech_ms": min_speech_ms,
                "merge_gap_ms": merge_gap_ms,
            },
        },
        "input_row_count": 0,
        "output_row_count": 0,
        "generated_segment_count": 0,
        "skipped_row_count": 0,
        "error_row_count": 0,
    }

    for row_index, source in enumerate(_load_jsonl(input_manifest)):
        summary["input_row_count"] += 1
        base = dict(source)
        source_status = str(source.get("status", "missing"))
        source_audio = source.get("audio_path")
        if source_status not in READY:
            base["asr_preprocess"] = {"status": "skipped", "reason": f"source status is {source_status}"}
            rows_out.append(base)
            summary["skipped_row_count"] += 1
            continue
        if not source_audio:
            base["status"] = "error"
            base["asr_preprocess"] = {"status": "error", "error": "source row has no audio_path"}
            rows_out.append(base)
            summary["error_row_count"] += 1
            continue

        input_path = Path(str(source_audio))
        if not input_path.is_absolute():
            # Historical manifests use both repository-relative paths and
            # paths relative to the manifest directory.  Mirror ASR CLI's
            # lookup, with the repository root as the final candidate.
            candidates = [input_manifest.parent / input_path,
                          input_manifest.parent.parent / input_path,
                          ROOT / input_path]
            input_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[-1]).resolve()
        try:
            pcm, info = read_input(input_path)
            if use_vad:
                ranges = find_speech_ranges(
                    pcm,
                    threshold=threshold,
                    frame_ms=frame_ms,
                    padding_ms=padding_ms,
                    silence_finalize_ms=silence_finalize_ms,
                    min_speech_ms=min_speech_ms,
                    merge_gap_ms=merge_gap_ms,
                )
                if not ranges:
                    ranges = [AudioRange(0, info.duration_ms)]
                    reason = "vad_empty_fallback_full_audio"
                else:
                    reason = "vad"
            else:
                ranges = [AudioRange(0, info.duration_ms)]
                reason = "full_audio"

            # Keep each source in its own directory.  This avoids collisions
            # when several TTS runs reuse the same sample_id/audio filename.
            sample_label = _safe_name(str(source.get("sample_id") or source.get("id") or f"row-{row_index:06d}"))
            segment_dir = audio_dir / f"{row_index:06d}-{sample_label}"
            segments = write_segments(pcm, info, segment_dir, ranges, reason=reason)
            for segment in segments:
                item = dict(source)
                item["audio_path"] = str((segment_dir / str(segment["file"])).resolve())
                item["source_audio_path"] = str(input_path.resolve())
                item["source_tts_status"] = source_status
                item["status"] = "generated"
                item["asr_preprocess"] = {
                    "status": "generated",
                    "segment_index": int(segment["index"]),
                    "start_ms": int(segment["start_ms"]),
                    "end_ms": int(segment["end_ms"]),
                    "duration_ms": int(segment["duration_ms"]),
                    "reason": str(segment["reason"]),
                    "input_sha256": info.pcm_sha256,
                }
                rows_out.append(item)
                summary["generated_segment_count"] += 1
        except (OSError, ValueError) as exc:
            base["status"] = "error"
            base["source_audio_path"] = str(input_path)
            base["asr_preprocess"] = {"status": "error", "error": str(exc)}
            rows_out.append(base)
            summary["error_row_count"] += 1

    if not rows_out:
        raise ValueError("ASR preprocess produced an empty manifest")
    summary["output_row_count"] = len(rows_out)
    manifest = target / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(target / "summary.json", summary)
    print(
        f"ASR preprocess: input={summary['input_row_count']} "
        f"segments={summary['generated_segment_count']} "
        f"skipped={summary['skipped_row_count']} errors={summary['error_row_count']} "
        f"manifest={manifest}"
    )
    return manifest


def _augmentation_input_manifest(config: dict[str, Any], output: Path) -> Path:
    """Return the TTS manifest to augment, optionally from a prior experiment."""
    settings = config.get("augmentation")
    if isinstance(settings, dict) and settings.get("input_manifest"):
        manifest = _path(settings["input_manifest"], ROOT)
        _require_manifest(manifest, "augmentation.input_manifest")
        return manifest
    return output / "tts" / "manifest.jsonl"


def _run_asr(config: dict[str, Any], manifest: Path, output: Path) -> list[dict[str, Any]]:
    _require_manifest(manifest, "ASR input")
    asr = _mapping(config.get("asr"), "asr")
    conditions = asr.get("conditions", [{"name": "default"}])
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("asr.conditions must be a non-empty array")
    default_repeats = int(asr.get("repeats", 1))
    if default_repeats < 1:
        raise ValueError("asr.repeats must be at least one")
    expected = len(_load_jsonl(manifest))
    manifest_rows = _load_jsonl(manifest)
    results: list[dict[str, Any]] = []
    for raw in conditions:
        condition = _mapping(raw, "asr condition")
        name = _safe_name(str(condition.get("name", "condition")))
        repeats = int(condition.get("repeats", default_repeats))
        if repeats < 1:
            raise ValueError(f"asr condition {name} repeats must be at least one")
        condition_dir = output / "results" / name
        condition_dir.mkdir(parents=True, exist_ok=True)
        run_fingerprint = _asr_run_fingerprint(manifest_rows, asr, condition)
        pending: list[int] = []
        for repeat in range(1, repeats + 1):
            result = condition_dir / f"run-{repeat:02d}.jsonl"
            if _valid_result(result, expected, fingerprint=run_fingerprint):
                print(f"Reusing ASR result: {result}")
                results.append(_asr_result_descriptor(asr, condition, name, repeat, result))
            else:
                pending.append(repeat)
        if not pending:
            continue
        with _asr_service(asr, condition, output / "configs", condition_dir / "service.log") as url:
            for repeat in pending:
                result = condition_dir / f"run-{repeat:02d}.jsonl"
                code = _transcribe_manifest(_load_jsonl(manifest), manifest_path=manifest, output_path=result, url=url,
                    model=str(asr.get("model", "qwen3-asr")), language=_condition_optional(asr, condition, "language"),
                    language_from_manifest=bool(_condition_value(asr, condition, "language_from_manifest", False)), prompt=_condition_optional(asr, condition, "prompt"),
                    api_key=_optional(asr.get("api_key")), timeout_seconds=float(asr.get("timeout_seconds", 180)),
                    continue_on_error=bool(asr.get("continue_on_error", True)), workers=int(asr.get("workers", 1)),
                    backend=str(asr.get("backend", "local_wsl")),
                    no_proxy=bool(asr.get("no_proxy", False)))
                rows = _load_jsonl(result)
                if len(rows) != expected:
                    raise RuntimeError(f"{result} expected {expected} rows, got {len(rows)}")
                _write_json(_asr_result_metadata_path(result), {
                    "fingerprint": run_fingerprint,
                    "input_count": expected,
                    "sampling": _sampling(condition),
                    "language": _condition_optional(asr, condition, "language"),
                    "prompt": _condition_optional(asr, condition, "prompt"),
                    "backend": str(asr.get("backend", "local_wsl")),
                    "no_proxy": bool(asr.get("no_proxy", False)),
                })
                if code:
                    print(f"WARNING: ASR {name} run {repeat} has row errors")
                results.append(_asr_result_descriptor(asr, condition, name, repeat, result))
    return results


@contextmanager
def _asr_service(asr: dict[str, Any], condition: dict[str, Any], configs: Path, log_path: Path) -> Iterator[str]:
    service = _mapping(asr.get("service", {}), "asr.service")
    mode = str(service.get("mode", "external")).lower()
    backend = str(asr.get("backend", "local_wsl")).strip() or "local_wsl"
    if backend not in {"local_wsl", "openai_http"}:
        raise ValueError(f"unsupported asr.backend: {backend}")
    host, port = str(service.get("host", "127.0.0.1")), int(service.get("port", 8756))
    url = str(asr.get("url", f"http://{host}:{port}/v1/audio/transcriptions"))
    parsed_url = urlsplit(url)
    if "url" in asr and parsed_url.hostname:
        host = parsed_url.hostname
        port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
    if mode == "external":
        _wait_port(host, port, float(asr.get("wait_seconds", 300)))
        yield url
        return
    if backend == "openai_http":
        raise ValueError("asr.service.mode must be 'external' for the openai_http backend")
    if mode != "managed":
        raise ValueError("asr.service.mode must be 'managed' or 'external'")
    if _port_open(host, port):
        raise RuntimeError(f"ASR port {host}:{port} is already in use; managed mode will not replace another service")
    base = str(service.get("base_config", "config/service-ideahub-random.yaml"))
    base_text = _wsl_read(service, base)
    effective = _replace_sampling(base_text, _sampling(condition))
    configs.mkdir(parents=True, exist_ok=True)
    generated = configs / f"service-{_safe_name(str(condition.get('name', 'condition')))}.yaml"
    generated.write_text(effective, encoding="utf-8")
    command = ["wsl.exe", "-d", str(service.get("distribution", "Ubuntu-24.04")), "--cd",
               str(service.get("working_directory", "/home/jsqdc/qwen3-asr")),
               str(service.get("python", "/home/jsqdc/miniconda3/envs/qwenasr/bin/python")), "-m",
               str(service.get("module", "qwenasr_service.main")), "--config", _windows_to_wsl(generated)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    started = Service(process, handle, host=host, port=port)
    try:
        _wait_service(started, log_path, host, port, _sampling(condition), float(asr.get("wait_seconds", 300)))
        yield url
    finally:
        _stop_service(started)


def _build_report(entries: list[Any], config: dict[str, Any], result_paths: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = _candidate_terms(entries, config)
    runs, grouped = [], []
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in result_paths:
        rows = _load_jsonl(item["path"])
        metric = _metrics(rows, candidates)
        metric.update(_prefixed_gt_metrics(_ground_truth_confusion_rows(entries, rows)))
        metric.update({
            key: (str(item[key]) if key == "path" else item[key])
            for key in ("condition", "sampling", "language", "prompt", "repeat", "path") if key in item
        })
        runs.append(metric); by_condition[str(item["condition"])].append(metric)
        for group_type, getter in _groupers().items():
            for group_value, group_rows in _partition(rows, getter).items():
                record = _metrics(group_rows, candidates)
                record.update(_prefixed_gt_metrics(_ground_truth_confusion_rows(entries, group_rows)))
                record.update({"condition": item["condition"], "repeat": item.get("repeat"),
                               "group_type": group_type, "group_value": group_value})
                grouped.append(record)
    configurations = []
    for name, values in by_condition.items():
        all_rows = [row for value in values for row in _load_jsonl(Path(value["path"]))]
        record = _metrics(all_rows, candidates)
        record.update(_prefixed_gt_metrics(_ground_truth_confusion_rows(entries, all_rows)))
        record.update({"condition": name, "sampling": values[0].get("sampling", {}),
                       "language": values[0].get("language"), "prompt": values[0].get("prompt"),
                       "repeats": len(values), "variable_audio_count": _variable_audio_count(values)})
        configurations.append(record)
    hits = [{"condition": record["condition"], **hit} for record in configurations for hit in record["matched_candidates"]]
    ground_truth = _ground_truth_confusion_metrics(entries, result_paths)
    target_keys = {_candidate_key(entry.canonical_text) for entry in entries}
    return {"terms": [entry.canonical_text for entry in entries], "candidate_lexicon": {"normalized_count": len(candidates),
            "non_target_count": len(set(candidates) - target_keys)}, "runs": runs,
            "configurations": configurations, "group_breakdown": grouped, "confusion_term_hits": hits,
            "ground_truth_confusion": ground_truth,
            "cmu_variant_summary": _cmu_variant_summary(entries, [
                row for item in result_paths for row in _load_jsonl(Path(item["path"]))
            ], candidates)}


def _ground_truth_confusion_metrics(entries: list[Any], result_paths: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure only a term's own GT-listed confusions, not a global candidate pool."""
    rows = [
        row
        for item in result_paths
        for row in _load_jsonl(Path(item["path"]))
    ]
    return _ground_truth_confusion_rows(entries, rows)


def _cmu_variant_summary(
    entries: list[Any], rows: list[dict[str, Any]], candidates: dict[str, list[str]]
) -> dict[str, Any]:
    """Summarize CMU baseline quality and per-rule mining effectiveness."""
    cmu_rows = [row for row in rows if "pronunciation_structure" in row]
    if not cmu_rows:
        return {}
    baseline = [row for row in cmu_rows if row.get("variant_kind") == "baseline"]
    variants = [row for row in cmu_rows if row.get("variant_kind") != "baseline"]
    known = {
        _ground_truth_key(entry.canonical_text): {
            _ground_truth_key(value) for value in entry.known_confusions
        }
        for entry in entries
    }
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in variants:
        by_rule[str(row.get("pronunciation_rule") or "(none)")].append(row)
    effectiveness = []
    for rule, rule_rows in sorted(by_rule.items()):
        success = gt_hit = canonical = over_distorted = 0
        for row in rule_rows:
            asr = row.get("asr") if isinstance(row.get("asr"), dict) else {}
            text = asr.get("text")
            if asr.get("status") not in {"success", "reused"} or not isinstance(text, str):
                continue
            success += 1
            comparison = row.get("comparison") if isinstance(row.get("comparison"), dict) else {}
            if comparison.get("compact_match"):
                canonical += 1
                continue
            expected = _ground_truth_key(str(row.get("expected_text", "")))
            if _ground_truth_key(text) in known.get(expected, set()):
                gt_hit += 1
            else:
                over_distorted += 1
        effectiveness.append({
            "rule": rule,
            "generated_count": len(rule_rows),
            "successful_asr_count": success,
            "canonical_event_count": canonical,
            "gt_hit_event_count": gt_hit,
            "gt_hit_rate": _rate(gt_hit, success),
            "ineffective_rate": _rate(canonical, success),
            "over_distorted_count": over_distorted,
            "over_distorted_rate": _rate(over_distorted, success),
        })
    return {
        "baseline": _metrics(baseline, candidates),
        "phoneme_variants": _metrics(variants, candidates),
        "rule_effectiveness": effectiveness,
        "asr_capabilities": {"top1": True, "nbest": False, "confidence": False, "timestamps": False},
    }


def _ground_truth_confusion_rows(
    entries: list[Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    known = {
        _ground_truth_key(entry.canonical_text): {
            _ground_truth_key(candidate): candidate for candidate in entry.known_confusions
        }
        for entry in entries
    }
    hits: Counter[tuple[str, str]] = Counter()
    present_terms: set[str] = set()
    rows_by_term: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        asr = row.get("asr") if isinstance(row.get("asr"), dict) else {}
        text = asr.get("text")
        term = str(row.get("expected_text", ""))
        term_key = _ground_truth_key(term)
        candidates = known.get(term_key, {})
        if candidates:
            present_terms.add(term_key)
            rows_by_term[term_key].append(row)
        if asr.get("status") == "success" and isinstance(text, str):
            candidate = candidates.get(_ground_truth_key(text))
            if candidate is not None:
                hits[(term, candidate)] += 1
    total = sum(len(known[term]) for term in present_terms)
    records = [
        {"term": term, "confusion": confusion, "hit_count": count}
        for (term, confusion), count in hits.most_common()
    ]
    hit_keys = {
        (_ground_truth_key(term), _ground_truth_key(confusion))
        for term, confusion in hits
    }
    misses: list[dict[str, Any]] = []
    for entry in entries:
        term_key = _ground_truth_key(entry.canonical_text)
        term_rows = rows_by_term.get(term_key, [])
        if not term_rows:
            continue
        successful = [
            row for row in term_rows
            if isinstance(row.get("asr"), dict) and row["asr"].get("status") in {"success", "reused"}
        ]
        transcripts: Counter[str] = Counter(
            str((row.get("asr") or {}).get("text", "(no transcript)"))
            for row in successful
        )
        rules = sorted({
            str(row.get("pronunciation_rule") or row.get("variant_kind") or "baseline")
            for row in term_rows
        })
        categories = sorted({
            str(row.get("confusion_category"))
            for row in term_rows if row.get("confusion_category")
        })
        canonical_outputs = sum(
            bool((row.get("comparison") or {}).get("compact_match"))
            for row in successful
        )
        top_outputs = " | ".join(
            f"{text} ({count})" for text, count in transcripts.most_common(8)
        )
        for confusion in entry.known_confusions:
            if (term_key, _ground_truth_key(confusion)) in hit_keys:
                continue
            misses.append({
                "term": entry.canonical_text,
                "confusion": confusion,
                "confusion_category": " | ".join(categories),
                "source_lines": ",".join(str(value) for value in entry.source_lines),
                "asr_sample_count": len(term_rows),
                "successful_asr_count": len(successful),
                "canonical_output_count": canonical_outputs,
                "rules_tried": " | ".join(rules),
                "top_asr_outputs": top_outputs,
            })
    return {
        "known_confusion_total": total,
        "matched_unique_confusion_count": len(hits),
        "matched_confusion_event_count": sum(hits.values()),
        "terms_with_hit_count": len({term for term, _ in hits}),
        "coverage_rate": _rate(len(hits), total),
        "hits": records,
        "misses": misses,
    }


def _prefixed_gt_metrics(value: dict[str, Any]) -> dict[str, Any]:
    return {
        f"gt_{key}": item
        for key, item in value.items()
        if key not in {"hits", "misses"}
    }


def _ground_truth_key(value: str) -> str:
    """GT comparison keeps word boundaries, matching parse_gt_file semantics."""
    return clean_transcript(value).casefold()


def _metrics(rows: list[dict[str, Any]], candidates: dict[str, list[str]]) -> dict[str, Any]:
    errors = exact = compact = 0; actuals: list[str] = []; wrong: Counter[str] = Counter(); hits: Counter[str] = Counter()
    targets = {_candidate_key(str(row.get("expected_text", ""))) for row in rows}
    for row in rows:
        asr = row.get("asr") if isinstance(row.get("asr"), dict) else {}
        if asr.get("status") == "error": errors += 1; continue
        comparison = row.get("comparison") if isinstance(row.get("comparison"), dict) else {}
        exact += bool(comparison.get("exact_match")); matched = bool(comparison.get("compact_match")); compact += matched
        text = asr.get("text")
        if isinstance(text, str):
            key = _candidate_key(text); actuals.append(key)
            if key in candidates: hits[key] += 1
            if not matched: wrong[_normalize(text)] += 1
        elif not matched: wrong["(no transcript)"] += 1
    confusion_keys = set(candidates) - targets
    return {"request_count": len(rows), "failure_count": errors, "exact_match_count": exact, "exact_match_rate": _rate(exact, len(rows)),
            "compact_match_count": compact, "compact_match_rate": _rate(compact, len(rows)), "distinct_transcript_count": len(set(actuals)),
            "top_error_transcripts": [{"transcript": v, "count": n} for v, n in wrong.most_common(10)],
            "matched_confusion_candidate_count": len(set(hits) & confusion_keys), "confusion_hit_count": sum(hits[key] for key in confusion_keys),
            "matched_candidates": [
                {"candidate": candidates[key][0], "aliases": candidates[key], "normalized_key": key,
                 "hit_count": count, "is_target_alias": key in targets}
                for key, count in hits.items()
            ]}


def _candidate_terms(entries: list[Any], config: dict[str, Any]) -> dict[str, list[str]]:
    values = [entry.canonical_text for entry in entries]
    values += [candidate for entry in entries for candidate in entry.known_confusions]
    candidate_file = config.get("candidate_terms_txt")
    if candidate_file:
        values += [line.strip() for line in _path(candidate_file, ROOT).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    grouped: dict[str, list[str]] = defaultdict(list)
    for value in values:
        key = _candidate_key(value)
        if key and value not in grouped[key]: grouped[key].append(value)
    return dict(grouped)


def _stage_switches(config: dict[str, Any], override: str | None) -> dict[str, bool]:
    settings = _mapping(config.get("stages", {}), "stages")
    selected = {name.strip() for name in override.split(",")} if override else None
    unknown = (selected or set()).difference(STAGES)
    if unknown: raise ValueError(f"unknown stages: {', '.join(sorted(unknown))}")
    return {stage: (stage in selected if selected is not None else bool(settings.get(stage, False))) for stage in STAGES}


def _validate_config(config: dict[str, Any], stages: dict[str, bool]) -> None:
    if not any(stages.values()): raise ValueError("at least one stage must be enabled")
    if stages["tts"] and not stages["pronunciation"]: pass  # samples may intentionally be reused
    if stages["asr"] and not _asr_manifest_sources(config) and not (stages["augmentation"] or stages["tts"]): pass
    if stages["asr"]:
        asr = _mapping(config.get("asr"), "asr")
        for item in asr.get("conditions", [{"name": "default"}]): _sampling(_mapping(item, "asr condition"))
    if stages["asr_preprocess"]:
        settings = _mapping(config.get("asr_preprocess", {}), "asr_preprocess")
        threshold = float(settings.get("threshold", 0.02))
        if not 0 <= threshold <= 1:
            raise ValueError("asr_preprocess.threshold must be between 0 and 1")
        for name in ("frame_ms", "padding_ms", "silence_finalize_ms", "min_speech_ms", "merge_gap_ms"):
            value = int(settings.get(name, {"frame_ms": 20, "padding_ms": 200,
                                            "silence_finalize_ms": 600, "min_speech_ms": 250,
                                            "merge_gap_ms": 300}[name]))
            if value < 0 or (name == "frame_ms" and value == 0):
                raise ValueError(f"asr_preprocess.{name} must be positive/non-negative")
    augmentation = config.get("augmentation")
    if isinstance(augmentation, dict):
        perturbations = augmentation.get("perturbations", [])
        if perturbations and not isinstance(perturbations, list):
            raise ValueError("augmentation.perturbations must be an array")
        for index, item in enumerate(perturbations):
            validate_perturbation(item, index)
    if stages["tts"]:
        for index, item in enumerate(_mapping(config.get("tts"), "tts").get("runs", [])):
            run = _mapping(item, f"tts.runs[{index}]")
            _require_file(_path(run.get("engine_config"), ROOT), f"TTS engine config for run {index + 1}")
            if run.get("python") is not None:
                _require_file(_path(run.get("python"), ROOT), f"TTS Python for run {index + 1}")


def _validate_reuse(config: dict[str, Any], output: Path, stages: dict[str, bool]) -> None:
    # Dry runs are intentionally side effect free.  Verify only artifacts that
    # must already exist because a consumer is enabled and producer is off.
    if stages["tts"] and not stages["pronunciation"]: _require_file(output / "samples.json", "reused samples")
    if stages["augmentation"] and not stages["tts"]:
        _require_manifest(_augmentation_input_manifest(config, output), "reused TTS")
    if (stages["asr"] or stages["asr_preprocess"]) and not stages["augmentation"] and not _asr_manifest_sources(config) and not stages["tts"]:
        _require_manifest(output / "asr-input-manifest.jsonl", "reused ASR input")
    if (stages["report"] or stages["dictionary_postprocess"]) and not stages["asr"] and not _existing_results(output):
        raise FileNotFoundError("reused ASR results are missing")


def _asr_manifest_sources(config: dict[str, Any]) -> list[Path]:
    asr = config.get("asr", {})
    values = asr.get("input_manifests", []) if isinstance(asr, dict) else []
    if not values: return []
    if not isinstance(values, list): raise ValueError("asr.input_manifests must be an array")
    paths = [_path(value, ROOT) for value in values]
    for path in paths: _require_manifest(path, "ASR source manifest")
    filter_text = _optional(asr.get("filter_text"))
    ready_only = bool(asr.get("ready_only", False))
    expected_rows = _optional_int(asr.get("expected_input_rows"), "asr.expected_input_rows")
    if expected_rows is not None:
        count = sum(
            1
            for path in paths
            for row in _load_jsonl(path)
            if (filter_text is None or str(row.get("text", "")) == filter_text)
            and (not ready_only or row.get("status") in READY)
        )
        if count != expected_rows:
            raise ValueError(f"ASR source manifests expected {expected_rows} rows after filter, got {count}")
    return paths


def _sampling(condition: dict[str, Any]) -> dict[str, Any]:
    result = {"temperature": float(condition.get("temperature", 0)), "top_p": float(condition.get("top_p", 1)),
              "top_k": int(condition.get("top_k", 0)), "min_p": float(condition.get("min_p", 0)), "seed": condition.get("seed")}
    if result["temperature"] < 0 or not 0 <= result["top_p"] <= 1 or result["top_k"] < 0 or not 0 <= result["min_p"] <= 1:
        raise ValueError(f"invalid sampling condition: {condition}")
    return result


def _condition_value(asr: dict[str, Any], condition: dict[str, Any], key: str, default: Any = None) -> Any:
    """Return a condition override, falling back to the experiment-wide ASR value."""
    return condition[key] if key in condition else asr.get(key, default)


def _condition_optional(asr: dict[str, Any], condition: dict[str, Any], key: str) -> str | None:
    return _optional(_condition_value(asr, condition, key))


def _replace_sampling(text: str, values: dict[str, Any]) -> str:
    for key, value in values.items():
        replacement = "null" if value is None else str(value).lower()
        text, count = re.subn(rf"(?m)^(\s*{re.escape(key)}:\s*).*?$", rf"\g<1>{replacement}", text, count=1)
        if count != 1: raise ValueError(f"ASR base config has no unique {key} field")
    return text


def _wsl_read(service: dict[str, Any], path: str) -> str:
    command = ["wsl.exe", "-d", str(service.get("distribution", "Ubuntu-24.04")), "--cd", str(service.get("working_directory", "/home/jsqdc/qwen3-asr")), "/bin/cat", path]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode: raise RuntimeError(f"could not read WSL ASR config {path}: {completed.stderr.strip()}")
    return completed.stdout


def _wait_service(service: Service, log_path: Path, host: str, port: int, sampling: dict[str, Any], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if service.process.poll() is not None: raise RuntimeError(f"WSL ASR exited early:\n{log[-4000:]}")
        if _port_open(host, port) and _sampling_in_log(log, sampling): return
        time.sleep(1)
    raise TimeoutError(f"WSL ASR did not become ready at {host}:{port}; log tail:\n{log[-2000:]}")


def _sampling_in_log(log: str, sampling: dict[str, Any]) -> bool:
    """Confirm the effective service decoding settings, independent of 0/0.0 formatting."""
    pattern = re.compile(
        r"Qwen decoding configured temperature=(\S+) top_p=(\S+) top_k=(\S+) min_p=(\S+) seed=(\S+)"
    )
    expected = (float(sampling["temperature"]), float(sampling["top_p"]), int(sampling["top_k"]),
                float(sampling["min_p"]), "None" if sampling["seed"] is None else str(sampling["seed"]))
    for match in pattern.finditer(log):
        try:
            actual = (float(match.group(1)), float(match.group(2)), int(match.group(3)), float(match.group(4)), match.group(5))
        except ValueError:
            continue
        if actual == expected:
            return True
    return False


def _stop_service(service: Service) -> None:
    try:
        if service.process.poll() is None:
            service.process.terminate()
            try: service.process.wait(timeout=20)
            except subprocess.TimeoutExpired: service.process.kill(); service.process.wait(timeout=10)
    finally: service.log.close()
    deadline = time.monotonic() + 30
    while _port_open(service.host, service.port):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"ASR port {service.host}:{service.port} remained open after stopping this run-owned service"
            )
        time.sleep(.5)


def _merge_manifests(
    paths: list[Path], target: Path, *, filter_text: str | None = None, ready_only: bool = False,
    expected_rows: int | None = None
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        count = 0
        for path in paths:
            for source_row in _load_jsonl(path):
                if filter_text is not None and str(source_row.get("text", "")) != filter_text:
                    continue
                if ready_only and source_row.get("status") not in READY:
                    continue
                row = dict(source_row)
                row.setdefault("benchmark_source_set", path.parent.name)
                row.setdefault("benchmark_source_manifest", str(path))
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    if not count: raise ValueError("combined ASR manifest is empty")
    if expected_rows is not None and count != expected_rows:
        raise ValueError(f"combined ASR manifest expected {expected_rows} rows, got {count}")


def _existing_results(output: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted((output / "results").glob("*/run-*.jsonl")):
        metadata_path = _asr_result_metadata_path(path)
        try:
            metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
        except (OSError, ValueError, json.JSONDecodeError):
            metadata = {}
        result.append({
            "condition": path.parent.name,
            "sampling": metadata.get("sampling", {}) if isinstance(metadata, dict) else {},
            "language": metadata.get("language") if isinstance(metadata, dict) else None,
            "prompt": metadata.get("prompt") if isinstance(metadata, dict) else None,
            "repeat": int(path.stem.split("-")[-1]),
            "path": path,
        })
    return result


def _asr_result_descriptor(
    asr: dict[str, Any], condition: dict[str, Any], name: str, repeat: int, path: Path
) -> dict[str, Any]:
    return {
        "condition": name,
        "sampling": _sampling(condition),
        "language": _condition_optional(asr, condition, "language"),
        "prompt": _condition_optional(asr, condition, "prompt"),
        "repeat": repeat,
        "path": path,
    }


def _asr_run_fingerprint(
    rows: list[dict[str, Any]], asr: dict[str, Any], condition: dict[str, Any]
) -> str:
    """Stable identity for an ASR result and the audio it was computed from.

    A condition directory is intentionally reusable, but only when both the
    decode settings and the ordered ASR input identity match.  This prevents a
    regenerated TTS variant from silently inheriting a transcript for an old
    WAV at the same experiment path.
    """
    identities = [
        {
            "sample_id": row.get("sample_id"),
            "text": row.get("text"),
            "audio_path": row.get("audio_path"),
            "phoneme_text": row.get("phoneme_text"),
            "input_mode": row.get("input_mode"),
            "pronunciation_variant_id": row.get("pronunciation_variant_id"),
            "pronunciation_rule": row.get("pronunciation_rule"),
            "augmentation": row.get("augmentation"),
        }
        for row in rows
    ]
    payload = {
        "inputs": identities,
        "sampling": _sampling(condition),
        "model": str(asr.get("model", "qwen3-asr")),
        "language": _condition_optional(asr, condition, "language"),
        "language_from_manifest": bool(_condition_value(asr, condition, "language_from_manifest", False)),
        "prompt": _condition_optional(asr, condition, "prompt"),
        "backend": str(asr.get("backend", "local_wsl")),
        "url": str(asr.get("url", "")),
        "no_proxy": bool(asr.get("no_proxy", False)),
    }
    import hashlib
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _asr_result_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _valid_result(path: Path, expected_rows: int, *, fingerprint: str | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        rows = _load_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    valid_rows = len(rows) == expected_rows and not any(
        isinstance(row.get("asr"), dict) and row["asr"].get("status") == "error" for row in rows
    )
    if not valid_rows:
        return False
    if fingerprint is None:
        return True
    metadata = _asr_result_metadata_path(path)
    try:
        value = _read_json(metadata)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("fingerprint") == fingerprint


def _groupers() -> dict[str, Any]:
    return {"source_set": lambda row: str(row.get("benchmark_source_set", row.get("source_manifest_path", "(none)"))),
            "prompt_or_voice": lambda row: str(row.get("tts_instruction_group") or (row.get("voice") or {}).get("name") or "(none)"),
            "augmentation": lambda row: str((row.get("augmentation") or {}).get("name", "(original)")),
            "confusion_category": lambda row: str(row.get("confusion_category") or "(canonical/none)"),
            "pronunciation_variant": lambda row: str(row.get("variant_kind") or "(canonical/none)"),
            "pronunciation_rule": lambda row: str(row.get("pronunciation_rule") or "(baseline/none)"),
            "input_mode": lambda row: str(row.get("input_mode") or "text"),
            "carrier_source": lambda row: str(
                (row.get("pronunciation_delta") or {}).get("carrier_source") or "(canonical/none)"
            )}

def _partition(rows: list[dict[str, Any]], getter: Any) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: result[getter(row)].append(row)
    return result

def _variable_audio_count(runs: list[dict[str, Any]]) -> int:
    all_values: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        for row in _load_jsonl(Path(run["path"])):
            if isinstance((asr := row.get("asr")), dict) and isinstance(asr.get("text"), str): all_values[str(row.get("audio_path"))].add(_normalize(asr["text"]))
    return sum(len(values) > 1 for values in all_values.values())

def _report_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"scope": "run", **item} for item in report["runs"]] + [{"scope": "condition", **item} for item in report["configurations"]]

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["scope"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})

def _print_plan(config: dict[str, Any], entries: list[Any], output: Path, stages: dict[str, bool]) -> None:
    condition_count = len(_mapping(config.get("asr", {}), "asr").get("conditions", [{"name": "default"}]))
    print(f"Experiment: terms={len(entries)} output={output}")
    print("Stages: " + ", ".join(f"{key}={'on' if value else 'reuse/off'}" for key, value in stages.items()))
    print(f"ASR conditions: {condition_count}")

def _run(command: list[str]) -> None:
    environment = os.environ.copy(); environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    code = subprocess.run(command, cwd=ROOT, env=environment).returncode
    if code: raise RuntimeError(f"command failed ({code}): {' '.join(command)}")

def _path(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser(); return path.resolve() if path.is_absolute() else (root / path).resolve()
def _required_path(config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip(): raise ValueError(f"config requires non-empty {key}")
    path = _path(value, ROOT); _require_file(path, key) if key == "input_txt" else None; return path
def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None: return {}
    if not isinstance(value, dict): raise ValueError(f"{name} must be an object")
    return value
def _read_json(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))
def _write_json(path: Path, value: Any) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def _require_file(path: Path, name: str) -> None:
    if not path.is_file(): raise FileNotFoundError(f"missing {name}: {path}")
def _require_manifest(path: Path, name: str) -> None:
    _require_file(path, name)
    if not _load_jsonl(path): raise ValueError(f"empty {name}: {path}")
def _optional(value: Any) -> str | None: return str(value).strip() or None if value is not None else None
def _optional_int(value: Any, name: str) -> int | None:
    if value is None: return None
    result = int(value)
    if result < 1: raise ValueError(f"{name} must be at least one")
    return result
def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-");
    if not result: raise ValueError(f"invalid name: {value!r}")
    return result
def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=.5): return True
    except OSError: return False
def _wait_port(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not _port_open(host, port):
        if time.monotonic() >= deadline: raise TimeoutError(f"ASR service did not open {host}:{port}")
        time.sleep(1)
def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve(); drive = resolved.drive.rstrip(":")
    if len(drive) != 1: raise ValueError(f"cannot map to WSL: {resolved}")
    return f"/mnt/{drive.lower()}/" + "/".join(resolved.parts[1:])
def _normalize(value: str) -> str: return unicodedata.normalize("NFKC", value).strip().casefold()
def _candidate_key(value: str) -> str: return "".join(ch for ch in _normalize(value) if ch.isalnum())
def _rate(a: int, b: int) -> float: return round(a / b, 6) if b else 0.0

if __name__ == "__main__":
    raise SystemExit(main())
