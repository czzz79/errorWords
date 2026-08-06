from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..asr_cli import _load_jsonl, _transcribe_manifest, _wait_for_service
from ..augmentation.cli import run_config as run_augmentation
from ..pronunciation.generator import (
    PronunciationError,
    PronunciationSample,
    arpabet_to_cosyvoice,
    generate_variants,
    load_rules,
)
from ..pronunciation.gt_directed import (
    PREPROCESSABLE_CATEGORIES,
    add_homophone_metadata,
    build_gt_directed_variants,
    classify_entries,
    write_classification_outputs,
)
from ..pronunciation.fuzzy_pinyin import build_fuzzy_pinyin_variants
from ..pronunciation.english_reading import build_english_reading_variants
from ..pronunciation.english_cmu import build_english_cmu_samples
from .dictionary_postprocess import (
    DEFAULT_ENGLISH_DICTIONARY,
    DEFAULT_JIEBA_DICTIONARY,
    DEFAULT_STOPWORD_DICTIONARY,
    DEFAULT_WUSONG_DICTIONARY,
    MIN_JIEBA_CHARACTER_FREQUENCY,
    MIN_WUSONG_WEIGHT,
    process as run_dictionary_postprocess,
)


DEFAULT_CONFIG = "src/error_words_tts/confusion/configs/cosyvoice3.json"


@dataclass(slots=True)
class GtEntry:
    sample_id: str
    canonical_text: str
    known_confusions: list[str]
    source_lines: list[int]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate ASR confusion words for every GT in a text file"
    )
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG, help="Pipeline JSON config")
    parser.add_argument("--dry-run", action="store_true", help="Prepare inputs without loading models")
    args = parser.parse_args()
    return run_config(Path(args.config), dry_run=args.dry_run)


def run_config(config_path: Path, *, dry_run: bool = False) -> int:
    config = _load_config(config_path)
    root = _project_root(config_path)
    input_path = _resolve_path(root, config["input_txt"])
    engine_config = _resolve_path(root, config["engine_config"])
    output_dir = _resolve_path(root, config["output_dir"])
    entries = parse_gt_file(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.json"
    pronunciation_summary = _write_samples(
        entries,
        samples_path,
        str(config.get("sample_language", "Auto")),
        root=root,
        output_dir=output_dir,
        settings=_as_mapping(config.get("pronunciation", {}), "pronunciation"),
    )
    engine_name = _read_engine_name(engine_config)
    print(
        f"GT input ready: terms={len(entries)} engine={engine_name} "
        f"samples={samples_path}"
    )
    if pronunciation_summary is not None:
        print(
            "Pronunciation preprocessing complete: "
            f"variants={pronunciation_summary['variant_count']} "
            f"skipped={pronunciation_summary['skipped_count']} "
            f"errors={pronunciation_summary['error_count']}"
        )
    if dry_run:
        return 0

    tts_dir = output_dir / "tts"
    tts_manifest = tts_dir / "manifest.jsonl"
    asr_settings = _as_mapping(config.get("asr", {}), "asr")
    asr_backend = str(asr_settings.get("backend", "local_wsl")).strip() or "local_wsl"
    if asr_backend not in {"local_wsl", "openai_http"}:
        raise ValueError(f"unsupported asr.backend: {asr_backend}")
    service_settings = _as_mapping(asr_settings.get("service", {}), "asr.service")
    manage_service = bool(service_settings.get("manage", asr_backend == "local_wsl"))
    if asr_backend == "openai_http" and manage_service:
        raise ValueError("asr.service.manage must be false for the openai_http backend")
    service_started = False
    service_stopped = False

    try:
        if manage_service:
            _stop_asr_service(service_settings)
            service_stopped = True

        tts_exit = _run_module(
            root,
            "error_words_tts.tts.cli",
            [
                "--samples",
                str(samples_path),
                "--engine-config",
                str(engine_config),
                "--output-dir",
                str(tts_dir),
                "--continue-on-error",
            ],
        )
        _require_usable_manifest(tts_manifest, "TTS", tts_exit)

        manifests = [tts_manifest] if bool(config.get("include_original_audio", True)) else []
        augmentation = _as_mapping(config.get("augmentation", {}), "augmentation")
        if bool(augmentation.get("enabled", True)):
            augmentation_dir = output_dir / "augmentation"
            augmentation_config = output_dir / "augmentation-config.json"
            _write_augmentation_config(
                augmentation_config,
                tts_manifest=tts_manifest,
                output_dir=augmentation_dir,
                settings=augmentation,
            )
            augmentation_exit = run_augmentation(augmentation_config)
            augmentation_manifest = augmentation_dir / "manifest.jsonl"
            _require_usable_manifest(augmentation_manifest, "augmentation", augmentation_exit)
            manifests.append(augmentation_manifest)

        if not manifests:
            raise ValueError("at least one of include_original_audio or augmentation.enabled must be true")
        asr_manifest = output_dir / "asr-input-manifest.jsonl"
        _merge_manifests(manifests, asr_manifest)

        if manage_service:
            _start_asr_service(service_settings, output_dir / "asr-service.log")
            service_started = True

        url = str(
            asr_settings.get(
                "url", "http://127.0.0.1:8756/v1/audio/transcriptions"
            )
        ).strip()
        _wait_for_service(url, float(asr_settings.get("wait_seconds", 300)))
        asr_rows = _load_jsonl(asr_manifest)
        asr_output = output_dir / "asr-results.jsonl"
        asr_exit = _transcribe_manifest(
            asr_rows,
            manifest_path=asr_manifest,
            output_path=asr_output,
            url=url,
            model=str(asr_settings.get("model", "qwen3-asr")),
            language=_optional_string(asr_settings.get("language")),
            language_from_manifest=bool(asr_settings.get("language_from_manifest", False)),
            prompt=_optional_string(asr_settings.get("prompt")),
            api_key=_optional_string(asr_settings.get("api_key")),
            timeout_seconds=float(asr_settings.get("timeout_seconds", 180)),
            continue_on_error=bool(asr_settings.get("continue_on_error", True)),
            workers=int(asr_settings.get("workers", 1)),
            backend=asr_backend,
            no_proxy=bool(asr_settings.get("no_proxy", False)),
        )
        results = _load_jsonl(asr_output)
        output_txt = output_dir / "confusion-words.txt"
        summary_path = output_dir / "summary.json"
        write_confusion_outputs(
            entries,
            results,
            output_txt=output_txt,
            summary_path=summary_path,
            include_known_confusions=bool(config.get("include_known_confusions", False)),
            engine_name=engine_name,
        )
        dictionary_settings = _as_mapping(
            config.get("dictionary_postprocess", {}), "dictionary_postprocess"
        )
        if bool(dictionary_settings.get("enabled", False)):
            _run_dictionary_postprocess(
                root=root,
                output_dir=output_dir,
                input_path=output_txt,
                settings=dictionary_settings,
            )
        print(
            f"Confusion generation complete: terms={len(entries)} "
            f"output={output_txt} summary={summary_path}"
        )
        return asr_exit
    finally:
        if manage_service and service_stopped and not service_started:
            try:
                _start_asr_service(service_settings, output_dir / "asr-service.log")
                print("ASR service was restored after an interrupted TTS run")
            except (OSError, ValueError) as exc:
                print(f"WARNING: failed to restore ASR service: {exc}", file=sys.stderr)


def parse_gt_file(path: Path) -> list[GtEntry]:
    if not path.is_file():
        raise ValueError(f"GT input file does not exist: {path}")
    merged: dict[str, GtEntry] = {}
    order: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = [part.strip() for part in stripped.split("|")]
        canonical = fields[0]
        known = [part for part in fields[1:] if part]
        key = _candidate_key(canonical)
        if not key:
            raise ValueError(f"{path}:{line_number} has an empty GT")
        if key not in merged:
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
            merged[key] = GtEntry(
                sample_id=f"gt-{len(order) + 1:04d}-{digest}",
                canonical_text=canonical,
                known_confusions=[],
                source_lines=[],
            )
            order.append(key)
        entry = merged[key]
        entry.source_lines.append(line_number)
        seen = {_candidate_key(value) for value in entry.known_confusions}
        for candidate in known:
            candidate_key = _candidate_key(candidate)
            if candidate_key and candidate_key != key and candidate_key not in seen:
                entry.known_confusions.append(candidate)
                seen.add(candidate_key)
    if not order:
        raise ValueError(f"GT input file contains no terms: {path}")
    return [merged[key] for key in order]


def write_confusion_outputs(
    entries: list[GtEntry],
    results: list[dict[str, Any]],
    *,
    output_txt: Path,
    summary_path: Path,
    include_known_confusions: bool,
    engine_name: str,
) -> None:
    by_id = {entry.sample_id: entry for entry in entries}
    frequencies: dict[str, Counter[str]] = {entry.sample_id: Counter() for entry in entries}
    displays: dict[str, dict[str, str]] = {entry.sample_id: {} for entry in entries}
    first_seen: dict[str, dict[str, int]] = {entry.sample_id: {} for entry in entries}
    successful: Counter[str] = Counter()

    for result_index, row in enumerate(results):
        sample_id = str(row.get("sample_id", ""))
        entry = by_id.get(sample_id)
        if entry is None:
            continue
        asr = row.get("asr", {})
        if not isinstance(asr, dict) or asr.get("status") not in {"success", "reused"}:
            continue
        successful[sample_id] += 1
        candidate = clean_transcript(str(asr.get("text", "")))
        key = _candidate_key(candidate)
        if not key or key == _candidate_key(entry.canonical_text):
            continue
        frequencies[sample_id][key] += 1
        displays[sample_id].setdefault(key, candidate)
        first_seen[sample_id].setdefault(key, result_index)

    text_lines: list[str] = []
    term_summaries: list[dict[str, Any]] = []
    for entry in entries:
        ranked_keys = sorted(
            frequencies[entry.sample_id],
            key=lambda key: (-frequencies[entry.sample_id][key], first_seen[entry.sample_id][key]),
        )
        generated = [displays[entry.sample_id][key] for key in ranked_keys]
        output_candidates = list(generated)
        if include_known_confusions:
            seen = {_candidate_key(value) for value in output_candidates}
            for candidate in entry.known_confusions:
                key = _candidate_key(candidate)
                if key not in seen:
                    output_candidates.append(candidate)
                    seen.add(key)
        text_lines.append(_format_confusion_line(entry.canonical_text, output_candidates))

        known_by_key = {_candidate_key(value): value for value in entry.known_confusions}
        matched_keys = [key for key in ranked_keys if key in known_by_key]
        term_summaries.append(
            {
                "sample_id": entry.sample_id,
                "canonical_text": entry.canonical_text,
                "source_lines": entry.source_lines,
                "successful_asr_count": successful[entry.sample_id],
                "generated_confusions": [
                    {
                        "text": displays[entry.sample_id][key],
                        "count": frequencies[entry.sample_id][key],
                    }
                    for key in ranked_keys
                ],
                "known_confusions": entry.known_confusions,
                "matched_known_confusions": [known_by_key[key] for key in matched_keys],
                "known_coverage": (
                    len(matched_keys) / len(known_by_key) if known_by_key else None
                ),
            }
        )

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    total_generated = sum(len(item["generated_confusions"]) for item in term_summaries)
    total_known = sum(len(item["known_confusions"]) for item in term_summaries)
    total_matched = sum(len(item["matched_known_confusions"]) for item in term_summaries)
    summary = {
        "engine": engine_name,
        "term_count": len(entries),
        "generated_confusion_count": total_generated,
        "known_confusion_count": total_known,
        "matched_known_confusion_count": total_matched,
        "known_coverage": total_matched / total_known if total_known else None,
        "include_known_confusions_in_txt": include_known_confusions,
        "terms": term_summaries,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_transcript(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    without_punctuation = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(without_punctuation.split())


def _format_confusion_line(canonical: str, candidates: list[str]) -> str:
    return "|".join([canonical, *candidates])


def _candidate_key(value: str) -> str:
    return clean_transcript(value).casefold()


def compare_confusion_files(
    all_confusions_path: Path | str,
    hit_confusions_path: Path | str,
) -> dict[str, Any]:
    """Compare all expected confusion words with the words hit by ASR.

    Both files use the format ``canonical|candidate1|candidate2``. The first
    file is the complete set to measure; the second file is the observed ASR
    output set. Matching is done per canonical term and ignores punctuation,
    surrounding whitespace, and case.
    """
    all_entries = parse_gt_file(Path(all_confusions_path))
    hit_entries = parse_gt_file(Path(hit_confusions_path))
    hit_by_canonical = {
        _candidate_key(entry.canonical_text): {
            _candidate_key(candidate): candidate
            for candidate in entry.known_confusions
        }
        for entry in hit_entries
    }

    term_results: list[dict[str, Any]] = []
    total_candidates = 0
    total_matched = 0
    terms_with_hits = 0

    for entry in all_entries:
        all_by_key = {
            _candidate_key(candidate): candidate
            for candidate in entry.known_confusions
        }
        observed_by_key = hit_by_canonical.get(_candidate_key(entry.canonical_text), {})
        matched_keys = [key for key in all_by_key if key in observed_by_key]
        missing_keys = [key for key in all_by_key if key not in observed_by_key]
        matched = [all_by_key[key] for key in matched_keys]
        missing = [all_by_key[key] for key in missing_keys]

        total_candidates += len(all_by_key)
        total_matched += len(matched)
        if matched:
            terms_with_hits += 1

        term_results.append(
            {
                "canonical": entry.canonical_text,
                "total_count": len(all_by_key),
                "matched_count": len(matched),
                "hit_rate": len(matched) / len(all_by_key) if all_by_key else None,
                "matched": matched,
                "missing": missing,
            }
        )

    return {
        "all_file": str(Path(all_confusions_path)),
        "hit_file": str(Path(hit_confusions_path)),
        "term_count": len(all_entries),
        "total_confusion_count": total_candidates,
        "matched_confusion_count": total_matched,
        "hit_rate": total_matched / total_candidates if total_candidates else None,
        "terms_with_hits": terms_with_hits,
        "terms_without_hits": len(all_entries) - terms_with_hits,
        "terms": term_results,
    }


def _write_samples(
    entries: list[GtEntry],
    path: Path,
    language: str,
    *,
    root: Path,
    output_dir: Path,
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    mode = str(settings.get("mode", "rule_based")).strip().casefold()
    if bool(settings.get("enabled", False)) and mode == "english_cmu":
        rows, summary = build_english_cmu_samples(
            entries, root=root, output_dir=output_dir, settings=settings
        )
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary
    rows = (
        [_canonical_sample_row(entry, language) for entry in entries]
        if bool(settings.get("include_canonical", True))
        else []
    )
    if not bool(settings.get("enabled", False)):
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return None

    if mode == "gt_directed":
        classification_rows = classify_entries(entries)
        add_homophone_metadata(rows, classification_rows)

    variants, summary = _generate_pronunciation_variants(
        entries,
        language=language,
        root=root,
        output_dir=output_dir,
        settings=settings,
    )
    requested_origins = settings.get("include_variant_origins")
    if requested_origins is not None:
        if not isinstance(requested_origins, list) or not requested_origins:
            raise ValueError("pronunciation.include_variant_origins must be a non-empty array")
        allowed = {str(value).strip() for value in requested_origins if str(value).strip()}
        if not allowed:
            raise ValueError("pronunciation.include_variant_origins must contain a non-empty origin")
        variants = [
            row for row in variants
            if _variant_origin_matches(
                str((row.get("pronunciation_delta") or {}).get("origin", "")), allowed
            )
        ]
        summary["variant_count"] = len(variants)
        summary["included_variant_origins"] = sorted(allowed)
    rows.extend(variants)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _variant_origin_matches(origin: str, allowed: set[str]) -> bool:
    """Match a requested origin even when GT observation metadata is appended.

    Fuzzy-pinyin variants retain ``+gt`` in their origin to record that the
    directional change was observed in the supplied GT set.  That suffix is
    metadata, not a separate generator, so filters such as
    ``gt_target_component_tone`` must include both forms.
    """
    return origin in allowed or origin.removesuffix("+gt") in allowed


def _canonical_sample_row(entry: GtEntry, language: str) -> dict[str, Any]:
    return {
        "id": entry.sample_id,
        "text": entry.canonical_text,
        "canonical_text": entry.canonical_text,
        "tts_text": entry.canonical_text,
        "source_text": entry.canonical_text,
        "text_source": "canonical",
        "pronunciation_processed": False,
        "pronunciation_rule": None,
        "pronunciation_variant_id": None,
        "language": language,
        "tags": ["gt", f"source-lines:{','.join(map(str, entry.source_lines))}"],
    }


def _generate_pronunciation_variants(
    entries: list[GtEntry],
    *,
    language: str,
    root: Path,
    output_dir: Path,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode = str(settings.get("mode", "rule_based")).strip().casefold()
    if mode == "gt_directed":
        return _generate_gt_directed_variants(
            entries,
            root=root,
            output_dir=output_dir,
            settings=settings,
        )
    if mode == "fuzzy_pinyin":
        return _generate_fuzzy_pinyin_variants(
            entries,
            root=root,
            output_dir=output_dir,
            settings=settings,
        )
    if mode == "english_reading":
        return _generate_english_reading_variants(entries, output_dir=output_dir, settings=settings)
    if mode == "english_cmu":
        return _generate_english_cmu_variants(
            entries, root=root, output_dir=output_dir, settings=settings
        )
    if mode != "rule_based":
        raise ValueError(
            "pronunciation.mode must be 'rule_based', 'gt_directed', 'fuzzy_pinyin', "
            "'english_reading', or 'english_cmu'"
        )
    rule_files = settings.get("rule_files")
    if not isinstance(rule_files, list) or not rule_files:
        raise ValueError("pronunciation.rule_files must be a non-empty array when enabled")
    rules = load_rules([_resolve_path(root, value) for value in rule_files])
    enabled_rules = settings.get("enabled_rules", "all")
    if enabled_rules != "all":
        if not isinstance(enabled_rules, list) or not enabled_rules:
            raise ValueError("pronunciation.enabled_rules must be 'all' or a non-empty array")
        requested = {str(value) for value in enabled_rules}
        known = {rule.rule_id for rule in rules}
        unknown = requested.difference(known)
        if unknown:
            raise ValueError(f"unknown pronunciation rules: {', '.join(sorted(unknown))}")
        rules = [rule for rule in rules if rule.rule_id in requested]
    max_rule_count = int(settings.get("max_rule_count", 1))
    if max_rule_count not in {1, 2}:
        raise ValueError("pronunciation.max_rule_count must be 1 or 2")
    renderabilities = settings.get("renderabilities", ["text_approximation"])
    if not isinstance(renderabilities, list) or not renderabilities:
        raise ValueError("pronunciation.renderabilities must be a non-empty array")
    allowed_renderabilities = {str(value) for value in renderabilities}
    variants_path = output_dir / "pronunciation" / "pronunciation-variants.jsonl"
    summary_path = output_dir / "pronunciation" / "summary.json"
    variants_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped = 0
    errors = 0
    with variants_path.open("w", encoding="utf-8") as variants_file:
        for entry in entries:
            # Text approximation deletes visible Han characters.  Applying it to
            # mixed-language terms would silently discard their English portion.
            if not _is_pure_han_text(entry.canonical_text):
                skipped += 1
                continue
            sample = _syllable_deletion_sample(entry)
            try:
                generated, _ = generate_variants(sample, rules, max_rule_count=max_rule_count)
            except PronunciationError as exc:
                errors += 1
                print(
                    f"WARNING: pronunciation preprocessing skipped {entry.canonical_text!r}: {exc}",
                    file=sys.stderr,
                )
                continue
            for variant in generated:
                if variant.get("tts_renderability") not in allowed_renderabilities:
                    continue
                row = _pronunciation_sample_row(entry, language, variant)
                rows.append(row)
                variants_file.write(json.dumps(variant, ensure_ascii=False) + "\n")

    summary = {
        "term_count": len(entries),
        "variant_count": len(rows),
        "skipped_count": skipped,
        "error_count": errors,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows, summary


def _generate_gt_directed_variants(
    entries: list[GtEntry],
    *,
    root: Path,
    output_dir: Path,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_categories = settings.get("target_categories", sorted(PREPROCESSABLE_CATEGORIES))
    if not isinstance(raw_categories, list) or not raw_categories:
        raise ValueError("pronunciation.target_categories must be a non-empty array")
    target_categories = {str(value) for value in raw_categories}
    unknown = target_categories.difference(
        {
            "normalization_alias",
            "number_reading_mixed",
            "cross_language_transliteration",
            "english_word_acronym",
            "zh_same_pinyin",
            "zh_single_syllable",
            "zh_multi_syllable_near",
            "zh_distant_semantic",
            "other_review",
        }
    )
    if unknown:
        raise ValueError(
            "unknown pronunciation.target_categories: " + ", ".join(sorted(unknown))
        )
    classification_dir_value = settings.get("classification_output_dir")
    classification_dir = (
        _resolve_path(root, classification_dir_value)
        if classification_dir_value is not None
        else output_dir / "pronunciation" / "gt-by-type"
    )
    classification_summary = write_classification_outputs(entries, classification_dir)
    classification_rows = classify_entries(entries)
    rows = build_gt_directed_variants(
        entries,
        classification_rows,
        target_categories=target_categories,
    )
    variants_path = output_dir / "pronunciation" / "pronunciation-variants.jsonl"
    summary_path = output_dir / "pronunciation" / "summary.json"
    variants_path.parent.mkdir(parents=True, exist_ok=True)
    with variants_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "term_count": len(entries),
        "variant_count": len(rows),
        "skipped_count": 0,
        "error_count": 0,
        "mode": "gt_directed",
        "target_categories": sorted(target_categories),
        "classification_pair_count": classification_summary["pair_count"],
        "preprocessable_pair_count": classification_summary["preprocessable_pair_count"],
        "classification_output_dir": str(classification_dir.resolve()),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return rows, summary


def _generate_fuzzy_pinyin_variants(
    entries: list[GtEntry],
    *,
    root: Path,
    output_dir: Path,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = build_fuzzy_pinyin_variants(entries, root=root, settings=settings)
    variants_path = output_dir / "pronunciation" / "pronunciation-variants.jsonl"
    summary_path = output_dir / "pronunciation" / "summary.json"
    variants_path.parent.mkdir(parents=True, exist_ok=True)
    with variants_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary.update(
        {
            "mode": "fuzzy_pinyin",
            "variants_path": str(variants_path.resolve()),
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return rows, summary


def _generate_english_reading_variants(entries: list[GtEntry], *, output_dir: Path, settings: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = build_english_reading_variants(entries, settings=settings)
    variants_path = output_dir / "pronunciation" / "pronunciation-variants.jsonl"
    summary_path = output_dir / "pronunciation" / "summary.json"
    variants_path.parent.mkdir(parents=True, exist_ok=True)
    with variants_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary.update({"mode": "english_reading", "variants_path": str(variants_path.resolve())})
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows, summary


def _generate_english_cmu_variants(
    entries: list[GtEntry],
    *,
    root: Path,
    output_dir: Path,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return build_english_cmu_samples(entries, root=root, output_dir=output_dir, settings=settings)


def _pronunciation_sample_row(
    entry: GtEntry,
    language: str,
    variant: dict[str, Any],
) -> dict[str, Any]:
    rule = variant.get("rule", {})
    rule_id = str(rule.get("rule_id", "")).strip() if isinstance(rule, dict) else ""
    variant_pronunciation = variant.get("variant_pronunciation")
    pronunciation_phonemes = None
    if isinstance(variant_pronunciation, dict) and variant_pronunciation.get("alphabet") == "arpabet":
        pronunciation_phonemes = arpabet_to_cosyvoice(variant_pronunciation)
    row = {
        "id": entry.sample_id,
        "text": entry.canonical_text,
        "canonical_text": entry.canonical_text,
        "tts_text": str(variant["display_text"]),
        "source_text": entry.canonical_text,
        "text_source": "pronunciation_variant",
        "pronunciation_processed": True,
        "pronunciation_rule": rule_id or None,
        "pronunciation_variant_id": str(variant["variant_id"]),
        "language": language,
        "tags": [
            "gt",
            f"source-lines:{','.join(map(str, entry.source_lines))}",
            "pronunciation:processed",
            f"pronunciation-rule:{rule_id}",
        ],
    }
    row["target_confusions"] = list(entry.known_confusions)
    row["confusion_category"] = "english_word_acronym"
    if pronunciation_phonemes:
        row["pronunciation_phonemes"] = pronunciation_phonemes
        row["pronunciation_alphabet"] = "arpabet-cosyvoice"
        row["base_pronunciation"] = variant.get("base_pronunciation")
        row["variant_pronunciation"] = variant_pronunciation
    return row


def _is_pure_han_text(value: str) -> bool:
    return bool(value) and all("\u3400" <= character <= "\u9fff" for character in value)


def _syllable_deletion_sample(entry: GtEntry) -> PronunciationSample:
    """Build a traceable Chinese sample for visible syllable deletion.

    The deletion variant only needs character boundaries.  Supplying pypinyin's
    default reading keeps those boundaries explicit without rejecting a term
    merely because a character has other dictionary readings.
    """
    try:
        from pypinyin import Style, pinyin
    except ImportError as exc:  # pragma: no cover - a declared project dependency
        raise PronunciationError("install pronunciation dependencies: pip install pypinyin") from exc
    readings = pinyin(
        entry.canonical_text,
        style=Style.TONE3,
        heteronym=False,
        neutral_tone_with_five=True,
        errors="default",
    )
    syllables = [values[0].replace("u:", "v") for values in readings]
    return PronunciationSample(
        sample_id=entry.sample_id,
        canonical_text=entry.canonical_text,
        synthesis_text=entry.canonical_text,
        language="Chinese",
        tags=("gt", f"source-lines:{','.join(map(str, entry.source_lines))}"),
        pronunciation={"alphabet": "pinyin-tone3", "syllables": syllables},
    )


def _run_dictionary_postprocess(
    *,
    root: Path,
    output_dir: Path,
    input_path: Path,
    settings: dict[str, Any],
) -> None:
    def resource_path(value: Any, default: Path) -> Path:
        if value is None:
            return default
        return _resolve_path(root, value)

    run_dictionary_postprocess(
        input_path=input_path,
        tagged_output_path=output_dir / "confusion-words-dict-tagged.txt",
        removed_output_path=(
            output_dir / "confusion-words-dict-removed-single-character.txt"
        ),
        detail_path=output_dir / "confusion-words-dict.review.jsonl",
        stopword_path=resource_path(settings.get("stopword_dictionary"), DEFAULT_STOPWORD_DICTIONARY),
        wusong_path=resource_path(settings.get("wusong_dictionary"), DEFAULT_WUSONG_DICTIONARY),
        english_path=resource_path(settings.get("english_dictionary"), DEFAULT_ENGLISH_DICTIONARY),
        min_wusong_weight=int(settings.get("min_wusong_weight", MIN_WUSONG_WEIGHT)),
        jieba_path=resource_path(settings.get("jieba_dictionary"), DEFAULT_JIEBA_DICTIONARY),
        min_jieba_character_frequency=int(
            settings.get("min_jieba_character_frequency", MIN_JIEBA_CHARACTER_FREQUENCY)
        ),
    )


def _write_augmentation_config(
    path: Path,
    *,
    tts_manifest: Path,
    output_dir: Path,
    settings: dict[str, Any],
) -> None:
    perturbations = settings.get("perturbations")
    if not isinstance(perturbations, list) or not perturbations:
        raise ValueError("augmentation.perturbations must be a non-empty array")
    payload = {
        "input_manifest": str(tts_manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": int(settings.get("seed", 20260717)),
        "continue_on_error": bool(settings.get("continue_on_error", True)),
        "perturbations": perturbations,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _merge_manifests(paths: list[Path], output_path: Path) -> None:
    row_count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for path in paths:
            for row in _load_jsonl(path):
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_count += 1
    if row_count == 0:
        raise ValueError("combined ASR manifest is empty")
    print(f"ASR input manifest ready: rows={row_count} output={output_path}")


def _run_module(root: Path, module: str, arguments: list[str]) -> int:
    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [sys.executable, "-m", module, *arguments]
    print("Running:", " ".join(shlex.quote(part) for part in command))
    return subprocess.run(command, cwd=root, env=env, check=False).returncode


def _require_usable_manifest(path: Path, stage: str, exit_code: int) -> None:
    if not path.is_file():
        raise RuntimeError(f"{stage} did not create a manifest: {path}")
    rows = _load_jsonl(path)
    usable = sum(row.get("status") in {"generated", "cached"} for row in rows)
    if usable == 0:
        raise RuntimeError(f"{stage} produced no usable audio (exit code {exit_code})")
    if exit_code:
        print(f"WARNING: {stage} exited with {exit_code}, continuing with {usable} usable rows")


def _stop_asr_service(settings: dict[str, Any]) -> None:
    _require_windows_wsl()
    distro = str(settings.get("distribution", "Ubuntu-24.04"))
    process_match = str(settings.get("process_match", "qwenasr_service.main"))
    safe_pattern = _pkill_pattern(process_match)
    command = [
        "wsl.exe",
        "-d",
        distro,
        "--",
        "bash",
        "-lc",
        f"pkill -f {shlex.quote(safe_pattern)} || true",
    ]
    subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    print(f"ASR service stopped in WSL distribution {distro}")


def _start_asr_service(settings: dict[str, Any], log_path: Path) -> subprocess.Popen[Any]:
    _require_windows_wsl()
    distro = str(settings.get("distribution", "Ubuntu-24.04"))
    working_directory = str(settings.get("working_directory", "/home/jsqdc/qwen3-asr"))
    python_path = str(
        settings.get("python", "/home/jsqdc/miniconda3/envs/qwenasr/bin/python")
    )
    module = str(settings.get("module", "qwenasr_service.main"))
    service_config = str(settings.get("config", "config/service.yaml"))
    command = [
        "wsl.exe",
        "-d",
        distro,
        "--cd",
        working_directory,
        python_path,
        "-m",
        module,
        "--config",
        service_config,
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    try:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
    finally:
        log.close()
    print(f"ASR service starting in WSL: pid={process.pid} log={log_path}")
    return process


def _pkill_pattern(value: str) -> str:
    return f"[{value[0]}]{value[1:]}" if value else value


def _require_windows_wsl() -> None:
    if os.name != "nt":
        raise ValueError("automatic ASR service management currently requires Windows and WSL")


def _read_engine_name(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not str(payload.get("engine", "")).strip():
        raise ValueError(f"engine config has no engine: {path}")
    return str(payload["engine"]).strip()


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("confusion pipeline config must be a JSON object")
    for field in ("input_txt", "engine_config", "output_dir"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"confusion pipeline config requires non-empty {field}")
    return payload


def _as_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for parent in resolved.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return resolved.parent


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


if __name__ == "__main__":
    raise SystemExit(main())
