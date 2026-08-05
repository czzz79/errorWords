from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .generator import PronunciationError, generate_variants, load_rules, load_samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate rule-based pronunciation variants")
    parser.add_argument(
        "--config",
        default="src/error_words_tts/pronunciation/configs/ideahub-rules.json",
        help="Pronunciation experiment config",
    )
    args = parser.parse_args()
    return run_config(Path(args.config))


def run_config(config_path: Path) -> int:
    config = _load_config(config_path)
    root = _project_root(config_path)
    samples_path = _resolve_path(root, config["samples"])
    output_path = _resolve_path(root, config["output"])
    summary_path = _resolve_path(root, config.get("summary", output_path.with_name("summary.json")))
    rule_paths = [_resolve_path(root, value) for value in config["rule_files"]]

    samples = load_samples(samples_path)
    rules = load_rules(rule_paths)
    max_rule_count = int(config.get("max_rule_count", 1))
    enabled = config.get("enabled_rules", "all")
    if enabled != "all":
        selected = set(enabled)
        unknown = selected.difference(rule.rule_id for rule in rules)
        if unknown:
            raise PronunciationError(f"unknown enabled_rules: {', '.join(sorted(unknown))}")
        rules = [rule for rule in rules if rule.rule_id in selected]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_counts: Counter[str] = Counter()
    sample_summaries = []
    variant_count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for sample in samples:
            variants, counts = generate_variants(
                sample,
                rules,
                max_rule_count=max_rule_count,
            )
            for row in variants:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            all_counts.update(counts)
            variant_count += len(variants)
            sample_summaries.append(
                {
                    "sample_id": sample.sample_id,
                    "canonical_text": sample.canonical_text,
                    "variant_count": len(variants),
                    "rule_counts": counts,
                }
            )

    summary = {
        "status": "success",
        "samples_path": str(samples_path),
        "output_path": str(output_path),
        "sample_count": len(samples),
        "variant_count": variant_count,
        "max_rule_count": max_rule_count,
        "rule_counts": dict(sorted(all_counts.items())),
        "samples": sample_summaries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Pronunciation variants complete: samples={len(samples)} "
        f"variants={variant_count} output={output_path}"
    )
    print(f"Summary: {summary_path}")
    return 0


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PronunciationError("pronunciation config must be a JSON object")
    for field in ("samples", "output", "rule_files"):
        if field not in payload:
            raise PronunciationError(f"pronunciation config is missing {field}")
    if not isinstance(payload["rule_files"], list) or not payload["rule_files"]:
        raise PronunciationError("pronunciation config rule_files must be a non-empty list")
    enabled = payload.get("enabled_rules", "all")
    if enabled != "all" and (not isinstance(enabled, list) or not enabled):
        raise PronunciationError("enabled_rules must be 'all' or a non-empty list")
    max_rule_count = payload.get("max_rule_count", 1)
    if isinstance(max_rule_count, bool) or max_rule_count not in {1, 2}:
        raise PronunciationError("max_rule_count must be 1 or 2")
    return payload


def _resolve_path(root: Path, value: Any) -> Path:
    path = value if isinstance(value, Path) else Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for parent in resolved.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return resolved.parent


if __name__ == "__main__":
    raise SystemExit(main())
