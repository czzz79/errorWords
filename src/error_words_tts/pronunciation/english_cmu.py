"""Structured, target-blind English/initialism CMU pronunciation preprocessing."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .cosyvoice_serializer import serialize_cosyvoice_arpabet
from .english_g2p import LETTER_PHONEMES, letter_subunits, load_overrides, resolve_token


_PARTS = re.compile(r"[A-Za-z]+|\d+|\s+|-|.")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[A-Z]+")
_VOWEL = re.compile(r"^[A-Z]+[012]$")


def build_english_cmu_samples(
    entries: Iterable[Any],
    *,
    root: Path,
    output_dir: Path,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build baseline and single-rule CMU samples from canonical term text only."""
    rules_path = _resolve(root, settings.get("rules_path", "src/error_words_tts/pronunciation/rules/english-cmu.json"))
    overrides_path = _resolve(root, settings.get("overrides_path", "resources/pronunciation/english-pronunciation-overrides.json"))
    rules = load_rules(rules_path)
    enabled_rules = settings.get("enabled_rules", "all")
    if enabled_rules != "all":
        if not isinstance(enabled_rules, list) or not enabled_rules:
            raise ValueError("pronunciation.enabled_rules must be 'all' or a non-empty array")
        requested = {str(value) for value in enabled_rules}
        known = {str(rule["rule_id"]) for rule in rules}
        unknown = requested.difference(known)
        if unknown:
            raise ValueError("unknown English CMU rules: " + ", ".join(sorted(unknown)))
        rules = [rule for rule in rules if rule["rule_id"] in requested]
    overrides = load_overrides(overrides_path)
    boundary_modes = _load_boundary_modes(settings.get("boundary_modes", {}))
    baseline_only = bool(settings.get("baseline_only", False))
    if baseline_only:
        rules = []
    include_boundary_variants = bool(settings.get("include_boundary_variants", False))
    include_text_baseline = bool(settings.get("include_text_baseline", False))
    max_variants = int(settings.get("max_variants_per_term", 32))
    if max_variants < 1:
        raise ValueError("pronunciation.max_variants_per_term must be positive")

    entries = list(entries)
    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    structures: list[dict[str, Any]] = []
    for entry in entries:
        base = build_structure(str(entry.canonical_text), overrides=overrides)
        _apply_boundary_modes(base, boundary_modes)
        if base["status"] != "ready":
            unresolved.append({
                "sample_id": entry.sample_id,
                "canonical_text": entry.canonical_text,
                "unresolved": base.get("unresolved", []),
            })
            continue
        if include_text_baseline:
            rows.append(_text_baseline_row(entry, base))
        variants = generate_variants(
            base, rules, include_boundary_variants=include_boundary_variants
        )
        for index, variant in enumerate(variants):
            if index >= max_variants:
                break
            sample = _sample_row(entry, base, variant)
            rows.append(sample)
            structures.append({
                "sample_id": entry.sample_id,
                "canonical_text": entry.canonical_text,
                "variant_id": sample["pronunciation_variant_id"],
                "structure": variant["structure"],
                "rule": variant["rule"],
            })

    pronunciation_dir = output_dir / "pronunciation"
    pronunciation_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(pronunciation_dir / "pronunciation-variants.jsonl", rows)
    _write_jsonl(pronunciation_dir / "structured-pronunciations.jsonl", structures)
    _write_jsonl(pronunciation_dir / "unresolved.jsonl", unresolved)
    summary = {
        "mode": "english_cmu",
        "generator": "structured_english_cmu",
        "term_count": len(entries),
        "resolved_term_count": len(entries) - len(unresolved),
        "unresolved_term_count": len(unresolved),
        "baseline_count": sum(row["variant_kind"] == "baseline" for row in rows),
        "variant_count": len(rows),
        "rules_path": str(rules_path.resolve()),
        "overrides_path": str(overrides_path.resolve()),
        "include_boundary_variants": include_boundary_variants,
        "include_text_baseline": include_text_baseline,
        "baseline_only": baseline_only,
        "boundary_modes": boundary_modes,
        "structured_pronunciations_path": str((pronunciation_dir / "structured-pronunciations.jsonl").resolve()),
        "unresolved_path": str((pronunciation_dir / "unresolved.jsonl").resolve()),
    }
    (pronunciation_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return rows, summary


def build_structure(text: str, *, overrides: dict[str, list[str]]) -> dict[str, Any]:
    nodes = parse_term(text)
    unresolved: list[dict[str, Any]] = []
    if not any(node.get("kind") == "token" for node in nodes):
        unresolved.append({"text": text, "token_type": "none", "reason": "no_supported_english_tokens"})
    for node in nodes:
        if node["kind"] != "token":
            continue
        resolution = resolve_token(node["text"], node["token_type"], overrides=overrides)
        if resolution["status"] != "ready":
            unresolved.append({
                "text": node["text"], "token_type": node["token_type"],
                "reason": resolution["reason"],
            })
            continue
        node["phones"] = resolution["phones"]
        node["pronunciation_source"] = resolution["source"]
        if "subunits" in resolution:
            node["subunits"] = resolution["subunits"]
    return {
        "raw_text": text,
        "alphabet": "arpabet",
        "nodes": nodes,
        "status": "unresolved" if unresolved else "ready",
        "unresolved": unresolved,
    }


def parse_term(text: str) -> list[dict[str, Any]]:
    """Parse term text while preserving lexical and visual boundaries."""
    nodes: list[dict[str, Any]] = []
    previous_token = False
    for match in _PARTS.finditer(text):
        part = match.group(0)
        if part.isspace():
            _append_boundary(nodes, "space")
            previous_token = False
            continue
        if part == "-":
            _append_boundary(nodes, "hyphen")
            previous_token = False
            continue
        if part.isdigit():
            for index, digit in enumerate(part):
                if index:
                    _append_boundary(nodes, "alnum")
                elif previous_token:
                    _append_boundary(nodes, "alnum")
                nodes.append({"kind": "token", "text": digit, "token_type": "number"})
                previous_token = True
            continue
        if part.isalpha() and part.isascii():
            fragments = _split_alpha(part)
            for index, fragment in enumerate(fragments):
                if index:
                    _append_boundary(nodes, "camel_case")
                elif previous_token:
                    _append_boundary(nodes, "alnum")
                token_type = "acronym" if fragment.isupper() else "word"
                node: dict[str, Any] = {"kind": "token", "text": fragment, "token_type": token_type}
                if token_type == "acronym":
                    node["letters"] = list(fragment)
                nodes.append(node)
                previous_token = True
            continue
        _append_boundary(nodes, "symbol")
        previous_token = False
    while nodes and nodes[-1]["kind"] == "boundary":
        nodes.pop()
    return nodes


def generate_variants(
    base: dict[str, Any],
    rules: list[dict[str, Any]],
    *,
    include_boundary_variants: bool,
) -> list[dict[str, Any]]:
    """Return one baseline plus deterministic single-rule variants."""
    result = [{"structure": deepcopy(base), "rule": None, "variant_kind": "baseline"}]
    seen = {_structure_key(base)}
    for rule in rules:
        if rule.get("level") == "boundary" and not include_boundary_variants:
            continue
        for structure, applied in _apply_rule(base, rule):
            key = _structure_key(structure)
            if key in seen:
                continue
            seen.add(key)
            result.append({
                "structure": structure,
                "rule": {**rule, "applied_change": applied},
                "variant_kind": str(rule["rule_id"]),
            })
    return result


def load_rules(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"English CMU rules must be a non-empty JSON array: {path}")
    seen: set[str] = set()
    rules: list[dict[str, Any]] = []
    for index, value in enumerate(payload):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: rule {index} must be an object")
        rule_id = str(value.get("rule_id", "")).strip()
        operation = str(value.get("operation", "")).strip()
        if not rule_id or not operation or rule_id in seen:
            raise ValueError(f"{path}: invalid or duplicate rule {rule_id!r}")
        seen.add(rule_id)
        rules.append(value)
    return rules


def _apply_rule(base: dict[str, Any], rule: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    operation = rule["operation"]
    if operation == "letter_spelling":
        return _letter_spelling(base, rule)
    if operation == "connected_boundary":
        return _connected_boundary(base, rule)
    results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for node_index, node in enumerate(base["nodes"]):
        if node.get("kind") != "token":
            continue
        phones = node.get("phones", [])
        if operation == "final_schwa_deletion":
            if not phones or phones[-1] != "AH0" or not any(_VOWEL.fullmatch(phone) for phone in phones[:-1]):
                continue
            results.append(_edit(base, node_index, len(phones) - 1, [], rule))
        elif operation == "final_cluster_delete_penultimate":
            if len(phones) < 3 or _VOWEL.fullmatch(phones[-1]) or _VOWEL.fullmatch(phones[-2]):
                continue
            results.append(_edit(base, node_index, len(phones) - 2, [], rule))
        elif operation == "final_substitution":
            if phones and phones[-1] == rule.get("source"):
                results.append(_edit(base, node_index, len(phones) - 1, [str(rule["target"])], rule))
        elif operation == "substitution":
            source = str(rule.get("source", ""))
            target = str(rule.get("target", ""))
            for phone_index, phone in enumerate(phones):
                if phone == source:
                    results.append(_edit(base, node_index, phone_index, [target], rule))
        elif operation == "initial_deletion":
            source = str(rule.get("source", ""))
            if phones and phones[0] == source and len(phones) > 1:
                results.append(_edit(base, node_index, 0, [], rule))
    return results


def _edit(
    base: dict[str, Any], node_index: int, phone_index: int, replacement: list[str], rule: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    structure = deepcopy(base)
    node = structure["nodes"][node_index]
    before = node["phones"][phone_index]
    node["phones"][phone_index : phone_index + 1] = replacement
    return structure, {
        "token_index": node_index,
        "token": node["text"],
        "phone_index": phone_index,
        "source": before,
        "target": replacement[0] if replacement else "",
        "left_context": node["phones"][phone_index - 1] if phone_index else None,
        "right_context": node["phones"][phone_index] if phone_index < len(node["phones"]) else None,
    }


def _letter_spelling(base: dict[str, Any], rule: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    results = []
    for node_index, node in enumerate(base["nodes"]):
        if node.get("kind") != "token" or node.get("token_type") != "word" or not node["text"].isalpha():
            continue
        letters = node["text"].upper()
        try:
            subunits = letter_subunits(letters)
        except ValueError:
            continue
        structure = deepcopy(base)
        target = structure["nodes"][node_index]
        before = list(target["phones"])
        target["phones"] = [phone for unit in subunits for phone in unit["phones"]]
        target["pronunciation_source"] = "letter_names"
        target["subunits"] = subunits
        results.append((structure, {
            "token_index": node_index, "token": node["text"], "phone_index": None,
            "source": before, "target": list(target["phones"]), "left_context": None, "right_context": None,
        }))
    return results


def _connected_boundary(base: dict[str, Any], rule: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    results = []
    for node_index, node in enumerate(base["nodes"]):
        if node.get("kind") != "boundary":
            continue
        structure = deepcopy(base)
        structure["nodes"][node_index]["mode"] = "connected"
        results.append((structure, {
            "boundary_index": node_index, "source": "default", "target": "connected",
        }))
    return results


def _sample_row(entry: Any, base: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    structure = variant["structure"]
    rule = variant["rule"]
    identity = {"sample_id": entry.sample_id, "structure": structure, "rule": rule}
    variant_id = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    rule_id = str(rule["rule_id"]) if rule else None
    return {
        "id": entry.sample_id,
        "text": entry.canonical_text,
        "canonical_text": entry.canonical_text,
        "tts_text": entry.canonical_text,
        "source_text": entry.canonical_text,
        "text_source": "canonical_cmu" if rule is None else "english_cmu_variant",
        "pronunciation_processed": rule is not None,
        "pronunciation_rule": rule_id,
        "pronunciation_variant_id": variant_id,
        "pronunciation_structure": structure,
        "base_pronunciation": base,
        "variant_pronunciation": structure,
        "phoneme_text": serialize_cosyvoice_arpabet(structure),
        "input_mode": "phoneme",
        "variant_kind": variant["variant_kind"],
        "pronunciation_delta": {
            "origin": "canonical_only", "type": "baseline" if rule is None else rule["operation"],
            "rule": rule,
        },
        "target_confusions": list(entry.known_confusions),
        "confusion_category": "english_word_acronym",
        "language": "English",
        "tags": ["gt", f"source-lines:{','.join(map(str, entry.source_lines))}", "pronunciation:english-cmu"],
    }


def _text_baseline_row(entry: Any, structure: dict[str, Any]) -> dict[str, Any]:
    variant_id = hashlib.sha256(
        json.dumps([entry.sample_id, "text_baseline", structure], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "id": entry.sample_id,
        "text": entry.canonical_text,
        "canonical_text": entry.canonical_text,
        "tts_text": entry.canonical_text,
        "source_text": entry.canonical_text,
        "text_source": "canonical_text",
        "pronunciation_processed": False,
        "pronunciation_rule": None,
        "pronunciation_variant_id": variant_id,
        "pronunciation_structure": structure,
        "base_pronunciation": structure,
        "variant_pronunciation": structure,
        "input_mode": "text",
        "variant_kind": "text_baseline",
        "pronunciation_delta": {"origin": "canonical_only", "type": "text_baseline", "rule": None},
        "target_confusions": list(entry.known_confusions),
        "confusion_category": "english_word_acronym",
        "language": "English",
        "tags": ["gt", f"source-lines:{','.join(map(str, entry.source_lines))}", "pronunciation:english-text-baseline"],
    }


def _append_boundary(nodes: list[dict[str, Any]], boundary_type: str) -> None:
    if not nodes or nodes[-1].get("kind") == "boundary":
        return
    nodes.append({"kind": "boundary", "boundary_type": boundary_type, "mode": "default"})


def _load_boundary_modes(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("pronunciation.boundary_modes must be an object")
    result = {str(key): str(mode) for key, mode in value.items()}
    unknown = set(result.values()).difference({"default", "connected"})
    if unknown:
        raise ValueError("unsupported pronunciation boundary mode: " + ", ".join(sorted(unknown)))
    return result


def _apply_boundary_modes(structure: dict[str, Any], boundary_modes: dict[str, str]) -> None:
    for node in structure.get("nodes", []):
        if node.get("kind") != "boundary":
            continue
        boundary_type = str(node.get("boundary_type", ""))
        if boundary_type in boundary_modes:
            node["mode"] = boundary_modes[boundary_type]


def _split_alpha(value: str) -> list[str]:
    return _CAMEL.findall(value) or [value]


def _structure_key(structure: dict[str, Any]) -> str:
    return json.dumps(structure, ensure_ascii=False, sort_keys=True)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()
