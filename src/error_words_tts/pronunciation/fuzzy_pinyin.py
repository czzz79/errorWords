"""Text-level Chinese fuzzy-pinyin preprocessing for TTS carrier generation.

The module deliberately creates ordinary Chinese ``tts_text`` carrier terms.
It does not ask a TTS model to follow a phoneme instruction.  A carrier is
accepted only when converting it back to tone-marked pinyin yields exactly the
single-rule-mutated target pronunciation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pypinyin import Style, lazy_pinyin


_HAN_MIN = "\u3400"
_HAN_MAX = "\u9fff"
_INITIALS = (
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "z", "c", "s", "y", "w",
)


@dataclass(frozen=True, slots=True)
class FuzzyRule:
    rule_type: str
    source: str
    target: str
    level: int
    origin: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.rule_type, self.source, self.target)


@dataclass(frozen=True, slots=True)
class FuzzyMutation:
    rule: FuzzyRule
    syllable_index: int
    source_pinyin: tuple[str, ...]
    perturbed_pinyin: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CarrierCandidate:
    text: str
    score: int
    source: str


def load_fuzzy_rules(path: Path) -> list[FuzzyRule]:
    """Read directional fuzzy-pinyin rules from the documented CSV format."""
    lines = [
        line for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    reader = csv.DictReader(lines)
    required = {"type", "source", "target", "level", "origin"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError(f"{path}: needs CSV columns {', '.join(sorted(required))}")
    rules: list[FuzzyRule] = []
    seen: set[tuple[str, str, str]] = set()
    for line_number, row in enumerate(reader, start=2):
        rule_type = str(row["type"] or "").strip().casefold()
        source = _normalize_plain(str(row["source"] or ""))
        target = _normalize_plain(str(row["target"] or ""))
        origin = str(row["origin"] or "").strip() or "public"
        try:
            level = int(str(row["level"] or ""))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: level must be an integer") from exc
        if rule_type not in {"initial", "final"} or not source or not target or level < 1:
            raise ValueError(f"{path}:{line_number}: invalid fuzzy-pinyin rule")
        rule = FuzzyRule(rule_type, source, target, level, origin)
        if rule.key not in seen:
            seen.add(rule.key)
            rules.append(rule)
    if not rules:
        raise ValueError(f"{path}: no fuzzy-pinyin rules")
    return rules


def build_fuzzy_pinyin_variants(
    entries: Iterable[Any],
    *,
    root: Path,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate one-rule Chinese carrier variants for GT/pipeline entries."""
    entries = list(entries)
    rules_path = _configured_path(
        root, settings.get("rules_path", "resources/chinese-fuzzy-pinyin-rules.csv")
    )
    rules = load_fuzzy_rules(rules_path)
    max_rule_level = int(settings.get("max_rule_level", 3))
    if max_rule_level not in {1, 2, 3}:
        raise ValueError("pronunciation.max_rule_level must be 1, 2, or 3")
    max_carriers = int(settings.get("max_carriers_per_pinyin", 3))
    max_variants = int(settings.get("max_variants_per_term", 30))
    if max_carriers < 1 or max_variants < 1:
        raise ValueError("fuzzy-pinyin carrier and term limits must be positive")
    overrides = _load_overrides(root, settings.get("pronunciation_overrides", {}))
    active_rules = [rule for rule in rules if rule.level <= max_rule_level]
    derive_gt_components = bool(settings.get("derive_gt_component_rules", False))
    use_gt_target_tones = bool(settings.get("use_gt_target_tones", False))
    derive_gt_syllable_edits = bool(settings.get("derive_gt_syllable_edits", False))
    observed = _observed_rule_keys(entries, active_rules, overrides)

    prepared: list[tuple[Any, tuple[str, ...], list[FuzzyMutation]]] = []
    required_plain: set[tuple[str, ...]] = set()
    skipped = 0
    derived_rule_keys: set[tuple[str, str, str]] = set()
    for entry in entries:
        canonical = str(entry.canonical_text)
        if not _is_pure_han(canonical):
            skipped += 1
            continue
        source_pinyin = _tones(canonical, overrides.get(canonical))
        mutations = enumerate_single_rule_mutations(source_pinyin, active_rules)
        if derive_gt_components:
            targeted = _derive_gt_component_mutations(
                entry, source_pinyin, use_target_tones=use_gt_target_tones
            )
            mutations = _dedupe_mutations([*mutations, *targeted])
            derived_rule_keys.update(item.rule.key for item in targeted)
        if derive_gt_syllable_edits:
            edits = _derive_gt_syllable_edit_mutations(
                entry, source_pinyin, use_target_tones=use_gt_target_tones
            )
            mutations = _dedupe_mutations([*mutations, *edits])
            derived_rule_keys.update(item.rule.key for item in edits)
        if not mutations:
            continue
        prepared.append((entry, source_pinyin, mutations))
        required_plain.update(tuple(_tone_less(value) for value in item.perturbed_pinyin) for item in mutations)

    lexicon = CarrierLexicon.from_resources(
        required_plain,
        rime_path=_configured_path(root, settings.get("carrier_rime_dictionary", "resources/wusong-base.dict.yaml")),
        high_frequency_path=_configured_path(root, settings.get("carrier_high_frequency_words", "resources/high_frequency_words.txt")),
        jieba_path=_configured_path(root, settings.get("carrier_jieba_dictionary", "resources/jieba-dict.txt")),
    )

    rows: list[dict[str, Any]] = []
    terms_with_carrier_ids: set[str] = set()
    carrier_count = 0
    for entry, source_pinyin, mutations in prepared:
        emitted_for_term = 0
        seen_perturbed: set[tuple[str, ...]] = set()
        for mutation in mutations:
            if emitted_for_term >= max_variants:
                break
            if mutation.perturbed_pinyin in seen_perturbed:
                continue
            seen_perturbed.add(mutation.perturbed_pinyin)
            excluded = set(entry.known_confusions) if bool(settings.get("exclude_known_confusion_texts", False)) else set()
            carriers = lexicon.carriers_for(
                mutation.perturbed_pinyin, limit=max_carriers, excluded_texts=excluded
            )
            if not carriers:
                continue
            terms_with_carrier_ids.add(str(entry.sample_id))
            for carrier_rank, carrier in enumerate(carriers, start=1):
                if emitted_for_term >= max_variants:
                    break
                rows.append(
                    _variant_row(
                        entry,
                        source_pinyin=source_pinyin,
                        mutation=mutation,
                        carrier=carrier,
                        carrier_rank=carrier_rank,
                        observed_in_gt=mutation.rule.key in observed,
                    )
                )
                emitted_for_term += 1
                carrier_count += 1

    return rows, {
        "term_count": len(entries),
        "variant_count": len(rows),
        "terms_with_carrier_count": len(terms_with_carrier_ids),
        "skipped_non_chinese_count": skipped,
        "rule_count": len(active_rules),
        "static_rule_count": len(rules),
        "derived_gt_component_rule_count": len(derived_rule_keys),
        "derive_gt_component_rules": derive_gt_components,
        "use_gt_target_tones": use_gt_target_tones,
        "derive_gt_syllable_edits": derive_gt_syllable_edits,
        "observed_gt_direction_count": len(observed),
        "carrier_count": carrier_count,
        "carrier_lexicon": lexicon.summary,
        "rules_path": str(rules_path.resolve()),
    }


def enumerate_single_rule_mutations(
    source_pinyin: tuple[str, ...], rules: Iterable[FuzzyRule]
) -> list[FuzzyMutation]:
    """Enumerate every one-syllable, one-directional-rule mutation."""
    results: list[FuzzyMutation] = []
    seen: set[tuple[tuple[str, ...], tuple[str, str, str], int]] = set()
    ordered_rules = sorted(rules, key=lambda item: (item.level, item.rule_type, item.source, item.target))
    for index, syllable in enumerate(source_pinyin):
        initial, final, tone = _split_syllable(syllable)
        for rule in ordered_rules:
            if rule.rule_type == "initial" and initial == rule.source:
                changed = f"{rule.target}{final}{tone}"
            elif rule.rule_type == "final" and final == rule.source:
                changed = f"{initial}{rule.target}{tone}"
            else:
                continue
            perturbed = (*source_pinyin[:index], changed, *source_pinyin[index + 1 :])
            key = (perturbed, rule.key, index)
            if key not in seen:
                seen.add(key)
                results.append(FuzzyMutation(rule, index, source_pinyin, perturbed))
    return results


def _derive_gt_component_mutations(
    entry: Any, source: tuple[str, ...], *, use_target_tones: bool
) -> list[FuzzyMutation]:
    """Generate only the one-component pinyin changes requested by this GT row.

    The GT target supplies a phonetic neighbour, not synthesis text.  Each
    By default the mutation retains the source tone.  ``use_target_tones`` is
    an explicit tone-relaxation experiment and emits only the newly reachable
    target-tone reading.  Full-syllable substitutions and insert/delete edits
    are excluded.
    """
    results: list[FuzzyMutation] = []
    for confusion in entry.known_confusions:
        target_text = str(confusion)
        if not _is_pure_han(target_text):
            continue
        target = _tones(target_text, None)
        if len(source) != len(target):
            continue
        changed = [
            index
            for index, (left, right) in enumerate(zip(source, target))
            if _tone_less(left) != _tone_less(right)
        ]
        if len(changed) != 1:
            continue
        index = changed[0]
        source_initial, source_final, source_tone = _split_syllable(source[index])
        target_initial, target_final, _ = _split_syllable(target[index])
        target_tone = _split_syllable(target[index])[2]
        tone = target_tone if use_target_tones else source_tone
        if source_initial != target_initial and source_final == target_final:
            origin = "gt_target_component_tone" if use_target_tones and tone != source_tone else "gt_target_component"
            rule = FuzzyRule("initial", source_initial, target_initial, 1, origin)
            changed_syllable = f"{target_initial}{source_final}{tone}"
        elif source_initial == target_initial and source_final != target_final:
            origin = "gt_target_component_tone" if use_target_tones and tone != source_tone else "gt_target_component"
            rule = FuzzyRule("final", source_final, target_final, 1, origin)
            changed_syllable = f"{source_initial}{target_final}{tone}"
        else:
            continue
        perturbed = (*source[:index], changed_syllable, *source[index + 1 :])
        results.append(FuzzyMutation(rule, index, source, perturbed))
    return _dedupe_mutations(results)


def _derive_gt_syllable_edit_mutations(
    entry: Any, source: tuple[str, ...], *, use_target_tones: bool
) -> list[FuzzyMutation]:
    """Derive one insertion/deletion edit when all remaining syllable bases align."""
    results: list[FuzzyMutation] = []
    for confusion in entry.known_confusions:
        target_text = str(confusion)
        if not _is_pure_han(target_text):
            continue
        target = _tones(target_text, None)
        edit = _one_syllable_sequence_edit(source, target)
        if edit is None:
            continue
        kind, index = edit
        if kind == "delete":
            perturbed = target if use_target_tones else (*source[:index], *source[index + 1 :])
            rule = FuzzyRule("syllable_delete", _tone_less(source[index]), "", 1, "gt_target_syllable_edit")
        else:
            perturbed = target if use_target_tones else (*source[:index], target[index], *source[index:])
            rule = FuzzyRule("syllable_insert", "", _tone_less(target[index]), 1, "gt_target_syllable_edit")
        results.append(FuzzyMutation(rule, index, source, tuple(perturbed)))
    return _dedupe_mutations(results)


def _one_syllable_sequence_edit(
    source: tuple[str, ...], target: tuple[str, ...]
) -> tuple[str, int] | None:
    source_plain = tuple(_tone_less(value) for value in source)
    target_plain = tuple(_tone_less(value) for value in target)
    if len(source) == len(target) + 1:
        for index in range(len(source)):
            if source_plain[:index] + source_plain[index + 1 :] == target_plain:
                return "delete", index
    if len(target) == len(source) + 1:
        for index in range(len(target)):
            if target_plain[:index] + target_plain[index + 1 :] == source_plain:
                return "insert", index
    return None


def _dedupe_mutations(values: Iterable[FuzzyMutation]) -> list[FuzzyMutation]:
    chosen: dict[tuple[tuple[str, ...], tuple[str, str, str], int], FuzzyMutation] = {}
    for value in values:
        key = (value.perturbed_pinyin, value.rule.key, value.syllable_index)
        chosen.setdefault(key, value)
    return list(chosen.values())


class CarrierLexicon:
    """High-frequency Chinese carrier lookup indexed by tone-less pinyin."""

    def __init__(self, entries: dict[tuple[str, ...], list[CarrierCandidate]]) -> None:
        self._entries = entries
        self.summary = {
            "pinyin_keys": len(entries),
            "candidate_count": sum(len(values) for values in entries.values()),
        }

    @classmethod
    def from_resources(
        cls,
        wanted: set[tuple[str, ...]],
        *,
        rime_path: Path,
        high_frequency_path: Path,
        jieba_path: Path,
    ) -> "CarrierLexicon":
        entries: dict[tuple[str, ...], list[CarrierCandidate]] = defaultdict(list)
        # Whole-term candidates are preferred.  Single-syllable keys are also
        # indexed so that a tone-checked high-frequency character fallback can
        # be assembled when no suitable whole term exists.
        single_wanted = {(syllable,) for key in wanted for syllable in key}
        indexed_keys = wanted | single_wanted
        if rime_path.is_file():
            for raw_line in rime_path.read_text(encoding="utf-8-sig").splitlines():
                fields = raw_line.strip().split()
                if len(fields) < 3 or not fields[-1].isdigit():
                    continue
                word = fields[0]
                key = tuple(_normalize_plain(value) for value in fields[1:-1])
                if key in indexed_keys and _is_pure_han(word):
                    entries[key].append(CarrierCandidate(word, int(fields[-1]), "rime"))
        if high_frequency_path.is_file():
            for rank, raw_line in enumerate(high_frequency_path.read_text(encoding="utf-8-sig").splitlines()):
                word = raw_line.strip().split("\t", 1)[0].strip()
                if not word or word.startswith("#") or not _is_pure_han(word):
                    continue
                key = tuple(_tone_less(value) for value in _tones(word, None))
                if key in indexed_keys:
                    entries[key].append(CarrierCandidate(word, 1_000_000 - rank, "high_frequency"))
        if jieba_path.is_file():
            for raw_line in jieba_path.read_text(encoding="utf-8-sig").splitlines():
                fields = raw_line.strip().split()
                if len(fields) < 2 or len(fields[0]) != 1 or not fields[1].isdigit():
                    continue
                word = fields[0]
                if not _is_pure_han(word):
                    continue
                key = tuple(_tone_less(value) for value in _tones(word, None))
                if key in single_wanted:
                    entries[key].append(CarrierCandidate(word, int(fields[1]), "jieba_character"))
        return cls({key: _dedupe_candidates(values) for key, values in entries.items()})

    def carriers_for(
        self,
        target_pinyin: tuple[str, ...],
        *,
        limit: int,
        excluded_texts: set[str] | None = None,
    ) -> list[CarrierCandidate]:
        excluded = excluded_texts or set()
        key = tuple(_tone_less(value) for value in target_pinyin)
        valid = [
            candidate
            for candidate in self._entries.get(key, [])
            if tuple(_tones(candidate.text, None)) == target_pinyin and candidate.text not in excluded
        ]
        valid = _dedupe_candidates(valid)
        if len(valid) < limit:
            fallback = self._character_fallback(target_pinyin)
            if fallback is not None and fallback.text not in excluded and fallback.text not in {item.text for item in valid}:
                valid.append(fallback)
        return valid[:limit]

    def _character_fallback(self, target_pinyin: tuple[str, ...]) -> CarrierCandidate | None:
        pieces: list[CarrierCandidate] = []
        for syllable in target_pinyin:
            key = (_tone_less(syllable),)
            choices = [
                candidate
                for candidate in self._entries.get(key, [])
                if len(candidate.text) == 1 and tuple(_tones(candidate.text, None)) == (syllable,)
            ]
            if not choices:
                return None
            pieces.append(_dedupe_candidates(choices)[0])
        text = "".join(item.text for item in pieces)
        if tuple(_tones(text, None)) != target_pinyin:
            return None
        return CarrierCandidate(text, min(item.score for item in pieces), "character_fallback")


def _variant_row(
    entry: Any,
    *,
    source_pinyin: tuple[str, ...],
    mutation: FuzzyMutation,
    carrier: CarrierCandidate,
    carrier_rank: int,
    observed_in_gt: bool,
) -> dict[str, Any]:
    canonical = str(entry.canonical_text)
    rule = mutation.rule
    rule_name = f"fuzzy_{rule.rule_type}_{rule.source}_to_{rule.target}"
    variant_id = hashlib.sha256(
        json.dumps(
            [canonical, mutation.perturbed_pinyin, carrier.text, rule.key, mutation.syllable_index],
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:16]
    rule_origin = f"{rule.origin}+gt" if observed_in_gt else rule.origin
    delta = {
        "type": rule.rule_type,
        "source": rule.source,
        "target": rule.target,
        "level": rule.level,
        "origin": rule_origin,
        "observed_in_gt": observed_in_gt,
        "syllable_index": mutation.syllable_index,
        "original_pinyin": list(source_pinyin),
        "perturbed_pinyin": list(mutation.perturbed_pinyin),
        "carrier_term": carrier.text,
        "carrier_source": carrier.source,
        "carrier_rank": carrier_rank,
    }
    return {
        "id": entry.sample_id,
        "text": canonical,
        "canonical_text": canonical,
        "tts_text": carrier.text,
        "source_text": canonical,
        "text_source": "fuzzy_pinyin_carrier",
        "pronunciation_processed": True,
        "pronunciation_rule": f"zh.{rule_name}",
        "pronunciation_variant_id": variant_id,
        "pronunciation_instruction": None,
        "target_confusions": list(entry.known_confusions),
        "confusion_category": "zh_fuzzy_pinyin",
        "pronunciation_delta": delta,
        "variant_kind": rule_name,
        "language": "Chinese",
        "tags": [
            "gt",
            f"source-lines:{','.join(map(str, entry.source_lines))}",
            "pronunciation:fuzzy-pinyin",
            f"rule-level:{rule.level}",
            f"rule-origin:{rule_origin}",
            f"carrier-source:{carrier.source}",
        ],
    }


def _observed_rule_keys(
    entries: Iterable[Any], rules: Iterable[FuzzyRule], overrides: dict[str, tuple[str, ...]]
) -> set[tuple[str, str, str]]:
    rules = list(rules)
    observed: set[tuple[str, str, str]] = set()
    for entry in entries:
        source = str(entry.canonical_text)
        if not _is_pure_han(source):
            continue
        source_pinyin = _tones(source, overrides.get(source))
        for confusion in entry.known_confusions:
            target = str(confusion)
            if not _is_pure_han(target):
                continue
            target_pinyin = _tones(target, None)
            if len(source_pinyin) != len(target_pinyin):
                continue
            changed = [
                index for index, values in enumerate(zip(source_pinyin, target_pinyin))
                if values[0] != values[1]
            ]
            if len(changed) != 1:
                continue
            index = changed[0]
            source_initial, source_final, _ = _split_syllable(source_pinyin[index])
            target_initial, target_final, _ = _split_syllable(target_pinyin[index])
            for rule in rules:
                if (
                    rule.rule_type == "initial"
                    and source_initial == rule.source
                    and target_initial == rule.target
                    and source_final == target_final
                ) or (
                    rule.rule_type == "final"
                    and source_final == rule.source
                    and target_final == rule.target
                    and source_initial == target_initial
                ):
                    observed.add(rule.key)
    return observed


def _load_overrides(root: Path, value: Any) -> dict[str, tuple[str, ...]]:
    if not value:
        return {}
    payload = value
    if isinstance(value, str):
        path = _configured_path(root, value)
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pronunciation.pronunciation_overrides must be a mapping or JSON path")
    overrides: dict[str, tuple[str, ...]] = {}
    for term, syllables in payload.items():
        if not isinstance(syllables, list) or not syllables:
            raise ValueError(f"pronunciation override for {term!r} must be a non-empty pinyin list")
        values = tuple(_normalize_tone(str(item)) for item in syllables)
        if any(not _has_tone(item) for item in values):
            raise ValueError(f"pronunciation override for {term!r} must use tone3 pinyin")
        overrides[str(term)] = values
    return overrides


def _tones(text: str, override: tuple[str, ...] | None) -> tuple[str, ...]:
    if override is not None:
        return override
    values = lazy_pinyin(
        text,
        style=Style.TONE3,
        neutral_tone_with_five=True,
        errors="default",
    )
    result = tuple(_normalize_tone(value) for value in values)
    if not result or any(not _has_tone(value) for value in result):
        raise ValueError(f"cannot derive tone3 pinyin for {text!r}")
    return result


def _split_syllable(syllable: str) -> tuple[str, str, str]:
    normalized = _normalize_tone(syllable)
    tone = normalized[-1]
    body = normalized[:-1]
    for initial in _INITIALS:
        if body.startswith(initial):
            return initial, body[len(initial) :], tone
    return "", body, tone


def _normalize_tone(value: str) -> str:
    return value.strip().casefold().replace("u:", "v").replace("ü", "v")


def _normalize_plain(value: str) -> str:
    return _normalize_tone(value).rstrip("12345")


def _tone_less(value: str) -> str:
    return value[:-1] if _has_tone(value) else value


def _has_tone(value: str) -> bool:
    return len(value) >= 2 and value[-1] in "12345"


def _is_pure_han(value: str) -> bool:
    return bool(value) and all(_HAN_MIN <= character <= _HAN_MAX for character in value)


def _configured_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _dedupe_candidates(values: Iterable[CarrierCandidate]) -> list[CarrierCandidate]:
    best: dict[str, CarrierCandidate] = {}
    for value in values:
        previous = best.get(value.text)
        if previous is None or value.score > previous.score:
            best[value.text] = value
    return sorted(best.values(), key=lambda item: (-item.score, item.text))
