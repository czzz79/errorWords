from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


class PronunciationError(ValueError):
    """A sample or rule cannot be converted into a reliable pronunciation."""


_ARPABET_VOWEL = re.compile(r"^[A-Z]+([012])$")
_PINYIN_TONE = re.compile(r"^([a-züv]+)([1-5])$", re.IGNORECASE)
_ENGLISH_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_HAN_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_PINYIN_INITIALS = (
    "zh",
    "ch",
    "sh",
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "j",
    "q",
    "x",
    "r",
    "z",
    "c",
    "s",
    "y",
    "w",
)

_LETTER_PHONEMES = {
    "a": ["EY1"],
    "b": ["B", "IY1"],
    "c": ["S", "IY1"],
    "d": ["D", "IY1"],
    "e": ["IY1"],
    "f": ["EH1", "F"],
    "g": ["JH", "IY1"],
    "h": ["EY1", "CH"],
    "i": ["AY1"],
    "j": ["JH", "EY1"],
    "k": ["K", "EY1"],
    "l": ["EH1", "L"],
    "m": ["EH1", "M"],
    "n": ["EH1", "N"],
    "o": ["OW1"],
    "p": ["P", "IY1"],
    "q": ["K", "Y", "UW1"],
    "r": ["AA1", "R"],
    "s": ["EH1", "S"],
    "t": ["T", "IY1"],
    "u": ["Y", "UW1"],
    "v": ["V", "IY1"],
    "w": ["D", "AH1", "B", "AH0", "L", "Y", "UW0"],
    "x": ["EH1", "K", "S"],
    "y": ["W", "AY1"],
    "z": ["Z", "IY1"],
}


@dataclass(frozen=True, slots=True)
class RuleDefinition:
    rule_id: str
    language: str
    operation: str
    tags: tuple[str, ...]
    tts_renderability: str

    @classmethod
    def from_mapping(cls, value: Any, path: Path, index: int) -> "RuleDefinition":
        if not isinstance(value, dict):
            raise PronunciationError(f"{path}: rule {index} must be a JSON object")
        rule_id = str(value.get("rule_id", "")).strip()
        language = str(value.get("language", "")).strip().lower()
        operation = str(value.get("operation", "")).strip()
        renderability = str(value.get("tts_renderability", "")).strip()
        tags = value.get("tags", [])
        if not rule_id or not operation:
            raise PronunciationError(f"{path}: rule {index} needs rule_id and operation")
        if language not in {"english", "chinese"}:
            raise PronunciationError(f"{path}: unsupported rule language: {language}")
        if renderability not in {"text_approximation", "phoneme_required"}:
            raise PronunciationError(
                f"{path}: unsupported tts_renderability for {rule_id}: {renderability}"
            )
        if not isinstance(tags, list):
            raise PronunciationError(f"{path}: {rule_id}.tags must be a JSON array")
        return cls(
            rule_id=rule_id,
            language=language,
            operation=operation,
            tags=tuple(str(tag) for tag in tags),
            tts_renderability=renderability,
        )


@dataclass(frozen=True, slots=True)
class PronunciationSample:
    sample_id: str
    canonical_text: str
    synthesis_text: str
    language: str
    tags: tuple[str, ...]
    pronunciation: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "PronunciationSample":
        if not isinstance(value, dict):
            raise PronunciationError(f"sample at index {index} must be a JSON object")
        synthesis_text = str(value.get("text", "")).strip()
        canonical_text = str(value.get("canonical_text", synthesis_text)).strip()
        sample_id = str(value.get("id", f"sample-{index + 1:03d}")).strip()
        language = str(value.get("language", "Auto")).strip()
        tags = value.get("tags", [])
        pronunciation = value.get("pronunciation", {})
        if not sample_id or not synthesis_text or not canonical_text:
            raise PronunciationError(f"sample at index {index} has an empty id or text")
        if not isinstance(tags, list):
            raise PronunciationError(f"sample {sample_id} tags must be a JSON array")
        if not isinstance(pronunciation, dict):
            raise PronunciationError(f"sample {sample_id} pronunciation must be an object")
        return cls(
            sample_id=sample_id,
            canonical_text=canonical_text,
            synthesis_text=synthesis_text,
            language=language,
            tags=tuple(str(tag) for tag in tags),
            pronunciation=pronunciation,
        )


def load_samples(path: Path) -> list[PronunciationSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise PronunciationError(f"samples must be a non-empty JSON array: {path}")
    return [PronunciationSample.from_mapping(item, index) for index, item in enumerate(payload)]


def load_rules(paths: Iterable[Path]) -> list[RuleDefinition]:
    rules: list[RuleDefinition] = []
    seen: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise PronunciationError(f"rules must be a JSON array: {path}")
        for index, value in enumerate(payload):
            rule = RuleDefinition.from_mapping(value, path, index)
            if rule.rule_id in seen:
                raise PronunciationError(f"duplicate rule_id: {rule.rule_id}")
            seen.add(rule.rule_id)
            rules.append(rule)
    if not rules:
        raise PronunciationError("no pronunciation rules were loaded")
    return rules


def generate_variants(
    sample: PronunciationSample,
    rules: Iterable[RuleDefinition],
    *,
    max_rule_count: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if max_rule_count not in {1, 2}:
        raise PronunciationError("max_rule_count must be 1 or 2")
    language = _normalize_language(sample)
    base = _resolve_pronunciation(sample, language)
    applicable_rules = [rule for rule in rules if rule.language == language]
    candidates: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    seen_pronunciations: set[str] = set()
    first_steps: list[
        tuple[dict[str, Any], list[dict[str, Any]], str, str]
    ] = []
    for rule in applicable_rules:
        produced = _apply_rule(sample, base, rule)
        kept = 0
        for variant, operation, display_text in produced:
            rule_chain = [_rule_step(rule, operation)]
            first_steps.append(
                (variant, rule_chain, display_text, rule.tts_renderability)
            )
            key = json.dumps(variant, ensure_ascii=False, sort_keys=True)
            if key in seen_pronunciations or variant == base:
                continue
            seen_pronunciations.add(key)
            candidates.append(
                _variant_row(
                    sample,
                    base,
                    variant,
                    rule_chain,
                    display_text,
                    rule.tts_renderability,
                )
            )
            kept += 1
        counts[rule.rule_id] = kept
    if max_rule_count == 2:
        composed_count = 0
        for intermediate, first_chain, _, _ in first_steps:
            for rule in applicable_rules:
                for variant, operation, display_text in _apply_rule(sample, intermediate, rule):
                    if variant == intermediate or variant == base:
                        continue
                    key = json.dumps(variant, ensure_ascii=False, sort_keys=True)
                    if key in seen_pronunciations:
                        continue
                    seen_pronunciations.add(key)
                    rule_chain = [*first_chain, _rule_step(rule, operation)]
                    candidates.append(
                        _variant_row(
                            sample,
                            base,
                            variant,
                            rule_chain,
                            display_text,
                            "phoneme_required",
                        )
                    )
                    composed_count += 1
        counts["composed.2"] = composed_count
    return candidates, counts


def _normalize_language(sample: PronunciationSample) -> str:
    lowered = sample.language.casefold()
    if lowered.startswith("en") or lowered == "english":
        return "english"
    if lowered.startswith("zh") or lowered in {"chinese", "mandarin"}:
        return "chinese"
    if lowered == "auto":
        return "chinese" if _HAN_CHARACTER.search(sample.synthesis_text) else "english"
    raise PronunciationError(f"sample {sample.sample_id}: unsupported language {sample.language!r}")


def _resolve_pronunciation(sample: PronunciationSample, language: str) -> dict[str, Any]:
    override = sample.pronunciation
    if language == "english":
        return _resolve_english(sample, override)
    return _resolve_chinese(sample, override)


def _resolve_english(sample: PronunciationSample, override: dict[str, Any]) -> dict[str, Any]:
    alphabet = str(override.get("alphabet", "arpabet")).casefold()
    raw_override_tokens = override.get("override_tokens")
    if raw_override_tokens is not None:
        if alphabet != "arpabet" or not isinstance(raw_override_tokens, list) or not raw_override_tokens:
            raise PronunciationError(
                f"sample {sample.sample_id}: English override_tokens require alphabet=arpabet"
            )
        tokens = []
        for index, item in enumerate(raw_override_tokens):
            if not isinstance(item, dict):
                raise PronunciationError(
                    f"sample {sample.sample_id}: override token {index} must be an object"
                )
            text = str(item.get("text", "")).strip()
            phonemes = _clean_string_list(item.get("phonemes"), "phonemes", sample.sample_id)
            if not text:
                raise PronunciationError(
                    f"sample {sample.sample_id}: override token {index} has empty text"
                )
            tokens.append({"text": text, "phonemes": phonemes})
        return {"alphabet": "arpabet", "tokens": tokens, "boundaries": ["word"] * (len(tokens) - 1)}

    token_hints = override.get("tokens")
    if token_hints is not None:
        words = _clean_string_list(token_hints, "tokens", sample.sample_id)
    else:
        words = _split_english_words(sample.synthesis_text)
    if not words:
        raise PronunciationError(f"sample {sample.sample_id}: no English words found")
    try:
        import cmudict
    except ImportError as exc:
        raise PronunciationError("install pronunciation dependencies: pip install cmudict pypinyin") from exc
    dictionary = cmudict.dict()
    tokens = []
    for word in words:
        entries = dictionary.get(word.casefold())
        if not entries:
            # CMUdict intentionally has limited acronym/initialism coverage.
            # For an all-uppercase token, use canonical English letter names;
            # this remains target-blind and is preferable to dropping the
            # complete term from the CMU experiment.
            if word.isalpha() and word.isupper() and all(
                character.casefold() in _LETTER_PHONEMES for character in word
            ):
                spelled: list[str] = []
                for character in word.casefold():
                    spelled.extend(_LETTER_PHONEMES[character])
                tokens.append({"text": word, "phonemes": spelled})
                continue
            raise PronunciationError(
                f"sample {sample.sample_id}: CMU Dictionary has no pronunciation for {word!r}; "
                "add pronunciation.override_tokens"
            )
        tokens.append({"text": word, "phonemes": list(entries[0])})
    return {"alphabet": "arpabet", "tokens": tokens, "boundaries": ["word"] * (len(tokens) - 1)}


def _resolve_chinese(sample: PronunciationSample, override: dict[str, Any]) -> dict[str, Any]:
    alphabet = str(override.get("alphabet", "pinyin-tone3")).casefold()
    raw_syllables = override.get("syllables")
    characters = [character for character in sample.synthesis_text if _HAN_CHARACTER.fullmatch(character)]
    if raw_syllables is not None:
        if alphabet != "pinyin-tone3":
            raise PronunciationError(
                f"sample {sample.sample_id}: Chinese syllables require alphabet=pinyin-tone3"
            )
        syllables = _clean_string_list(raw_syllables, "syllables", sample.sample_id)
        if len(syllables) != len(characters):
            raise PronunciationError(
                f"sample {sample.sample_id}: pronunciation.syllables must match Han character count"
            )
    else:
        if not characters:
            raise PronunciationError(f"sample {sample.sample_id}: no Chinese characters found")
        try:
            from pypinyin import Style, pinyin
        except ImportError as exc:
            raise PronunciationError("install pronunciation dependencies: pip install cmudict pypinyin") from exc
        readings = pinyin(
            "".join(characters),
            style=Style.TONE3,
            heteronym=True,
            neutral_tone_with_five=True,
            errors="default",
        )
        ambiguous = [characters[index] for index, values in enumerate(readings) if len(values) != 1]
        if ambiguous:
            joined = "、".join(ambiguous)
            raise PronunciationError(
                f"sample {sample.sample_id}: ambiguous Chinese pronunciation for {joined}; "
                "add pronunciation.syllables"
            )
        syllables = [values[0].replace("u:", "v") for values in readings]
    for syllable in syllables:
        if not _PINYIN_TONE.fullmatch(syllable):
            raise PronunciationError(
                f"sample {sample.sample_id}: invalid pinyin-tone3 syllable {syllable!r}"
            )
    tokens = [
        {"text": character, "phonemes": [syllable]}
        for character, syllable in zip(characters, syllables)
    ]
    return {"alphabet": "pinyin-tone3", "tokens": tokens, "boundaries": ["syllable"] * (len(tokens) - 1)}


def _split_english_words(text: str) -> list[str]:
    expanded = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return _ENGLISH_WORD.findall(expanded)


def _clean_string_list(value: Any, field: str, sample_id: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise PronunciationError(f"sample {sample_id}: pronunciation.{field} must be non-empty")
    cleaned = [str(item).strip() for item in value]
    if any(not item for item in cleaned):
        raise PronunciationError(f"sample {sample_id}: pronunciation.{field} contains an empty item")
    return cleaned


def _apply_rule(
    sample: PronunciationSample,
    base: dict[str, Any],
    rule: RuleDefinition,
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    operations: dict[
        str,
        Callable[[PronunciationSample, dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any], str]]],
    ] = {
        "en_unstressed_vowel_reduction": _en_unstressed_vowel_reduction,
        "en_h_deletion": _en_h_deletion,
        "en_final_consonant_deletion": _en_final_consonant_deletion,
        "en_final_stop_devoicing": _en_final_stop_devoicing,
        "en_consonant_substitution": _en_consonant_substitution,
        "en_connected_boundary": _en_connected_boundary,
        "en_letter_spelling": _en_letter_spelling,
        "en_unstressed_syllable_deletion": _en_unstressed_syllable_deletion,
        "zh_initial_n_l": lambda s, b: _zh_initial_substitution(s, b, {"n": "l", "l": "n"}),
        "zh_retroflex_flattening": lambda s, b: _zh_initial_substitution(
            s, b, {"zh": "z", "ch": "c", "sh": "s"}
        ),
        "zh_nasal_final_confusion": _zh_nasal_final_confusion,
        "zh_nasal_final_weakening": _zh_nasal_final_weakening,
        "zh_tone_substitution": _zh_tone_substitution,
        "zh_tone_neutralization": _zh_tone_neutralization,
        "zh_syllable_deletion": _zh_syllable_deletion,
    }
    function = operations.get(rule.operation)
    if function is None:
        raise PronunciationError(f"unsupported pronunciation operation: {rule.operation}")
    return function(sample, base)


def _en_unstressed_vowel_reduction(sample: PronunciationSample, base: dict[str, Any]):
    results = []
    for token_index, token in enumerate(base["tokens"]):
        for unit_index, phoneme in enumerate(token["phonemes"]):
            match = _ARPABET_VOWEL.fullmatch(phoneme)
            if match and match.group(1) == "0" and phoneme != "AH0":
                variant = deepcopy(base)
                variant["tokens"][token_index]["phonemes"][unit_index] = "AH0"
                operation = _operation("replace", token_index, unit_index, [phoneme], ["AH0"])
                results.append((variant, operation, sample.synthesis_text))
    return results


def _en_h_deletion(sample: PronunciationSample, base: dict[str, Any]):
    results = []
    for token_index, token in enumerate(base["tokens"]):
        for unit_index, phoneme in enumerate(token["phonemes"]):
            if phoneme != "HH":
                continue
            variant = deepcopy(base)
            del variant["tokens"][token_index]["phonemes"][unit_index]
            text_tokens = [item["text"] for item in base["tokens"]]
            if text_tokens[token_index].casefold().startswith("h"):
                text_tokens[token_index] = text_tokens[token_index][1:]
            operation = _operation("delete", token_index, unit_index, [phoneme], [])
            results.append((variant, operation, " ".join(text_tokens)))
    return results


def _en_final_consonant_deletion(sample: PronunciationSample, base: dict[str, Any]):
    results = []
    for token_index, token in enumerate(base["tokens"]):
        phonemes = token["phonemes"]
        if not phonemes or _is_arpabet_vowel(phonemes[-1]):
            continue
        variant = deepcopy(base)
        removed = variant["tokens"][token_index]["phonemes"].pop()
        operation = _operation("delete", token_index, len(phonemes) - 1, [removed], [])
        results.append((variant, operation, sample.synthesis_text))
    return results


def _en_final_stop_devoicing(sample: PronunciationSample, base: dict[str, Any]):
    mapping = {"B": "P", "D": "T", "G": "K", "V": "F", "Z": "S", "JH": "CH", "DH": "TH"}
    results = []
    for token_index, token in enumerate(base["tokens"]):
        phonemes = token["phonemes"]
        replacement = mapping.get(phonemes[-1]) if phonemes else None
        if replacement is None:
            continue
        variant = deepcopy(base)
        variant["tokens"][token_index]["phonemes"][-1] = replacement
        text_tokens = [item["text"] for item in base["tokens"]]
        if text_tokens[token_index].casefold().endswith("b") and replacement == "P":
            text_tokens[token_index] = text_tokens[token_index][:-1] + "p"
        operation = _operation(
            "replace", token_index, len(phonemes) - 1, [phonemes[-1]], [replacement]
        )
        results.append((variant, operation, " ".join(text_tokens)))
    return results


def _en_consonant_substitution(sample: PronunciationSample, base: dict[str, Any]):
    mapping = {"B": "P", "P": "B", "D": "T", "T": "D", "R": "L", "L": "R"}
    results = []
    for token_index, token in enumerate(base["tokens"]):
        for unit_index, phoneme in enumerate(token["phonemes"]):
            replacement = mapping.get(phoneme)
            if replacement is None:
                continue
            variant = deepcopy(base)
            variant["tokens"][token_index]["phonemes"][unit_index] = replacement
            operation = _operation("replace", token_index, unit_index, [phoneme], [replacement])
            results.append((variant, operation, sample.synthesis_text))
    return results


def _en_connected_boundary(sample: PronunciationSample, base: dict[str, Any]):
    results = []
    for boundary_index, boundary in enumerate(base.get("boundaries", [])):
        if boundary != "word":
            continue
        variant = deepcopy(base)
        variant["boundaries"][boundary_index] = "connected"
        operation = {
            "type": "replace_boundary",
            "boundary_index": boundary_index,
            "from": [boundary],
            "to": ["connected"],
        }
        display = "".join(item["text"] for item in base["tokens"])
        results.append((variant, operation, display))
    return results


def _en_letter_spelling(sample: PronunciationSample, base: dict[str, Any]):
    results = []
    for token_index, token in enumerate(base["tokens"]):
        letters = [character.casefold() for character in token["text"] if character.isalpha()]
        if not letters or any(letter not in _LETTER_PHONEMES for letter in letters):
            continue
        spelled: list[str] = []
        for letter in letters:
            spelled.extend(_LETTER_PHONEMES[letter])
        variant = deepcopy(base)
        original = list(variant["tokens"][token_index]["phonemes"])
        variant["tokens"][token_index]["phonemes"] = spelled
        text_tokens = [item["text"] for item in base["tokens"]]
        text_tokens[token_index] = " ".join(character.upper() for character in letters)
        operation = {
            "type": "spell_letters",
            "token_index": token_index,
            "from": original,
            "to": spelled,
        }
        results.append((variant, operation, " ".join(text_tokens)))
    return results


def _en_unstressed_syllable_deletion(sample: PronunciationSample, base: dict[str, Any]):
    results = []
    for token_index, token in enumerate(base["tokens"]):
        phonemes = token["phonemes"]
        nuclei = [index for index, phoneme in enumerate(phonemes) if _is_arpabet_vowel(phoneme)]
        if len(nuclei) <= 1:
            continue
        for nucleus_position, nucleus_index in enumerate(nuclei):
            match = _ARPABET_VOWEL.fullmatch(phonemes[nucleus_index])
            if match is None or match.group(1) != "0":
                continue
            start = 0 if nucleus_position == 0 else nuclei[nucleus_position - 1] + 1
            end = nucleus_index + 1
            removed = phonemes[start:end]
            variant = deepcopy(base)
            del variant["tokens"][token_index]["phonemes"][start:end]
            if not variant["tokens"][token_index]["phonemes"]:
                continue
            operation = {
                "type": "delete_syllable",
                "token_index": token_index,
                "unit_index": start,
                "from": removed,
                "to": [],
            }
            results.append((variant, operation, sample.synthesis_text))
    return results


def _zh_initial_substitution(
    sample: PronunciationSample,
    base: dict[str, Any],
    mapping: dict[str, str],
):
    results = []
    for token_index, token in enumerate(base["tokens"]):
        syllable = token["phonemes"][0]
        initial, final, tone = _split_pinyin(syllable)
        replacement = mapping.get(initial)
        if replacement is None:
            continue
        changed = f"{replacement}{final}{tone}"
        variant = deepcopy(base)
        variant["tokens"][token_index]["phonemes"] = [changed]
        operation = _operation("replace", token_index, 0, [syllable], [changed])
        results.append((variant, operation, _pinyin_display(variant)))
    return results


def _zh_nasal_final_confusion(sample: PronunciationSample, base: dict[str, Any]):
    pairs = (("ing", "in"), ("eng", "en"), ("ang", "an"), ("in", "ing"), ("en", "eng"), ("an", "ang"))
    results = []
    for token_index, token in enumerate(base["tokens"]):
        syllable = token["phonemes"][0]
        initial, final, tone = _split_pinyin(syllable)
        replacement = next(
            (final[: -len(suffix)] + target for suffix, target in pairs if final.endswith(suffix)),
            None,
        )
        if replacement is None:
            continue
        changed = f"{initial}{replacement}{tone}"
        variant = deepcopy(base)
        variant["tokens"][token_index]["phonemes"] = [changed]
        operation = _operation("replace", token_index, 0, [syllable], [changed])
        results.append((variant, operation, _pinyin_display(variant)))
    return results


def _zh_nasal_final_weakening(sample: PronunciationSample, base: dict[str, Any]):
    results = []
    for token_index, token in enumerate(base["tokens"]):
        syllable = token["phonemes"][0]
        initial, final, tone = _split_pinyin(syllable)
        if final.endswith("ng"):
            weakened = final[:-2]
        elif final.endswith("n"):
            weakened = final[:-1]
        else:
            continue
        if not weakened:
            continue
        changed = f"{initial}{weakened}{tone}"
        variant = deepcopy(base)
        variant["tokens"][token_index]["phonemes"] = [changed]
        operation = _operation("delete_final_nasal", token_index, 0, [syllable], [changed])
        results.append((variant, operation, _pinyin_display(variant)))
    return results


def _zh_tone_neutralization(sample: PronunciationSample, base: dict[str, Any]):
    results = []
    for token_index, token in enumerate(base["tokens"]):
        syllable = token["phonemes"][0]
        initial, final, tone = _split_pinyin(syllable)
        if tone == "5":
            continue
        changed = f"{initial}{final}5"
        variant = deepcopy(base)
        variant["tokens"][token_index]["phonemes"] = [changed]
        operation = _operation("replace_tone", token_index, 0, [tone], ["5"])
        results.append((variant, operation, _pinyin_display(variant)))
    return results


def _zh_tone_substitution(sample: PronunciationSample, base: dict[str, Any]):
    results = []
    for token_index, token in enumerate(base["tokens"]):
        syllable = token["phonemes"][0]
        initial, final, tone = _split_pinyin(syllable)
        if tone == "5":
            continue
        for replacement in ("1", "2", "3", "4"):
            if replacement == tone:
                continue
            changed = f"{initial}{final}{replacement}"
            variant = deepcopy(base)
            variant["tokens"][token_index]["phonemes"] = [changed]
            operation = _operation("replace_tone", token_index, 0, [tone], [replacement])
            results.append((variant, operation, _pinyin_display(variant)))
    return results


def _zh_syllable_deletion(sample: PronunciationSample, base: dict[str, Any]):
    if len(base["tokens"]) <= 1:
        return []
    results = []
    for token_index, token in enumerate(base["tokens"]):
        variant = deepcopy(base)
        removed = variant["tokens"].pop(token_index)
        if variant["boundaries"]:
            boundary_index = min(token_index, len(variant["boundaries"]) - 1)
            variant["boundaries"].pop(boundary_index)
        display = "".join(item["text"] for item in variant["tokens"])
        operation = {
            "type": "delete_syllable",
            "token_index": token_index,
            "from": removed["phonemes"],
            "to": [],
        }
        results.append((variant, operation, display))
    return results


def _variant_row(
    sample: PronunciationSample,
    base: dict[str, Any],
    variant: dict[str, Any],
    rule_chain: list[dict[str, Any]],
    display_text: str,
    tts_renderability: str,
) -> dict[str, Any]:
    if len(rule_chain) == 1:
        rule_summary = rule_chain[0]
    else:
        rule_summary = {
            "rule_id": "composed.2",
            "operation": "composed",
            "tags": list(
                dict.fromkeys(
                    ["composition:2"]
                    + [tag for step in rule_chain for tag in step["tags"]]
                )
            ),
            "applied_change": {
                "type": "composed",
                "steps": [step["applied_change"] for step in rule_chain],
            },
        }
    identity = {
        "sample_id": sample.sample_id,
        "variant_pronunciation": variant,
        "rule_chain": rule_chain,
    }
    variant_id = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "variant_id": variant_id,
        "sample_id": sample.sample_id,
        "canonical_text": sample.canonical_text,
        "synthesis_text": sample.synthesis_text,
        "display_text": display_text,
        "language": sample.language,
        "sample_tags": list(sample.tags),
        "alphabet": base["alphabet"],
        "base_pronunciation": base,
        "variant_pronunciation": variant,
        "rule": rule_summary,
        "rule_chain": rule_chain,
        "tts_renderability": tts_renderability,
        "source": "rule_generated",
    }


def _rule_step(rule: RuleDefinition, operation: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_id": rule.rule_id,
        "operation": rule.operation,
        "tags": list(rule.tags),
        "applied_change": operation,
    }


def _operation(
    operation_type: str,
    token_index: int,
    unit_index: int,
    before: list[str],
    after: list[str],
) -> dict[str, Any]:
    return {
        "type": operation_type,
        "token_index": token_index,
        "unit_index": unit_index,
        "from": before,
        "to": after,
    }


def _is_arpabet_vowel(phoneme: str) -> bool:
    return _ARPABET_VOWEL.fullmatch(phoneme) is not None


def _split_pinyin(syllable: str) -> tuple[str, str, str]:
    match = _PINYIN_TONE.fullmatch(syllable)
    if match is None:
        raise PronunciationError(f"invalid pinyin-tone3 syllable: {syllable}")
    base, tone = match.groups()
    initial = next((item for item in _PINYIN_INITIALS if base.startswith(item)), "")
    return initial, base[len(initial) :], tone


def _pinyin_display(pronunciation: dict[str, Any]) -> str:
    return " ".join(token["phonemes"][0] for token in pronunciation["tokens"])


def arpabet_to_cosyvoice(pronunciation: dict[str, Any]) -> str:
    """Render an ARPAbet pronunciation using CosyVoice special tokens.

    CosyVoice's tokenizer registers tokens such as ``[B]`` and ``[AY1]``.
    Keep word boundaries as spaces; the caller must disable the normal text
    frontend when passing this representation to the model.
    """
    if pronunciation.get("alphabet") != "arpabet":
        raise PronunciationError("CosyVoice ARPAbet rendering requires alphabet=arpabet")
    rendered: list[str] = []
    for token in pronunciation.get("tokens", []):
        phonemes = token.get("phonemes", [])
        if not isinstance(phonemes, list) or not phonemes:
            raise PronunciationError("ARPAbet pronunciation contains an empty token")
        rendered.append("".join(f"[{str(phoneme).upper()}]" for phoneme in phonemes))
    if not rendered:
        raise PronunciationError("ARPAbet pronunciation contains no tokens")
    return " ".join(rendered)
