from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pypinyin import Style, lazy_pinyin


CATEGORY_FILES = {
    "normalization_alias": "normalization-alias.txt",
    "number_reading_mixed": "number-reading-mixed.txt",
    "cross_language_transliteration": "cross-language-transliteration.txt",
    "english_word_acronym": "english-word-acronym.txt",
    "zh_same_pinyin": "zh-same-pinyin.txt",
    "zh_single_syllable": "zh-single-syllable.txt",
    "zh_multi_syllable_near": "zh-multi-syllable-near.txt",
    "zh_distant_semantic": "zh-distant-semantic.txt",
    "other_review": "other-review.txt",
}

PREPROCESSABLE_CATEGORIES = {
    "number_reading_mixed",
    "cross_language_transliteration",
    "english_word_acronym",
    "zh_same_pinyin",
    "zh_single_syllable",
}

_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"\d")
_NUMBER_TOKEN = re.compile(r"[a-z]+|\d+|[\u3400-\u9fff]|.", re.IGNORECASE)
_CAMEL_BOUNDARY_1 = re.compile(r"(?<=[a-z])(?=[A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_SCRIPT_BOUNDARY = re.compile(
    r"(?<=[A-Za-z0-9])(?=[\u3400-\u9fff])|(?<=[\u3400-\u9fff])(?=[A-Za-z0-9])"
)
_ALNUM_BOUNDARY = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
_PINYIN = re.compile(r"^([a-zv]+?)([1-5])?$", re.IGNORECASE)
_INITIALS = (
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "z", "c", "s", "y", "w",
)

_CHINESE_NUMERALS = set("零〇一二两三四五六七八九十百千万亿幺")
_ENGLISH_NUMBER_WORDS = {
    "zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty",
    "seventy", "eighty", "ninety", "hundred", "thousand", "million",
}
_ZH_DIGITS = "零一二三四五六七八九"


def classify_entries(entries: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        for candidate_index, confusion in enumerate(entry.known_confusions):
            row = classify_pair(entry.canonical_text, confusion)
            row.update(
                {
                    "pair_id": _stable_id(entry.canonical_text, confusion),
                    "sample_id": entry.sample_id,
                    "source_lines": list(entry.source_lines),
                    "candidate_index": candidate_index,
                }
            )
            rows.append(row)
    return rows


def classify_pair(canonical: str, confusion: str) -> dict[str, Any]:
    canonical_normalized = _normalize(canonical)
    confusion_normalized = _normalize(confusion)
    canonical_scripts = _script_profile(canonical)
    confusion_scripts = _script_profile(confusion)
    canonical_pinyin = _pinyin(canonical, tone=False) if _is_pure_han(canonical) else []
    confusion_pinyin = _pinyin(confusion, tone=False) if _is_pure_han(confusion) else []
    canonical_pinyin_tone = _pinyin(canonical, tone=True) if canonical_pinyin else []
    confusion_pinyin_tone = _pinyin(confusion, tone=True) if confusion_pinyin else []
    distance: int | None = None
    normalized_distance: float | None = None

    if _compact(canonical) == _compact(confusion):
        category = "normalization_alias"
        reason = "NFKC/casefold 后移除空格和标点，两侧完全相同"
    elif _is_number_reading_pair(canonical, confusion):
        category = "number_reading_mixed"
        reason = "至少一侧含阿拉伯数字，数字归一化后非数字骨架一致"
    elif _is_cross_language(canonical_scripts, confusion_scripts):
        category = "cross_language_transliteration"
        reason = "中文、英文或中英混合书写类型发生切换"
    elif not canonical_scripts["han"] and not confusion_scripts["han"] and (
        canonical_scripts["latin"] or confusion_scripts["latin"]
    ):
        category = "english_word_acronym"
        reason = "两侧均无汉字且至少一侧含拉丁字母"
    elif canonical_pinyin and confusion_pinyin:
        distance = _edit_distance(canonical_pinyin, confusion_pinyin)
        normalized_distance = distance / max(len(canonical_pinyin), len(confusion_pinyin), 1)
        if distance == 0:
            category = "zh_same_pinyin"
            reason = "忽略声调后的逐字拼音序列相同"
        elif distance == 1:
            category = "zh_single_syllable"
            reason = "忽略声调后的拼音音节序列编辑距离为 1"
        elif normalized_distance <= 0.75:
            category = "zh_multi_syllable_near"
            reason = "多音节变化且归一化音节编辑距离不超过 0.75"
        else:
            category = "zh_distant_semantic"
            reason = "纯中文但归一化音节编辑距离大于 0.75"
    else:
        category = "other_review"
        reason = "复杂数字、符号或混合形式，无法可靠归入其他类别"

    tone_differences = [
        index
        for index, (source, target) in enumerate(zip(canonical_pinyin_tone, confusion_pinyin_tone))
        if source != target and _strip_tone(source) == _strip_tone(target)
    ]
    return {
        "canonical_text": canonical,
        "confusion_text": confusion,
        "canonical_normalized": canonical_normalized,
        "confusion_normalized": confusion_normalized,
        "canonical_scripts": canonical_scripts,
        "confusion_scripts": confusion_scripts,
        "canonical_pinyin": canonical_pinyin,
        "confusion_pinyin": confusion_pinyin,
        "canonical_pinyin_tone": canonical_pinyin_tone,
        "confusion_pinyin_tone": confusion_pinyin_tone,
        "syllable_edit_distance": distance,
        "normalized_syllable_distance": normalized_distance,
        "tone_difference_indices": tone_differences,
        "category": category,
        "reason": reason,
    }


def write_classification_outputs(entries: Iterable[Any], output_dir: Path) -> dict[str, Any]:
    entry_list = list(entries)
    rows = classify_entries(entry_list)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    for category, filename in CATEGORY_FILES.items():
        _write_pair_txt(output_dir / filename, by_category.get(category, []))
    preprocessable = [row for row in rows if row["category"] in PREPROCESSABLE_CATEGORIES]
    _write_pair_txt(output_dir / "preprocessable.txt", preprocessable)

    details_path = output_dir / "classification.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter(row["category"] for row in rows)
    summary = {
        "term_count": len(entry_list),
        "pair_count": len(rows),
        "preprocessable_pair_count": len(preprocessable),
        "category_counts": {category: counts.get(category, 0) for category in CATEGORY_FILES},
        "category_files": CATEGORY_FILES,
        "preprocessable_categories": sorted(PREPROCESSABLE_CATEGORIES),
        "classification_details": str(details_path.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_gt_directed_variants(
    entries: Iterable[Any],
    classification_rows: list[dict[str, Any]],
    *,
    target_categories: set[str] | None = None,
) -> list[dict[str, Any]]:
    allowed = target_categories or set(PREPROCESSABLE_CATEGORIES)
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entries_by_sample = {entry.sample_id: entry for entry in entries}
    for row in classification_rows:
        if row["category"] in allowed:
            by_sample[row["sample_id"]].append(row)

    variants: list[dict[str, Any]] = []
    for sample_id, pairs in by_sample.items():
        entry = entries_by_sample[sample_id]
        single_pairs = [row for row in pairs if row["category"] == "zh_single_syllable"]
        for pair in single_pairs:
            variants.extend(_single_syllable_variants(entry, pair))
        for category, builder in (
            ("english_word_acronym", _english_variants),
            ("cross_language_transliteration", _cross_language_variants),
            ("number_reading_mixed", _number_variants),
        ):
            targets = [row for row in pairs if row["category"] == category]
            if targets:
                variants.extend(builder(entry, targets))
    return variants


def add_homophone_metadata(
    canonical_rows: list[dict[str, Any]], classification_rows: list[dict[str, Any]]
) -> None:
    targets: dict[str, list[str]] = defaultdict(list)
    for row in classification_rows:
        if row["category"] == "zh_same_pinyin":
            targets[row["sample_id"]].append(row["confusion_text"])
    for row in canonical_rows:
        values = targets.get(str(row.get("id")))
        if values:
            row.update(
                {
                    "target_confusions": values,
                    "confusion_category": "zh_same_pinyin",
                    "variant_kind": "canonical_homophone_baseline",
                    "pronunciation_delta": {"type": "same_pinyin", "tone_ignored": True},
                }
            )


def _single_syllable_variants(entry: Any, pair: dict[str, Any]) -> list[dict[str, Any]]:
    delta = _single_syllable_delta(pair)
    position = int(delta.get("index", 0)) + 1
    weak_instruction = (
        f"保持输入文本和字数不变，只将第{position}个音节发得较轻、较短且略微含混，"
        "并与相邻音节自然连读，不要改成另一个词。"
    )
    operation = str(delta["type"])
    if operation == "tone_change":
        transition_instruction = (
            f"保持输入文本不变，将第{position}个音节的声调弱化为不明确的过渡调值，"
            "声母和韵母保持不变。"
        )
        transition_kind = "tone_transition"
    elif operation == "initial_change":
        transition_instruction = (
            f"保持输入文本不变，将第{position}个音节的声母发得含混，"
            f"听感介于 {delta['source_initial'] or '零声母'} 和 {delta['target_initial'] or '零声母'} 之间，"
            "韵母保持不变。"
        )
        transition_kind = "initial_transition"
    elif operation == "final_change":
        transition_instruction = (
            f"保持输入文本不变，将第{position}个音节的韵母或尾音发得含混，"
            f"听感介于 {delta['source_final']} 和 {delta['target_final']} 之间，声母保持不变。"
        )
        transition_kind = "final_transition"
    elif operation in {"syllable_insertion", "syllable_deletion"}:
        transition_instruction = (
            f"保持输入文本不变，在第{position}个音节附近快速连读并轻微吞音，"
            "不要增加新词，也不要完整删除原词。"
        )
        transition_kind = "syllable_boundary_reduction"
    else:
        transition_instruction = (
            f"保持输入文本不变，将第{position}个音节整体发得不精确，"
            f"发音从 {delta.get('source_syllable') or '无'} 向 {delta.get('target_syllable') or '无'} 轻微偏移，"
            "但不要直接说出目标词。"
        )
        transition_kind = "syllable_transition"
    return [
        _variant_row(
            entry,
            tts_text=entry.canonical_text,
            targets=[pair["confusion_text"]],
            category="zh_single_syllable",
            variant_kind="changed_syllable_weakening",
            delta=delta,
            instruction=weak_instruction,
        ),
        _variant_row(
            entry,
            tts_text=entry.canonical_text,
            targets=[pair["confusion_text"]],
            category="zh_single_syllable",
            variant_kind=transition_kind,
            delta=delta,
            instruction=transition_instruction,
        ),
    ]


def _english_variants(entry: Any, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical = entry.canonical_text
    candidates = [
        ("camelcase_word_split", _split_camel_and_alnum(canonical), None),
        ("letter_spelling", _spell_short_leading_token(_split_camel_and_alnum(canonical)), None),
        ("connected_word_boundary", re.sub(r"[\s_-]+", "", canonical), "将英文词边界连读，不要增加内容。"),
        ("english_natural_reading", canonical, "按自然英文单词读法朗读，不要逐字母拼读。"),
    ]
    return _deduplicated_text_variants(entry, pairs, "english_word_acronym", candidates, limit=4)


def _cross_language_variants(entry: Any, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical = entry.canonical_text
    candidates = [
        ("script_boundary_split", _split_script_boundaries(canonical), None),
        ("letter_spelling", _spell_short_leading_token(_split_script_boundaries(canonical)), None),
        ("english_natural_reading", canonical, "英文部分按自然英文读法，中文部分按普通话读法，语言切换处不停顿。"),
        ("chinese_accented_english", canonical, "保持文本不变，用普通话说话人的中文口音读英文部分，字母和音节边界略微含混。"),
    ]
    return _deduplicated_text_variants(
        entry, pairs, "cross_language_transliteration", candidates, limit=4
    )


def _number_variants(entry: Any, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical = entry.canonical_text
    candidates = [
        ("number_chinese_digits", _spell_acronyms(_replace_digit_runs(canonical, _chinese_digit_reading)), None),
        ("number_chinese_yao", _spell_acronyms(_replace_digit_runs(canonical, _chinese_yao_reading)), None),
        ("number_chinese_cardinal", _spell_acronyms(_replace_digit_runs(canonical, _chinese_cardinal)), None),
    ]
    return _deduplicated_text_variants(entry, pairs, "number_reading_mixed", candidates, limit=3)


def _deduplicated_text_variants(
    entry: Any,
    pairs: list[dict[str, Any]],
    category: str,
    candidates: list[tuple[str, str, str | None]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    targets = [pair["confusion_text"] for pair in pairs]
    target_texts = {_normalize(value) for value in targets}
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for kind, text, instruction in candidates:
        text = " ".join(text.split()).strip()
        key = (_normalize(text), instruction or "")
        if not text or key in seen:
            continue
        if _normalize(text) in target_texts:
            # Do not synthesize a complete GT target verbatim.  Spaced letter
            # or script-boundary readings remain eligible because they are
            # independently derived from the canonical form.
            continue
        seen.add(key)
        rows.append(
            _variant_row(
                entry,
                tts_text=text,
                targets=targets,
                category=category,
                variant_kind=kind,
                delta={"type": kind, "derived_from": "canonical_text"},
                instruction=instruction,
            )
        )
        if len(rows) >= limit:
            break
    return rows


def _variant_row(
    entry: Any,
    *,
    tts_text: str,
    targets: list[str],
    category: str,
    variant_kind: str,
    delta: dict[str, Any],
    instruction: str | None,
) -> dict[str, Any]:
    variant_id = _stable_id(entry.canonical_text, category, variant_kind, tts_text, targets)
    return {
        "id": entry.sample_id,
        "text": entry.canonical_text,
        "canonical_text": entry.canonical_text,
        "tts_text": tts_text,
        "source_text": entry.canonical_text,
        "text_source": "gt_directed_pronunciation",
        "pronunciation_processed": True,
        "pronunciation_rule": f"gt.{variant_kind}",
        "pronunciation_variant_id": variant_id,
        "pronunciation_instruction": instruction,
        "target_confusions": list(targets),
        "confusion_category": category,
        "pronunciation_delta": delta,
        "variant_kind": variant_kind,
        "language": "Auto",
        "tags": [
            "gt",
            f"source-lines:{','.join(map(str, entry.source_lines))}",
            "pronunciation:gt-directed",
            f"confusion-category:{category}",
            f"variant-kind:{variant_kind}",
        ],
    }


def _single_syllable_delta(pair: dict[str, Any]) -> dict[str, Any]:
    source = list(pair["canonical_pinyin_tone"])
    target = list(pair["confusion_pinyin_tone"])
    source_plain = [_strip_tone(value) for value in source]
    target_plain = [_strip_tone(value) for value in target]
    if len(source) == len(target):
        index = next(i for i, values in enumerate(zip(source_plain, target_plain)) if values[0] != values[1])
        source_initial, source_final, source_tone = _pinyin_parts(source[index])
        target_initial, target_final, target_tone = _pinyin_parts(target[index])
        if source_plain[index] == target_plain[index] and source_tone != target_tone:
            change_type = "tone_change"
        elif source_initial != target_initial and source_final == target_final:
            change_type = "initial_change"
        elif source_initial == target_initial and source_final != target_final:
            change_type = "final_change"
        else:
            change_type = "syllable_substitution"
        return {
            "type": change_type,
            "index": index,
            "source_syllable": source[index],
            "target_syllable": target[index],
            "source_initial": source_initial,
            "target_initial": target_initial,
            "source_final": source_final,
            "target_final": target_final,
            "source_tone": source_tone,
            "target_tone": target_tone,
        }
    if len(source) > len(target):
        index = _first_alignment_gap(source_plain, target_plain)
        return {
            "type": "syllable_deletion",
            "index": index,
            "source_syllable": source[index] if index < len(source) else None,
            "target_syllable": None,
        }
    index = _first_alignment_gap(target_plain, source_plain)
    return {
        "type": "syllable_insertion",
        "index": index,
        "source_syllable": None,
        "target_syllable": target[index] if index < len(target) else None,
    }


def _first_alignment_gap(longer: list[str], shorter: list[str]) -> int:
    for index in range(len(longer)):
        if longer[:index] + longer[index + 1 :] == shorter:
            return index
    return min(len(shorter), len(longer) - 1)


def _pinyin_parts(value: str) -> tuple[str, str, str | None]:
    match = _PINYIN.fullmatch(value.replace("ü", "v"))
    plain = match.group(1) if match else _strip_tone(value)
    tone = match.group(2) if match else None
    initial = next((item for item in _INITIALS if plain.startswith(item)), "")
    return initial, plain[len(initial) :], tone


def _pinyin(value: str, *, tone: bool) -> list[str]:
    style = Style.TONE3 if tone else Style.NORMAL
    return [
        item.replace("u:", "v").replace("ü", "v")
        for item in lazy_pinyin(
            value,
            style=style,
            neutral_tone_with_five=tone,
            errors="default",
        )
    ]


def _strip_tone(value: str) -> str:
    return re.sub(r"[1-5]$", "", value).replace("ü", "v")


def _edit_distance(source: list[str], target: list[str]) -> int:
    previous = list(range(len(target) + 1))
    for source_index, source_value in enumerate(source, 1):
        current = [source_index]
        for target_index, target_value in enumerate(target, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[target_index] + 1,
                    previous[target_index - 1] + (source_value != target_value),
                )
            )
        previous = current
    return previous[-1]


def _script_profile(value: str) -> dict[str, bool]:
    return {
        "han": bool(_HAN.search(value)),
        "latin": bool(_LATIN.search(value)),
        "digit": bool(_DIGIT.search(value)),
        "other_alnum": any(character.isalnum() and not (_HAN.match(character) or _LATIN.match(character) or character.isdigit()) for character in value),
    }


def _is_cross_language(source: dict[str, bool], target: dict[str, bool]) -> bool:
    if source["han"] != target["han"]:
        return True
    return (source["han"] and source["latin"]) != (target["han"] and target["latin"])


def _is_pure_han(value: str) -> bool:
    alnum = [character for character in value if character.isalnum()]
    return bool(alnum) and all(bool(_HAN.fullmatch(character)) for character in alnum)


def _is_number_reading_pair(canonical: str, confusion: str) -> bool:
    if not (_DIGIT.search(canonical) or _DIGIT.search(confusion)):
        return False
    source = _number_skeleton(canonical)
    target = _number_skeleton(confusion)
    return "#" in source and source == target


def _number_skeleton(value: str) -> str:
    result: list[str] = []
    in_number = False
    for token in _NUMBER_TOKEN.findall(_normalize(value)):
        is_number = token.isdigit() or token in _CHINESE_NUMERALS or token in _ENGLISH_NUMBER_WORDS
        if is_number:
            if not in_number:
                result.append("#")
            in_number = True
        else:
            in_number = False
            if token.isalnum():
                result.append(token)
    return "".join(result)


def _split_camel_and_alnum(value: str) -> str:
    value = _CAMEL_BOUNDARY_2.sub(" ", value)
    value = _CAMEL_BOUNDARY_1.sub(" ", value)
    return _ALNUM_BOUNDARY.sub(" ", value)


def _split_script_boundaries(value: str) -> str:
    return _split_camel_and_alnum(_SCRIPT_BOUNDARY.sub(" ", value))


def _spell_acronyms(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if len(token) <= 1:
            return token
        if token.isupper():
            return " ".join(token)
        return token

    return re.sub(r"[A-Za-z]+", replace, value)


def _spell_short_leading_token(value: str) -> str:
    value = _spell_acronyms(value)
    match = re.search(r"[A-Za-z]+", value)
    if match is None:
        return value
    token = match.group(0)
    if len(token) > 4 or token.isupper():
        return value
    spelled = " ".join(character.upper() for character in token)
    return value[: match.start()] + spelled + value[match.end() :]


def _replace_digit_runs(value: str, converter: Any) -> str:
    replaced = re.sub(r"\d+", lambda match: f" {converter(match.group(0))} ", value)
    return " ".join(replaced.split())


def _chinese_digit_reading(digits: str) -> str:
    return "".join(_ZH_DIGITS[int(value)] for value in digits)


def _chinese_yao_reading(digits: str) -> str:
    return "".join("幺" if value == "1" else _ZH_DIGITS[int(value)] for value in digits)


def _chinese_cardinal(digits: str) -> str:
    number = int(digits)
    if number == 0:
        return "零"
    if number >= 100_000_000:
        return _chinese_digit_reading(digits)
    units = ("", "十", "百", "千", "万", "十", "百", "千")
    values = str(number)
    parts: list[str] = []
    zero_pending = False
    length = len(values)
    for index, value in enumerate(values):
        digit = int(value)
        position = length - index - 1
        if digit == 0:
            zero_pending = bool(parts)
            continue
        if zero_pending:
            parts.append("零")
            zero_pending = False
        parts.append(_ZH_DIGITS[digit])
        parts.append(units[position])
    text = "".join(parts)
    if text.startswith("一十"):
        text = text[1:]
    return text


def _write_pair_txt(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(f"{row['canonical_text']}|{row['confusion_text']}\n" for row in rows),
        encoding="utf-8",
    )


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _compact(value: str) -> str:
    return "".join(character for character in _normalize(value) if character.isalnum())


def _stable_id(*values: Any) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
