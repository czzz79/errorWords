"""Deterministic frequency tagging for ASR confusion words."""

from __future__ import annotations

import json
import re
import urllib.request
import unicodedata
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT_DIR / "outputs/confusion/cosyvoice3/confusion-words.txt"
DEFAULT_TAGGED_OUTPUT = ROOT_DIR / "outputs/confusion/cosyvoice3/confusion-words-dict-tagged.txt"
DEFAULT_REMOVED_OUTPUT = (
    ROOT_DIR
    / "outputs/confusion/cosyvoice3/confusion-words-dict-removed-single-character.txt"
)
DEFAULT_DETAILS = ROOT_DIR / "outputs/confusion/cosyvoice3/confusion-words-dict.review.jsonl"
DEFAULT_STOPWORD_DICTIONARY = ROOT_DIR / "resources/stopwords-zh-public.txt"
DEFAULT_WUSONG_DICTIONARY = ROOT_DIR / "resources/wusong-base.dict.yaml"
DEFAULT_ENGLISH_DICTIONARY = ROOT_DIR / "resources/english-top-10000.txt"
DEFAULT_JIEBA_DICTIONARY = ROOT_DIR / "resources/jieba-dict.txt"
PUBLIC_WUSONG_URL = "https://raw.githubusercontent.com/iDvel/rime-ice/main/cn_dicts/base.dict.yaml"
PUBLIC_ENGLISH_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
MIN_WUSONG_WEIGHT = 5000
MIN_JIEBA_CHARACTER_FREQUENCY = 200000
HAN_TEXT_RE = re.compile(r"^[\u4e00-\u9fff]+$")


def word_key(value: str) -> str:
    """Normalize a word for exact dictionary membership checking."""
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(normalized.split()).casefold()


def load_dictionary(path: Path) -> set[str]:
    """Load one word per line, optionally followed by a frequency."""
    words: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Supports: word, word<TAB>frequency, and word frequency.
        if "\t" in line:
            word = line.split("\t", 1)[0].strip()
        else:
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2 and _is_number(parts[1]):
                word = parts[0].strip()
            else:
                word = line

        key = word_key(word)
        if key:
            words.add(key)
    return words


def ensure_public_wusong_dictionary(path: Path) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在下载雾凇词库: {PUBLIC_WUSONG_URL}")
    request = urllib.request.Request(
        PUBLIC_WUSONG_URL,
        headers={"User-Agent": "error-words-tts/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()
    path.write_bytes(content)


def ensure_public_english_dictionary(path: Path) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在下载公开英文高频词表: {PUBLIC_ENGLISH_URL}")
    request = urllib.request.Request(
        PUBLIC_ENGLISH_URL,
        headers={"User-Agent": "error-words-tts/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()
    path.write_bytes(content)


def load_wusong_dictionary(path: Path, min_weight: int) -> set[str]:
    """Load Rime/Wusong ``word pinyin weight`` entries above a threshold."""
    words: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = raw_line.strip().split()
        if len(fields) < 3 or not fields[-1].isdigit():
            continue
        if int(fields[-1]) >= min_weight:
            key = word_key(fields[0])
            if key:
                words.add(key)
    return words


def load_english_dictionary(path: Path) -> set[str]:
    return {
        line.strip().casefold()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def load_jieba_character_frequencies(path: Path) -> dict[str, int]:
    """Load single-Han-character frequencies from the bundled jieba dictionary."""
    frequencies: dict[str, int] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = raw_line.strip().split()
        if len(fields) < 2 or len(fields[0]) != 1 or not fields[1].isdigit():
            continue
        character = fields[0]
        if HAN_TEXT_RE.fullmatch(character):
            frequencies[character] = int(fields[1])
    return frequencies


def is_high_frequency_character_combination(
    candidate_key: str,
    character_frequencies: dict[str, int],
    min_character_frequency: int,
) -> bool:
    """Return whether every Han character in a multi-character candidate is frequent."""
    return (
        len(candidate_key) >= 2
        and bool(HAN_TEXT_RE.fullmatch(candidate_key))
        and all(
            character_frequencies.get(character, 0) >= min_character_frequency
            for character in candidate_key
        )
    )


def _is_number(value: str) -> bool:
    return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value))


def parse_confusion_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split("|")]
        canonical = fields[0]
        if not canonical:
            continue

        candidates: list[str] = []
        seen: set[str] = set()
        canonical_key = word_key(canonical)
        for candidate in fields[1:]:
            key = word_key(candidate)
            if key and key not in seen and key != canonical_key:
                candidates.append(candidate)
                seen.add(key)
        rows.append(
            {
                "line_number": line_number,
                "canonical": canonical,
                "candidates": candidates,
            }
        )
    return rows


def process(
    input_path: Path,
    tagged_output_path: Path,
    removed_output_path: Path,
    detail_path: Path,
    stopword_path: Path,
    wusong_path: Path,
    english_path: Path,
    min_wusong_weight: int,
    jieba_path: Path = DEFAULT_JIEBA_DICTIONARY,
    min_jieba_character_frequency: int = MIN_JIEBA_CHARACTER_FREQUENCY,
) -> None:
    ensure_public_wusong_dictionary(wusong_path)
    ensure_public_english_dictionary(english_path)
    dictionary = load_dictionary(stopword_path)
    wusong_words = load_wusong_dictionary(wusong_path, min_wusong_weight)
    dictionary.update(wusong_words)
    english_words = load_english_dictionary(english_path)
    character_frequencies = load_jieba_character_frequencies(jieba_path)
    rows = parse_confusion_file(input_path)

    tagged_output_path.parent.mkdir(parents=True, exist_ok=True)
    removed_output_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        tagged_output_path.open("w", encoding="utf-8") as tagged_file,
        removed_output_path.open("w", encoding="utf-8") as removed_file,
        detail_path.open("w", encoding="utf-8") as detail_file,
    ):
        tagged_file.write("#TERM_IMPORT_V2\n")
        for index, row in enumerate(tqdm(rows, desc="词典标记", unit="行", dynamic_ncols=True), 1):
            candidates = row["candidates"]
            tagged_candidates: list[dict[str, Any]] = []
            removed_single_characters: list[str] = []
            for candidate in candidates:
                candidate_key = word_key(candidate)
                if len(candidate_key) == 1:
                    removed_single_characters.append(candidate)
                    continue

                is_english_word = bool(re.fullmatch(r"[a-z]+", candidate_key))
                dictionary_matched = candidate_key in dictionary or (
                    is_english_word and candidate_key in english_words
                )
                character_combination_matched = is_high_frequency_character_combination(
                    candidate_key,
                    character_frequencies,
                    min_jieba_character_frequency,
                )
                tagged_candidates.append(
                    {
                        "word": candidate,
                        "frequency": (
                            "HIGH"
                            if dictionary_matched or character_combination_matched
                            else "LOW"
                        ),
                        "dictionary_matched": dictionary_matched,
                        "character_combination_matched": character_combination_matched,
                    }
                )

            tagged_fields = [row["canonical"]]
            for candidate in tagged_candidates:
                tagged_fields.extend([candidate["word"], candidate["frequency"]])
            output_line = "|".join(tagged_fields)
            removed_line = "|".join([row["canonical"], *removed_single_characters])
            record = {
                "index": index,
                "line_number": row["line_number"],
                "canonical": row["canonical"],
                "input_candidates": candidates,
                "removed_single_characters": removed_single_characters,
                "tagged_candidates": tagged_candidates,
                "output_line": output_line,
                "removed_line": removed_line,
            }
            tagged_file.write(output_line + "\n")
            tagged_file.flush()
            removed_file.write(removed_line + "\n")
            removed_file.flush()
            detail_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            detail_file.flush()

    print(f"输入文件: {input_path}")
    print(f"停用词典: {stopword_path}")
    print(
        f"雾凇词库: {wusong_path}（权重 >= {min_wusong_weight}，"
        f"{len(wusong_words)} 个词）"
    )
    print(f"英文高频词表: {english_path}（{len(english_words)} 个词）")
    print(
        f"单字高频词典: {jieba_path}（{len(character_frequencies)} 个字，"
        f"阈值 >= {min_jieba_character_frequency}）"
    )
    print(f"合并后词典: {len(dictionary)} 个词")
    print(f"高低频标记文件: {tagged_output_path}")
    print(f"单字移除文件: {removed_output_path}")
    print(f"详情文件: {detail_path}")
    print(f"处理行数: {len(rows)}")


def main() -> int:
    # 直接修改这里的路径和阈值即可，不依赖命令行参数。
    process(
        DEFAULT_INPUT,
        DEFAULT_TAGGED_OUTPUT,
        DEFAULT_REMOVED_OUTPUT,
        DEFAULT_DETAILS,
        DEFAULT_STOPWORD_DICTIONARY,
        DEFAULT_WUSONG_DICTIONARY,
        DEFAULT_ENGLISH_DICTIONARY,
        MIN_WUSONG_WEIGHT,
        DEFAULT_JIEBA_DICTIONARY,
        MIN_JIEBA_CHARACTER_FREQUENCY,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
