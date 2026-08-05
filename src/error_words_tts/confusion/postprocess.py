"""LLM post-processing for generated ASR confusion words."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tqdm import tqdm


DEFAULT_INPUT = Path(
    r"C:\Users\jsqdc\Desktop\workspace\errorWords\outputs\confusion\cosyvoice3\confusion-words.txt"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\jsqdc\Desktop\workspace\errorWords\outputs\confusion\cosyvoice3\confusion-words-llm.txt"
)
DEFAULT_DETAILS = Path(
    r"C:\Users\jsqdc\Desktop\workspace\errorWords\outputs\confusion\cosyvoice3\confusion-words-llm-v2.review.jsonl"
)
MAX_WORKERS = 4
SYSTEM_PROMPT = """你是一个严格且保守的 ASR 术语混淆词清理器。
输入是一条标准术语和 ASR 生成的候选识别词。
你的唯一任务是找出明显应该删除的候选词。

只删除以下类型：
1. 明显的高频普通词、疑问词、功能词、口语词或泛化词，例如：
   “很多”“然后”“我们”“这个”“怎样”“怎么”“什么”“哪里”“如何”、
   “可以”“现在”“已经”“如果”“因为”“所以”“但是”“可能”、
   “一个”“一些”“没有”“不是”“就是”“还有”“他们”等；
2. 纯标点、空字符串、无意义片段。

必须非常保守：
1. 上述高频词如果与标准术语存在明显的近音、同音、谐音或字形关系，仍然必须保留；
2. 高频词如果只是独立的普通词，且与标准术语没有合理读音或字形关系，就删除；
3. 其他候选只要可能是标准术语的正常 ASR 识别结果，就必须保留；
4. 不确定时必须保留。

例如：标准术语“森伢”的候选中，“怎样”是普通疑问词且没有合理的近音或字形关系，应放入 drop；“生涯”“森牙”等可能是近音或字形相关结果，应保留。

不要输出理由，不要输出 keep 列表，只输出最终要删除的候选词，必须是严格 JSON，不要 Markdown，不要额外说明：
{
  "drop": ["要删除的词1", "要删除的词2"]
}
"""

def candidate_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(normalized.split()).casefold()


def parse_input(path: Path) -> list[dict[str, Any]]:
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
        for candidate in fields[1:]:
            key = candidate_key(candidate)
            if key and key not in seen and key != candidate_key(canonical):
                candidates.append(candidate)
                seen.add(key)
        rows.append({"line_number": line_number, "canonical": canonical, "candidates": candidates})
    return rows


def completion_endpoint(api_url: str | None, host: str | None, port: int | None) -> str:
    if api_url:
        endpoint = api_url.strip().rstrip("/")
        if endpoint.endswith("/chat/completions"):
            return endpoint
        if endpoint.endswith("/v1"):
            return f"{endpoint}/chat/completions"
        return f"{endpoint}/v1/chat/completions"

    if not host:
        raise ValueError("请提供 --api-url，或提供 --host 和 --port")
    return f"http://{host.strip().rstrip('/') }:{port or 8000}/v1/chat/completions"


def call_model(
    endpoint: str,
    api_key: str,
    model: str,
    canonical: str,
    candidates: list[str],
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    user_prompt = json.dumps(
        {
            "canonical": canonical,
            "candidates": candidates,
        },
        ensure_ascii=False,
    )
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    request_headers = {"Content-Type": "application/json"}
    if api_key:
        request_headers["Authorization"] = f"Bearer {api_key}"

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=request_headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            tqdm.write(f"LLM 原始输出 [{canonical}]: {content}")
            return parse_model_json(content)
        except (OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"LLM 请求失败: {last_error}") from last_error


def parse_model_json(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    text = str(content).strip()
    if text.startswith("```"):
        text = text.removeprefix("```").removeprefix("json").strip()
        if text.endswith("```"):
            text = text.removesuffix("```").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if isinstance(parsed, list):
        parsed = {"drop": parsed}
    if not isinstance(parsed, dict):
        raise ValueError("LLM 返回结果不是 JSON 对象或数组")
    return parsed


def decision_items(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            items.append({"word": item, "reason": "模型未提供理由"})
        elif isinstance(item, dict):
            word = item.get("word", item.get("candidate", item.get("text", "")))
            if word:
                items.append({"word": str(word), "reason": str(item.get("reason", ""))})
    return items


def apply_decision(
    candidates: list[str],
    result: dict[str, Any],
    *,
    on_missing: str = "keep",
) -> tuple[list[str], list[dict[str, str]]]:
    keep_items = decision_items(result.get("keep"))
    drop_items = decision_items(result.get("drop"))
    keep_keys = {candidate_key(item["word"]) for item in keep_items}
    drop_by_key = {candidate_key(item["word"]): item["reason"] for item in drop_items}
    known_keys = keep_keys | set(drop_by_key)

    kept: list[str] = []
    decisions: list[dict[str, str]] = []
    for candidate in candidates:
        key = candidate_key(candidate)
        if key in drop_by_key:
            decisions.append({"word": candidate, "decision": "drop", "reason": drop_by_key[key]})
        elif key in keep_keys:
            kept.append(candidate)
            decisions.append({"word": candidate, "decision": "keep", "reason": "模型判定保留"})
        elif on_missing == "drop":
            decisions.append({"word": candidate, "decision": "drop", "reason": "模型未返回该候选"})
        else:
            kept.append(candidate)
            decisions.append({"word": candidate, "decision": "keep", "reason": "模型未返回该候选，按 keep 策略保留"})

    return kept, decisions


def process_one(
    index: int,
    total: int,
    row: dict[str, Any],
    *,
    endpoint: str,
    api_key: str,
    model: str,
    timeout_seconds: float,
    retries: int,
    sleep_seconds: float,
    on_error: str,
) -> dict[str, Any]:
    canonical = row["canonical"]
    candidates = row["candidates"]
    input_line = "|".join([canonical, *candidates])
    tqdm.write(f"[{index}/{total}] 输入: {input_line}")
    detail: dict[str, Any] = {
        "index": index,
        "line_number": row["line_number"],
        "canonical": canonical,
        "input_candidates": candidates,
    }

    if not candidates:
        kept: list[str] = []
        detail.update({"kept": kept, "decisions": [], "elapsed_seconds": 0.0})
    else:
        started_at = time.perf_counter()
        try:
            model_result = call_model(
                endpoint,
                api_key,
                model,
                canonical,
                candidates,
                timeout_seconds,
                retries,
            )
            # 模型只返回 drop；没有返回的候选一律保留，避免误删正常 ASR 结果。
            kept, decisions = apply_decision(candidates, model_result, on_missing="keep")
            detail.update(
                {
                    "kept": kept,
                    "decisions": decisions,
                    "model_result": model_result,
                    "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                }
            )
        except Exception as exc:
            elapsed_seconds = round(time.perf_counter() - started_at, 3)
            kept = [] if on_error == "drop" else candidates
            detail.update(
                {
                    "kept": kept,
                    "decisions": [],
                    "error": str(exc),
                    "elapsed_seconds": elapsed_seconds,
                }
            )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    output_line = "|".join([canonical, *detail["kept"]])
    detail["output_line"] = output_line
    removed_count = len(candidates) - len(detail["kept"])
    tqdm.write(
        f"[{index}/{total}] 输出: {output_line} | "
        f"保留 {len(detail['kept'])} 个，删除 {removed_count} 个 | "
        f"耗时 {detail['elapsed_seconds']:.2f} 秒"
    )
    return detail


def load_completed(detail_path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not detail_path.is_file():
        return completed

    with detail_path.open("r", encoding="utf-8") as detail_file:
        for raw_line in detail_file:
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                # 断线时最后一行可能未写完整，忽略它即可重新处理。
                continue
            index = record.get("index") if isinstance(record, dict) else None
            if isinstance(index, int) and isinstance(record.get("output_line"), str):
                completed[index] = record
    return completed


def process(
    input_path: Path,
    output_path: Path,
    detail_path: Path,
    *,
    endpoint: str,
    api_key: str,
    model: str,
    timeout_seconds: float,
    retries: int,
    sleep_seconds: float,
    on_error: str,
) -> None:
    rows = parse_input(input_path)
    total = len(rows)

    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"审查详情(JSONL): {detail_path}")
    print(f"模型: {model}")
    print(f"接口: {endpoint}")
    print(f"待处理行数: {total}")
    print(f"并发 worker 数: {MAX_WORKERS}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(detail_path)
    completed = {index: record for index, record in completed.items() if 1 <= index <= total}
    if completed:
        tqdm.write(f"检测到 {len(completed)} 条已保存记录，将跳过并续跑。")

    next_to_write = 1
    with output_path.open("w", encoding="utf-8") as output_file, \
            detail_path.open("a", encoding="utf-8") as detail_file, \
            tqdm(
                total=total,
                initial=len(completed),
                desc="处理术语",
                unit="行",
                dynamic_ncols=True,
            ) as progress:

        def flush_ready_output() -> None:
            nonlocal next_to_write
            while next_to_write in completed:
                output_file.write(completed[next_to_write]["output_line"] + "\n")
                output_file.flush()
                next_to_write += 1

        # 续跑时先把已完成且连续的记录恢复到 TXT。
        flush_ready_output()

        future_to_index = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for index, row in enumerate(rows, 1):
                if index in completed:
                    continue
                future = executor.submit(
                    process_one,
                    index,
                    total,
                    row,
                    endpoint=endpoint,
                    api_key=api_key,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    sleep_seconds=sleep_seconds,
                    on_error=on_error,
                )
                future_to_index[future] = index

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    record = future.result()
                except Exception as exc:
                    row = rows[index - 1]
                    kept = row["candidates"] if on_error == "keep" else []
                    record = {
                        "index": index,
                        "line_number": row["line_number"],
                        "canonical": row["canonical"],
                        "input_candidates": row["candidates"],
                        "kept": kept,
                        "decisions": [],
                        "error": str(exc),
                        "elapsed_seconds": 0.0,
                        "output_line": "|".join([row["canonical"], *kept]),
                    }

                completed[index] = record
                detail_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                detail_file.flush()
                flush_ready_output()
                progress.update(1)



def main() -> int:
    parser = argparse.ArgumentParser(description="Use an LLM to filter generated ASR confusion words")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--details", type=Path, help="审查详情 JSONL，默认使用 review.jsonl")
    parser.add_argument("--api-url", default=os.getenv("LLM_API_URL"), help="API base URL or full chat completions URL")
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    parser.add_argument("--host", default=os.getenv("LLM_HOST"), help="内网 OpenAI 兼容服务地址")
    parser.add_argument("--port", type=int, default=int(os.getenv("LLM_PORT", "8000")))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen2.5-7b-instruct"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--on-error", choices=("keep", "drop"), default="keep")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"输入文件不存在: {args.input}")
    if args.retries < 0:
        parser.error("--retries 不能小于 0")

    details = args.details or DEFAULT_DETAILS

    endpoint = completion_endpoint(args.api_url, args.host, args.port)
    process(
        args.input,
        args.output,
        details,
        endpoint=endpoint,
        api_key=args.api_key,
        model=args.model,
        timeout_seconds=args.timeout,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        on_error=args.on_error,
    )
    print(f"完成: output={args.output} details={details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
