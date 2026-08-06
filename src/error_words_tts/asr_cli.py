from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import socket
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def main() -> int:
    args = _build_parser().parse_args()
    manifest_path = Path(args.tts_manifest)
    rows = _load_jsonl(manifest_path)
    output_path = Path(args.output) if args.output else manifest_path.with_name("asr-results.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.wait_seconds > 0:
        _wait_for_service(args.url, args.wait_seconds)
    return _transcribe_manifest(
        rows,
        manifest_path=manifest_path,
        output_path=output_path,
        url=args.url,
        model=args.model,
        language=args.language,
        language_from_manifest=args.language_from_manifest,
        prompt=args.prompt,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
        continue_on_error=args.continue_on_error,
        workers=args.workers,
        backend=args.backend,
        no_proxy=args.no_proxy,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe every WAV referenced by a TTS manifest")
    parser.add_argument("--tts-manifest", required=True, help="TTS manifest.jsonl to consume")
    parser.add_argument("--output", default=None, help="Result JSONL; defaults beside the TTS manifest")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8756/v1/audio/transcriptions",
        help="Qwen3-ASR OpenAI-compatible transcription endpoint",
    )
    parser.add_argument(
        "--backend",
        choices=("local_wsl", "openai_http"),
        default="local_wsl",
        help="ASR backend profile; both profiles use the multipart transcription API",
    )
    parser.add_argument("--model", default="qwen3-asr", help="Compatibility model field")
    parser.add_argument("--language", default=None, help="Optional fixed ASR language hint")
    parser.add_argument(
        "--language-from-manifest",
        action="store_true",
        help="Use each TTS row's language unless it is Auto",
    )
    parser.add_argument("--prompt", default=None, help="Optional ASR context prompt")
    parser.add_argument("--api-key", default=None, help="Optional Bearer API key")
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Bypass HTTP(S) proxy settings (useful for private-network ASR endpoints)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--wait-seconds", type=float, default=0.0, help="Wait for the ASR TCP port")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent ASR requests")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"TTS manifest is empty: {path}")
    return rows


def _transcribe_manifest(
    rows: list[dict[str, Any]],
    *,
    manifest_path: Path,
    output_path: Path,
    url: str,
    model: str,
    language: str | None,
    language_from_manifest: bool,
    prompt: str | None,
    timeout_seconds: float,
    continue_on_error: bool,
    api_key: str | None = None,
    workers: int = 1,
    backend: str = "local_wsl",
    no_proxy: bool = False,
) -> int:
    if workers < 1:
        raise ValueError("ASR workers must be at least 1")
    if workers > 1:
        return _transcribe_manifest_parallel(
            rows,
            manifest_path=manifest_path,
            output_path=output_path,
            url=url,
            model=model,
            language=language,
            language_from_manifest=language_from_manifest,
            prompt=prompt,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            continue_on_error=continue_on_error,
            workers=workers,
            backend=backend,
            no_proxy=no_proxy,
        )
    transcriptions: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}
    transcribable_count = 0
    error_count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for row in rows:
            result = _base_result(row)
            audio_path, path_error = _resolve_audio_path(row.get("audio_path"), manifest_path)
            if row.get("status") not in {"generated", "cached"}:
                result["asr"] = {
                    "status": "skipped",
                    "error": f"TTS status is {row.get('status', 'missing')}",
                }
                _write_result(output, result)
                continue
            if path_error:
                result["asr"] = {"status": "error", "error": path_error}
                _write_result(output, result)
                error_count += 1
                if not continue_on_error:
                    return 1
                continue

            transcribable_count += 1
            row_language = _select_language(row, language, language_from_manifest)
            cache_key = (str(audio_path.resolve()), row_language, prompt)
            cached = transcriptions.get(cache_key)
            if cached is not None:
                asr_result = dict(cached)
                asr_result["status"] = "reused"
            else:
                try:
                    asr_result = _invoke_transcribe_audio(
                        audio_path,
                        url=url,
                        model=model,
                        language=row_language,
                        prompt=prompt,
                        api_key=api_key,
                        timeout_seconds=timeout_seconds,
                        backend=backend,
                        no_proxy=no_proxy,
                    )
                    transcriptions[cache_key] = dict(asr_result)
                except (OSError, ValueError) as exc:
                    asr_result = {"status": "error", "error": str(exc), "service_url": url}
                    error_count += 1
            result["asr"] = asr_result
            transcript = asr_result.get("text")
            if isinstance(transcript, str):
                result["comparison"] = _compare_text(str(row.get("text", "")), transcript)
            _write_result(output, result)
            status = asr_result["status"]
            _print_console_safe(
                f"ASR {status}: {audio_path.name} -> "
                f"{transcript or asr_result.get('error', '')}"
            )
            if status == "error" and not continue_on_error:
                return 1
    if transcribable_count == 0:
        print("ASR error: the TTS manifest contains no generated audio")
        return 1
    print(f"ASR complete: rows={len(rows)} audio={len(transcriptions)} errors={error_count} output={output_path}")
    return 1 if error_count else 0


def _transcribe_manifest_parallel(
    rows: list[dict[str, Any]],
    *,
    manifest_path: Path,
    output_path: Path,
    url: str,
    model: str,
    language: str | None,
    language_from_manifest: bool,
    prompt: str | None,
    api_key: str | None,
    timeout_seconds: float,
    continue_on_error: bool,
    workers: int,
    backend: str,
    no_proxy: bool,
) -> int:
    """Transcribe rows concurrently while preserving manifest output order."""
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qwen-asr") as executor:
        futures = [
            executor.submit(
                _transcribe_one_row,
                row,
                manifest_path=manifest_path,
                url=url,
                model=model,
                language=language,
                language_from_manifest=language_from_manifest,
                prompt=prompt,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                backend=backend,
                no_proxy=no_proxy,
            )
            for row in rows
        ]
        transcribable_count = 0
        error_count = 0
        with output_path.open("w", encoding="utf-8") as output:
            for index, future in enumerate(futures, start=1):
                result, row_transcribable, row_error = future.result()
                transcribable_count += row_transcribable
                error_count += row_error
                _write_result(output, result)
                asr_result = result.get("asr", {})
                transcript = asr_result.get("text") if isinstance(asr_result, dict) else None
                audio_path = result.get("audio_path")
                _print_console_safe(
                    f"ASR {asr_result.get('status', 'unknown') if isinstance(asr_result, dict) else 'unknown'}: "
                    f"{Path(str(audio_path)).name} -> {transcript or (asr_result.get('error', '') if isinstance(asr_result, dict) else '')}"
                )
                if index == 1 or index == len(rows) or index % max(1, len(rows) // 100) == 0:
                    elapsed = max(0.0, time.perf_counter() - started)
                    remaining = elapsed / index * (len(rows) - index) if index else 0.0
                    print(
                        f"ASR [{index}/{len(rows)} {index * 100 / len(rows):5.1f}%] "
                        f"workers={workers} elapsed={_format_duration(elapsed)} "
                        f"eta={_format_duration(remaining)}",
                        flush=True,
                    )
                if row_error and not continue_on_error:
                    return 1
    if transcribable_count == 0:
        print("ASR error: the TTS manifest contains no generated audio")
        return 1
    print(
        f"ASR complete: rows={len(rows)} audio={transcribable_count} "
        f"errors={error_count} output={output_path}"
    )
    return 1 if error_count else 0


def _transcribe_one_row(
    row: dict[str, Any],
    *,
    manifest_path: Path,
    url: str,
    model: str,
    language: str | None,
    language_from_manifest: bool,
    prompt: str | None,
    api_key: str | None,
    timeout_seconds: float,
    backend: str = "local_wsl",
    no_proxy: bool = False,
) -> tuple[dict[str, Any], int, int]:
    result = _base_result(row)
    audio_path, path_error = _resolve_audio_path(row.get("audio_path"), manifest_path)
    if row.get("status") not in {"generated", "cached"}:
        result["asr"] = {
            "status": "skipped",
            "error": f"TTS status is {row.get('status', 'missing')}",
        }
        return result, 0, 0
    if path_error:
        result["asr"] = {"status": "error", "error": path_error}
        return result, 0, 1
    assert audio_path is not None
    row_language = _select_language(row, language, language_from_manifest)
    try:
        asr_result = _invoke_transcribe_audio(
            audio_path,
            url=url,
            model=model,
            language=row_language,
            prompt=prompt,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            backend=backend,
            no_proxy=no_proxy,
        )
    except (OSError, ValueError) as exc:
        asr_result = {"status": "error", "error": str(exc), "service_url": url}
    result["asr"] = asr_result
    transcript = asr_result.get("text")
    if isinstance(transcript, str):
        result["comparison"] = _compare_text(str(row.get("text", "")), transcript)
    return result, 1, int(asr_result.get("status") == "error")


def _invoke_transcribe_audio(audio_path: Path, **kwargs: Any) -> dict[str, Any]:
    """Call the client while keeping the old monkeypatch/call signature valid."""
    backend = kwargs.pop("backend", "local_wsl")
    no_proxy = bool(kwargs.pop("no_proxy", False))
    if backend == "local_wsl" and not no_proxy:
        return _transcribe_audio(audio_path, **kwargs)
    return _transcribe_audio(audio_path, backend=backend, no_proxy=no_proxy, **kwargs)


def _resolve_audio_path(value: Any, manifest_path: Path) -> tuple[Path, str | None]:
    if not isinstance(value, str) or not value.strip():
        return Path(), "TTS row has no audio_path"
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((manifest_path.parent / path, manifest_path.parent.parent / path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate, None
    return path, f"audio file does not exist: {path}"


def _select_language(
    row: dict[str, Any],
    fixed_language: str | None,
    language_from_manifest: bool,
) -> str | None:
    if fixed_language:
        return fixed_language
    if not language_from_manifest:
        return None
    value = str(row.get("language", "")).strip()
    return value if value and value.lower() != "auto" else None


def _transcribe_audio(
    audio_path: Path,
    *,
    url: str,
    model: str,
    language: str | None,
    prompt: str | None,
    timeout_seconds: float,
    api_key: str | None = None,
    backend: str = "local_wsl",
    no_proxy: bool = False,
) -> dict[str, Any]:
    if backend not in {"local_wsl", "openai_http"}:
        raise ValueError(f"unsupported ASR backend: {backend}")
    fields = {"model": model, "response_format": "json"}
    if language:
        fields["language"] = language
    if prompt:
        fields["prompt"] = prompt
    body, content_type = _encode_multipart(fields, audio_path)
    headers = {"Content-Type": content_type, "Accept": "application/json"}
    resolved_api_key = (api_key or os.getenv("QWEN_ASR_API_KEY", "")).strip()
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if no_proxy
        else urllib.request
    )
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
            status_code = response.status
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OSError(f"ASR HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OSError(f"ASR service unavailable at {url}: {exc.reason}") from exc
    payload = json.loads(response_body)
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise ValueError(f"unexpected ASR response: {response_body[:500]}")
    cleaned_text = _clean_response_text(payload["text"])
    result = {
        "status": "success",
        "text": cleaned_text,
        "raw_text": payload["text"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "http_status": status_code,
        "service_url": url,
        "model": model,
        "language": language,
        "prompt": prompt,
        "backend": backend,
        "no_proxy": no_proxy,
    }
    if isinstance(payload.get("language"), str):
        result["detected_language"] = payload["language"]
    if isinstance(payload.get("usage"), dict):
        result["usage"] = payload["usage"]
    return result


_ASR_TEXT_MARKER_RE = re.compile(
    r"^\s*(?:language\s*[:=]?\s*[^<\r\n]+)?<asr_text>\s*",
    re.IGNORECASE,
)


def _clean_response_text(value: str) -> str:
    """Remove provider wrappers while retaining the server text separately.

    The internal Qwen3-ASR deployment currently returns values such as
    ``language English<asr_text>A P Y key.``.  Local WSL responses are plain
    text, so this function is intentionally a no-op for ordinary transcripts.
    """
    cleaned = _ASR_TEXT_MARKER_RE.sub("", value, count=1)
    return cleaned.strip()


def _encode_multipart(fields: dict[str, str], audio_path: Path) -> tuple[bytes, str]:
    boundary = f"----errorWords{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            )
        )
    content_type = mimetypes.guess_type(audio_path.name)[0] or "audio/wav"
    chunks.extend(
        (
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8"),
            audio_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        )
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _base_result(row: dict[str, Any]) -> dict[str, Any]:
    result = {
        "sample_id": row.get("sample_id"),
        "expected_text": row.get("text"),
        "audio_path": row.get("audio_path"),
        "tts_engine": row.get("engine"),
        "voice": row.get("voice"),
        "instruction": row.get("instruction"),
        "sample_tags": row.get("sample_tags", row.get("tags", [])),
        "tts_status": row.get("status"),
    }
    for field in (
        "source_text",
        "tts_text",
        "text_source",
        "pronunciation_processed",
        "prompt_version",
        "tts_instruction_group",
        "tts_instruction_text",
        "canonical_text",
        "pronunciation_variant_id",
        "pronunciation_rule",
        "base_pronunciation",
        "variant_pronunciation",
        "tts_renderability",
        "render_method",
        "target_confusions",
        "confusion_category",
        "pronunciation_delta",
        "variant_kind",
        "pronunciation_instruction",
        "phoneme_text",
        "input_mode",
        "pronunciation_structure",
        "source_audio_path",
        "source_manifest_path",
        "source_tts_status",
        "augmentation",
    ):
        if field in row:
            result[field] = row[field]
    return result


def _compare_text(expected: str, actual: str) -> dict[str, Any]:
    expected_normalized = _normalize_text(expected)
    actual_normalized = _normalize_text(actual)
    return {
        "expected_normalized": expected_normalized,
        "actual_normalized": actual_normalized,
        "exact_match": expected_normalized == actual_normalized,
        "compact_match": _compact_text(expected_normalized) == _compact_text(actual_normalized),
    }


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _compact_text(value: str) -> str:
    return "".join(character for character in value if character.isalnum())


def _write_result(output: Any, row: dict[str, Any]) -> None:
    output.write(json.dumps(row, ensure_ascii=False) + "\n")
    output.flush()


def _print_console_safe(message: Any) -> None:
    """Print arbitrary ASR text even when the Windows console uses GBK."""
    encoding = sys.stdout.encoding or "utf-8"
    safe_message = str(message).encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe_message)


def _wait_for_service(url: str, wait_seconds: float) -> None:
    parsed = urlsplit(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    deadline = time.monotonic() + wait_seconds
    print(f"Waiting for ASR service at {host}:{port} ...")
    while True:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                print("ASR service port is ready")
                return
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"ASR service did not open {host}:{port} within {wait_seconds}s")
            time.sleep(2.0)


if __name__ == "__main__":
    raise SystemExit(main())
