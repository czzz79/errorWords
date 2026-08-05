"""Probe an OpenAI-compatible ASR endpoint without third-party dependencies.

Examples:
  uv run python tools/probe_asr_endpoint.py
  uv run python tools/probe_asr_endpoint.py --audio sample.wav --api-key 'gpus...6204'
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe an OpenAI-compatible ASR service")
    parser.add_argument("--base-url", default="http://200.4.188.200:18000/v1")
    parser.add_argument("--audio", type=Path, help="Optional WAV file for a real transcription request")
    parser.add_argument("--model", default="Qwen3-ASR")
    parser.add_argument("--api-key", default=os.getenv("QWEN_ASR_API_KEY", ""))
    parser.add_argument("--language", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        print(f"Invalid base URL: {base}", file=sys.stderr)
        return 2
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    print(f"[1/3] TCP {parsed.hostname}:{port}")
    try:
        with socket.create_connection((parsed.hostname, port), timeout=args.timeout):
            print("      PASS: TCP port is reachable")
    except OSError as exc:
        print(f"      FAIL: {exc}")
        return 1

    print(f"[2/3] GET {base}/models")
    headers = _auth_headers(args.api_key)
    try:
        status, body = _request(f"{base}/models", b"", headers, args.timeout, method="GET")
        print(f"      HTTP {status}: {body[:500]}")
    except OSError as exc:
        print(f"      WARN: /models unavailable ({exc})")

    if args.audio is None:
        print("[3/3] POST skipped: no --audio supplied")
        return 0
    if not args.audio.is_file():
        print(f"Audio file does not exist: {args.audio}", file=sys.stderr)
        return 2

    print(f"[3/3] POST {base}/audio/transcriptions")
    fields = {"model": args.model, "response_format": "json"}
    if args.language:
        fields["language"] = args.language
    if args.prompt:
        fields["prompt"] = args.prompt
    body, content_type = _multipart(fields, args.audio)
    headers = {"Content-Type": content_type, "Accept": "application/json", **_auth_headers(args.api_key)}
    try:
        status, response = _request(
            f"{base}/audio/transcriptions", body, headers, args.timeout, method="POST"
        )
        print(f"      HTTP {status}: {response[:1000]}")
        payload = json.loads(response)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            print("      WARN: response JSON has no string 'text' field")
            return 1
        print(f"      PASS: transcript={payload['text']!r}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"      FAIL: {exc}")
        return 1
    return 0


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _multipart(fields: dict[str, str], audio: Path) -> tuple[bytes, str]:
    boundary = f"----errorWordsProbe{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{audio.name}"\r\n'
            "Content-Type: audio/wav\r\n\r\n".encode(),
            audio.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _request(url: str, body: bytes, headers: dict[str, str], timeout: float, *, method: str) -> tuple[int, str]:
    request = urllib.request.Request(url, data=body or None, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OSError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OSError(str(exc.reason)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
