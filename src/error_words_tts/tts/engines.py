from __future__ import annotations

import io
import json
import os
import sys
import hashlib
import mimetypes
import threading
import uuid
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .audio import write_normalized_wav, write_raw_pcm16_wav
from .models import SpeechSynthesisRequest, SpeechSynthesisResult


class TtsEngineError(RuntimeError):
    """A recoverable error for one optional TTS engine."""


class TtsEngine(ABC):
    name: str

    @abstractmethod
    def synthesize(self, request: SpeechSynthesisRequest, output_path: Path) -> SpeechSynthesisResult:
        raise NotImplementedError


class Qwen3TtsEngine(TtsEngine):
    name = "qwen3-tts"

    def __init__(
        self,
        model: str | None = None,
        device_map: str | None = None,
        dtype: str = "bfloat16",
        mode: str = "custom_voice",
    ) -> None:
        self.model_name = model or os.getenv("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        self.device_map = device_map or os.getenv("QWEN3_TTS_DEVICE", "cuda:0")
        self.dtype_name = dtype
        self.mode = mode.strip().lower()
        self._model: Any | None = None

    def synthesize(self, request: SpeechSynthesisRequest, output_path: Path) -> SpeechSynthesisResult:
        model = self._load_model()
        language = request.language if request.language != "Auto" else "Auto"
        try:
            if self.mode == "voice_design":
                wavs, sample_rate = model.generate_voice_design(
                    text=request.text,
                    language=language,
                    instruct=request.instruction or "",
                )
                speaker = None
            else:
                speaker = request.speaker or os.getenv("QWEN3_TTS_SPEAKER", "Vivian")
                wavs, sample_rate = model.generate_custom_voice(
                    text=request.text,
                    language=language,
                    speaker=speaker,
                    instruct=request.instruction,
                )
        except Exception as exc:  # optional third-party runtime
            raise TtsEngineError(f"Qwen3-TTS synthesis failed: {exc}") from exc
        rate, duration_ms = write_normalized_wav(wavs[0], int(sample_rate), output_path)
        return SpeechSynthesisResult(
            audio_path=str(output_path),
            engine=self.name,
            sample_rate=rate,
            duration_ms=duration_ms,
            metadata={
                "model": self.model_name,
                "mode": self.mode,
                "speaker": speaker,
                "language": language,
            },
        )

    def synthesize_batch(
        self,
        requests: list[SpeechSynthesisRequest],
        output_paths: list[Path],
    ) -> list[SpeechSynthesisResult]:
        """Generate a batch for VoiceDesign/CustomVoice models."""
        if not requests:
            return []
        if len(requests) != len(output_paths):
            raise ValueError("Qwen3-TTS request/output batch sizes differ")
        model = self._load_model()
        language = [request.language if request.language != "Auto" else "Auto" for request in requests]
        try:
            if self.mode == "voice_design":
                wavs, sample_rate = model.generate_voice_design(
                    text=[request.text for request in requests],
                    language=language,
                    instruct=[request.instruction or "" for request in requests],
                )
            else:
                speakers = [request.speaker or os.getenv("QWEN3_TTS_SPEAKER", "Vivian") for request in requests]
                wavs, sample_rate = model.generate_custom_voice(
                    text=[request.text for request in requests],
                    language=language,
                    speaker=speakers,
                    instruct=[request.instruction or "" for request in requests],
                )
        except Exception as exc:
            raise TtsEngineError(f"Qwen3-TTS batch synthesis failed: {exc}") from exc
        if len(wavs) != len(output_paths):
            raise TtsEngineError(
                f"Qwen3-TTS returned {len(wavs)} waveforms for {len(output_paths)} requests"
            )
        results: list[SpeechSynthesisResult] = []
        for request, wav, output_path in zip(requests, wavs, output_paths):
            rate, duration_ms = write_normalized_wav(wav, int(sample_rate), output_path)
            results.append(
                SpeechSynthesisResult(
                    audio_path=str(output_path),
                    engine=self.name,
                    sample_rate=rate,
                    duration_ms=duration_ms,
                    metadata={
                        "model": self.model_name,
                        "mode": self.mode,
                        "speaker": request.speaker,
                        "language": request.language if request.language != "Auto" else "Auto",
                    },
                )
            )
        return results

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise TtsEngineError("install the qwen extra with: pip install -e '.[qwen]'") from exc
        dtype = getattr(torch, self.dtype_name, None)
        if dtype is None:
            raise TtsEngineError(f"unsupported torch dtype: {self.dtype_name}")
        try:
            self._model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map=self.device_map,
            )
        except Exception as exc:  # model download and device errors
            raise TtsEngineError(f"failed to load Qwen3-TTS model: {exc}") from exc
        return self._model


class CosyVoice3Engine(TtsEngine):
    name = "cosyvoice3"

    def __init__(
        self,
        model_dir: str | None = None,
        repo_dir: str | None = None,
        prompt_wav: str | None = None,
        prompt_text: str | None = None,
        phoneme_input: dict[str, Any] | None = None,
    ) -> None:
        self.model_dir = model_dir or os.getenv("COSYVOICE3_MODEL_DIR", "")
        self.repo_dir = repo_dir or os.getenv("COSYVOICE_REPO_DIR", "")
        self.prompt_wav = prompt_wav or os.getenv("COSYVOICE_PROMPT_WAV", "")
        self.prompt_text = prompt_text or os.getenv("COSYVOICE_PROMPT_TEXT", "")
        phoneme_input = phoneme_input or {}
        if not isinstance(phoneme_input, dict):
            raise ValueError("CosyVoice phoneme_input must be an object")
        self.phoneme_input_enabled = bool(phoneme_input.get("enabled", False))
        self.phoneme_input_format = str(
            phoneme_input.get("format", "cosyvoice_arpabet")
        ).strip().lower()
        self.phoneme_text_frontend = bool(phoneme_input.get("text_frontend", False))
        if self.phoneme_input_enabled and self.phoneme_input_format != "cosyvoice_arpabet":
            raise ValueError(
                "CosyVoice phoneme_input.format must be 'cosyvoice_arpabet'"
            )
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._voice_cache_lock = threading.Lock()
        # CosyVoice keeps the model in one GPU-backed instance.  Its generator
        # is not safe to advance concurrently from several worker threads.
        # Keep the outer TTS worker pool usable, but serialize model inference
        # so a larger worker count cannot corrupt the shared generator.
        self._inference_lock = threading.Lock()
        self._voice_cache: dict[str, str] = {}

    def synthesize(self, request: SpeechSynthesisRequest, output_path: Path) -> SpeechSynthesisResult:
        if request.phoneme_text:
            if not self.phoneme_input_enabled:
                raise TtsEngineError(
                    "CosyVoice phoneme input was supplied but disabled in engine config"
                )
            if self.phoneme_input_format != "cosyvoice_arpabet":
                raise TtsEngineError(
                    "CosyVoice phoneme_input.format must be 'cosyvoice_arpabet'"
                )
            synthesis_text = request.phoneme_text
            text_frontend = self.phoneme_text_frontend
            input_mode = "phoneme"
        else:
            synthesis_text = request.text
            text_frontend = True
            input_mode = "text"
        model = self._load_model()
        if not self.prompt_wav:
            raise TtsEngineError("COSYVOICE_PROMPT_WAV is required for CosyVoice3")
        try:
            if request.instruction:
                instruction = _cosyvoice_instruction(request.instruction)
                # CosyVoice3 instruct2 uses its ``instruct_text`` as the model
                # prompt.  A cached zero-shot speaker would retain the prompt
                # text used at registration and silently override it, causing
                # the reference transcript to be synthesized before the term.
                # Keep this ID empty so frontend_instruct2 receives the actual
                # official instruction prompt for every request.
                voice_id = ""
                outputs = model.inference_instruct2(
                    synthesis_text,
                    instruction,
                    self.prompt_wav,
                    zero_shot_spk_id=voice_id,
                    stream=False,
                    text_frontend=text_frontend,
                )
            else:
                if not self.prompt_text:
                    raise TtsEngineError("COSYVOICE_PROMPT_TEXT is required without an instruction")
                voice_id = self._cached_voice_id(model, self.prompt_text)
                outputs = model.inference_zero_shot(
                    synthesis_text,
                    self.prompt_text,
                    self.prompt_wav,
                    zero_shot_spk_id=voice_id,
                    stream=False,
                    text_frontend=text_frontend,
                )
            with self._inference_lock:
                first_output = next(iter(outputs), None)
            if first_output is None:
                raise TtsEngineError(
                    "CosyVoice3 returned no audio; input text may be unsupported "
                    "or normalized to an empty string"
                )
            waveform = first_output["tts_speech"]
            sample_rate = int(model.sample_rate)
        except TtsEngineError:
            raise
        except Exception as exc:  # optional third-party runtime
            raise TtsEngineError(f"CosyVoice3 synthesis failed: {exc}") from exc
        rate, duration_ms = write_normalized_wav(waveform, sample_rate, output_path)
        return SpeechSynthesisResult(
            audio_path=str(output_path),
            engine=self.name,
            sample_rate=rate,
            duration_ms=duration_ms,
            metadata={
                "model_dir": self.model_dir,
                "prompt_wav": self.prompt_wav,
                "cached_reference_voice": voice_id,
                "prompt_mode": "instruct2" if request.instruction else "zero_shot",
                "tts_instruction_text": instruction if request.instruction else None,
                "prompt_text": self.prompt_text,
                "phoneme_input": bool(request.phoneme_text),
                "input_mode": input_mode,
                "synthesis_text": synthesis_text,
            },
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            if not self.model_dir:
                raise TtsEngineError("COSYVOICE3_MODEL_DIR is required for CosyVoice3")
            if self.repo_dir:
                repo_path = str(Path(self.repo_dir).resolve())
                if repo_path not in sys.path:
                    sys.path.insert(0, repo_path)
                matcha_path = str(Path(repo_path, "third_party", "Matcha-TTS").resolve())
                if matcha_path not in sys.path:
                    sys.path.insert(0, matcha_path)
            try:
                from cosyvoice.cli.cosyvoice import AutoModel
                from cosyvoice.cli import frontend as cosyvoice_frontend
            except ImportError as exc:
                raise TtsEngineError(
                    "CosyVoice is not importable; set COSYVOICE_REPO_DIR to its checkout"
                ) from exc
            try:
                # torchaudio 2.9+ routes load() through TorchCodec even when the
                # caller requests the soundfile backend.  CosyVoice only needs a
                # normal PCM reader here, so keep its documented soundfile path
                # working without adding TorchCodec/FFmpeg to Windows installs.
                cosyvoice_frontend.load_wav = _load_cosyvoice_wav
                self._model = AutoModel(model_dir=self.model_dir)
            except Exception as exc:  # model download and device errors
                raise TtsEngineError(f"failed to load CosyVoice3 model: {exc}") from exc
            return self._model

    def _cached_voice_id(self, model: Any, prompt_text: str) -> str:
        cached = self._voice_cache.get(prompt_text)
        if cached is not None:
            return cached
        with self._voice_cache_lock:
            cached = self._voice_cache.get(prompt_text)
            if cached is not None:
                return cached
            digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
            voice_id = f"error-words-{digest}"
            try:
                model.add_zero_shot_spk(prompt_text, self.prompt_wav, voice_id)
            except Exception as exc:
                raise TtsEngineError(f"failed to cache CosyVoice reference voice: {exc}") from exc
            self._voice_cache[prompt_text] = voice_id
            return voice_id


class CosyVoice3ApiEngine(TtsEngine):
    """CosyVoice3 client for the repository's FastAPI runtime server."""

    name = "cosyvoice3-api"

    def __init__(
        self,
        api_url: str | None = None,
        model: str | None = None,
        api_protocol: str = "official",
        voice_id: str | None = None,
        api_key: str | None = None,
        api_key_header: str = "Authorization",
        api_key_prefix: str = "Bearer ",
        prompt_wav: str | None = None,
        prompt_text: str | None = None,
        sample_rate: int = 24_000,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.api_url = (api_url or os.getenv("COSYVOICE3_API_URL", "")).strip().rstrip("/")
        self.model = (model or os.getenv("COSYVOICE3_MODEL", "")).strip()
        self.api_protocol = (api_protocol or "official").strip().lower()
        self.voice_id = (voice_id or os.getenv("COSYVOICE3_VOICE_ID", "")).strip()
        self.api_key = (api_key or os.getenv("COSYVOICE3_API_KEY", "")).strip()
        self.api_key_header = str(api_key_header or "Authorization").strip()
        self.api_key_prefix = str(api_key_prefix or "")
        self.prompt_wav = prompt_wav or os.getenv("COSYVOICE_PROMPT_WAV", "")
        self.prompt_text = prompt_text or os.getenv("COSYVOICE_PROMPT_TEXT", "")
        self.sample_rate = int(sample_rate)
        self.timeout_seconds = float(timeout_seconds)

    def synthesize(self, request: SpeechSynthesisRequest, output_path: Path) -> SpeechSynthesisResult:
        if self.api_protocol == "yuntts":
            return self._synthesize_yuntts(request, output_path)
        if not self.api_url:
            raise TtsEngineError("COSYVOICE3_API_URL is required for CosyVoice3 API")
        if not self.prompt_wav:
            raise TtsEngineError("COSYVOICE_PROMPT_WAV is required for CosyVoice3 API")
        prompt_path = Path(self.prompt_wav)
        if not prompt_path.is_file():
            raise TtsEngineError(f"CosyVoice prompt wav does not exist: {prompt_path}")

        if request.instruction:
            instruction = cosyvoice_instruction_text(request.instruction)
            route = "/inference_instruct2"
            fields = {"tts_text": request.text, "instruct_text": instruction}
            prompt_mode = "instruct2"
            prompt_value = instruction
        else:
            if not self.prompt_text:
                raise TtsEngineError("COSYVOICE_PROMPT_TEXT is required without an instruction")
            route = "/inference_zero_shot"
            fields = {"tts_text": request.text, "prompt_text": self.prompt_text}
            prompt_mode = "zero_shot"
            prompt_value = self.prompt_text

        if self.model:
            fields["model"] = self.model

        try:
            pcm_bytes = _post_cosyvoice_multipart(
                f"{self.api_url}{route}",
                fields=fields,
                file_field="prompt_wav",
                file_path=prompt_path,
                timeout_seconds=self.timeout_seconds,
                api_key=self.api_key,
                api_key_header=self.api_key_header,
                api_key_prefix=self.api_key_prefix,
            )
        except (OSError, ValueError) as exc:
            raise TtsEngineError(f"CosyVoice3 API synthesis failed: {exc}") from exc

        rate, duration_ms = write_raw_pcm16_wav(
            pcm_bytes,
            output_path,
            sample_rate=self.sample_rate,
        )
        return SpeechSynthesisResult(
            audio_path=str(output_path),
            engine=self.name,
            sample_rate=rate,
            duration_ms=duration_ms,
            metadata={
                "api_url": self.api_url,
                "model": self.model or None,
                "api_key_configured": bool(self.api_key),
                "prompt_wav": self.prompt_wav,
                "prompt_mode": prompt_mode,
                "tts_instruction_text": prompt_value if request.instruction else None,
                "prompt_text": self.prompt_text if not request.instruction else None,
            },
        )

    def _synthesize_yuntts(
        self,
        request: SpeechSynthesisRequest,
        output_path: Path,
    ) -> SpeechSynthesisResult:
        """Call YunTTS's JSON CosyVoice synthesis endpoint."""
        if not self.api_url:
            raise TtsEngineError("YunTTS api_url is required")
        if not self.model:
            raise TtsEngineError("YunTTS model is required")
        if not self.voice_id:
            raise TtsEngineError(
                "YunTTS voice_id is required; create a voice first or use a system voice"
            )
        if not self.api_key:
            raise TtsEngineError("YunTTS api_key is required")

        payload: dict[str, Any] = {
            "model": self.model,
            "voice": self.voice_id,
            "text": request.text,
            "format": "wav",
            "sample_rate": 24_000,
            "stream": False,
        }
        if request.instruction:
            payload["instruction"] = request.instruction[:100]
        try:
            response = _post_json(
                self.api_url,
                payload=payload,
                headers=self._api_headers(),
                timeout_seconds=self.timeout_seconds,
            )
            audio_url = (
                response.get("data", {})
                .get("audio", {})
                .get("url")
            )
            if not isinstance(audio_url, str) or not audio_url.strip():
                raise ValueError(f"YunTTS response has no audio URL: {response}")
            audio_bytes = _download_bytes(
                audio_url,
                headers=self._api_headers(),
                timeout_seconds=self.timeout_seconds,
            )
            import soundfile as sf

            samples, sample_rate = sf.read(
                io.BytesIO(audio_bytes), dtype="float32", always_2d=True
            )
            rate, duration_ms = write_normalized_wav(samples, int(sample_rate), output_path)
        except (OSError, ValueError, ImportError) as exc:
            raise TtsEngineError(f"YunTTS CosyVoice synthesis failed: {exc}") from exc
        return SpeechSynthesisResult(
            audio_path=str(output_path),
            engine=self.name,
            sample_rate=rate,
            duration_ms=duration_ms,
            metadata={
                "api_url": self.api_url,
                "api_protocol": self.api_protocol,
                "model": self.model,
                "voice_id": self.voice_id,
                "api_key_configured": True,
            },
        )

    def _api_headers(self) -> dict[str, str]:
        return {
            self.api_key_header: f"{self.api_key_prefix}{self.api_key}",
            "Accept": "application/json",
        }


class AzureSpeechEngine(TtsEngine):
    name = "azure"

    def __init__(
        self,
        key: str | None = None,
        region: str | None = None,
        voice: str | None = None,
    ) -> None:
        self.key = key or os.getenv("AZURE_SPEECH_KEY", "")
        self.region = region or os.getenv("AZURE_SPEECH_REGION", "")
        self.voice = voice or os.getenv("AZURE_SPEECH_VOICE", "")

    def synthesize(self, request: SpeechSynthesisRequest, output_path: Path) -> SpeechSynthesisResult:
        if not self.key or not self.region:
            raise TtsEngineError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION are required")
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:
            raise TtsEngineError("install the azure extra with: pip install -e '.[azure]'") from exc
        speech_config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
        )
        voice = request.speaker or self.voice or _default_azure_voice(request.language)
        speech_config.speech_synthesis_voice_name = voice
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        try:
            result = (
                synthesizer.speak_ssml_async(request.ssml).get()
                if request.ssml
                else synthesizer.speak_text_async(request.text).get()
            )
        except Exception as exc:  # optional third-party runtime
            raise TtsEngineError(f"Azure Speech synthesis failed: {exc}") from exc
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = getattr(result, "cancellation_details", None)
            message = getattr(details, "error_details", "unknown Azure error")
            raise TtsEngineError(str(message))
        rate, duration_ms = write_raw_pcm16_wav(result.audio_data, output_path)
        return SpeechSynthesisResult(
            audio_path=str(output_path),
            engine=self.name,
            sample_rate=rate,
            duration_ms=duration_ms,
            metadata={"voice": voice, "region": self.region},
        )


def _default_azure_voice(language: str) -> str:
    if language.lower().startswith("en"):
        return "en-US-JennyNeural"
    return "zh-CN-XiaoxiaoNeural"


def _post_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json", **headers}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OSError(f"HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise OSError(f"service unavailable: {exc.reason}") from exc
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"API returned non-JSON response: {body[:500]}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"API returned an unexpected JSON value: {body[:500]}")
    if result.get("code") not in (None, 200, "200"):
        raise OSError(f"API error: {result}")
    return result


def _download_bytes(
    url: str,
    *,
    headers: dict[str, str],
    timeout_seconds: float,
) -> bytes:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OSError(f"audio download HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise OSError(f"audio download unavailable: {exc.reason}") from exc
    if not body:
        raise ValueError("audio download returned an empty payload")
    return body


def _post_cosyvoice_multipart(
    url: str,
    *,
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
    timeout_seconds: float,
    api_key: str = "",
    api_key_header: str = "Authorization",
    api_key_prefix: str = "Bearer ",
) -> bytes:
    boundary = f"----errorWordsCosyVoice{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            (
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            )
        )
    content_type = mimetypes.guess_type(file_path.name)[0] or "audio/wav"
    chunks.extend(
        (
            f"--{boundary}\r\n".encode("ascii"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        )
    )
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Accept": "application/octet-stream",
    }
    if api_key:
        headers[api_key_header] = f"{api_key_prefix}{api_key}"
    request = urllib.request.Request(
        url,
        data=b"".join(chunks),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OSError(f"HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise OSError(f"service unavailable: {exc.reason}") from exc
    if not body:
        raise ValueError("CosyVoice API returned an empty audio payload")
    return body


def cosyvoice_instruction_text(instruction: str) -> str:
    marker = "<|endofprompt|>"
    cleaned = instruction.strip()
    if marker in cleaned:
        return cleaned
    return f"You are a helpful assistant. {cleaned}{marker}"


def _cosyvoice_instruction(instruction: str) -> str:
    """Backward-compatible private alias for the CosyVoice prompt formatter."""
    return cosyvoice_instruction_text(instruction)


def _load_cosyvoice_wav(wav: Any, target_sr: int, min_sr: int = 16_000) -> Any:
    import soundfile as sf
    import torch
    import torchaudio

    samples, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
    speech = torch.from_numpy(samples.T.copy()).mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        if sample_rate < min_sr:
            raise ValueError(
                f"wav sample rate {sample_rate} must be greater than or equal to {min_sr}"
            )
        speech = torchaudio.functional.resample(speech, sample_rate, target_sr)
    return speech
