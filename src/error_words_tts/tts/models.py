from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TermSample:
    """One text sample that should be synthesized."""

    sample_id: str
    text: str
    language: str = "Auto"
    tags: tuple[str, ...] = ()
    source_text: str | None = None
    canonical_text: str | None = None
    text_source: str = "canonical"
    pronunciation_processed: bool = False
    pronunciation_rule: str | None = None
    pronunciation_variant_id: str | None = None
    pronunciation_instruction: str | None = None
    phoneme_text: str | None = None
    pronunciation_structure: dict[str, Any] | None = None
    base_pronunciation: dict[str, Any] | None = None
    variant_pronunciation: dict[str, Any] | None = None
    target_confusions: tuple[str, ...] = ()
    confusion_category: str | None = None
    pronunciation_delta: dict[str, Any] | None = None
    variant_kind: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any], index: int) -> "TermSample":
        if not isinstance(value, dict):
            raise ValueError(f"sample at index {index} must be a JSON object")
        text = str(value.get("tts_text", value.get("text", ""))).strip()
        if not text:
            raise ValueError(f"sample at index {index} has empty text")
        source_text = str(value.get("source_text", value.get("text", text))).strip()
        if not source_text:
            raise ValueError(f"sample at index {index} has empty source_text")
        sample_id = str(value.get("id", f"sample-{index + 1:03d}")).strip()
        if not sample_id:
            raise ValueError(f"sample at index {index} has empty id")
        raw_tags = value.get("tags", [])
        if not isinstance(raw_tags, list):
            raise ValueError(f"sample at index {index} tags must be a JSON array")
        tags = tuple(str(tag).strip() for tag in raw_tags if str(tag).strip())
        raw_targets = value.get("target_confusions", [])
        if not isinstance(raw_targets, list):
            raise ValueError(f"sample at index {index} target_confusions must be a JSON array")
        raw_delta = value.get("pronunciation_delta")
        if raw_delta is not None and not isinstance(raw_delta, dict):
            raise ValueError(f"sample at index {index} pronunciation_delta must be an object")
        structured_fields = {
            field: value.get(field)
            for field in ("pronunciation_structure", "base_pronunciation", "variant_pronunciation")
        }
        for field, raw in structured_fields.items():
            if raw is not None and not isinstance(raw, dict):
                raise ValueError(f"sample at index {index} {field} must be an object")
        canonical_text = str(
            value.get("canonical_text", value.get("text", source_text))
        ).strip()
        if not canonical_text:
            raise ValueError(f"sample at index {index} has empty canonical_text")
        return cls(
            sample_id=sample_id,
            text=text,
            language=str(value.get("language", "Auto")),
            tags=tags,
            source_text=source_text,
            canonical_text=canonical_text,
            text_source=str(value.get("text_source", "canonical")).strip() or "canonical",
            pronunciation_processed=bool(value.get("pronunciation_processed", False)),
            pronunciation_rule=(
                str(value["pronunciation_rule"]).strip()
                if value.get("pronunciation_rule") is not None
                else None
            ),
            pronunciation_variant_id=(
                str(value["pronunciation_variant_id"]).strip()
                if value.get("pronunciation_variant_id") is not None
                else None
            ),
            pronunciation_instruction=(
                str(value["pronunciation_instruction"]).strip()
                if value.get("pronunciation_instruction") is not None
                else None
            ),
            phoneme_text=(
                str(value.get("phoneme_text", value.get("pronunciation_phonemes"))).strip()
                if value.get("phoneme_text", value.get("pronunciation_phonemes")) is not None
                else None
            ),
            pronunciation_structure=structured_fields["pronunciation_structure"],
            base_pronunciation=structured_fields["base_pronunciation"],
            variant_pronunciation=structured_fields["variant_pronunciation"],
            target_confusions=tuple(str(item) for item in raw_targets),
            confusion_category=(
                str(value["confusion_category"]).strip()
                if value.get("confusion_category") is not None
                else None
            ),
            pronunciation_delta=raw_delta,
            variant_kind=(
                str(value["variant_kind"]).strip()
                if value.get("variant_kind") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class TaggedVoice:
    """One engine-native speaker, voice, or reference voice."""

    kind: str
    name: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InstructionPreset:
    """One engine-native instruction plus searchable semantic tags."""

    name: str
    tags: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EngineRunConfig:
    """A parsed config for exactly one TTS engine."""

    engine: str
    settings: dict[str, Any]
    voices: tuple[TaggedVoice, ...]
    instructions: tuple[InstructionPreset, ...]


@dataclass(frozen=True, slots=True)
class SpeechSynthesisRequest:
    text: str
    language: str = "Auto"
    speaker: str | None = None
    instruction: str | None = None
    phoneme_text: str | None = None
    ssml: str | None = None


@dataclass(frozen=True, slots=True)
class SpeechSynthesisResult:
    audio_path: str
    engine: str
    sample_rate: int
    duration_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)
