"""Rule-based pronunciation generation and TTS rendering adapters."""

from .generator import (
    PronunciationError,
    PronunciationSample,
    RuleDefinition,
    generate_variants,
    load_rules,
    load_samples,
)

__all__ = [
    "PronunciationError",
    "PronunciationSample",
    "RuleDefinition",
    "generate_variants",
    "load_rules",
    "load_samples",
]
