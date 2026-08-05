from __future__ import annotations

import pytest

from error_words_tts.pronunciation.cosyvoice_serializer import (
    CosyVoiceSerializerError,
    serialize_cosyvoice_arpabet,
)


def test_serializer_wraps_arpabet_and_preserves_default_boundaries() -> None:
    structure = {
        "status": "ready",
        "nodes": [
            {"kind": "token", "text": "Idea", "phones": ["AY0", "D", "IY1", "AH0"]},
            {"kind": "boundary", "boundary_type": "camel_case", "mode": "default"},
            {"kind": "token", "text": "UI", "phones": ["Y", "UW1", "AY1"]},
        ],
    }
    assert serialize_cosyvoice_arpabet(structure) == "[AY0][D][IY1][AH0] [Y][UW1][AY1]"


def test_serializer_connects_verified_connected_boundary() -> None:
    structure = {
        "status": "ready",
        "nodes": [
            {"kind": "token", "text": "Idea", "phones": ["AY0"]},
            {"kind": "boundary", "boundary_type": "camel_case", "mode": "connected"},
            {"kind": "token", "text": "UI", "phones": ["AY1"]},
        ],
    }
    assert serialize_cosyvoice_arpabet(structure) == "[AY0][AY1]"


def test_serializer_rejects_unverified_boundary_mode() -> None:
    structure = {
        "status": "ready",
        "nodes": [
            {"kind": "token", "text": "Idea", "phones": ["AY0"]},
            {"kind": "boundary", "boundary_type": "camel_case", "mode": "short_pause"},
            {"kind": "token", "text": "UI", "phones": ["AY1"]},
        ],
    }
    with pytest.raises(CosyVoiceSerializerError, match="not verified"):
        serialize_cosyvoice_arpabet(structure)
