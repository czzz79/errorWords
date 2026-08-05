"""Serialize structured English ARPAbet for CosyVoice special tokens."""

from __future__ import annotations

import re
from typing import Any


_PHONE = re.compile(r"^[A-Z]{1,3}[012]?$")


class CosyVoiceSerializerError(ValueError):
    pass


def serialize_cosyvoice_arpabet(structure: dict[str, Any]) -> str:
    """Return a ``[AA1][B]`` CosyVoice input string.

    ``default`` boundaries serialize as one space.  ``connected`` boundaries
    serialize without a separator, which gives the acoustic model a continuous
    phone stream while preserving the lexical-boundary metadata for audit.
    ``short_pause`` remains deliberately unsupported until a model-supported
    pause representation is verified.
    """
    if structure.get("status") != "ready":
        raise CosyVoiceSerializerError("cannot serialize an unresolved pronunciation")
    pieces: list[str] = []
    pending_separator: str | None = None
    saw_token = False
    for node in structure.get("nodes", []):
        kind = node.get("kind")
        if kind == "boundary":
            boundary_mode = str(node.get("mode", "default"))
            if boundary_mode == "default":
                pending_separator = " "
            elif boundary_mode == "connected":
                pending_separator = ""
            else:
                raise CosyVoiceSerializerError(
                    f"CosyVoice boundary mode is not verified: {boundary_mode}"
                )
            continue
        if kind != "token":
            continue
        phones = node.get("phones")
        if not isinstance(phones, list) or not phones:
            raise CosyVoiceSerializerError("token has no phones")
        normalized = [str(phone).upper() for phone in phones]
        if any(not _PHONE.fullmatch(phone) for phone in normalized):
            raise CosyVoiceSerializerError(f"unsupported ARPAbet sequence: {phones!r}")
        if saw_token:
            pieces.append(" " if pending_separator is None else pending_separator)
        pieces.append("".join(f"[{phone}]" for phone in normalized))
        saw_token = True
        pending_separator = None
    if not saw_token:
        raise CosyVoiceSerializerError("pronunciation structure contains no tokens")
    return "".join(pieces)
