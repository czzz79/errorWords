from __future__ import annotations

import json
from pathlib import Path

import pytest

from error_words_tts.tts.cli import (
    _build_engine,
    _load_engine_config,
    _make_request,
    _prepare_generation_row,
)
from error_words_tts.tts.engines import CosyVoice3Engine, TtsEngineError
from error_words_tts.tts.models import (
    EngineRunConfig,
    InstructionPreset,
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
    TaggedVoice,
    TermSample,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeCosyModel:
    sample_rate = 24_000

    def __init__(self) -> None:
        self.calls = []

    def inference_instruct2(self, text, instruction, prompt_wav, **kwargs):
        self.calls.append(("instruct2", text, instruction, prompt_wav, kwargs))
        return iter(({"tts_speech": object()},))

    def inference_zero_shot(self, text, prompt_text, prompt_wav, **kwargs):
        self.calls.append(("zero_shot", text, prompt_text, prompt_wav, kwargs))
        return iter(({"tts_speech": object()},))


def _engine_config(enabled: bool) -> EngineRunConfig:
    return EngineRunConfig(
        "cosyvoice3",
        {
            "phoneme_input": {
                "enabled": enabled,
                "format": "cosyvoice_arpabet",
                "text_frontend": False,
            }
        },
        (TaggedVoice("reference_voice", "reference"),),
        (InstructionPreset("neutral", parameters={"instruct": "Speak clearly."}),),
    )


def test_cosyvoice_config_files_load() -> None:
    old = _load_engine_config(
        ROOT / "src/error_words_tts/tts/configs/cosyvoice3-neutral-clear.json"
    )
    cmu = _load_engine_config(
        ROOT / "src/error_words_tts/tts/configs/cosyvoice3-cmu-neutral-clear.json"
    )
    assert old.engine == cmu.engine == "cosyvoice3"
    assert "phoneme_input" not in old.settings
    assert cmu.settings["phoneme_input"]["enabled"] is True


def test_make_request_uses_explicit_phoneme_text() -> None:
    sample = TermSample(
        sample_id="box",
        text="Box",
        language="English",
        phoneme_text="[B][AA1][K][S]",
    )
    request = _make_request(
        sample,
        InstructionPreset("neutral", parameters={"instruct": "Speak clearly."}),
        TaggedVoice("reference_voice", "reference"),
        "cosyvoice3",
    )
    assert request.text == "Box"
    assert request.phoneme_text == "[B][AA1][K][S]"


def test_cosyvoice_normal_text_uses_frontend(monkeypatch, tmp_path) -> None:
    model = FakeCosyModel()
    engine = CosyVoice3Engine(
        prompt_wav="prompt.wav",
        phoneme_input={"enabled": True, "format": "cosyvoice_arpabet", "text_frontend": False},
    )
    engine._model = model
    monkeypatch.setattr(
        "error_words_tts.tts.engines.write_normalized_wav",
        lambda waveform, sample_rate, output_path: (16_000, 1),
    )

    engine.synthesize(
        SpeechSynthesisRequest(text="Box", instruction="Speak clearly."),
        tmp_path / "text.wav",
    )

    assert model.calls[0][1] == "Box"
    assert model.calls[0][4]["text_frontend"] is True


def test_cosyvoice_phoneme_text_disables_frontend(monkeypatch, tmp_path) -> None:
    model = FakeCosyModel()
    engine = CosyVoice3Engine(
        prompt_wav="prompt.wav",
        phoneme_input={"enabled": True, "format": "cosyvoice_arpabet", "text_frontend": False},
    )
    engine._model = model
    monkeypatch.setattr(
        "error_words_tts.tts.engines.write_normalized_wav",
        lambda waveform, sample_rate, output_path: (16_000, 1),
    )

    result = engine.synthesize(
        SpeechSynthesisRequest(
            text="Box",
            phoneme_text="[B][AA1][K][S]",
            instruction="Speak clearly.",
        ),
        tmp_path / "phoneme.wav",
    )

    assert model.calls[0][1] == "[B][AA1][K][S]"
    assert model.calls[0][4]["text_frontend"] is False
    assert result.metadata["input_mode"] == "phoneme"


def test_disabled_phoneme_input_fails_before_model_call(tmp_path) -> None:
    model = FakeCosyModel()
    engine = CosyVoice3Engine(prompt_wav="prompt.wav", phoneme_input={"enabled": False})
    engine._model = model

    with pytest.raises(TtsEngineError, match="disabled in engine config"):
        engine.synthesize(
            SpeechSynthesisRequest(text="Box", phoneme_text="[B][AA1][K][S]"),
            tmp_path / "phoneme.wav",
        )
    assert model.calls == []


def test_build_engine_passes_phoneme_settings() -> None:
    config = _load_engine_config(
        ROOT / "src/error_words_tts/tts/configs/cosyvoice3-cmu-neutral-clear.json"
    )
    engine = _build_engine(config)
    assert isinstance(engine, CosyVoice3Engine)
    assert engine.phoneme_input_enabled is True
    assert engine.phoneme_input_format == "cosyvoice_arpabet"


def test_phoneme_variants_have_distinct_tts_cache_paths(tmp_path) -> None:
    config = _engine_config(True)
    engine = _build_engine(config)
    instruction = config.instructions[0]
    voice = config.voices[0]
    baseline = TermSample("box", "Box", language="English", phoneme_text="[B][AA1][K][S]")
    variant = TermSample("box", "Box", language="English", phoneme_text="[B][AA1][S]")

    _, baseline_path = _prepare_generation_row(
        baseline, instruction, voice, config, engine, tmp_path
    )
    _, variant_path = _prepare_generation_row(
        variant, instruction, voice, config, engine, tmp_path
    )

    assert baseline_path != variant_path
