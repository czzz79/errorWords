from __future__ import annotations

import json

import pytest

from error_words_tts.tts.cli import (
    _generate_one,
    _job_description,
    _load_engine_config,
    _make_request,
    _run_generation,
)
from error_words_tts.tts.engines import CosyVoice3Engine, Qwen3TtsEngine
from error_words_tts.tts.models import (
    EngineRunConfig,
    InstructionPreset,
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
    TaggedVoice,
    TermSample,
)


class FakeEngine:
    name = "qwen3-tts"

    def __init__(self) -> None:
        self.call_count = 0

    def synthesize(self, request, output_path):
        self.call_count += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"audio")
        return SpeechSynthesisResult(
            audio_path=str(output_path),
            engine=self.name,
            sample_rate=16_000,
            duration_ms=1,
        )


def test_qwen_config_keeps_native_speaker_and_instruction_tags(tmp_path) -> None:
    path = tmp_path / "qwen.json"
    path.write_text(
        json.dumps(
            {
                "engine": "qwen3-tts",
                "model": "local-model",
                "speakers": [{"name": "Vivian", "tags": ["voice:female"]}],
                "instructions": [
                    {"name": "fast", "instruct": "Speak fast.", "tags": ["pace:fast"]}
                ],
            }
        ),
        encoding="utf-8",
    )

    config = _load_engine_config(path)

    assert config.engine == "qwen3-tts"
    assert config.voices == (TaggedVoice("speaker", "Vivian", ("voice:female",)),)
    assert config.instructions[0].parameters == {"instruct": "Speak fast."}
    assert config.instructions[0].tags == ("pace:fast",)


def test_qwen_voice_design_config_uses_instruction_instead_of_speaker(tmp_path) -> None:
    path = tmp_path / "qwen-voice-design.json"
    path.write_text(
        json.dumps(
            {
                "engine": "qwen3-tts",
                "mode": "voice_design",
                "model": "local-voice-design-model",
                "instructions": [
                    {
                        "name": "deep_slow",
                        "instruct": "A man with a deep voice, speaking slowly.",
                        "tags": ["identity:male", "pace:slow"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = _load_engine_config(path)

    assert config.settings["mode"] == "voice_design"
    assert config.voices == (TaggedVoice("voice_design", "instruction-defined"),)
    assert config.instructions[0].parameters == {
        "instruct": "A man with a deep voice, speaking slowly."
    }


def test_qwen_voice_design_rejects_speakers_field(tmp_path) -> None:
    path = tmp_path / "qwen-voice-design.json"
    path.write_text(
        json.dumps(
            {
                "engine": "qwen3-tts",
                "mode": "voice_design",
                "speakers": [],
                "instructions": [{"name": "neutral", "instruct": "A neutral voice."}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain speakers"):
        _load_engine_config(path)


def test_qwen_voice_design_requires_non_empty_instruct(tmp_path) -> None:
    path = tmp_path / "qwen-voice-design.json"
    path.write_text(
        json.dumps(
            {
                "engine": "qwen3-tts",
                "mode": " voice_design ",
                "instructions": [{"name": "missing"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="require non-empty instruct fields: missing"):
        _load_engine_config(path)


def test_qwen_voice_design_request_and_manifest_have_no_voice() -> None:
    sample = TermSample(sample_id="term", text="ideahub", language="English")
    instruction = InstructionPreset(
        name="deep_slow",
        tags=("identity:male", "pace:slow"),
        parameters={"instruct": "A man with a deep voice, speaking slowly."},
    )
    voice = TaggedVoice("voice_design", "instruction-defined")
    config = EngineRunConfig(
        "qwen3-tts",
        {"mode": "voice_design"},
        (voice,),
        (instruction,),
    )

    request = _make_request(sample, instruction, voice, "qwen3-tts")
    row = _job_description(sample, instruction, voice, config)

    assert request.speaker is None
    assert request.instruction == "A man with a deep voice, speaking slowly."
    assert row["voice"] is None


def test_job_description_records_text_provenance_and_effective_cosyvoice_prompt() -> None:
    sample = TermSample(
        sample_id="term",
        text="志为",
        language="Auto",
        source_text="志伟",
        text_source="pronunciation_variant",
        pronunciation_processed=True,
        pronunciation_rule="zh.syllable_deletion",
        pronunciation_variant_id="variant-001",
    )
    instruction = InstructionPreset(
        name="official_fast",
        tags=("control:official", "pace:fast"),
        parameters={"instruct": "请用尽可能快地语速说一句话。"},
    )
    config = EngineRunConfig(
        "cosyvoice3",
        {"prompt_version": "cosyvoice3-official-v1"},
        (TaggedVoice("reference_voice", "reference"),),
        (instruction,),
    )

    row = _job_description(sample, instruction, config.voices[0], config)

    assert row["source_text"] == "志伟"
    assert row["tts_text"] == "志为"
    assert row["text_source"] == "pronunciation_variant"
    assert row["pronunciation_processed"] is True
    assert row["pronunciation_rule"] == "zh.syllable_deletion"
    assert row["pronunciation_variant_id"] == "variant-001"
    assert row["tts_instruction_group"] == "official_fast"
    assert row["tts_instruction_text"] == (
        "You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>"
    )
    assert row["prompt_version"] == "cosyvoice3-official-v1"
    assert row["augmentation"] == {"name": "none", "parameters": {}}


def test_base_zero_shot_manifest_has_no_instruction_text() -> None:
    sample = TermSample(sample_id="term", text="玮彬")
    instruction = InstructionPreset("base_zero_shot", parameters={})
    config = EngineRunConfig(
        "cosyvoice3",
        {"prompt_version": "cosyvoice3-official-v1"},
        (TaggedVoice("reference_voice", "reference"),),
        (instruction,),
    )

    row = _job_description(sample, instruction, config.voices[0], config)

    assert row["tts_instruction_group"] == "base_zero_shot"
    assert row["tts_instruction_text"] is None


def test_qwen_voice_design_engine_calls_generate_voice_design(monkeypatch, tmp_path) -> None:
    class FakeQwenModel:
        def __init__(self) -> None:
            self.voice_design_calls = []

        def generate_voice_design(self, **kwargs):
            self.voice_design_calls.append(kwargs)
            return ([object()], 24_000)

        def generate_custom_voice(self, **kwargs):
            raise AssertionError("VoiceDesign must not call generate_custom_voice")

    model = FakeQwenModel()
    engine = Qwen3TtsEngine(model="local-model", mode="voice_design")
    engine._model = model
    monkeypatch.setattr(
        "error_words_tts.tts.engines.write_normalized_wav",
        lambda waveform, sample_rate, output_path: (16_000, 321),
    )
    request = _make_request(
        TermSample(sample_id="term", text="ideahub", language="English"),
        InstructionPreset("deep", parameters={"instruct": "A deep male voice."}),
        TaggedVoice("voice_design", "instruction-defined"),
        "qwen3-tts",
    )

    result = engine.synthesize(request, tmp_path / "voice.wav")

    assert model.voice_design_calls == [
        {"text": "ideahub", "language": "English", "instruct": "A deep male voice."}
    ]
    assert result.metadata["mode"] == "voice_design"
    assert result.metadata["speaker"] is None


def test_cosyvoice_instruct2_uses_instruction_as_its_prompt(
    monkeypatch, tmp_path
) -> None:
    class FakeCosyVoiceModel:
        def __init__(self) -> None:
            self.sample_rate = 24_000
            self.added_speakers = []
            self.instruct_calls = []

        def add_zero_shot_spk(self, prompt_text, prompt_wav, voice_id) -> None:
            self.added_speakers.append((prompt_text, prompt_wav, voice_id))

        def inference_instruct2(self, text, instruction, prompt_wav, **kwargs):
            self.instruct_calls.append((text, instruction, prompt_wav, kwargs))
            return iter(({"tts_speech": object()},))

    model = FakeCosyVoiceModel()
    prompt_text = "You are a helpful assistant.<|endofprompt|>参考音频文本。"
    engine = CosyVoice3Engine(prompt_wav="prompt.wav", prompt_text=prompt_text)
    engine._model = model
    monkeypatch.setattr(
        "error_words_tts.tts.engines.write_normalized_wav",
        lambda waveform, sample_rate, output_path: (16_000, 321),
    )

    result = engine.synthesize(
        SpeechSynthesisRequest(text="备件", instruction="请清晰说出给定文本。"),
        tmp_path / "voice.wav",
    )

    assert model.added_speakers == []
    assert model.instruct_calls[0][0] == "备件"
    assert model.instruct_calls[0][1] == (
        "You are a helpful assistant. 请清晰说出给定文本。<|endofprompt|>"
    )
    assert model.instruct_calls[0][3]["zero_shot_spk_id"] == ""
    assert result.metadata["prompt_text"] == prompt_text




def test_azure_config_rejects_qwen_instruct_field(tmp_path) -> None:
    path = tmp_path / "azure.json"
    path.write_text(
        json.dumps(
            {
                "engine": "azure",
                "voices": [{"name": "en-US-JennyNeural"}],
                "instructions": [{"name": "fast", "instruct": "Speak fast."}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported azure instruction fields: instruct"):
        _load_engine_config(path)


def test_azure_ssml_template_uses_escaped_text() -> None:
    sample = TermSample(sample_id="term", text="A & B")
    instruction = InstructionPreset(
        name="fast",
        parameters={"ssml_template": '<speak><voice name="{voice}">{text}</voice></speak>'},
    )
    voice = TaggedVoice("voice", "en-US-JennyNeural")

    request = _make_request(sample, instruction, voice, "azure")

    assert request.ssml == '<speak><voice name="en-US-JennyNeural">A &amp; B</voice></speak>'


def test_instruction_name_and_tags_do_not_affect_cache(tmp_path) -> None:
    engine = FakeEngine()
    sample = TermSample(sample_id="term", text="ideahub")
    voice = TaggedVoice("speaker", "Vivian", ("first-tag",))
    first_instruction = InstructionPreset("fast", ("first-tag",), {"instruct": "Speak fast."})
    second_instruction = InstructionPreset("quick", ("second-tag",), {"instruct": "Speak fast."})
    config = EngineRunConfig("qwen3-tts", {"model": "model"}, (voice,), (first_instruction,))

    first = _generate_one(sample, first_instruction, voice, config, engine, tmp_path)
    second = _generate_one(sample, second_instruction, voice, config, engine, tmp_path)

    assert first["audio_path"] == second["audio_path"]
    assert second["status"] == "cached"
    assert engine.call_count == 1


def test_native_instruction_parameter_affects_cache(tmp_path) -> None:
    engine = FakeEngine()
    sample = TermSample(sample_id="term", text="ideahub")
    voice = TaggedVoice("speaker", "Vivian")
    first_instruction = InstructionPreset("delivery", parameters={"instruct": "Speak slowly."})
    second_instruction = InstructionPreset("delivery", parameters={"instruct": "Speak quickly."})
    config = EngineRunConfig("qwen3-tts", {"model": "model"}, (voice,), (first_instruction,))

    first = _generate_one(sample, first_instruction, voice, config, engine, tmp_path)
    second = _generate_one(sample, second_instruction, voice, config, engine, tmp_path)

    assert first["audio_path"] != second["audio_path"]
    assert engine.call_count == 2


def test_batch_generation_prints_progress_and_summary(tmp_path, capsys) -> None:
    engine = FakeEngine()
    samples = [
        TermSample(sample_id="one", text="术语一"),
        TermSample(sample_id="two", text="TermTwo"),
    ]
    config = EngineRunConfig(
        "qwen3-tts",
        {"model": "model"},
        (TaggedVoice("speaker", "Vivian"),),
        (InstructionPreset("neutral", parameters={"instruct": "Speak."}),),
    )

    exit_code = _run_generation(
        samples,
        config,
        engine,
        tmp_path / "manifest.jsonl",
        continue_on_error=False,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "TTS [1/2  50.0%]" in output
    assert "TTS [2/2 100.0%]" in output
    assert "TTS complete: jobs=2" in output
