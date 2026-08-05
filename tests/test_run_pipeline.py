from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import run_pipeline
from error_words_tts.augmentation.audio import apply_perturbation, validate_perturbation
from error_words_tts.confusion.cli import GtEntry


def _config(tmp_path: Path, stages: dict[str, bool]) -> Path:
    terms = tmp_path / "terms.txt"
    terms.write_text("IdeaHub|ID Hub\n", encoding="utf-8")
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({"input_txt": str(terms), "output_dir": str(tmp_path / "output"), "stages": stages}), encoding="utf-8")
    return path


def test_disabled_tts_requires_reusable_samples(tmp_path: Path) -> None:
    config = _config(tmp_path, {"tts": True})
    with pytest.raises(FileNotFoundError, match="reused samples"):
        run_pipeline.run_pipeline(config, dry_run=True)


def test_augmentation_can_reuse_an_explicit_prior_tts_manifest(tmp_path: Path) -> None:
    source = tmp_path / "prior" / "manifest.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"sample_id":"box","status":"generated","audio_path":"box.wav"}\n', encoding="utf-8")
    config = _config(tmp_path, {"augmentation": True})
    value = json.loads(config.read_text(encoding="utf-8"))
    value["augmentation"] = {"input_manifest": str(source)}
    config.write_text(json.dumps(value), encoding="utf-8")
    run_pipeline.run_pipeline(config, dry_run=True)


def test_disabled_asr_requires_reusable_results(tmp_path: Path) -> None:
    config = _config(tmp_path, {"report": True})
    with pytest.raises(FileNotFoundError, match="reused ASR results"):
        run_pipeline.run_pipeline(config, dry_run=True)


def test_chain_applies_steps_in_declared_order() -> None:
    values = np.array([0.2, -0.6, 0.8, -0.1], dtype=np.float32)
    chain = validate_perturbation({"name": "clip-then-speed", "type": "chain", "steps": [
        {"name": "clip", "type": "hard_clip", "threshold": 0.5},
        {"name": "slow", "type": "speed", "factor": 0.5},
    ]}, 0)
    expected = apply_perturbation(
        apply_perturbation(values, 16_000, chain["steps"][0], seed=1 ^ 0x9E3779B1),
        16_000, chain["steps"][1], seed=1 ^ (2 * 0x9E3779B1),
    )
    assert np.allclose(apply_perturbation(values, 16_000, chain, seed=1), expected)


def test_managed_asr_refuses_occupied_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_pipeline, "_port_open", lambda host, port: True)
    asr = {"service": {"mode": "managed", "host": "127.0.0.1", "port": 8756}}
    with pytest.raises(RuntimeError, match="already in use"):
        with run_pipeline._asr_service(asr, {"name": "test"}, Path("unused"), Path("unused.log")):
            pass


def test_managed_asr_starts_and_stops_only_its_own_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Process:
        pid = 123
        terminated = False
        def poll(self): return None
        def terminate(self): self.terminated = True
        def wait(self, timeout): return 0

    process = Process()
    monkeypatch.setattr(run_pipeline, "_port_open", lambda host, port: False)
    monkeypatch.setattr(run_pipeline, "_wsl_read", lambda service, path: "temperature: 0\ntop_p: 1\ntop_k: 0\nmin_p: 0\nseed: null\n")
    monkeypatch.setattr(run_pipeline.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(run_pipeline, "_wait_service", lambda *args, **kwargs: None)
    asr = {"service": {"mode": "managed", "host": "127.0.0.1", "port": 8756}}
    with run_pipeline._asr_service(asr, {"name": "random", "temperature": .5, "top_p": .9, "top_k": 10, "min_p": .1}, tmp_path / "configs", tmp_path / "service.log"):
        assert not process.terminated
    assert process.terminated
    assert "temperature: 0.5" in (tmp_path / "configs" / "service-random.yaml").read_text(encoding="utf-8")


def test_valid_result_requires_expected_successful_rows(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text('{"asr":{"status":"success"}}\n', encoding="utf-8")
    assert run_pipeline._valid_result(path, 1)
    assert not run_pipeline._valid_result(path, 2)
    path.write_text('{"asr":{"status":"error"}}\n', encoding="utf-8")
    assert not run_pipeline._valid_result(path, 1)


def test_valid_result_requires_matching_input_fingerprint_when_provided(tmp_path: Path) -> None:
    path = tmp_path / "run-01.jsonl"
    path.write_text('{"asr":{"status":"success"}}\n', encoding="utf-8")
    assert not run_pipeline._valid_result(path, 1, fingerprint="new-input")
    run_pipeline._asr_result_metadata_path(path).write_text(
        json.dumps({"fingerprint": "old-input"}), encoding="utf-8"
    )
    assert not run_pipeline._valid_result(path, 1, fingerprint="new-input")
    run_pipeline._asr_result_metadata_path(path).write_text(
        json.dumps({"fingerprint": "new-input"}), encoding="utf-8"
    )
    assert run_pipeline._valid_result(path, 1, fingerprint="new-input")


def test_ground_truth_misses_include_rules_and_observed_outputs() -> None:
    entries = [GtEntry("box", "Box", ["Pox", "Fox"], [7])]
    rows = [
        {
            "expected_text": "Box",
            "pronunciation_rule": "en.consonant.b_to_p",
            "confusion_category": "english_word_acronym",
            "asr": {"status": "success", "text": "Pox"},
            "comparison": {"compact_match": False},
        },
        {
            "expected_text": "Box",
            "variant_kind": "baseline",
            "confusion_category": "english_word_acronym",
            "asr": {"status": "success", "text": "Box"},
            "comparison": {"compact_match": True},
        },
    ]
    result = run_pipeline._ground_truth_confusion_rows(entries, rows)
    assert result["matched_unique_confusion_count"] == 1
    assert result["misses"] == [{
        "term": "Box", "confusion": "Fox", "confusion_category": "english_word_acronym",
        "source_lines": "7", "asr_sample_count": 2, "successful_asr_count": 2,
        "canonical_output_count": 1,
        "rules_tried": "baseline | en.consonant.b_to_p",
        "top_asr_outputs": "Pox (1) | Box (1)",
    }]


def test_external_asr_never_stops_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_pipeline, "_wait_port", lambda *args: None)
    asr = {"url": "http://example.test/transcribe", "service": {"mode": "external"}}
    with run_pipeline._asr_service(asr, {"name": "test"}, Path("unused"), Path("unused.log")) as url:
        assert url == "http://example.test/transcribe"


def test_service_log_sampling_match_accepts_float_rendering() -> None:
    log = "Qwen decoding configured temperature=0.0 top_p=1.0 top_k=0 min_p=0.0 seed=None"
    assert run_pipeline._sampling_in_log(log, {"temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0, "seed": None})
