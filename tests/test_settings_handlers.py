"""Tests for pure handler logic — no Gradio imports needed."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MODEL_CONFIGS, PROVIDER_OLLAMA, PROVIDER_OPENROUTER, AVAILABLE_MODELS


def test_model_configs_has_all_models():
    for model in AVAILABLE_MODELS + ["qwen3.5:9b"]:
        assert model in MODEL_CONFIGS, f"Missing config for {model}"


def test_model_configs_keys():
    required = {"supports_temperature", "supports_top_p",
                "supports_freq_penalty", "supports_pres_penalty",
                "recommended", "tooltip"}
    for model, cfg in MODEL_CONFIGS.items():
        assert required <= cfg.keys(), f"{model} missing keys: {required - cfg.keys()}"


def test_recommended_has_all_slider_keys():
    slider_keys = {"temperature", "top_p", "max_tokens", "freq_pen", "pres_pen"}
    for model, cfg in MODEL_CONFIGS.items():
        assert slider_keys <= cfg["recommended"].keys(), \
            f"{model} recommended missing: {slider_keys - cfg['recommended'].keys()}"


def test_qwen_presence_penalty_is_nonzero():
    # Qwen3.5 needs presence_penalty=1.5 to prevent repetition (per official docs)
    assert MODEL_CONFIGS["qwen3.5:9b"]["recommended"]["pres_pen"] == 1.5


def test_qwen_max_tokens():
    assert MODEL_CONFIGS["qwen3.5:9b"]["recommended"]["max_tokens"] == 32768
