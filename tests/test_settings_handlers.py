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


# --- Task 2: Input Templates ---

from llm.user_prompts import INPUT_TEMPLATES


def test_input_templates_structure():
    assert len(INPUT_TEMPLATES) >= 1
    for t in INPUT_TEMPLATES:
        assert "label" in t and "text" in t
        assert isinstance(t["label"], str) and len(t["label"]) > 0
        assert isinstance(t["text"], str) and len(t["text"]) > 0


# --- Task 3: Provider → model cascade ---

def _model_choices_for_provider(provider):
    """Pure logic extracted from the Gradio handler for testability."""
    if provider == PROVIDER_OLLAMA:
        return ["qwen3.5:9b"], "qwen3.5:9b", False
    return AVAILABLE_MODELS, AVAILABLE_MODELS[0], True


def test_model_choices_ollama():
    choices, value, interactive = _model_choices_for_provider(PROVIDER_OLLAMA)
    assert choices == ["qwen3.5:9b"]
    assert value == "qwen3.5:9b"
    assert interactive is False


def test_model_choices_openrouter():
    choices, value, interactive = _model_choices_for_provider(PROVIDER_OPENROUTER)
    assert choices == AVAILABLE_MODELS
    assert value == AVAILABLE_MODELS[0]
    assert interactive is True
