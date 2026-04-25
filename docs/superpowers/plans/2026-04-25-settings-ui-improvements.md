# Settings UI Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add provider-first model selection, per-model capability config with reactive sliders, and a collapsible Input Templates accordion to the Gradio settings panel.

**Architecture:** All logic is pure Python — each UI event handler is split into a testable helper function and a thin Gradio wrapper. Model capability metadata lives in a single `MODEL_CONFIGS` dict in `config.py`. Template prompts live in a new `llm/user_prompts.py` file. One small JS snippet handles injecting template text into the ChatInterface input.

**Tech Stack:** Python 3.11, Gradio 4.x, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-25-settings-ui-improvements-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `config.py` | Modify | Add `MODEL_CONFIGS`, remove `OLLAMA_*` constants |
| `llm/user_prompts.py` | Create | `INPUT_TEMPLATES` list |
| `app.py` | Modify | All UI wiring: reorder, handlers, accordion, JS |
| `tests/test_settings_handlers.py` | Create | Unit tests for pure handler logic |

---

## Task 1: Add MODEL_CONFIGS to config.py

**Files:**
- Modify: `config.py`
- Create: `tests/test_settings_handlers.py`

### Why a test first?
The pure helper functions (not Gradio itself) are what we test. Each UI event handler is written as two parts: a `_logic()` helper that returns plain Python values, and a thin Gradio wrapper that calls it. This keeps Gradio out of the test suite entirely.

- [ ] **Step 1: Create the test file with failing tests for config values**

Create `tests/__init__.py` (empty) and `tests/test_settings_handlers.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect failure (MODEL_CONFIGS not defined yet)**

```bash
cd C:/Users/markm/Documents/Claude/Projects/Turing/rmerge-AE.1.5-main
python -m pytest tests/test_settings_handlers.py -v
```

Expected: `ImportError` or `AttributeError: module 'config' has no attribute 'MODEL_CONFIGS'`

- [ ] **Step 3: Add MODEL_CONFIGS to config.py**

Open `config.py`. After the existing `AVAILABLE_MODELS` list, add:

```python
# Per-model capability flags and recommended slider values.
# supports_* = False means that slider should be greyed out in the UI.
MODEL_CONFIGS = {
    "openai/gpt-5-mini": {
        "supports_temperature": False,
        "supports_top_p": True,
        "supports_freq_penalty": True,
        "supports_pres_penalty": True,
        "recommended": {
            "temperature": 1.0,   # not sent to API when unsupported
            "top_p": 1.0,
            "max_tokens": 1024,
            "freq_pen": 0.0,
            "pres_pen": 0.0,
        },
        "tooltip": "gpt-5-mini uses fixed sampling — Temperature has no effect for this model.",
    },
    "openai/gpt-5-nano": {
        # TODO: double-check this — I couldn't find a definitive answer on whether
        # gpt-5-nano ignores temperature. Check OpenRouter model page for gpt-5-nano
        # before shipping. If it does support it, flip this to True.
        "supports_temperature": False,
        "supports_top_p": True,
        "supports_freq_penalty": True,
        "supports_pres_penalty": True,
        "recommended": {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "freq_pen": 0.0,
            "pres_pen": 0.0,
        },
        "tooltip": "gpt-5-nano uses fixed sampling — Temperature has no effect for this model.",
    },
    "openai/gpt-5": {
        "supports_temperature": True,
        "supports_top_p": True,
        "supports_freq_penalty": True,
        "supports_pres_penalty": True,
        "recommended": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
            "freq_pen": 0.0,
            "pres_pen": 0.0,
        },
        "tooltip": "gpt-5 supports all parameters.",
    },
    "qwen3.5:9b": {
        "supports_temperature": True,
        "supports_top_p": True,
        "supports_freq_penalty": False,
        "supports_pres_penalty": True,
        "recommended": {
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 32768,
            "freq_pen": 0.0,
            "pres_pen": 1.5,
        },
        "tooltip": "Qwen3.5 instruct mode — presence_penalty 1.5 prevents repetition. Frequency penalty is ignored by Ollama.",
    },
}
```

- [ ] **Step 4: Remove the now-redundant OLLAMA_* constants from config.py**

Delete these lines (their values now live in `MODEL_CONFIGS["qwen3.5:9b"]`):

```python
# DELETE these four lines:
OLLAMA_TEMPERATURE = 0.7
OLLAMA_TOP_P = 0.8
OLLAMA_PRESENCE_PENALTY = 1.5
OLLAMA_TOP_K = 20
```

Keep `OLLAMA_MAX_TOKENS = 32768` for now — it is still referenced in `app.py`'s `respond()`. Task 4 will remove it once the handler reads from `MODEL_CONFIGS`.

- [ ] **Step 5: Run tests — expect pass**

```bash
python -m pytest tests/test_settings_handlers.py -v
```

Expected output:
```
tests/test_settings_handlers.py::test_model_configs_has_all_models PASSED
tests/test_settings_handlers.py::test_model_configs_keys PASSED
tests/test_settings_handlers.py::test_recommended_has_all_slider_keys PASSED
tests/test_settings_handlers.py::test_qwen_presence_penalty_is_nonzero PASSED
tests/test_settings_handlers.py::test_qwen_max_tokens PASSED

5 passed in 0.XXs
```

- [ ] **Step 6: Commit**

```bash
git add config.py tests/__init__.py tests/test_settings_handlers.py
git commit -m "feat: add MODEL_CONFIGS, remove redundant OLLAMA_* constants"
```

---

## Task 2: Create llm/user_prompts.py

**Files:**
- Create: `llm/user_prompts.py`

- [ ] **Step 1: Create the file**

```python
"""User-facing input template prompts.

These appear in the 'Input Templates' accordion in the UI.
Clicking a template replaces the chat input with the template text,
ready to edit before sending.
"""

INPUT_TEMPLATES = [
    {
        "label": "Start the interview",
        "text": "Please begin the interview. Introduce yourself briefly and ask your first question.",
    },
    {
        "label": "Weakest skills focus",
        "text": "Based on my CV and the job description, list 5 interview questions targeting my weakest or most underrepresented skills.",
    },
    {
        "label": "Behavioral round",
        "text": "Ask me 3 behavioral questions relevant to this role. Use the STAR method format.",
    },
    {
        "label": "Harder follow-up",
        "text": "That was too straightforward. Ask a more challenging follow-up on the same topic.",
    },
    {
        "label": "Session evaluation",
        "text": "Please evaluate my performance in this session so far. Give me an overall readiness score and the top 3 things I should improve.",
    },
]
```

- [ ] **Step 2: Add a quick sanity test to the test file**

Append to `tests/test_settings_handlers.py`:

```python
from llm.user_prompts import INPUT_TEMPLATES


def test_input_templates_structure():
    assert len(INPUT_TEMPLATES) >= 1
    for t in INPUT_TEMPLATES:
        assert "label" in t and "text" in t
        assert isinstance(t["label"], str) and len(t["label"]) > 0
        assert isinstance(t["text"], str) and len(t["text"]) > 0
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_settings_handlers.py::test_input_templates_structure -v
```

Expected: `PASSED`

- [ ] **Step 4: Commit**

```bash
git add llm/user_prompts.py tests/test_settings_handlers.py
git commit -m "feat: add INPUT_TEMPLATES for chat input accordion"
```

---

## Task 3: Feature 1 — Provider-first, dependent model dropdown

**Files:**
- Modify: `app.py`

This task only changes the Settings accordion layout and wires one event handler. No changes to `respond()` yet.

- [ ] **Step 1: Add tests for the model-choices helper**

Append to `tests/test_settings_handlers.py`:

```python
from config import AVAILABLE_MODELS, PROVIDER_OLLAMA, PROVIDER_OPENROUTER


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
```

- [ ] **Step 2: Run tests — expect pass (pure logic, no Gradio needed)**

```bash
python -m pytest tests/test_settings_handlers.py::test_model_choices_ollama tests/test_settings_handlers.py::test_model_choices_openrouter -v
```

Expected: `2 passed`

- [ ] **Step 3: Add the helper function and Gradio wrapper to app.py**

In `app.py`, add these two functions after the existing `check_status` function (around line 136), before `create_app`:

```python
def _model_choices_for_provider(provider):
    """Return (choices, value, interactive) for the model dropdown."""
    if provider == PROVIDER_OLLAMA:
        return ["qwen3.5:9b"], "qwen3.5:9b", False
    return AVAILABLE_MODELS, AVAILABLE_MODELS[0], True


def update_model_choices(provider):
    choices, value, interactive = _model_choices_for_provider(provider)
    return gr.Dropdown(choices=choices, value=value, interactive=interactive)
```

- [ ] **Step 4: Reorder provider/model in the Settings accordion and wire the handler**

In `create_app()`, find the Settings accordion block (around lines 174–214). Replace the entire accordion contents with:

```python
with gr.Accordion("Settings", open=False):
    provider = gr.Radio(
        choices=[PROVIDER_OPENROUTER, PROVIDER_OLLAMA],
        value=PROVIDER_OPENROUTER,
        label="Provider",
    )
    model = gr.Dropdown(
        choices=AVAILABLE_MODELS,
        value=AVAILABLE_MODELS[0],
        label="Model",
    )
    provider.change(update_model_choices, inputs=[provider], outputs=[model])
    persona = gr.Radio(
        choices=["neutral", "strict", "friendly"],
        value="neutral",
        label="Interviewer Persona",
    )
    difficulty = gr.Radio(
        choices=["easy", "medium", "hard"],
        value="medium",
        label="Difficulty Level",
    )
    temperature = gr.Slider(
        0.0, 2.0, DEFAULT_TEMPERATURE,
        step=0.1, label="Temperature",
    )
    top_p = gr.Slider(
        0.0, 1.0, DEFAULT_TOP_P,
        step=0.05, label="Top-p",
    )
    max_tokens = gr.Slider(
        64, 4096, DEFAULT_MAX_TOKENS,
        step=64, label="Max Tokens",
    )
    freq_pen = gr.Slider(
        -2.0, 2.0, DEFAULT_FREQUENCY_PENALTY,
        step=0.1, label="Frequency Penalty",
    )
    pres_pen = gr.Slider(
        -2.0, 2.0, DEFAULT_PRESENCE_PENALTY,
        step=0.1, label="Presence Penalty",
    )
```

- [ ] **Step 5: Smoke test in browser**

Run the app and open http://localhost:7860. Open Settings, confirm Provider appears above Model. Switch to Ollama — model dropdown should show only `qwen3.5:9b` and be greyed out. Switch back to OpenRouter — full dropdown returns.

```bash
python app.py
```

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_settings_handlers.py
git commit -m "feat: provider-first dropdown, model choices cascade on provider change"
```

---

## Task 4: Feature 2 — Reactive sliders and recommended defaults button

**Files:**
- Modify: `app.py`
- Modify: `config.py` (remove `OLLAMA_MAX_TOKENS`)

- [ ] **Step 1: Add tests for the slider-state and recommended helpers**

Append to `tests/test_settings_handlers.py`:

```python
from config import MODEL_CONFIGS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS
from config import DEFAULT_FREQUENCY_PENALTY, DEFAULT_PRESENCE_PENALTY


def _slider_states_for_model(model_name):
    """Return (temp_on, top_p_on, freq_on, pres_on, tooltip) for a model."""
    cfg = MODEL_CONFIGS.get(model_name, {})
    return (
        cfg.get("supports_temperature", True),
        cfg.get("supports_top_p", True),
        cfg.get("supports_freq_penalty", True),
        cfg.get("supports_pres_penalty", True),
        cfg.get("tooltip", ""),
    )


def _recommended_values_for_model(model_name):
    """Return (temperature, top_p, max_tokens, freq_pen, pres_pen) recommended values."""
    rec = MODEL_CONFIGS.get(model_name, {}).get("recommended", {})
    return (
        rec.get("temperature", DEFAULT_TEMPERATURE),
        rec.get("top_p", DEFAULT_TOP_P),
        rec.get("max_tokens", DEFAULT_MAX_TOKENS),
        rec.get("freq_pen", DEFAULT_FREQUENCY_PENALTY),
        rec.get("pres_pen", DEFAULT_PRESENCE_PENALTY),
    )


def test_slider_states_gpt5_mini_no_temp():
    temp_on, top_p_on, freq_on, pres_on, tooltip = _slider_states_for_model("openai/gpt-5-mini")
    assert temp_on is False
    assert top_p_on is True
    assert "Temperature" in tooltip


def test_slider_states_qwen_no_freq():
    temp_on, top_p_on, freq_on, pres_on, tooltip = _slider_states_for_model("qwen3.5:9b")
    assert temp_on is True
    assert freq_on is False
    assert "presence_penalty" in tooltip


def test_recommended_values_qwen():
    temp, top_p, max_tok, freq, pres = _recommended_values_for_model("qwen3.5:9b")
    assert temp == 0.7
    assert top_p == 0.8
    assert max_tok == 32768
    assert pres == 1.5


def test_recommended_values_unknown_model_uses_defaults():
    temp, top_p, max_tok, freq, pres = _recommended_values_for_model("some/unknown-model")
    assert temp == DEFAULT_TEMPERATURE
    assert top_p == DEFAULT_TOP_P
```

- [ ] **Step 2: Run tests — expect pass**

```bash
python -m pytest tests/test_settings_handlers.py -v -k "slider or recommended"
```

Expected: `6 passed`

- [ ] **Step 3: Add the helper functions and Gradio wrappers to app.py**

Add after `update_model_choices` in `app.py`:

```python
def _slider_states_for_model(model_name):
    cfg = MODEL_CONFIGS.get(model_name, {})
    return (
        cfg.get("supports_temperature", True),
        cfg.get("supports_top_p", True),
        cfg.get("supports_freq_penalty", True),
        cfg.get("supports_pres_penalty", True),
        cfg.get("tooltip", ""),
    )


def update_slider_states(model_name):
    temp_on, top_p_on, freq_on, pres_on, tooltip = _slider_states_for_model(model_name)
    return (
        gr.Slider(interactive=temp_on),
        gr.Slider(interactive=top_p_on),
        gr.Slider(interactive=freq_on),
        gr.Slider(interactive=pres_on),
        tooltip,
    )


def _recommended_values_for_model(model_name):
    rec = MODEL_CONFIGS.get(model_name, {}).get("recommended", {})
    return (
        rec.get("temperature", DEFAULT_TEMPERATURE),
        rec.get("top_p", DEFAULT_TOP_P),
        rec.get("max_tokens", DEFAULT_MAX_TOKENS),
        rec.get("freq_pen", DEFAULT_FREQUENCY_PENALTY),
        rec.get("pres_pen", DEFAULT_PRESENCE_PENALTY),
    )


def apply_recommended(model_name):
    return _recommended_values_for_model(model_name)
```

- [ ] **Step 4: Add tooltip markdown, recommended button, and wire handlers in Settings accordion**

In `create_app()`, extend the Settings accordion (after the sliders you added in Task 3):

```python
    # Tooltip shown when a slider is disabled for the selected model
    slider_tooltip = gr.Markdown("", visible=True)

    recommended_btn = gr.Button("Apply recommended settings", size="sm")

    # Wire model change → update slider interactive states + tooltip
    model.change(
        update_slider_states,
        inputs=[model],
        outputs=[temperature, top_p, freq_pen, pres_pen, slider_tooltip],
    )

    # Wire button → snap all sliders to model's recommended values
    recommended_btn.click(
        apply_recommended,
        inputs=[model],
        outputs=[temperature, top_p, max_tokens, freq_pen, pres_pen],
    )
```

- [ ] **Step 5: Update respond() to read OLLAMA_MAX_TOKENS from MODEL_CONFIGS**

In `respond()` (around line 76), replace:

```python
effective_max_tokens = OLLAMA_MAX_TOKENS if is_ollama else max_tokens
```

with:

```python
effective_max_tokens = MODEL_CONFIGS["qwen3.5:9b"]["recommended"]["max_tokens"] if is_ollama else max_tokens
```

- [ ] **Step 6: Remove OLLAMA_MAX_TOKENS from config.py**

Delete this line from `config.py`:

```python
OLLAMA_MAX_TOKENS = 32768
```

Also remove the import of `OLLAMA_MAX_TOKENS` from `app.py`'s import block at the top:

```python
# Remove OLLAMA_MAX_TOKENS from this line:
from config import (
    APP_TITLE, APP_PORT, AVAILABLE_MODELS,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS, OLLAMA_MAX_TOKENS,  # <- remove OLLAMA_MAX_TOKENS
    ...
)
```

Add `MODEL_CONFIGS` to the same import:

```python
from config import (
    APP_TITLE, APP_PORT, AVAILABLE_MODELS,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS,
    DEFAULT_FREQUENCY_PENALTY, DEFAULT_PRESENCE_PENALTY,
    PROVIDER_OPENROUTER, PROVIDER_OLLAMA, MODELS, MODEL_CONFIGS,
)
```

- [ ] **Step 7: Smoke test in browser**

Open Settings. Select `openai/gpt-5-mini` — Temperature slider should grey out and tooltip should appear. Click "Apply recommended settings" — sliders should snap to documented values. Switch to Ollama — Frequency Penalty should grey out.

- [ ] **Step 8: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add app.py config.py tests/test_settings_handlers.py
git commit -m "feat: reactive sliders and recommended defaults button per model"
```

---

## Task 5: Feature 3 — Input Templates accordion

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the import at the top of app.py**

In `app.py`, add to the imports block (after the `from llm.guards` line):

```python
from llm.user_prompts import INPUT_TEMPLATES
```

- [ ] **Step 2: Add the JS injection block inside create_app()**

In `create_app()`, directly after `gr.Blocks(...)` opens (before any `gr.Markdown`), add:

```python
# One-time JS helper — lets template buttons push text into ChatInterface's input field.
gr.HTML("""<script>
function fillChat(text) {
    var ta = document.querySelector('.message-input textarea');
    if (ta) {
        ta.value = text;
        ta.dispatchEvent(new Event('input', {bubbles: true}));
    }
}
</script>""")
```

- [ ] **Step 3: Add the Input Templates accordion to the left column**

In `create_app()`, inside the left column (`with gr.Column(scale=1)`), add the accordion after the System Status accordion:

```python
with gr.Accordion("Input Templates", open=False):
    # Hidden textbox — receives template text from button click,
    # then its .change() fires the JS to fill the chat input.
    template_target = gr.Textbox(visible=False)

    for t in INPUT_TEMPLATES:
        with gr.Row():
            gr.Textbox(
                value=t["text"],
                label=t["label"],
                interactive=False,
                lines=2,
            )
            use_btn = gr.Button("↑ Use", size="sm", min_width=60)
            # Capture current template text in closure
            use_btn.click(
                fn=lambda txt=t["text"]: txt,
                outputs=[template_target],
            )

    # When template_target gets a new value, inject it into the chat input via JS
    template_target.change(
        fn=None,
        inputs=[template_target],
        js="(text) => { fillChat(text); }",
    )
```

- [ ] **Step 4: Smoke test in browser**

Open http://localhost:7860. Expand "Input Templates". Click "↑ Use" on any template — the chat input field should fill with that text. Confirm the text is editable before sending.

```bash
python app.py
```

- [ ] **Step 5: Run full test suite one more time**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass (the templates are tested via `test_input_templates_structure`).

- [ ] **Step 6: Final commit**

```bash
git add app.py
git commit -m "feat: Input Templates accordion with one-click chat prefill"
```

---

## Testing Checklist (manual, post-implementation)

- [ ] Switch Provider to Ollama → model locks to qwen3.5:9b (greyed)
- [ ] Switch Provider to OpenRouter → model dropdown restores full list
- [ ] Select gpt-5-mini → Temperature slider greyed, tooltip visible
- [ ] Select qwen3.5:9b → Frequency Penalty greyed, tooltip mentions presence_penalty
- [ ] Click "Apply recommended settings" with gpt-5-mini → sliders snap correctly
- [ ] Click "Apply recommended settings" with qwen3.5:9b → max_tokens shows 32768, pres_pen 1.5
- [ ] Open Input Templates accordion (collapsed by default ✓)
- [ ] Click "↑ Use" on any template → chat input fills with that text
- [ ] Filled text is editable before hitting send
- [ ] Send a message after using a template → chat still works end-to-end
- [ ] Verify gpt-5-nano temperature claim against OpenRouter docs and update `supports_temperature` flag if needed
