# Settings UI Improvements — Design Spec

**Date:** 2026-04-25  
**Project:** Interview Practice App (Turing College Sprint 1)  
**Session scope:** Session 3 — Chat + Guards  
**Status:** Approved

---

## Overview

Three UI improvements to `app.py` that stay within Gradio's Python-native model, require no new dependencies, and keep the code fully readable. All changes touch at most three files.

---

## Feature 1 — Provider-first, dependent model dropdown

### What changes
The Settings accordion currently shows **Model → Provider**. This is inverted — you must pick a provider before a model makes sense. The new order is **Provider → Model**.

The model dropdown becomes reactive: when the user switches to Ollama, it collapses to a single locked entry (`qwen3.5:9b`, non-interactive). When OpenRouter is selected, it restores the full `AVAILABLE_MODELS` list as an interactive dropdown.

### Implementation
Add `update_model_choices(provider)` to `app.py`:

```python
def update_model_choices(provider):
    if provider == PROVIDER_OLLAMA:
        return gr.Dropdown(choices=["qwen3.5:9b"], value="qwen3.5:9b", interactive=False)
    return gr.Dropdown(choices=AVAILABLE_MODELS, value=AVAILABLE_MODELS[0], interactive=True)
```

Wire it with `provider.change(update_model_choices, inputs=[provider], outputs=[model])`.

The model row stays visible in both states — the user always sees which model is active.

### Files
- `app.py` — reorder components, add handler, add `.change()` wire

---

## Feature 2 — Model capability config + recommended defaults

### What changes
Each model has a known set of supported parameters and recommended values. Currently this knowledge lives nowhere — sliders are always active and defaults are global constants.

Two new behaviours:
1. **Reactive sliders** — when the selected model doesn't support a parameter (e.g. temperature on gpt-5-mini), that slider becomes `interactive=False` and visually greyed out. A one-line markdown note below the sliders explains why.
2. **"Apply recommended settings" button** — snaps all sliders to the model's documented recommended values in one click.

### MODEL_CONFIGS structure (added to `config.py`)

```python
MODEL_CONFIGS = {
    "openai/gpt-5-mini": {
        "supports_temperature": False,
        "supports_top_p": True,
        "supports_freq_penalty": True,
        "supports_pres_penalty": True,
        "recommended": {
            "temperature": 1.0,   # placeholder — not sent when unsupported
            "top_p": 1.0,
            "max_tokens": 1024,
            "freq_pen": 0.0,
            "pres_pen": 0.0,
        },
        "tooltip": "gpt-5-mini uses fixed sampling — Temperature has no effect for this model.",
    },
    "openai/gpt-5-nano": {
        "supports_temperature": False,   # verify against OpenRouter docs before coding
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

The existing `OLLAMA_TEMPERATURE`, `OLLAMA_TOP_P`, `OLLAMA_PRESENCE_PENALTY`, `OLLAMA_MAX_TOKENS` constants are **removed** — their values now live in `MODEL_CONFIGS["qwen3.5:9b"]["recommended"]`. References in `app.py`'s `respond()` are updated to read from `MODEL_CONFIGS`.

### Handler signatures

```python
def update_slider_states(model_name):
    cfg = MODEL_CONFIGS.get(model_name, {})
    tooltip = cfg.get("tooltip", "")
    rec = cfg.get("recommended", {})
    return (
        gr.Slider(interactive=cfg.get("supports_temperature", True)),
        gr.Slider(interactive=cfg.get("supports_top_p", True)),
        gr.Slider(interactive=cfg.get("supports_freq_penalty", True)),
        gr.Slider(interactive=cfg.get("supports_pres_penalty", True)),
        tooltip,   # goes to a gr.Markdown component
    )

def apply_recommended(model_name):
    rec = MODEL_CONFIGS.get(model_name, {}).get("recommended", {})
    return (
        rec.get("temperature", DEFAULT_TEMPERATURE),
        rec.get("top_p", DEFAULT_TOP_P),
        rec.get("max_tokens", DEFAULT_MAX_TOKENS),
        rec.get("freq_pen", DEFAULT_FREQUENCY_PENALTY),
        rec.get("pres_pen", DEFAULT_PRESENCE_PENALTY),
    )
```

Both handlers are wired to `model.change(...)`. The recommended button fires `apply_recommended` on click.

### Files
- `config.py` — add `MODEL_CONFIGS`, remove `OLLAMA_*` constants
- `app.py` — add handlers, add tooltip markdown, add recommended button, update `respond()` to read from `MODEL_CONFIGS`

---

## Feature 3 — Input Templates accordion

### What changes
A new collapsible accordion **"Input Templates"** added to the left column, below System Status, collapsed by default. It contains a fixed list of read-only textboxes — one per template. Clicking a textbox injects its text into the ChatInterface input field, replacing whatever is there.

### New file: `llm/user_prompts.py`

```python
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

### Chat input injection

Gradio's ChatInterface owns its textbox internally, so populating it from outside requires one JS call. A single `gr.HTML` block is added once to the layout:

```python
gr.HTML("""<script>
function fillChat(text) {
    var ta = document.querySelector('.message-input textarea');
    if (ta) { ta.value = text; ta.dispatchEvent(new Event('input', {bubbles:true})); }
}
</script>""")
```

Each template is rendered as a read-only `gr.Textbox(value=t["text"], interactive=False, label=t["label"])` for display, with a companion `gr.Button("↑ Use")` beside it. Clicking the button returns the template text into a hidden `gr.Textbox(visible=False)`; that textbox's `.change()` fires a `gr.HTML` update that calls `fillChat(text)` to inject into the chat input.

### Accordion placement in left column

```
Setup
├── CV / Resume       [open]
├── Job Description   [open]
├── Settings          [closed]
├── Input Templates   [closed]   ← new
└── System Status     [closed]
```

### Files
- `llm/user_prompts.py` — new file, `INPUT_TEMPLATES` list
- `app.py` — import `INPUT_TEMPLATES`, add accordion, add JS block, add handlers

---

## What is NOT changing

- No new Python dependencies
- No CSS theming or custom styling
- No changes to `llm/router.py`, `llm/guards.py`, `db/`, or `utils/`
- No changes to the six system prompts in `llm/prompts.py`
- No multi-page or tab restructuring

---

## Testing checklist

- [ ] Switching Provider → Ollama: model dropdown locks to qwen3.5:9b
- [ ] Switching Provider → OpenRouter: model dropdown restores full list
- [ ] Selecting gpt-5-mini: Temperature slider greys out, tooltip appears
- [ ] Selecting qwen3.5:9b: Frequency Penalty greys out, tooltip appears
- [ ] "Apply recommended settings" snaps all sliders correctly per model
- [ ] Clicking a template in Input Templates accordion fills chat input
- [ ] Filled chat input is editable before sending
- [ ] Chat still works end-to-end after all changes
