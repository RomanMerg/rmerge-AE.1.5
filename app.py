"""Interview Practice App - Gradio 6.x compatible."""

import re
import logging
import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

from config import (
    APP_TITLE, APP_PORT, AVAILABLE_MODELS,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS, OLLAMA_MAX_TOKENS,
    DEFAULT_FREQUENCY_PENALTY, DEFAULT_PRESENCE_PENALTY,
    PROVIDER_OPENROUTER, PROVIDER_OLLAMA, MODELS,
)
from llm.router import chat, chat_stream, smoke_test
from llm.guards import validate_cv, validate_jd, check_prompt_injection
from db.connection import init_db, test_connection
from utils.cost_tracker import SessionCostTracker

cost_tracker = SessionCostTracker()
logger = logging.getLogger("app")

THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.DOTALL)


def strip_think_tags(text):
    """Remove <think>...</think> blocks from thinking models."""
    cleaned = THINK_RE.sub("", text)
    # Handle partial/unclosed think tag during streaming
    idx = cleaned.find("<think>")
    if idx != -1:
        cleaned = cleaned[:idx]
    return cleaned.strip()


def respond(message, history, cv_text, jd_text, model, provider,
            temperature, top_p, max_tokens, freq_pen, pres_pen,
            persona, difficulty):
    """Handle chat messages with streaming."""
    safe, reason = check_prompt_injection(message)
    if not safe:
        yield "Input blocked: " + reason
        return

    system_parts = [
        "You are a " + persona + " technical interviewer.",
        "Difficulty level: " + difficulty + ".",
    ]
    if cv_text and cv_text.strip():
        system_parts.append("\nCandidate CV:\n" + cv_text[:2000])
    if jd_text and jd_text.strip():
        system_parts.append("\nJob description:\n" + jd_text[:2000])
    system_parts.append(
        "\nAsk interview questions one at a time. "
        "After the candidate answers, provide brief feedback "
        "and ask the next question. Stay in character."
    )

    messages = [{"role": "system", "content": "\n".join(system_parts)}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    if provider == PROVIDER_OLLAMA:
        actual_model = MODELS["chat_local"]
    else:
        actual_model = model
    logger.info("Chat: provider=%s model=%s", provider, actual_model)

    full_response = ""
    is_ollama = (provider == PROVIDER_OLLAMA)
    effective_max_tokens = OLLAMA_MAX_TOKENS if is_ollama else max_tokens
    try:
        for chunk in chat_stream(
            messages=messages,
            model=actual_model,
            provider=provider,
            temperature=temperature,
            top_p=top_p,
            max_tokens=effective_max_tokens,
            frequency_penalty=freq_pen,
            presence_penalty=pres_pen,
        ):
            full_response += chunk
            if is_ollama:
                display = strip_think_tags(full_response)
            else:
                display = full_response
            if display:
                yield display
    except Exception as e:
        logger.error("Chat error: %s: %s", type(e).__name__, e, exc_info=True)
        yield "Error (" + type(e).__name__ + "): " + str(e)
        return

    if is_ollama:
        full_response = strip_think_tags(full_response)

    if not full_response:
        logger.warning("LLM returned empty response")
        yield "(No response from model. Try a different provider.)"
        return

    logger.info("Response: %d chars", len(full_response))
    est_in = sum(len(m["content"]) for m in messages) // 4
    est_out = len(full_response) // 4
    cost_tracker.add(actual_model, est_in, est_out)


def handle_cv_upload(file):
    """Extract text from uploaded PDF."""
    if file is None:
        return ""
    try:
        import pymupdf4llm
        return pymupdf4llm.to_markdown(file)
    except Exception as e:
        return "Error parsing PDF: " + str(e) + "\n\nPaste your CV text manually."


def check_status():
    """Check all service connections."""
    parts = []
    if test_connection():
        parts.append("PostgreSQL: Connected")
    else:
        parts.append("PostgreSQL: Not connected")
    parts.append("OpenRouter: " + smoke_test(PROVIDER_OPENROUTER))
    parts.append("Ollama: " + smoke_test(PROVIDER_OLLAMA))
    parts.append("\n" + cost_tracker.summary())
    return "\n\n".join(parts)


def create_app():
    with gr.Blocks(title=APP_TITLE) as app:
        gr.Markdown("# " + APP_TITLE)
        gr.Markdown(
            "Practice for your next interview with AI-powered feedback. "
            "Upload your CV, paste the job description, and start practicing."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Setup")

                with gr.Accordion("CV / Resume", open=True):
                    cv_upload = gr.File(
                        label="Upload CV (PDF)",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    cv_text = gr.Textbox(
                        label="CV Text",
                        placeholder="Upload PDF or paste CV here...",
                        lines=8,
                    )
                    cv_upload.change(
                        handle_cv_upload,
                        inputs=[cv_upload],
                        outputs=[cv_text],
                    )

                with gr.Accordion("Job Description", open=True):
                    jd_text = gr.Textbox(
                        label="Job Description",
                        placeholder="Paste job description here...",
                        lines=8,
                    )

                with gr.Accordion("Settings", open=False):
                    model = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value=AVAILABLE_MODELS[0],
                        label="Model (OpenRouter)",
                    )
                    provider = gr.Radio(
                        choices=[PROVIDER_OPENROUTER, PROVIDER_OLLAMA],
                        value=PROVIDER_OPENROUTER,
                        label="Provider",
                    )
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

                with gr.Accordion("System Status", open=False):
                    status_btn = gr.Button("Check Connections")
                    status_out = gr.Textbox(label="Status", lines=8)
                    status_btn.click(check_status, outputs=[status_out])

            with gr.Column(scale=2):
                gr.Markdown("### Interview Practice Chat")
                gr.ChatInterface(
                    fn=respond,
                    additional_inputs=[
                        cv_text, jd_text, model, provider,
                        temperature, top_p, max_tokens,
                        freq_pen, pres_pen, persona, difficulty,
                    ],
                    fill_height=True,
                )

    return app


if __name__ == "__main__":
    print("Starting " + APP_TITLE + "...")
    init_db()
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=APP_PORT, share=False)
