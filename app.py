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
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS,
    DEFAULT_FREQUENCY_PENALTY, DEFAULT_PRESENCE_PENALTY,
    PROVIDER_OPENROUTER, PROVIDER_OLLAMA, MODELS, MODEL_CONFIGS,
)
from llm.router import chat, chat_stream, smoke_test
from llm.guards import validate_cv, validate_jd, check_prompt_injection, classify_input
from llm.judge import evaluate_session
from db.queries import store_evaluation, store_chat_message
from llm.user_prompts import INPUT_TEMPLATES
from db.connection import init_db, test_connection
from utils.cost_tracker import SessionCostTracker
from parsers.pipeline import run_pipeline, PipelineResult

cost_tracker = SessionCostTracker()
logger = logging.getLogger("app")

THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.DOTALL)


def strip_think_tags(text):
    """Remove <think>...</think> blocks from thinking models."""
    cleaned = THINK_RE.sub("", text)
    idx = cleaned.find("<think>")
    if idx != -1:
        cleaned = cleaned[:idx]
    return cleaned.strip()


def _build_system_parts(cv_text, jd_text, persona, difficulty, pipeline_result, chat_mode="practice"):
    """Build system prompt parts — extracted for testability."""

    if chat_mode == "study":
        parts = [
            "You are a highly skilled technical candidate demonstrating strong interview answers.",
            "Difficulty level: " + difficulty + ".",
        ]
        if pipeline_result is not None:
            cv = pipeline_result.cv
            jd = pipeline_result.jd
            parts.append(
                f"\nYou are interviewing for: {jd.role_title} ({jd.role_level}) "
                f"at a {jd.company_type} company."
            )
            if jd.summary:
                parts.append(f"Role: {jd.summary}")
            parts.append("\nYour background: " + ", ".join(cv.technical_skills[:15]))
            if cv.years_of_experience:
                parts.append(f"Years of experience: {cv.years_of_experience}")
            if cv.summary:
                parts.append(f"Professional summary: {cv.summary}")
        else:
            if cv_text and cv_text.strip():
                parts.append("\nYour background:\n" + cv_text[:2000])
            if jd_text and jd_text.strip():
                parts.append("\nTarget role:\n" + jd_text[:1000])
        parts.append(
            "\nWhen the user sends a prompt or interview question, respond as a strong candidate would."
            " Use your background to give specific, structured answers."
            " Lead with your strongest evidence, acknowledge gaps honestly, be concise but thorough."
        )
        return parts

    if chat_mode == "interview":
        parts = [
            "You are a professional hiring manager conducting a formal job interview.",
            "Difficulty level: " + difficulty + ".",
        ]
        if pipeline_result is not None:
            jd = pipeline_result.jd
            gaps = pipeline_result.gaps
            parts.append(
                f"\nYou are interviewing for: {jd.role_title} ({jd.role_level}) "
                f"at a {jd.company_type} company."
            )
            if jd.summary:
                parts.append(f"Role: {jd.summary}")
            if gaps.gaps:
                parts.append(
                    "Key areas to probe: " + ", ".join(g.requirement for g in gaps.gaps[:5])
                )
            parts.append(f"Readiness score: {gaps.readiness_score}/100 (internal — do not reveal).")
        else:
            if cv_text and cv_text.strip():
                parts.append("\nCandidate CV:\n" + cv_text[:2000])
            if jd_text and jd_text.strip():
                parts.append("\nJob description:\n" + jd_text[:2000])
        parts.append(
            "\nConduct a realistic, formal interview. Ask one focused prompt at a time."
            " Do NOT provide feedback, hints, or coaching mid-session — stay strictly in character."
            " After 6–8 prompts offer to conclude. Save all evaluation for the end."
        )
        return parts

    # --- practice mode (default) ---
    parts = [
        "You are a " + persona + " technical interviewer.",
        "Difficulty level: " + difficulty + ".",
    ]
    if pipeline_result is not None:
        cv = pipeline_result.cv
        jd = pipeline_result.jd
        gaps = pipeline_result.gaps
        parts.append("\nCandidate skills: " + ", ".join(cv.technical_skills[:15]))
        if cv.years_of_experience:
            parts.append(f"Years of experience: {cv.years_of_experience}")
        parts.append(
            f"\nTarget role: {jd.role_title} ({jd.role_level}) at {jd.company_type} company"
        )
        parts.append(f"\nRole summary: {jd.summary}")
        if gaps.gaps:
            parts.append(
                "Key skill gaps to probe: " + ", ".join(g.requirement for g in gaps.gaps[:5])
            )
        parts.append(f"Readiness score: {gaps.readiness_score}/100")
    else:
        if cv_text and cv_text.strip():
            parts.append("\nCandidate CV:\n" + cv_text[:2000])
        if jd_text and jd_text.strip():
            parts.append("\nJob description:\n" + jd_text[:2000])
    parts.append(
        "\nAsk interview prompts one at a time. "
        "After the candidate answers, provide brief feedback "
        "and ask the next prompt. Stay in character."
    )
    return parts


def respond(message, history, cv_text, jd_text, model, provider,
            temperature, top_p, max_tokens, freq_pen, pres_pen,
            persona, difficulty, pipeline_result, chat_mode="practice"):
    """Handle chat messages with streaming."""
    safe, reason = check_prompt_injection(message)
    if safe:
        safe, reason = classify_input(message)
    logger.info("Guard: %s", reason)
    if not safe:
        yield "Input blocked: " + reason
        return

    system_parts = _build_system_parts(cv_text, jd_text, persona, difficulty, pipeline_result, chat_mode)

    messages = [{"role": "system", "content": "\n".join(system_parts)}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    if provider == PROVIDER_OLLAMA:
        actual_model = MODELS["chat_local"]
    else:
        actual_model = model
    logger.info("Chat: provider=%s model=%s", provider, actual_model)

    # Persist user message before streaming
    if pipeline_result is not None:
        try:
            store_chat_message(pipeline_result.session_id, "user", message)
        except Exception as e:
            logger.warning("Could not persist user message: %s", e)

    full_response = ""
    is_ollama = (provider == PROVIDER_OLLAMA)
    in_think_block = False          # suppress display while think block is open
    effective_max_tokens = MODEL_CONFIGS["qwen3.5:9b"]["recommended"]["max_tokens"] if is_ollama else max_tokens
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
                # Track whether we're inside an unclosed <think> block
                if "<think>" in full_response:
                    in_think_block = "</think>" not in full_response
                display = strip_think_tags(full_response)
                if display and not in_think_block:
                    yield display
            else:
                if full_response:
                    yield full_response
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
    last_cost = cost_tracker.add(actual_model, est_in, est_out)

    # Persist assistant message after streaming completes
    if pipeline_result is not None:
        try:
            store_chat_message(pipeline_result.session_id, "assistant", full_response, actual_model, last_cost)
        except Exception as e:
            logger.warning("Could not persist assistant message: %s", e)


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


def handle_confirm(cv_text, jd_text, difficulty):
    """Run the parsing pipeline when user clicks Analyze CV & JD.

    Yields 8-tuples: (status, pipeline_result, q0, q1, q2, q3, q4, gap_summary)
    """
    _empty = ("", None, "", "", "", "", "", "")

    if not cv_text or not cv_text.strip():
        yield ("Error: CV cannot be empty", None, "", "", "", "", "", "")
        return
    if not jd_text or not jd_text.strip():
        yield ("Error: Job description cannot be empty", None, "", "", "", "", "", "")
        return

    yield ("Analyzing CV and JD...", None, "", "", "", "", "", "")

    try:
        result = run_pipeline(cv_text, jd_text, difficulty)
        n_gaps = len(result.gaps.gaps)
        n_crit = sum(1 for g in result.gaps.gaps if g.severity.value == "critical")
        status = (
            f"✓ Ready — readiness score {result.gaps.readiness_score}/100, "
            f"{n_gaps} gaps ({n_crit} critical)"
        )
        gap_summary = (
            f"**Role:** {result.jd.role_title} ({result.jd.role_level})  \n"
            f"**Readiness:** {result.gaps.readiness_score}/100  \n"
            f"**Key gaps:** {', '.join(g.requirement for g in result.gaps.gaps[:3]) or 'none'}"
        )
        questions_padded = (result.questions + [""] * 5)[:5]
        yield (status, result, *questions_padded, gap_summary)
    except ValueError as e:
        yield (f"Error: {e}", None, "", "", "", "", "", "")
    except Exception as e:
        yield (f"Error ({type(e).__name__}): {e}", None, "", "", "", "", "", "")


def format_evaluation_md(evaluation) -> str:
    """Format SessionEvaluation as readable markdown."""
    lines = [f"## Overall Score: {evaluation.overall_score}/100", ""]
    lines.append(evaluation.overall_feedback)
    if evaluation.areas_to_study:
        lines.append("\n### Areas to Study")
        for area in evaluation.areas_to_study:
            lines.append(f"- {area}")
    if evaluation.answer_evaluations:
        lines.append("\n### Per-Answer Breakdown")
        for i, ae in enumerate(evaluation.answer_evaluations, 1):
            lines.append(f"\n**Q{i}: {ae.question}** — Quality: {ae.answer_quality}/10")
            lines.append(f"- **Strengths:** {ae.strengths}")
            lines.append(f"- **Weaknesses:** {ae.weaknesses}")
            lines.append(f"- **Improvement:** {ae.suggested_improvement}")
    return "\n".join(lines)


def handle_evaluate(history, pipeline_result):
    """Run LLM-as-Judge on the chat history and return formatted markdown."""
    if not history:
        return "No conversation to evaluate yet.", gr.Button(interactive=True)
    try:
        evaluation = evaluate_session(history, pipeline_result)
        if pipeline_result is not None and pipeline_result.session_id:
            try:
                store_evaluation(pipeline_result.session_id, evaluation)
            except Exception as e:
                logger.warning("Could not persist evaluation: %s", e)
        return format_evaluation_md(evaluation), gr.Button(interactive=True)
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        return f"Evaluation failed: {e}", gr.Button(interactive=True)


def _model_choices_for_provider(provider):
    """Return (choices, value, interactive) for the model dropdown."""
    if provider == PROVIDER_OLLAMA:
        return ["qwen3.5:9b"], "qwen3.5:9b", False
    return AVAILABLE_MODELS, AVAILABLE_MODELS[0], True


def update_model_choices(provider):
    choices, value, interactive = _model_choices_for_provider(provider)
    return gr.Dropdown(choices=choices, value=value, interactive=interactive)


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

                # --- Confirm pipeline trigger ---
                confirm_btn = gr.Button("Analyze CV & JD", variant="primary")
                confirm_status = gr.Markdown("")
                gap_summary_md = gr.Markdown("")

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
                        64, 32768, DEFAULT_MAX_TOKENS,
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
                    slider_tooltip = gr.Markdown("", visible=True)
                    recommended_btn = gr.Button("Apply recommended settings", size="sm")

                    model.change(
                        update_slider_states,
                        inputs=[model],
                        outputs=[temperature, top_p, freq_pen, pres_pen, slider_tooltip],
                    )
                    recommended_btn.click(
                        apply_recommended,
                        inputs=[model],
                        outputs=[temperature, top_p, max_tokens, freq_pen, pres_pen],
                    )

                with gr.Accordion("System Status", open=False):
                    status_btn = gr.Button("Check Connections")
                    status_out = gr.Textbox(label="Status", lines=8)
                    status_btn.click(check_status, outputs=[status_out])

            with gr.Column(scale=2):
                gr.Markdown("### Interview Practice Chat")
                chat_input = gr.Textbox(
                    placeholder="Type a message...",
                    show_label=False,
                    scale=7,
                )

                # pipeline_state must be declared before ChatInterface
                # so it can be passed as an additional_input
                pipeline_state = gr.State(None)

                chat_mode = gr.Radio(
                    choices=[
                        ("Practice — LLM coaches you through prompts and gives feedback after each answer", "practice"),
                        ("Study — send any prompt, LLM demonstrates a model candidate answer using your CV", "study"),
                        ("Interview — full simulation, LLM is the hiring manager (no mid-session coaching)", "interview"),
                    ],
                    value="practice",
                    label="Chat Mode",
                    info="Switch at any time — takes effect on the next message.",
                )

                # Explicit chatbot reference so evaluate button can read history
                chatbot = gr.Chatbot(height=500)

                gr.ChatInterface(
                    fn=respond,
                    chatbot=chatbot,
                    textbox=chat_input,
                    additional_inputs=[
                        cv_text, jd_text, model, provider,
                        temperature, top_p, max_tokens,
                        freq_pen, pres_pen, persona, difficulty,
                        pipeline_state, chat_mode,
                    ],
                    fill_height=True,
                )

                evaluate_btn = gr.Button(
                    "Evaluate Session",
                    variant="secondary",
                    interactive=False,
                )
                with gr.Accordion("Session Evaluation", open=False) as eval_accordion:
                    eval_output = gr.Markdown("*Run a practice session first, then click Evaluate.*")

                with gr.Accordion("Input Templates", open=False):
                    for t in INPUT_TEMPLATES:
                        with gr.Row():
                            gr.Textbox(
                                value=t["text"],
                                label=t["label"],
                                interactive=False,
                                lines=2,
                            )
                            use_btn = gr.Button("↑ Use", size="sm", min_width=60)
                            use_btn.click(
                                fn=lambda txt=t["text"]: txt,
                                outputs=[chat_input],
                            )

                with gr.Accordion("Generated Prompts", open=False):
                    gr.Markdown("*Run 'Analyze CV & JD' to generate personalised prompts.*")
                    gen_q_tbs = []
                    for i in range(5):
                        with gr.Row():
                            tb = gr.Textbox(
                                value="",
                                label=f"Prompt {i + 1}",
                                interactive=False,
                                lines=2,
                                placeholder="Will appear after analysis...",
                            )
                            use_btn = gr.Button("↑ Use", size="sm", min_width=60)
                            use_btn.click(fn=lambda v: v, inputs=[tb], outputs=[chat_input])
                            gen_q_tbs.append(tb)

        # Wire confirm button → pipeline → update status, state, questions, summary
        confirm_btn.click(
            fn=handle_confirm,
            inputs=[cv_text, jd_text, difficulty],
            outputs=[confirm_status, pipeline_state] + gen_q_tbs + [gap_summary_md],
        ).then(
            fn=lambda: gr.Button(interactive=True),
            outputs=[evaluate_btn],
        )

        # Wire evaluate button → judge → eval output
        evaluate_btn.click(
            fn=lambda: gr.Button(value="Evaluating...", interactive=False),
            outputs=[evaluate_btn],
        ).then(
            fn=handle_evaluate,
            inputs=[chatbot, pipeline_state],
            outputs=[eval_output, evaluate_btn],
        )

    return app


if __name__ == "__main__":
    print("Starting " + APP_TITLE + "...")
    init_db()
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=APP_PORT, share=False)
