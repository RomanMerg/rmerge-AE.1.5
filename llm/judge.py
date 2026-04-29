"""LLM-as-Judge: evaluate a completed interview session."""

import logging
from llm.router import chat
from llm.prompts import SESSION_EVALUATION
from db.models import SessionEvaluation
from parsers.pipeline import _parse_llm_json
from config import MODELS, PROVIDER_OPENROUTER

logger = logging.getLogger(__name__)


def evaluate_session(history: list[dict], pipeline_result) -> SessionEvaluation:
    """Send full chat history to judge model and return structured evaluation.

    history: list of {"role": "user"|"assistant", "content": str}
    pipeline_result: PipelineResult | None — for role/readiness context
    """
    # Build transcript — Q: for assistant (interviewer), A: for user (candidate)
    lines = []
    for msg in history:
        role = msg.get("role", "")
        content_raw = msg.get("content", "")
        # Gradio 6.x stores multimodal content as list[{"type": "text", "text": "..."}]
        if isinstance(content_raw, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content_raw
            ).strip()
        else:
            content = str(content_raw).strip()
        if role == "assistant":
            lines.append(f"Q: {content}")
        elif role == "user":
            lines.append(f"A: {content}")
    transcript = "\n\n".join(lines)

    # Build context block if pipeline ran
    context_parts = []
    if pipeline_result is not None:
        jd = pipeline_result.jd
        gaps = pipeline_result.gaps
        context_parts.append(f"Role: {jd.role_title} ({jd.role_level})")
        context_parts.append(f"Readiness score: {gaps.readiness_score}/100")
        if gaps.gaps:
            context_parts.append(
                "Key gaps: " + ", ".join(g.requirement for g in gaps.gaps[:5])
            )

    user_content = ""
    if context_parts:
        user_content = "\n".join(context_parts) + "\n\n"
    user_content += "Interview transcript:\n\n" + transcript

    result = chat(
        messages=[
            {"role": "system", "content": SESSION_EVALUATION},
            {"role": "user", "content": user_content},
        ],
        model=MODELS["judge"],
        provider=PROVIDER_OPENROUTER,
        max_tokens=4096,
        extra_body={"reasoning": {"effort": "low"}},
    )

    data = _parse_llm_json(result["content"], "Session evaluation")
    evaluation = SessionEvaluation(**data)
    logger.info(
        "[judge] evaluation complete: overall=%d, %d answers",
        evaluation.overall_score,
        len(evaluation.answer_evaluations),
    )
    return evaluation
