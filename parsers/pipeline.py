"""Parsing pipeline: CV → JD → embeddings → gap analysis → questions → DB."""

import json
import logging
from dataclasses import dataclass

from llm.router import chat, get_ollama_client
from llm.prompts import CV_EXTRACTION, JD_EXTRACTION, GAP_ANALYSIS, QUESTION_GENERATION
from db.models import CVProfile, JobDescription, GapAnalysis, RequirementSeverity
from db.queries import create_session, store_cv_profile, store_jd, store_gap_analysis
from config import MODELS, PROVIDER_OPENROUTER

logger = logging.getLogger(__name__)


def _parse_llm_json(raw: str, label: str) -> dict:
    """Strip markdown fences if present, then json.loads. Raises ValueError on failure."""
    content = raw.strip()
    if not content:
        raise ValueError(f"{label} failed: LLM returned empty response")
    # Strip ```json ... ``` fences that some models add despite json_object mode
    if content.startswith("```"):
        content = content.split("```", 2)[-1] if content.count("```") >= 2 else content
        content = content.lstrip("json").strip().rstrip("`").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"{label} failed: invalid JSON — {e}") from e


def parse_cv(cv_text: str) -> CVProfile:
    result = chat(
        messages=[
            {"role": "system", "content": CV_EXTRACTION},
            {"role": "user", "content": cv_text},
        ],
        model=MODELS["parse"],
        provider=PROVIDER_OPENROUTER,
        max_tokens=2048,
    )
    try:
        data = _parse_llm_json(result["content"], "CV parse")
        cv = CVProfile(**data)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"CV parse failed: {e}") from e
    logger.info(
        "[pipeline] CV parsed: %d skills, %dy exp, %d projects",
        len(cv.technical_skills),
        cv.years_of_experience or 0,
        len(cv.projects),
    )
    return cv


def parse_jd(jd_text: str) -> JobDescription:
    result = chat(
        messages=[
            {"role": "system", "content": JD_EXTRACTION},
            {"role": "user", "content": jd_text},
        ],
        model=MODELS["parse"],
        provider=PROVIDER_OPENROUTER,
        max_tokens=2048,
    )
    try:
        data = _parse_llm_json(result["content"], "JD parse")
        jd = JobDescription(**data)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"JD parse failed: {e}") from e
    logger.info(
        "[pipeline] JD parsed: %s @ %s, %d requirements",
        jd.role_title,
        jd.company_type,
        len(jd.requirements),
    )
    return jd


def generate_embeddings(text: str) -> list[float] | None:
    try:
        client = get_ollama_client()
        response = client.embeddings.create(
            model=MODELS["embed"],
            input=text[:8000],
        )
        embedding = response.data[0].embedding
        logger.info("[pipeline] Embedding: %d dims", len(embedding))
        return embedding
    except Exception as e:
        logger.warning("[pipeline] Embedding unavailable: %s", e)
        return None


def run_gap_analysis(cv: CVProfile, jd: JobDescription) -> GapAnalysis:
    context = json.dumps({"cv": cv.model_dump(), "job_description": jd.model_dump()})
    result = chat(
        messages=[
            {"role": "system", "content": GAP_ANALYSIS},
            {"role": "user", "content": f"Analyze this CV and job description:\n{context}"},
        ],
        model=MODELS["parse"],
        provider=PROVIDER_OPENROUTER,
        max_tokens=2048,
    )
    try:
        data = _parse_llm_json(result["content"], "Gap analysis")
        gaps = GapAnalysis(**data)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Gap analysis failed: {e}") from e
    logger.info(
        "[pipeline] Gap analysis: score=%d, %d gaps (%d critical), %d matches",
        gaps.readiness_score,
        len(gaps.gaps),
        sum(1 for g in gaps.gaps if g.severity == RequirementSeverity.CRITICAL),
        len(gaps.matching_skills),
    )
    return gaps


def generate_questions(gaps: GapAnalysis, difficulty: str, n: int = 5) -> list[str]:
    gap_list = "\n".join(
        f"- {g.requirement} (severity: {g.severity.value})" for g in gaps.gaps[:10]
    ) or "General technical and behavioural skills"
    user_content = (
        QUESTION_GENERATION.format(num_questions=n, difficulty=difficulty, gaps=gap_list)
        + '\n\nReturn ONLY a JSON object with key "questions" containing an array of strings.'
    )
    result = chat(
        messages=[{"role": "user", "content": user_content}],
        model=MODELS["parse"],
        provider=PROVIDER_OPENROUTER,
        max_tokens=1024,
    )
    try:
        data = _parse_llm_json(result["content"], "Question generation")
    except ValueError:
        raise
    try:
        questions = [str(q) for q in data["questions"][:n]]
        if not questions:
            raise ValueError("empty questions list")
    except KeyError:
        raise ValueError(
            f"Question generation failed — 'questions' key missing; got: {list(data.keys())}"
        )
    logger.info("[pipeline] Questions: %d generated for difficulty=%s", len(questions), difficulty)
    return questions


@dataclass
class PipelineResult:
    cv: CVProfile
    jd: JobDescription
    gaps: GapAnalysis
    questions: list[str]
    session_id: str


def run_pipeline(cv_text: str, jd_text: str, difficulty: str = "medium") -> PipelineResult:
    """Run the full parsing pipeline and persist results. Returns PipelineResult."""
    cv = parse_cv(cv_text)
    jd = parse_jd(jd_text)

    cv_embedding = generate_embeddings(cv_text)
    jd_embedding = generate_embeddings(jd_text)

    session_id = create_session()
    cv_id = store_cv_profile(session_id, cv_text, cv, cv_embedding)
    jd_id = store_jd(session_id, jd_text, jd, jd_embedding)

    gaps = run_gap_analysis(cv, jd)
    store_gap_analysis(session_id, cv_id, jd_id, gaps)

    questions = generate_questions(gaps, difficulty)
    logger.info("[pipeline] Complete: session=%s, score=%d", session_id, gaps.readiness_score)
    return PipelineResult(cv=cv, jd=jd, gaps=gaps, questions=questions, session_id=session_id)
