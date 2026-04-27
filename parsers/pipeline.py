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


def parse_cv(cv_text: str) -> CVProfile:
    result = chat(
        messages=[
            {"role": "system", "content": CV_EXTRACTION},
            {"role": "user", "content": cv_text},
        ],
        model=MODELS["parse"],
        provider=PROVIDER_OPENROUTER,
        response_format={"type": "json_object"},
        max_tokens=1024,
    )
    try:
        data = json.loads(result["content"])
        cv = CVProfile(**data)
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
        response_format={"type": "json_object"},
        max_tokens=1024,
    )
    try:
        data = json.loads(result["content"])
        jd = JobDescription(**data)
    except Exception as e:
        raise ValueError(f"JD parse failed: {e}") from e
    logger.info(
        "[pipeline] JD parsed: %s @ %s, %d requirements",
        jd.role_title,
        jd.company_type,
        len(jd.requirements),
    )
    return jd
