"""CRUD operations for session data."""

import json
import logging
from db.connection import execute_query
from db.models import CVProfile, JobDescription, GapAnalysis, SessionEvaluation

logger = logging.getLogger(__name__)


def _vec_literal(embedding: list[float] | None) -> str | None:
    if embedding is None:
        return None
    return "[" + ",".join(str(v) for v in embedding) + "]"


def create_session() -> str:
    rows = execute_query(
        "INSERT INTO sessions DEFAULT VALUES RETURNING id",
        fetch=True,
    )
    session_id = str(rows[0]["id"])
    logger.debug("[db] session created: %s", session_id)
    return session_id


def store_cv_profile(
    session_id: str,
    raw_text: str,
    cv: CVProfile,
    embedding: list[float] | None,
) -> str:
    rows = execute_query(
        """INSERT INTO cv_profiles (session_id, raw_text, parsed_data, embedding)
           VALUES (%s, %s, %s, %s::vector)
           RETURNING id""",
        (session_id, raw_text, json.dumps(cv.model_dump()), _vec_literal(embedding)),
        fetch=True,
    )
    cv_id = str(rows[0]["id"])
    logger.debug("[db] cv_profile stored: %s", cv_id)
    return cv_id


def store_jd(
    session_id: str,
    raw_text: str,
    jd: JobDescription,
    embedding: list[float] | None,
) -> str:
    rows = execute_query(
        """INSERT INTO job_descriptions (session_id, raw_text, parsed_data, embedding)
           VALUES (%s, %s, %s, %s::vector)
           RETURNING id""",
        (session_id, raw_text, json.dumps(jd.model_dump()), _vec_literal(embedding)),
        fetch=True,
    )
    jd_id = str(rows[0]["id"])
    logger.debug("[db] job_description stored: %s", jd_id)
    return jd_id


def store_gap_analysis(
    session_id: str,
    cv_id: str,
    jd_id: str,
    gaps: GapAnalysis,
) -> str:
    rows = execute_query(
        """INSERT INTO gap_analyses (session_id, cv_id, jd_id, analysis, readiness_score)
           VALUES (%s, %s, %s, %s, %s)
           RETURNING id""",
        (session_id, cv_id, jd_id, json.dumps(gaps.model_dump()), gaps.readiness_score),
        fetch=True,
    )
    gap_id = str(rows[0]["id"])
    logger.debug("[db] gap_analysis stored: %s score=%d", gap_id, gaps.readiness_score)
    return gap_id


def store_chat_message(
    session_id: str,
    role: str,
    content: str,
    model_used: str | None = None,
    cost_usd: float | None = None,
) -> str:
    """Store a single message in chat_messages table. Returns message_id."""
    rows = execute_query(
        """INSERT INTO chat_messages (session_id, role, content, model_used, cost_usd)
           VALUES (%s, %s, %s, %s, %s)
           RETURNING id""",
        (session_id, role, content, model_used, cost_usd),
        fetch=True,
    )
    msg_id = str(rows[0]["id"])
    logger.debug("[db] chat_message stored: %s role=%s", msg_id, role)
    return msg_id


def store_evaluation(session_id: str, evaluation: SessionEvaluation) -> str:
    rows = execute_query(
        """INSERT INTO evaluations (session_id, evaluation, overall_score)
           VALUES (%s, %s, %s)
           RETURNING id""",
        (session_id, json.dumps(evaluation.model_dump()), evaluation.overall_score),
        fetch=True,
    )
    eval_id = str(rows[0]["id"])
    logger.debug("[db] evaluation stored: %s score=%d", eval_id, evaluation.overall_score)
    return eval_id


def find_similar_cv(embedding: list[float], threshold: float = 0.85, limit: int = 5) -> list[dict]:
    """Find CV profiles with cosine similarity above threshold.

    Returns list of dicts with keys: session_id, cv_id, similarity_score.
    """
    vec = _vec_literal(embedding)
    rows = execute_query(
        """SELECT id, session_id, 1 - (embedding <=> %s::vector) AS similarity_score
           FROM cv_profiles
           WHERE embedding IS NOT NULL
             AND 1 - (embedding <=> %s::vector) > %s
           ORDER BY similarity_score DESC
           LIMIT %s""",
        (vec, vec, threshold, limit),
        fetch=True,
    )
    return [
        {"cv_id": str(r["id"]), "session_id": str(r["session_id"]), "similarity_score": r["similarity_score"]}
        for r in rows
    ]


def find_similar_jd(embedding: list[float], threshold: float = 0.85, limit: int = 5) -> list[dict]:
    """Find job descriptions with cosine similarity above threshold.

    Returns list of dicts with keys: session_id, jd_id, similarity_score.
    """
    vec = _vec_literal(embedding)
    rows = execute_query(
        """SELECT id, session_id, 1 - (embedding <=> %s::vector) AS similarity_score
           FROM job_descriptions
           WHERE embedding IS NOT NULL
             AND 1 - (embedding <=> %s::vector) > %s
           ORDER BY similarity_score DESC
           LIMIT %s""",
        (vec, vec, threshold, limit),
        fetch=True,
    )
    return [
        {"jd_id": str(r["id"]), "session_id": str(r["session_id"]), "similarity_score": r["similarity_score"]}
        for r in rows
    ]
