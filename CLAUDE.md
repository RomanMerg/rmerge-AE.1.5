# CLAUDE.md — Interview Practice App

## Project Overview
Turing College Sprint 1 project: CV-aware interview practice chatbot. Gradio frontend, OpenRouter + Ollama LLMs, PostgreSQL + pgvector storage.

**Full plan:** See `05-project-ultraplan.md` in the Obsidian vault (sprint-1-llm-fundamentals/).

**GitHub repo:** https://github.com/TuringCollegeSubmissions/rmerge-AE.1.5

## Architecture
- `app.py` — Gradio UI (ChatInterface + CV upload + JD input + Settings + Confirm button)
- `config.py` — env vars, model routing constants
- `llm/router.py` — OpenRouter + Ollama via openai SDK (same client, different base_url)
- `llm/prompts.py` — 6 system prompts (zero-shot, few-shot, CoT, role-play, structured, judge)
- `llm/guards.py` — input validation + LLM-based intent classification
- `parsers/pipeline.py` — parse_cv → parse_jd → embeddings → gap_analysis → questions → DB persist
- `db/connection.py` — PostgreSQL connection + init.sql auto-run
- `db/models.py` — Pydantic models (CVProfile, JobDescription, GapAnalysis, SessionEvaluation)
- `db/queries.py` — CRUD: create_session, store_cv_profile, store_jd, store_gap_analysis
- `utils/cost_tracker.py` — API cost estimation

## Model Routing
```python
MODELS = {
    "guard": "openai/gpt-5-nano",       # cheap, fast classification
    "parse": "openai/gpt-5-mini",       # structured extraction
    "chat": "openai/gpt-5-mini",        # interactive default
    "chat_local": "qwen3.5:9b",         # Ollama optional
    "judge": "openai/gpt-5",            # highest capability, 1x per session
    "embed": "nomic-embed-text-v2-moe", # Ollama, Matryoshka @ 256 dims
}
```

## Infrastructure
- PostgreSQL: `host.docker.internal:5432` → database `interview_app`
- Ollama: `http://host.docker.internal:11434` (qwen3.5:9b + nomic-embed-text-v2-moe)
- OpenRouter: external API, key in `.env`
- App: port 7860, Docker container

## Session Status
- Session 1 (Scaffold + Infra): COMPLETE
- Session 2 (Core Pipeline): IN PROGRESS — see `docs/superpowers/specs/2026-04-27-session2-core-pipeline-design.md`
- Session 3 (Chat + Guards): TODO — LLM-as-judge evaluation button, enhanced guards
- Session 4 (Polish): TODO — pgvector dedup query, cost tracking, submission prep

## Session 2 Plan
1. `parsers/pipeline.py` — parse_cv(), parse_jd(), generate_embeddings(), run_gap_analysis(), generate_questions(), run_pipeline() orchestrator
2. `db/queries.py` — create_session(), store_cv_profile(), store_jd(), store_gap_analysis()
3. `app.py` — Confirm button + status label + pipeline_state (gr.State) + dynamic Input Templates + gap summary as first chat message + structured system prompt when pipeline ran
4. Each pipeline function logs a one-liner summary; raises ValueError with context on failure

## Session 2 Key Decisions
- Single `parsers/pipeline.py` (not separate cv_parser/jd_parser) — approved
- No `instructor` lib — manual json.loads() + response_format={"type":"json_object"} on gpt-5-mini
- Parsing always uses MODELS["parse"] (OpenRouter) regardless of UI provider selection
- Embeddings generated in Session 2 (Ollama, silent NULL fallback); dedup query deferred to Session 4
- Confirm button is deliberate trigger — parsing does NOT happen on CV/JD text change
