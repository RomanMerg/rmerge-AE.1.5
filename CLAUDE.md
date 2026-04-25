# CLAUDE.md — Interview Practice App

## Project Overview
Turing College Sprint 1 project: CV-aware interview practice chatbot. Gradio frontend, OpenRouter + Ollama LLMs, PostgreSQL + pgvector storage.

**Full plan:** See `05-project-ultraplan.md` in the Obsidian vault (sprint-1-llm-fundamentals/).

**GitHub repo:** https://github.com/TuringCollegeSubmissions/rmerge-AE.1.5

## Architecture
- `app.py` — Gradio UI (ChatInterface + CV upload + JD input + Settings)
- `config.py` — env vars, model routing constants
- `llm/router.py` — OpenRouter + Ollama via openai SDK (same client, different base_url)
- `llm/prompts.py` — 6 system prompts (zero-shot, few-shot, CoT, role-play, structured, judge)
- `llm/guards.py` — input validation + LLM-based intent classification
- `db/connection.py` — PostgreSQL connection + init.sql auto-run
- `db/models.py` — Pydantic models (CVProfile, JobDescription, GapAnalysis, SessionEvaluation)
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
- Session 2 (Core Pipeline): TODO — CV/JD parsing, gap analysis, question generation
- Session 3 (Chat + Guards): TODO — personas, difficulty, guards, settings
- Session 4 (Polish): TODO — pgvector, judge, cost tracking, submission
