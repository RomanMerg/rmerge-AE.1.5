# Interview Practice App

A CV-aware interview practice chatbot built for Turing College Sprint 1 (LLM Fundamentals). The application compares a candidate's CV against a target job description, identifies skill gaps, generates targeted interview questions, and evaluates answers across a multi-turn conversation — all powered by a combination of OpenRouter-hosted and locally-running open-source LLMs.

**GitHub:** https://github.com/TuringCollegeSubmissions/rmerge-AE.1.5

---

## What It Does

1. **CV & Job Description Analysis** — Upload a PDF CV and paste a job description. The app extracts structured data from both using LLM-powered parsing (zero-shot and structured output prompts).
2. **Gap Analysis** — A chain-of-thought prompt identifies matching skills, partial matches, and critical gaps, producing a readiness score from 0–100.
3. **Interview Practice** — A multi-turn chat session with a configurable interviewer persona (neutral / strict / friendly) and difficulty level (easy / medium / hard). Questions are generated from the gap analysis using few-shot prompting.
4. **Session Evaluation** — An LLM-as-Judge evaluation (using the highest-capability model) scores each answer and provides an overall readiness summary with specific improvement recommendations.

---

## Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Frontend | Gradio (`gr.ChatInterface`) | Multi-turn chat, file upload, clean UI without custom CSS |
| LLM Primary | `openai/gpt-5-mini` via OpenRouter | Mandatory requirement; good balance of speed and capability |
| LLM Judge | `openai/gpt-5` via OpenRouter | Runs once per session; highest capability for evaluation |
| LLM Guard | `openai/gpt-5-nano` via OpenRouter | Cheapest model for simple intent classification |
| Local LLM | `qwen3.5:9b` via Ollama | Hard optional — open-source LLM, user-selectable |
| Embeddings | `nomic-embed-text-v2-moe` via Ollama | MoE architecture, Matryoshka @ 256 dims, free and local |
| PDF Parsing | `pymupdf4llm` | Converts PDF → LLM-optimised markdown |
| Storage | PostgreSQL + pgvector | Structured data + vector embeddings for deduplication |
| Python SDK | `openai` v1.x | Same client works for both OpenRouter and Ollama |
| Deployment | Docker Compose (app container only) | Connects to host PostgreSQL and host Ollama |

---

## Model Routing

Different models are used for different stages of the pipeline, optimised for cost and capability:

| Stage | Model | Provider |
|-------|-------|----------|
| Security guard (intent check) | `openai/gpt-5-nano` | OpenRouter |
| CV / JD parsing | `openai/gpt-5-mini` | OpenRouter |
| Gap analysis | `openai/gpt-5-mini` | OpenRouter |
| Interview chat (default) | `openai/gpt-5-mini` | OpenRouter |
| Interview chat (local) | `qwen3.5:9b` | Ollama |
| Embeddings | `nomic-embed-text-v2-moe` | Ollama |
| LLM-as-Judge evaluation | `openai/gpt-5` | OpenRouter |

---

## Assignment Coverage

### Mandatory Requirements

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1 | Define scope | CV vs JD matching, personalised interview prep |
| 2 | Front-end | Gradio (`gr.ChatInterface`) |
| 3 | OpenRouter API key | Environment variable, primary provider |
| 4 | Use specified model | `openai/gpt-5-mini` default; gpt-5-nano and gpt-5 also selectable |
| 5 | ≥5 system prompts | 6 prompts covering all major techniques (see below) |
| 6 | Tune ≥1 setting | Full settings panel: temperature, top-p, max tokens, frequency/presence penalty |
| 7 | ≥1 security guard | Input intent classifier + CV/JD content validation |

### Medium Optional Tasks (6 of 6)

- Expose all model settings as UI sliders ✅
- ≥2 structured JSON output formats ✅ (CV extraction + gap analysis)
- Show cost per prompt ✅
- Add job description field ✅
- Let user choose multiple LLM providers ✅ (OpenRouter / Ollama toggle)
- Simulate difficulty levels ✅ (easy / medium / hard)

### Hard Optional Tasks (4 of 4)

- Full multi-turn chatbot ✅ (`gr.ChatInterface`)
- Use open-source LLMs ✅ (Ollama + qwen3.5:9b)
- LLM-as-a-Judge ✅ (session evaluation with `openai/gpt-5`)
- Vector DB for deduplication ✅ (pgvector on PostgreSQL)

---

## System Prompts — 6 Techniques

### 1. Zero-Shot — CV Skill Extraction
Instructs the model to extract structured skills, experience, education, and projects from raw CV text and return valid JSON with no examples provided.

### 2. Few-Shot — Interview Question Generation
Provides three example gap→question pairs before asking the model to generate targeted questions for the candidate's specific gaps.

### 3. Chain-of-Thought — Gap Analysis
Asks the model to reason step-by-step: list all JD requirements, check each against the CV, categorise as MATCH / PARTIAL / GAP, and calculate a readiness score.

### 4. Role-Playing — Interviewer Persona
The model adopts a configurable interviewer character (strict / neutral / friendly) and stays in character throughout the multi-turn session.

### 5. Structured Output — JD Parsing
Instructs the model to return ONLY valid JSON matching a defined schema, used with `instructor` + Pydantic for reliable extraction.

### 6. Self-Consistency / Evaluation — LLM-as-Judge
A separate model call evaluates each Q&A pair in the session transcript independently, then produces an overall readiness score with specific improvement suggestions.

---

## Security Guards

### Guard 1: Input Intent Classifier
Every user message is classified by `gpt-5-nano` before processing. Categories: `LEGITIMATE`, `JAILBREAK`, `OFF_TOPIC`, `MALICIOUS`. Non-legitimate inputs are rejected before reaching the main model.

### Guard 2: CV/JD Content Validation
- Maximum input length enforced (10,000 chars for CV, 5,000 for JD)
- Prompt injection pattern detection in uploaded files
- PDF-parsed markdown sanitised before being sent to the LLM

### Guard 3: Output Validation
- LLM responses checked to stay within the interview domain
- Structured outputs validated against Pydantic schemas with retry logic

---

## Database Schema

PostgreSQL with the `pgvector` extension. All tables use UUIDs and are session-scoped.

- `sessions` — top-level session record
- `cv_profiles` — raw CV text, parsed JSON, and a 256-dimensional embedding
- `job_descriptions` — raw JD text, parsed JSON, and a 256-dimensional embedding
- `gap_analyses` — structured gap analysis JSON and readiness score
- `chat_messages` — full conversation history with model used and estimated cost
- `evaluations` — per-session LLM-as-Judge output

Vector embeddings use `nomic-embed-text-v2-moe` at 256 Matryoshka dimensions with an IVFFlat cosine similarity index. Before generating questions, the app queries pgvector for similar CV+JD combinations — if similarity exceeds 0.95, the model is instructed to produce novel questions not seen in prior sessions.

---

## UI Features

### Settings Panel
- **Provider** selector (OpenRouter / Ollama) — shown first; switches the model dropdown between the OpenRouter model list and the fixed local model (`qwen3.5:9b`)
- **Model** dropdown — dependent on provider selection; locked to `qwen3.5:9b` when Ollama is active
- **Per-model capability config** — sliders for unsupported parameters are greyed out automatically (e.g. Temperature is disabled for gpt-5-mini, which uses fixed sampling)
- **Apply recommended settings** button — snaps all sliders to documented defaults for the selected model
- Interviewer Persona and Difficulty Level selectors

### Input Templates
A collapsible accordion below the chat window containing pre-written prompt templates for common interview scenarios (start the interview, focus on weakest skills, request a behavioral round, ask for a harder follow-up, request a session evaluation). Each template can be loaded into the chat input with one click for editing before sending.

---

## Infrastructure

| Service | Host | Port |
|---------|------|------|
| PostgreSQL | localhost | 5432 |
| Ollama API | localhost | 11434 |
| Gradio App | localhost | 7860 |
| OpenRouter API | openrouter.ai | 443 |

The app runs as a single Docker container. PostgreSQL and Ollama run on the host machine; the container connects via `host.docker.internal`.

---

## Running Locally

**Prerequisites:** Docker, a running PostgreSQL instance with the `pgvector` extension, Ollama with `qwen3.5:9b` and `nomic-embed-text-v2-moe` pulled, and an OpenRouter API key.

```bash
# Clone the repo
git clone https://github.com/TuringCollegeSubmissions/rmerge-AE.1.5
cd rmerge-AE.1.5

# Create .env from the example
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY and DATABASE_URL

# Start the app
docker compose up --build
```

Open http://localhost:7860 once the container is running.

---

## Project Structure

```
rmerge-AE.1.5/
├── app.py              # Gradio UI — chat interface, settings panel, input templates
├── config.py           # Environment variables, model routing, per-model capability config
├── init.sql            # PostgreSQL schema (auto-run on first start)
├── llm/
│   ├── router.py       # OpenRouter + Ollama client (same openai SDK, different base_url)
│   ├── prompts.py      # All 6 system prompts
│   ├── guards.py       # Input validation and intent classification
│   ├── judge.py        # LLM-as-Judge session evaluation
│   └── user_prompts.py # Chat input template prompts shown in the UI
├── parsers/
│   ├── cv_parser.py    # PDF → markdown → structured JSON
│   └── jd_parser.py    # Job description extraction
├── db/
│   ├── connection.py   # PostgreSQL connection, pgvector setup
│   ├── models.py       # Pydantic models
│   └── queries.py      # CRUD and vector similarity queries
├── utils/
│   └── cost_tracker.py # OpenRouter API cost estimation
└── tests/
    └── test_settings_handlers.py  # Unit tests for UI handler logic
```
