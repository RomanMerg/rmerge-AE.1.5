"""Application configuration — loads from .env and defines model routing."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys & URLs ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/interview_app")

# --- Model Routing ---
# Each stage uses the optimal model for its task
MODELS = {
    "guard": os.getenv("GUARD_MODEL", "openai/gpt-5-nano"),
    "parse": os.getenv("DEFAULT_MODEL", "openai/gpt-5-mini"),
    "chat": os.getenv("DEFAULT_MODEL", "openai/gpt-5-mini"),
    "chat_local": os.getenv("LOCAL_CHAT_MODEL", "qwen3.5:9b"),
    "judge": os.getenv("JUDGE_MODEL", "openai/gpt-5"),
    "embed": os.getenv("EMBED_MODEL", "nomic-embed-text-v2-moe"),
}

# Available models for the UI dropdown
AVAILABLE_MODELS = [
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-5",
]

# Per-model capability flags and recommended slider values.
# supports_* = False means that slider should be greyed out in the UI.
MODEL_CONFIGS = {
    "openai/gpt-5-mini": {
        "supports_temperature": False,
        "supports_top_p": True,
        "supports_freq_penalty": True,
        "supports_pres_penalty": True,
        "recommended": {
            "temperature": 1.0,   # not sent to API when unsupported
            "top_p": 1.0,
            "max_tokens": 1024,
            "freq_pen": 0.0,
            "pres_pen": 0.0,
        },
        "tooltip": "gpt-5-mini uses fixed sampling — Temperature has no effect for this model.",
    },
    "openai/gpt-5-nano": {
        # TODO: double-check this — I couldn't find a definitive answer on whether
        # gpt-5-nano ignores temperature. Check OpenRouter model page for gpt-5-nano
        # before shipping. If it does support it, flip this to True.
        "supports_temperature": False,
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

# --- LLM Default Settings (OpenRouter) ---
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 1024
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0


# --- Providers ---
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_OLLAMA = "ollama"
DEFAULT_PROVIDER = PROVIDER_OPENROUTER

# --- App Settings ---
APP_TITLE = "Interview Practice App"
APP_PORT = 7860
MAX_CV_LENGTH = 10_000
MAX_JD_LENGTH = 5_000
EMBEDDING_DIMS = 256  # nomic-embed-text-v2-moe Matryoshka dimension
