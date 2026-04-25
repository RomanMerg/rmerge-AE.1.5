"""LLM Router - OpenRouter and Ollama via openai SDK."""
"""Ollama/Qwen3.5:9b(thinking)"""

import logging
from openai import OpenAI
from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OLLAMA_BASE_URL,
    MODELS, PROVIDER_OPENROUTER, PROVIDER_OLLAMA,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS,
    DEFAULT_FREQUENCY_PENALTY, DEFAULT_PRESENCE_PENALTY,
)

logger = logging.getLogger(__name__)

_openrouter_client = None
_ollama_client = None


def get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is None:
        logger.info("Creating OpenRouter client")
        _openrouter_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
    return _openrouter_client


def get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        url = OLLAMA_BASE_URL + "/v1"
        logger.info("Creating Ollama client: %s", url)
        _ollama_client = OpenAI(base_url=url, api_key="ollama")
    return _ollama_client


def get_client(provider):
    if provider == PROVIDER_OLLAMA:
        return get_ollama_client()
    return get_openrouter_client()


def chat(
    messages, model=None, provider=PROVIDER_OPENROUTER,
    temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
    max_tokens=DEFAULT_MAX_TOKENS,
    frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
    presence_penalty=DEFAULT_PRESENCE_PENALTY,
    response_format=None,
):
    """Non-streaming chat completion."""
    if model is None:
        model = MODELS["chat"]

    client = get_client(provider)
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if provider != PROVIDER_OLLAMA:
        kwargs["frequency_penalty"] = frequency_penalty
        kwargs["presence_penalty"] = presence_penalty
    else:
        kwargs["extra_body"] = {"options": {"think": False}}
    if response_format:
        kwargs["response_format"] = response_format

    logger.info("[chat] %s model=%s msgs=%d", provider, model, len(messages))

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        logger.error("[chat] FAILED %s: %s", provider, e)
        raise

    choice = response.choices[0]
    usage = response.usage
    tok = usage.total_tokens if usage else 0

    # Debug logging
    logger.debug("[chat] Full choice: %s", choice)
    logger.debug("[chat] Content: %s", choice.message.content)
    reasoning = getattr(choice.message, 'reasoning_content', None)
    logger.debug("[chat] Reasoning: %s", reasoning)

    # Handle empty content from thinking models
    content = choice.message.content or ""
    if not content and reasoning:
        logger.warning("[chat] Empty content but reasoning exists; using reasoning as fallback")
        content = reasoning

    if not content:
        logger.warning("[chat] Both content and reasoning are empty!")

    logger.info("[chat] OK tokens=%d", tok)

    return {
        "content": content,
        "model": response.model,
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": tok,
        },
        "finish_reason": choice.finish_reason,
    }


def chat_stream(
    messages, model=None, provider=PROVIDER_OPENROUTER,
    temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
    max_tokens=DEFAULT_MAX_TOKENS,
    frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
    presence_penalty=DEFAULT_PRESENCE_PENALTY,
):
    """Streaming chat. Yields content chunks."""
    if model is None:
        model = MODELS["chat"]

    client = get_client(provider)
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if provider != PROVIDER_OLLAMA:
        kwargs["frequency_penalty"] = frequency_penalty
        kwargs["presence_penalty"] = presence_penalty
    else:
        # Disable Qwen3 thinking mode via Ollama's options API
        kwargs["extra_body"] = {"options": {"think": False}}

    logger.info("[stream] %s model=%s", provider, model)

    try:
        stream = client.chat.completions.create(**kwargs)
        chunk_count = 0
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                chunk_count += 1
                yield delta.content
        logger.debug("[stream] received %d chunks", chunk_count)
    except Exception as e:
        logger.error("[stream] FAILED %s: %s", provider, e)
        yield "[LLM Error: " + str(e) + "]"


def smoke_test(provider=PROVIDER_OPENROUTER):
    """Quick connection test."""
    if provider == PROVIDER_OPENROUTER:
        mdl = MODELS["chat"]
    else:
        mdl = MODELS["chat_local"]
    msgs = [
        {"role": "system", "content": "Respond briefly."},
        {"role": "user", "content": "Say hello."},
    ]
    try:
        r = chat(messages=msgs, model=mdl, provider=provider, max_tokens=50)
        return r["model"] + ": " + r["content"]
    except Exception as e:
        return "Error: " + str(e)
