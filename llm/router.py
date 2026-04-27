"""LLM Router - OpenRouter and Ollama via openai SDK."""

import logging
from openai import OpenAI
from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OLLAMA_BASE_URL,
    MODELS, PROVIDER_OPENROUTER, PROVIDER_OLLAMA,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS,
    DEFAULT_FREQUENCY_PENALTY, DEFAULT_PRESENCE_PENALTY,
    OLLAMA_MAX_TOKENS, OLLAMA_TEMPERATURE, OLLAMA_TOP_P,
    OLLAMA_PRESENCE_PENALTY, OLLAMA_TOP_K,
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
    extra_body=None,
):
    """Non-streaming chat completion."""
    if model is None:
        model = MODELS["chat"]

    client = get_client(provider)
    if provider == PROVIDER_OLLAMA:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "max_tokens": OLLAMA_MAX_TOKENS,
            "presence_penalty": OLLAMA_PRESENCE_PENALTY,
            "extra_body": {
                "top_k": OLLAMA_TOP_K,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
    else:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
    if response_format:
        kwargs["response_format"] = response_format
    if extra_body and provider != PROVIDER_OLLAMA:
        # Merge with existing extra_body if present (e.g. Ollama already sets one)
        kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **extra_body}

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
    if provider == PROVIDER_OLLAMA:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "max_tokens": OLLAMA_MAX_TOKENS,
            "presence_penalty": OLLAMA_PRESENCE_PENALTY,
            "stream": True,
            "extra_body": {
                "top_k": OLLAMA_TOP_K,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
    else:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": True,
        }

    logger.info("[stream] %s model=%s", provider, model)

    try:
        stream = client.chat.completions.create(**kwargs)
        in_reasoning = False
        chunk_count = 0
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            # Ollama sends Qwen3 reasoning in delta.reasoning — wrap in tags so strip_think_tags filters it
            reasoning = getattr(delta, "reasoning", None)
            if reasoning:
                if not in_reasoning:
                    yield "<think>"
                    in_reasoning = True
                yield reasoning
            if delta.content:
                if in_reasoning:
                    yield "</think>"
                    in_reasoning = False
                chunk_count += 1
                yield delta.content
        if in_reasoning:
            yield "</think>"
        logger.info("[stream] done, %d content chunks", chunk_count)
    except Exception as e:
        logger.error("[stream] FAILED %s: %s", provider, e)
        yield "[LLM Error: " + str(e) + "]"


def chat_collect(
    messages, model=None, provider=PROVIDER_OPENROUTER,
    temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
    max_tokens=DEFAULT_MAX_TOKENS,
    frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
    presence_penalty=DEFAULT_PRESENCE_PENALTY,
) -> str:
    """Collect full streamed response into a single string.

    Reasoning models (gpt-5-mini, o4-mini) return None in choices[0].message.content
    on non-streaming calls but deliver content correctly via the streaming path.
    This function uses streaming internally and joins the content chunks.
    Think/reasoning tags are stripped automatically.
    """
    chunks = []
    for chunk in chat_stream(
        messages=messages,
        model=model,
        provider=provider,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    ):
        # skip reasoning tags emitted by Ollama path
        if chunk in ("<think>", "</think>"):
            continue
        chunks.append(chunk)
    content = "".join(chunks).strip()
    logger.info("[collect] %s model=%s len=%d", provider, model or MODELS["chat"], len(content))
    return content


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
