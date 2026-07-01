"""Provider-agnostic LLM client (Anthropic default, OpenAI optional).

Returns text plus token usage and an estimated USD cost so the tracer can log
it. Prices are approximate per-million-token rates; adjust in PRICES as needed.
"""
from dataclasses import dataclass

from app.config import settings

# USD per 1M tokens (input, output). Update if pricing changes.
PRICES = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
}


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    in_price, out_price = PRICES.get(model, (0.0, 0.0))
    return (in_tok * in_price + out_tok * out_price) / 1_000_000


def complete(system: str, user: str, max_tokens: int = 1024) -> LLMResult:
    provider = settings.llm_provider.lower()
    if provider == "groq":
        return _groq(system, user, max_tokens)
    if provider == "ollama":
        return _ollama(system, user, max_tokens)
    if provider == "anthropic":
        return _anthropic(system, user, max_tokens)
    if provider == "openai":
        return _openai(system, user, max_tokens)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


def _groq(system: str, user: str, max_tokens: int) -> LLMResult:
    """Groq — free hosted API, OpenAI-compatible, nothing to download.

    Get a free key at https://console.groq.com/keys. Free tier => cost 0.
    The same code path works for any OpenAI-compatible free endpoint
    (OpenRouter, Gemini's OpenAI shim, etc.) — just change GROQ_BASE_URL.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
        timeout=60,
        max_retries=2,
    )
    model = settings.groq_model
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,  # deterministic extraction
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    in_tok = getattr(usage, "prompt_tokens", 0) or 0
    out_tok = getattr(usage, "completion_tokens", 0) or 0
    return LLMResult(text, model, in_tok, out_tok, 0.0)


def _ollama(system: str, user: str, max_tokens: int) -> LLMResult:
    """Local Ollama via its OpenAI-compatible endpoint. Free, so cost is 0."""
    from openai import OpenAI

    client = OpenAI(base_url=f"{settings.ollama_host}/v1", api_key="ollama")
    model = settings.ollama_model
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,  # deterministic extraction
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    in_tok = getattr(usage, "prompt_tokens", 0) or 0
    out_tok = getattr(usage, "completion_tokens", 0) or 0
    return LLMResult(text, model, in_tok, out_tok, 0.0)


def _anthropic(system: str, user: str, max_tokens: int) -> LLMResult:
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    model = settings.anthropic_model
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    return LLMResult(text, model, in_tok, out_tok, _cost(model, in_tok, out_tok))


def _openai(system: str, user: str, max_tokens: int) -> LLMResult:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    model = settings.openai_model
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content or ""
    in_tok = resp.usage.prompt_tokens
    out_tok = resp.usage.completion_tokens
    return LLMResult(text, model, in_tok, out_tok, _cost(model, in_tok, out_tok))
