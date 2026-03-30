# =====================================================
# rag/llm.py
# Groq LLM client wrapper.
#
# WHY GROQ?
# - 100% FREE tier (generous limits)
# - Extremely fast (custom LPU hardware)
# - Supports LLaMA 3 — excellent quality
# - Simple API (OpenAI-compatible)
#
# FREE TIER LIMITS (as of 2024):
# - 30 requests/minute
# - 6000 tokens/minute
# - 500,000 tokens/day
# Plenty for development and demo.
# =====================================================

from groq import Groq
from loguru import logger
from ..core.config import settings


# Singleton Groq client
_groq_client: Groq | None = None


def get_groq_client() -> Groq:
    """Returns the Groq client, creating it if needed."""
    global _groq_client

    if _groq_client is None:
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not set in .env file. "
                "Get a free key at console.groq.com"
            )
        _groq_client = Groq(api_key=settings.groq_api_key)
        logger.info("Groq client initialized")

    return _groq_client


# ✅ Added fallback model list (no other logic touched)
_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]


def generate_answer(
    messages: list[dict],
    model: str = "llama-3.1-8b-instant",  # ✅ updated default model
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> tuple[str, dict]:
    """
    Generate an answer from the LLM.

    WHY temperature=0.1 (near zero)?
    We want FACTUAL answers from the context only.
    High temperature = creative but potentially hallucinates.
    Low temperature = sticks closely to provided context.
    For RAG, factuality > creativity.

    WHY llama3-8b-8192?
    - 8192 token context window (fits multiple chunks)
    - Fast on Groq hardware
    - Good instruction following
    - Free tier friendly

    Args:
        messages:    OpenAI-format message list
        model:       Groq model name
        max_tokens:  Max response length
        temperature: 0=deterministic, 1=creative

    Returns:
        (answer_text, usage_stats)
    """
    client = get_groq_client()

    # ✅ Try requested model first, then fallback models
    models_to_try = [model] + [m for m in _MODELS if m != model]

    for m in models_to_try:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            answer = response.choices[0].message.content

            # Usage stats for monitoring
            usage = {
                "prompt_tokens":     response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens":      response.usage.total_tokens,
                "model":             m,
            }

            logger.info(
                f"LLM response | tokens={usage['total_tokens']} "
                f"model={m}"
            )

            return answer, usage

        except Exception as e:
            logger.warning(f"Model {m} failed: {e}")
            continue

    logger.error("All Groq models failed")
    raise RuntimeError("All Groq models failed")