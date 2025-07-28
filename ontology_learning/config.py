from functools import cache

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.globals import set_llm_cache
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.cache import SQLiteCache


@cache
def chat_model(model: str, use_cache: bool = True) -> BaseChatModel:
    kwargs, provider = {}, "openai"
    if use_cache:
        set_llm_cache(SQLiteCache(database_path="./.cache"))

    if model in ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]:
        kwargs["temperature"] = 0.0
    if model == "llama-3.3-70B":
        model = "accounts/fireworks/models/llama-v3p3-70b-instruct"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "llama-4-maverick":
        model = "accounts/fireworks/models/llama4-maverick-instruct-basic"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "llama-4-scout":
        model = "accounts/fireworks/models/llama4-scout-instruct-basic"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "deepseek-v3":
        model = "accounts/fireworks/models/deepseek-v3"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "qwen-2.5-72B":
        model = "accounts/fireworks/models/qwen2p5-72b-instruct"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "mixtral-8x22b":
        model = "accounts/fireworks/models/mixtral-8x22b-instruct"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "claude-3.7":
        model = "claude-3-7-sonnet-20250219"
        provider = "anthropic"
        kwargs["temperature"] = 0.0
    if model == "claude-4":
        model = "claude-4-sonnet-20250514"
        provider = "anthropic"
        kwargs["temperature"] = 0.0
    if model == "gemini-2.0-flash":
        provider = "google_genai"
        kwargs["temperature"] = 0.0
    if model == "gemini-2.5-flash":
        provider = "google_genai"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )

    return init_chat_model(model, model_provider=provider, **kwargs)
