"""
Central configuration for the Electricity Bill Copilot.
All secrets are loaded from environment variables (.env) — never hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_FALLBACK_CHAIN = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
]


def _parse_fallback_chain() -> tuple[str, ...]:
    raw = os.getenv("GROQ_MODEL_FALLBACK_CHAIN", "")
    primary = os.getenv("GROQ_MODEL", "").strip()
    chain = [m.strip() for m in raw.split(",") if m.strip()] if raw.strip() else list(_DEFAULT_FALLBACK_CHAIN)
    if primary and primary not in chain:
        chain.insert(0, primary)
    return tuple(dict.fromkeys(chain))


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_model_fallback_chain: tuple[str, ...] = field(default_factory=_parse_fallback_chain)
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    app_title: str = "AI-Powered Electricity Bill Calculator & Comparison Chatbot"


settings = Settings()


def validate_config() -> list[str]:
    problems = []
    if not settings.groq_api_key:
        problems.append(
            "GROQ_API_KEY is not set. Create a `.env` file (see `.env.example`) "
            "and add your Groq API key, or set it as an environment variable."
        )
    return problems
