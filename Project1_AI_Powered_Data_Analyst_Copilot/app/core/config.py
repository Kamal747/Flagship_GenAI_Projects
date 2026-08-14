"""
Central configuration for the AI Data Analyst Copilot.
All secrets are loaded from environment variables (.env) — never hard-coded.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# Default ordered fallback chain: tried in order until one succeeds or all
# are exhausted. Override via GROQ_MODEL_FALLBACK_CHAIN (comma-separated) in .env.
_DEFAULT_FALLBACK_CHAIN = [
    "openai/gpt-oss-120b",
]


def _parse_fallback_chain() -> tuple[str, ...]:
    raw = os.getenv("GROQ_MODEL_FALLBACK_CHAIN", "")
    primary = os.getenv("GROQ_MODEL", "").strip()
    if raw.strip():
        chain = [m.strip() for m in raw.split(",") if m.strip()]
    else:
        chain = list(_DEFAULT_FALLBACK_CHAIN)
    # If GROQ_MODEL is explicitly set and not already in the chain, try it first.
    if primary and primary not in chain:
        chain.insert(0, primary)
    return tuple(dict.fromkeys(chain))  # de-dupe, preserve order


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_model_fallback_chain: tuple[str, ...] = field(default_factory=_parse_fallback_chain)
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "200"))
    sandbox_timeout_seconds: int = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "5"))
    max_rows_preview: int = int(os.getenv("MAX_ROWS_PREVIEW", "50"))
    large_dataset_row_threshold: int = int(os.getenv("LARGE_DATASET_ROW_THRESHOLD", "200000"))
    app_title: str = "AI-Powered Data Analyst Copilot"
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))


settings = Settings()


def validate_config() -> list[str]:
    """Returns a list of human-readable configuration problems, empty if OK."""
    problems = []
    if not settings.groq_api_key:
        problems.append(
            "GROQ_API_KEY is not set. Create a `.env` file (see `.env.example`) "
            "and add your Groq API key, or set it as an environment variable."
        )
    return problems

