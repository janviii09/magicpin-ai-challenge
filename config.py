"""
Vera 2.0 — Configuration
Env vars → typed config. No business logic here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Provider ─────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")        # openai | anthropic | gemini | deepseek
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "")                    # blank → provider default
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "25"))         # seconds; judge timeout is 30s

# ── Server ────────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8080"))

# ── Team metadata ─────────────────────────────────────────────
TEAM_NAME: str = os.getenv("TEAM_NAME", "Vera 2.0")
TEAM_MEMBERS: list[str] = os.getenv("TEAM_MEMBERS", "Janvi").split(",")
CONTACT_EMAIL: str = os.getenv("CONTACT_EMAIL", "")
VERSION: str = "1.0.0"

# ── Composition tuning ────────────────────────────────────────
MAX_COMPOSE_RETRIES: int = 2                                    # LLM retries before fallback
MAX_BODY_LENGTH: int = 600                                      # chars; WhatsApp readability
MIN_BODY_LENGTH: int = 30
MAX_ACTIONS_PER_TICK: int = 3                                   # don't spam

# ── Dataset location ─────────────────────────────────────────
DATASET_DIR: str = os.getenv("DATASET_DIR", "dataset")

# ── Provider defaults (model name if LLM_MODEL is blank) ─────
PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.5-flash",
    "deepseek": "deepseek-chat",
}

def get_model() -> str:
    return LLM_MODEL or PROVIDER_DEFAULTS.get(LLM_PROVIDER, "gpt-4o")
