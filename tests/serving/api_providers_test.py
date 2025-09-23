import os
from pathlib import Path

import pytest

from serving.api_providers import chat_completion


def ensure_cache_dir():
    root = Path(__file__).resolve().parents[2]
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def test_chat_completion_openai_live_call(monkeypatch):
    """Live API call via uniform chat_completion (OpenAI; skips without key)."""
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY not set; run `source env_setup.sh <OPENAI> [GEMINI]` and retry")

    try:
        import litellm  # type: ignore
        monkeypatch.setitem(chat_completion.__globals__, "completion", litellm.completion)
    except Exception:
        pytest.skip("litellm not available")

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    out = chat_completion(
        messages=[{"role": "user", "content": "Say hi in one word."}],
        provider="openai",
        model=model,
        temperature=0.0,
        max_tokens=5,
    )
    assert isinstance(out, str) and len(out.strip()) > 0

    cache_dir = ensure_cache_dir()
    out_file = cache_dir / "api_providers_openai_live_test_log.txt"
    with out_file.open("w", encoding="utf-8") as f:
        f.write("Provider: openai\n")
        f.write(f"Model: {model}\n")
        f.write("Response:\n")
        f.write(out)
    assert out_file.exists() and out_file.stat().st_size > 0


def test_chat_completion_gemini_live_call(monkeypatch):
    """Live API call via uniform chat_completion (Gemini; skips without key)."""
    key_env = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key_env:
        pytest.skip("GEMINI_API_KEY/GOOGLE_API_KEY not set; run `source env_setup.sh <OPENAI> <GEMINI>` and retry")

    try:
        import litellm  # type: ignore
        monkeypatch.setitem(chat_completion.__globals__, "completion", litellm.completion)
    except Exception:
        pytest.skip("litellm not available")

    model = os.environ.get("GEMINI_MODEL", "gemini/gemini-2.0-flash")
    try:
        out = chat_completion(
            messages=[{"role": "user", "content": "Say hi in one word."}],
            provider="gemini",
            model=model,
            temperature=0.0,
            max_tokens=5,
            api_key=key_env,
        )
    except Exception as e:
        # Skip on invalid key / ADC fallback errors so suite remains green without proper setup
        error_str = str(e)
        if "API key not valid" in error_str or "Default credentials were not found" in error_str:
            pytest.skip("Gemini not configured: invalid API key or missing ADC")
        raise
    assert isinstance(out, str) and len(out.strip()) > 0

    cache_dir = ensure_cache_dir()
    out_file = cache_dir / "api_providers_gemini_live_test_log.txt"
    with out_file.open("w", encoding="utf-8") as f:
        f.write("Provider: gemini\n")
        f.write(f"Model: {model}\n")
        f.write("Response:\n")
        f.write(out)
    assert out_file.exists() and out_file.stat().st_size > 0


