import os
from pathlib import Path
from typing import Dict, Any

import yaml
import pytest

from agent.sokobanAgent import SokobanAgent
from serving.api_providers import chat_completion, LLMProviderError


def ensure_cache_dir():
    root = Path(__file__).resolve().parents[2]
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_yaml_config_5_turns() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "agent" / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Pick the first sokobanAgent entry
    picked = None
    if isinstance(data, dict):
        for _, v in data.items():
            if isinstance(v, dict) and v.get("agent_type") == "sokobanAgent":
                picked = v
                break
    if picked is None:
        raise AssertionError("No sokobanAgent config found in agent/config.yaml")

    cfg = {
        "agent_config": picked.get("agent_config", {}).copy(),
        "env_config": picked.get("env_config", {}).copy(),
    }
    # Force 5 turns for this test
    cfg["agent_config"]["max_turns"] = 5
    return cfg


@pytest.fixture
def openai_config() -> Dict[str, Any]:
    return _load_yaml_config_5_turns()


@pytest.mark.integration
def test_sokoban_agent_with_openai_chat_completion_5_turns(openai_config):
    # Skip if no key available
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping OpenAI integration test")

    cache_dir = ensure_cache_dir()

    agent = None
    try:
        agent = SokobanAgent(config=openai_config, tag="openaiTest")
        env_out = agent.reset(seed=123)

        # Run up to 5 turns or until environment signals done
        prompts_history = []
        for turn_idx in range(5):
            messages = agent.get_llm_prompts(env_out)
            # Print and record prompts using repr for clarity
            try:
                print(f"[OpenAI Test] Turn {turn_idx+1} prompts: {repr(messages)}")
            except Exception:
                pass
            prompts_history.append(repr(messages))

            # Use OpenAI via wrapper; keep response brief and deterministic
            try:
                llm_response = chat_completion(
                    messages=messages,
                    provider="openai",
                    model="gpt-5-nano",            # default (gpt-4o-mini)
                    temperature=1.0,
                    max_tokens=128,
                )
            except LLMProviderError as e:
                pytest.skip(f"LLM provider unavailable: {e}")

            env_out = agent.get_env_outputs(llm_response)
            if env_out.truncated or env_out.terminated:
                break

        row = agent.get_final_rollout_states()
        assert isinstance(row, dict)
        assert "metrics" in row and "history" in row
        # Ensure we captured at least one raw response (printed inside agent as well)
        assert getattr(agent, "raw_response_list", [])

        # Log to cache directory as text (like sokobanAgent_test)
        out_file = cache_dir / "agent_api_serving_openai_test_log.txt"
        with out_file.open("w", encoding="utf-8") as f:
            f.write("OpenAI Integration Test Log\n")
            f.write("============================\n")
            f.write("metrics:\n")
            f.write(repr(row.get("metrics")))
            f.write("\n---\n")
            f.write("history:\n")
            f.write(repr(row.get("history")))
            f.write("\n---\n")
            f.write("prompts_history:\n")
            f.write(repr(prompts_history))

        assert out_file.exists() and out_file.stat().st_size > 0
    finally:
        try:
            if agent is not None:
                agent.close()
        except Exception:
            pass
