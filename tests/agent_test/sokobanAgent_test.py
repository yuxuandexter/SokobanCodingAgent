import os
from pathlib import Path

import pytest

from agent.sokobanAgent import SokobanAgent


def ensure_cache_dir():
    root = Path(__file__).resolve().parents[2]
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def basic_config():
    return {
        "agent_config": {
            "max_turns": 1,
            "max_actions_all_turns": 2,
            "max_actions_per_turn": 2,
            "max_tokens": 64,
            "format_penalty": -0.1,
            "enable_think": True,
            "system_prompt": "You are a helpful AI assistant.",
            "prompt": "Solve sokoban.",
            "action_separator": "||",
            "use_think_answer_token": False,
        },
        "env_config": {
            "grid_lookup": {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"},
            "action_lookup": {1: "Up", 2: "Down", 3: "Left", 4: "Right"},
            "render_mode": "text",
            "dim_room": (6, 6),
            "max_steps": 30,
            "num_boxes": 1,
            "search_depth": 50,
        },
    }


def test_sokoban_agent_rollout_and_logging(basic_config, tmp_path):
    cache_dir = ensure_cache_dir()

    agent = SokobanAgent(config=basic_config, tag="testAgent")

    # Reset agent and get initial prompts
    env_out = agent.reset(seed=123)
    prompts = agent.get_llm_prompts(env_out)
    assert isinstance(prompts, list) and len(prompts) >= 2

    # Fake a minimal model response with 1-2 actions
    llm_response = "<think>ok</think><answer>Up||Right</answer>"

    env_out2 = agent.get_env_outputs(llm_response)
    assert env_out2.state
    assert isinstance(env_out2.reward, float)
    assert isinstance(env_out2.info, dict)

    row = agent.get_final_rollout_states()
    assert "metrics" in row and "history" in row

    # Log to cache directory as text
    out_file = cache_dir / "sokoban_agent_test_log.txt"
    with out_file.open("w", encoding="utf-8") as f:
        f.write("SokobanAgent Test Log\n")
        f.write("======================\n")
        f.write("metrics:\n")
        f.write(str(row.get("metrics")))
        f.write("\n---\n")
        f.write("history:\n")
        f.write(str(row.get("history")))

    assert out_file.exists() and out_file.stat().st_size > 0