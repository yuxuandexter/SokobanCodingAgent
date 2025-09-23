import json
from pathlib import Path

import pytest

from env.sokobanEnv import SokobanEnv


def ensure_cache_dir():
    root = Path(__file__).resolve().parents[2]
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def env_config():
    return {
        "grid_lookup": {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"},
        "action_lookup": {1: "Up", 2: "Down", 3: "Left", 4: "Right"},
        "render_mode": "text",
        "dim_room": (6, 6),
        "max_steps": 20,
        "num_boxes": 1,
        "search_depth": 50,
    }


def test_sokoban_env_step_and_logging(env_config):
    cache_dir = ensure_cache_dir()
    env = SokobanEnv(env_config)
    obs = env.reset(seed=42)
    assert isinstance(obs, str) and len(obs) > 0

    # Try a couple of actions (1..4)
    for action in [1, 2, 3, 4]:
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, str)
        assert isinstance(reward, (int, float))
        assert isinstance(info, dict)
        if done:
            break

    # Log minimal info
    log = {"last_obs": obs, "done": done, "info": info}
    out_file = cache_dir / "sokoban_env_test_log.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(log, f)

    assert out_file.exists() and out_file.stat().st_size > 0
