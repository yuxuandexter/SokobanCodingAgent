import os
from pathlib import Path
from typing import Any, Dict

import pytest

from agent.sokobanAgent import SokobanAgent
from tools import build_default_tool_manager
from serving.api_providers import chat_completion


@pytest.fixture
def basic_config() -> Dict[str, Any]:
    return {
        "agent_config": {
            "max_turns": 3,
            "max_actions_all_turns": 10,
            "max_actions_per_turn": 5,
            "max_tokens": 64,
            "format_penalty": -0.1,
            "enable_think": False,
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


def test_iterate_function_calls_create_and_run(basic_config, tmp_path: Path):
    agent = SokobanAgent(config=basic_config, tag="toolFlowTest")
    agent.reset(seed=123)

    # Prepare a small script file path
    script_path = tmp_path / "hello_tool_flow.py"
    script_text = "print('hello from tool flow')\n"

    # Function-call response: create file, then run it
    llm_response = (
        f"<function=file_editor>\n"
        f"<parameter=command>create</parameter>\n"
        f"<parameter=path>{str(script_path)}</parameter>\n"
        f"<parameter=file_text>{script_text}</parameter>\n"
        f"</function>\n"
        f"<function=execute_bash>\n"
        f"<parameter=cmd>python3 {str(script_path)}</parameter>\n"
        f"</function>\n"
    )

    finished, env_out = agent.iterate_function_calls(llm_response)
    assert finished is False
    assert env_out is None
    assert script_path.exists()

    # Verify messages contain tool feedback
    msgs = agent.get_messages()
    user_chunks = [m["content"] for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    joined = "\n".join(user_chunks)
    assert "Execution output of [file_editor]" in joined
    assert "Execution output of [execute_bash]" in joined


def test_iterate_function_calls_with_finish_executes_actions(basic_config):
    agent = SokobanAgent(config=basic_config, tag="toolFlowFinish")
    env_out = agent.reset(seed=123)
    assert env_out is not None

    llm_response = (
        "<function=finish>\n"
        "<parameter=command>submit</parameter>\n"
        "<parameter=result>Right || Up</parameter>\n"
        "</function>\n"
    )
    finished, env_out2 = agent.iterate_function_calls(llm_response)
    assert finished is True
    assert env_out2 is not None
    assert hasattr(env_out2, "state")


def test_get_env_outputs_with_plain_action_string(basic_config):
    agent = SokobanAgent(config=basic_config, tag="plainActions")
    env_out = agent.reset(seed=123)
    assert env_out is not None

    out2 = agent.get_env_outputs("Right || Up || Left")
    assert out2 is not None and hasattr(out2, "state")


def test_tool_manager_direct_file_editor_and_execute_bash(tmp_path: Path):
    tm = build_default_tool_manager()

    script = tmp_path / "direct_tool_test.py"
    code = "print('direct tool ok')\n"

    # Create file
    r1 = tm.execute("file_editor", {"command": "create", "path": str(script), "file_text": code})
    assert r1.get("exit_code") == "0"
    assert script.exists()

    # Run file
    r2 = tm.execute("execute_bash", {"cmd": f"python3 {script}"})
    assert r2.get("exit_code") in {"0", 0}
    assert "direct tool ok" in r2.get("output", "")


def test_chat_completion_wrapper_monkeypatched(monkeypatch):
    # Monkeypatch LiteLLM completion to avoid external calls
    class Dummy:
        pass

    def fake_completion(**kwargs):
        return {"choices": [{"message": {"content": "ok"}}]}

    from serving import api_providers as ap
    monkeypatch.setattr(ap, "completion", fake_completion, raising=True)

    resp = chat_completion(messages=[{"role": "user", "content": "hi"}], provider="openai", model="gpt-test")
    assert resp == "ok"


