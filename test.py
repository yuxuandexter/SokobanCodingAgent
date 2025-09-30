import os
from pathlib import Path
import re
from pprint import pprint

from agent.config import get_sokoban_coding_agent_config, workspace_absolute_path as WORKSPACE_ROOT

# Cache directory and single log file for streaming logs
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
os.makedirs(str(CACHE_DIR), exist_ok=True)
LOG_FILE = str(CACHE_DIR / "tool_demo_log.txt")
# Truncate previous run
try:
    with open(LOG_FILE, "w", encoding="utf-8") as _f:
        _f.write("")
except Exception:
    pass

from serving.api_providers import chat_completion
from agent.sokobanAgent import SokobanAgent


def _user_messages_repr(messages):
    try:
        users = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        return repr(users)
    except Exception:
        return repr(messages)


def print_messages(label: str, messages):
    # Intentionally minimized: no-op for full prompts; we only print new tool feedback below
    pass

def append_log(content: str):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")
    except Exception:
        pass

def _load_env_from_dotenv():
    """Best-effort load of .env at repo root so OPENAI_API_KEY is available."""
    dotenv_path = PROJECT_ROOT / ".env"
    if not dotenv_path.exists():
        return
    try:
        with open(str(dotenv_path), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)
    except Exception:
        pass


def main():
    # Build config from config.py (prompts come from agent.prompts)
    conf = get_sokoban_coding_agent_config()
    agent_cfg = {
        "agent_config": conf["agent_config"],
        "env_config": conf["env_config"],
    }

    # Ensure the configured workspace directory exists for tool executions
    try:
        os.makedirs(WORKSPACE_ROOT, exist_ok=True)
    except Exception:
        pass

    agent = SokobanAgent(config=agent_cfg, tag="toolCallingDemo-script")

    # Reset environment; one env execution at the end after finish
    env_out = agent.reset(seed=123)

    # Build initial messages from agent state
    symbols = agent.env_config.get("grid_vocab", {})
    symbols_txt = ", ".join([f"{k}: {v}" for k, v in symbols.items()]) if symbols else ""
    actions_txt = ", ".join(agent.env_config.get("action_lookup", {}).values())
    initial_user = (
        f"{agent.prompt}\n\n"
        f"Initial Sokoban state:\n{env_out.state}\n\n"
        f"The meaning of each symbol is: {symbols_txt}\n"
        f"Your available actions are: {actions_txt}\n"
        f"Separator: '{agent.action_separator}'\n"
        f"Max actions total: {agent.max_actions_all_turns}\n"
    )
    messages = [
        {"role": "system", "content": agent.system_prompt},
        {"role": "user", "content": initial_user},
    ]
    agent.messages = messages[:]

    # Print succinct initial game state and config helpers
    print("\n=== Initial Game State ===")
    print(env_out.state)
    print("Symbols:", symbols_txt)
    print("Actions:", actions_txt)
    print("Separator:", agent.action_separator)
    print("Max actions total:", agent.max_actions_all_turns)
    print("Workspace root:", WORKSPACE_ROOT)

    # Do not print or log initial prompts; we only log new feedback per step

    MAX_ITER = 10
    provider = "openai"  # or "gemini"
    model = "gpt-5"     # explicitly use GPT-4o for OpenAI
    print(f"Model provider={provider} model={model or '(default)'}")
    append_log(f"=== Model provider={provider} model={model or '(default)'} ===")

    # Ensure API keys are available (use .env if present)
    _load_env_from_dotenv()
    if not os.getenv("OPENAI_API_KEY") and provider == "openai":
        msg = (
            "OPENAI_API_KEY is not set. Please add it to .env at repo root or run: "
            "source env_setup.sh <OPENAI_API_KEY>"
        )
        print(msg)
        append_log(msg)
        return

    # Prepare tool schemas for better function-calling adherence (if available)
    tm = getattr(agent, "tool_manager", None)
    tool_schemas = tm.get_schemas() if tm is not None else None

    final_env_out = None
    for step in range(MAX_ITER):
        print(f"\n=== Tool Step {step+1} ===")
        # Skip printing/logging full prompts; keep logs focused on feedback

        # Query the model
        extra_args = {"tools": tool_schemas} if tool_schemas else None
        llm_response_raw = chat_completion(
            messages=agent.messages,
            provider=provider,
            model=model,
            temperature=1,
            extra_args=extra_args,
        )
        if isinstance(llm_response_raw, str):
            llm_response = llm_response_raw
        elif llm_response_raw is None:
            llm_response = ""
        else:
            llm_response = str(llm_response_raw)

        if not llm_response:
            print("\nLLM returned empty response; continuing.")
            append_log(f"=== LLM returned empty response at step {step+1} ===")

        # Parse declared tool calls (names and parameter keys) for logging
        tool_names: list[str] = []
        try:
            tool_names = re.findall(r"<function\s*=\s*([^>]+)>", llm_response)
        except Exception:
            tool_names = []
        # Exclude finish/submit from the tool feedback mapping
        tool_names_no_finish = [n for n in tool_names if n.lower() not in {"finish", "submit"}]
        for fn in tool_names:
            append_log(f"=== Tool call step {step+1}: {fn} ===")
            print(f"[ToolCall] step {step+1}: {fn}")

        # Remember message length to capture newly added feedback after tool execution
        before_len = len(agent.get_messages())

        # Execute tool calls iteratively; if finish encountered it will run env
        finished, env_out = agent.iterate_function_calls(llm_response)
        if finished:
            final_env_out = env_out
            print("\nExecuted final actions in environment.")
            append_log(f"=== User messages before finish step {step+1} ===")
            append_log(_user_messages_repr(agent.get_messages()))
            if final_env_out is not None:
                append_log("=== Final observation ===")
                append_log(str(final_env_out.state))
            break

        # If the model replied with a plain action string, execute directly
        if llm_response and ("||" in llm_response) and ("<function=" not in llm_response) and ("<answer>" not in llm_response):
            final_env_out = agent.get_env_outputs(llm_response)
            print("\nExecuted actions from plain string.")
            append_log(f"=== Executed plain actions at step {step+1} ===")
            append_log(llm_response)
            break

        # Collect newly added user feedback messages and map to tool calls (best-effort by order)
        new_msgs = agent.get_messages()[before_len:]
        new_user_feedback = [m.get("content", "") for m in new_msgs if isinstance(m, dict) and m.get("role") == "user"]
        for idx, fn in enumerate(tool_names_no_finish):
            if idx < len(new_user_feedback):
                append_log(f"=== Feedback step {step+1}: {fn} ===")
                append_log(new_user_feedback[idx])

        # Print and log tool feedback (no full prompt history); fall back if name mapping unavailable
        if new_user_feedback:
            for idx, fb in enumerate(new_user_feedback):
                name = tool_names_no_finish[idx] if idx < len(tool_names_no_finish) else "?"
                print(f"\n[ToolFeedback] step {step+1} tool={name}\n{fb}")
                append_log(f"=== Feedback step {step+1}: {name} ===")
                append_log(fb)

        # Do not append any reminder; keep loop output minimal and focused on tools

    if final_env_out is None:
        print("\nNo finish received within iteration cap.")
    else:
        print("\nFinal Observation:\n", final_env_out.state)
        print("Reward:", final_env_out.reward)
        print("Info:", final_env_out.info)
        append_log("=== Final observation ===")
        append_log(str(final_env_out.state))

    # Show brief rollout metrics
    row = agent.get_final_rollout_states()
    print("\nMetrics:")
    pprint(row.get("metrics", {}))
    print("\nHistory length:", len(row.get("history", [])))


if __name__ == "__main__":
    main()

