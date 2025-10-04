import os
import shutil
import random
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor


def _run_single_process(agent_name: str, seed: int) -> Tuple[int, Dict[str, Any]]:
    # Local imports inside the process
    from agent import config as agent_config
    from agent.sokobanAgent import SokobanAgent

    # Per-run workspace directory
    base_workspace = Path(agent_config.workspace_absolute_path)
    run_workspace = base_workspace / f"{agent_name}_{seed}"
    run_workspace.mkdir(parents=True, exist_ok=True)

    # Override global workspace just for this process
    agent_config.workspace_absolute_path = str(run_workspace)

    # Cache logging
    project_root = Path(__file__).resolve().parents[1]
    cache_dir = project_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_file = cache_dir / "multi_thread_tool_demo_log.txt"

    def append_log(content: str) -> None:
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(content)
                if not content.endswith("\n"):
                    f.write("\n")
        except Exception:
            pass

    try:
        # Build agent
        conf = agent_config.get_sokoban_coding_agent_config()
        agent_cfg = {"agent_config": conf["agent_config"], "env_config": conf["env_config"]}
        agent = SokobanAgent(config=agent_cfg, seed=seed, tag=f"{agent_name}-{seed}")

        # Reset env
        env_out = agent.reset(seed=seed)

        # Build initial messages from agent state (same as test.py)
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
        agent.messages = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": initial_user},
        ]

        # Provider/model and tool schemas
        from serving.api_providers import chat_completion
        import re as _re

        provider = "openai"
        model = "gpt-5"

        tm = getattr(agent, "tool_manager", None)
        tool_schemas = tm.get_schemas() if tm is not None else None

        # Optional .env loader (best-effort)
        try:
            dotenv_path = Path(__file__).resolve().parents[1] / ".env"
            if dotenv_path.exists():
                for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
        except Exception:
            pass

        MAX_TURNS = getattr(agent, "max_turns", 1)
        MAX_STEPS = agent.agent_config.get("max_steps", 10)

        final_env_out = None
        finished = False
        for turn_idx in range(MAX_TURNS):
            step_calls = 0
            while step_calls < MAX_STEPS and not finished:
                step_calls += 1
                append_log(f"[async_group] turn={turn_idx+1}.{step_calls}")

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
                    append_log(f"[async_group] empty LLM response at {turn_idx+1}.{step_calls}")
                    continue

                # Log declared tool calls
                try:
                    tool_names = _re.findall(r"<function\s*=\s*([^>]+)>", llm_response)
                except Exception:
                    tool_names = []
                tool_names_no_finish = [n for n in tool_names if n.lower() not in {"finish", "submit"}]
                for fn in tool_names:
                    append_log(f"[async_group] Tool call {turn_idx+1}.{step_calls}: {fn}")

                before_len = len(agent.get_messages())

                finished, env_out = agent.execute_tool_call(llm_response)
                if finished:
                    final_env_out = env_out
                    append_log("[async_group] Executed final actions in environment.")
                    break

                # Fallback: plain action string
                if llm_response and ("||" in llm_response) and ("<function=" not in llm_response) and ("<answer>" not in llm_response):
                    final_env_out = agent.get_env_outputs(llm_response)
                    append_log(f"[async_group] Executed plain actions at {turn_idx+1}.{step_calls}")
                    finished = True
                    break

                # Collect newly added user feedback
                new_msgs = agent.get_messages()[before_len:]
                new_user_feedback = [m.get("content", "") for m in new_msgs if isinstance(m, dict) and m.get("role") == "user"]
                for idx, fn in enumerate(tool_names_no_finish):
                    if idx < len(new_user_feedback):
                        append_log(f"[async_group] Feedback {turn_idx+1}.{step_calls}: {fn}")
                        append_log(new_user_feedback[idx])

        # Summarize
        metrics = agent.get_final_rollout_states()
        reward_val = getattr(final_env_out, "reward", 0.0) if final_env_out is not None else 0.0
        append_log(f"[async_group] seed={seed} reward={reward_val}")

        return seed, {
            "finished": bool(finished),
            "reward": reward_val,
            "workspace": str(run_workspace),
            "metrics": metrics.get("metrics", {}),
        }
    finally:
        # Cleanup workspace
        try:
            if run_workspace.exists():
                shutil.rmtree(run_workspace)
        except Exception:
            pass


class AsyncAgentGroupBuilder:
    def __init__(self, seeds: List[int], agent_name: str = "sokobanAgent") -> None:
        self.seeds = seeds
        self.agent_name = agent_name

    async def run(self, max_workers: int = 4) -> List[Tuple[int, Dict[str, Any]]]:
        loop = asyncio.get_running_loop()
        results: List[Tuple[int, Dict[str, Any]]] = []
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            tasks = [
                loop.run_in_executor(pool, _run_single_process, self.agent_name, seed)
                for seed in self.seeds
            ]
            done = await asyncio.gather(*tasks, return_exceptions=True)
        for item in done:
            if isinstance(item, Exception):
                # Unknown seed on exception path
                results.append((-1, {"error": str(item)}))
            else:
                results.append(item)
        results.sort(key=lambda x: x[0])
        return results


