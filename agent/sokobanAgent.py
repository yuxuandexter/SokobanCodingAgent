# ─────────────────── IMPORTS ───────────────────
import random
import yaml
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from agent.agent_utils import SingleTurnTrajectory, MultiTurnTrajectory, EnvOutput, debug_printout_in_env_output
from agent.base_agent import BaseAgent
from env.sokobanEnv import SokobanEnv
from agent import register_agent
import re
import importlib
from pathlib import Path

# ─────────────────── SOKOBAN AGENT ───────────────────
@register_agent("sokobanAgent")
class SokobanAgent(BaseAgent):
    """
    Sokoban agent that manages environment interactions and conversation history.
    Compatible with SyncMultiTurnRollout interface.
    """
    
    def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
        super().__init__(config, group_id, agent_id, seed, tag)
        self.prompt = self._build_enhanced_prompt(self.prompt)
        self.initialize_env()

    def _build_enhanced_prompt(self, base_prompt):
        """Build enhanced prompt with environment info and emphatic format instructions."""
        enhanced_prompt = base_prompt
        
        if self.env_config.get("grid_vocab"):
            symbols = [f"{k}: {v}" for k, v in self.env_config["grid_vocab"].items()]
            grid_vocab = f"\nThe meaning of each symbol in the state is:\n {', '.join(symbols)}"
            enhanced_prompt += grid_vocab
        
        if self.env_config.get("action_lookup"):
            actions = list(self.env_config["action_lookup"].values())
            action_lookup_str = "\nYour available actions are:\n" + ", ".join(actions)
            enhanced_prompt += action_lookup_str

        enhanced_prompt += f"\nYou can make up to {self.max_actions_all_turns} actions, and each action is separated by '{self.action_separator}'."
        return enhanced_prompt

    def initialize_env(self):
        """Initialize the Sokoban environment."""
        self.env = SokobanEnv(self.env_config)
        # Initialize local tool manager for function-call protocol
        self.tool_manager = None
        # First, try normal import by module name
        try:
            mod = importlib.import_module("tools")
            if hasattr(mod, "build_default_tool_manager"):
                self.tool_manager = mod.build_default_tool_manager()
                self._cache_log("[init] Tool manager initialized via normal import")
                return
        except Exception:
            pass
        # Fallback: import by absolute path to repo-root/tools/__init__.py
        try:
            repo_root = Path(__file__).resolve().parents[1]
            tools_init = repo_root / "tools" / "__init__.py"
            if tools_init.exists():
                spec = importlib.util.spec_from_file_location("tools", str(tools_init))  # type: ignore[attr-defined]
                if spec and spec.loader:  # type: ignore[truthy-bool]
                    mod = importlib.util.module_from_spec(spec)  # type: ignore[attr-defined]
                    spec.loader.exec_module(mod)  # type: ignore[union-attr]
                    if hasattr(mod, "build_default_tool_manager"):
                        self.tool_manager = mod.build_default_tool_manager()
                        self._cache_log("[init] Tool manager initialized via path import")
        except Exception:
            self.tool_manager = None
            self._cache_log("[init] Tool manager unavailable")

    def _cache_log(self, text: str) -> None:
        """Append a log line to cache/agent_execution.log (best-effort)."""
        try:
            root = Path(__file__).resolve().parents[1]
            cache_dir = root / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            log_file = cache_dir / "agent_execution.log"
            with log_file.open("a", encoding="utf-8") as f:
                f.write(text.rstrip("\n") + "\n")
        except Exception:
            pass

    # ─────────────────── TOOL-CALL PROTOCOL HELPERS ───────────────────
    def _parse_function_blocks(self, text: str) -> List[str]:
        """Extract all <function=...>...</function> blocks from text in order."""
        if not isinstance(text, str):
            return []
        pattern = re.compile(r"<function\s*=\s*[^>]+>.*?</function>", re.DOTALL)
        return pattern.findall(text)

    def _parse_function_call(self, block: str) -> Tuple[str, Dict[str, str]]:
        """Parse a single <function=...>...</function> block into (name, params)."""
        try:
            fn_match = re.search(r"<function\s*=\s*([^>]+)>", block)
            function_name = fn_match.group(1).strip() if fn_match else ""
            params: Dict[str, str] = {}
            for key, val in re.findall(r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>", block, flags=re.DOTALL):
                params[key.strip()] = val.strip()
            return function_name, params
        except Exception:
            return "", {}

    def _format_tool_observation(self, function_name: str, tool_out: Dict[str, Any]) -> str:
        """Format tool output into a compact observation string for the next turn."""
        output = str(tool_out.get("output", ""))
        exit_code = str(tool_out.get("exit_code", ""))
        if function_name in {"execute_bash", "bash"}:
            return f"Exit code: {exit_code}\nExecution output of [{function_name}]:\n{output}"
        return f"Execution output of [{function_name}]:\n{output}"

    def _parse_actions_from_result(self, result_text: str) -> List[str]:
        """Extract the first line (before optional ---) and split into actions by configured separator."""
        if not isinstance(result_text, str):
            return []
        first_part = result_text.split("\n", 1)[0]
        first_part = first_part.split("---", 1)[0].strip()
        if not first_part:
            return []
        sep = self.action_separator
        return [a.strip() for a in first_part.split(sep) if a.strip()]

    # ─────────────────── SIMPLE ACTION PARSER OVERRIDE ───────────────────
    def parse_llm_response(self, llm_response, enable_think: bool = False):
        """Override: parse a simple, uniform action sequence.
        Supports three forms:
        1) Pure string: "Right || Up || Left"
        2) <answer>Right || Up</answer>
        3) <function=finish>... <parameter=result>Right || Up</parameter> </function>

        Returns (processed_llm_response, actions_list) where processed_llm_response is
        normalized to "<answer>Right || Up || Left</answer>" and actions_list is a
        list of action names.
        """
        text = str(llm_response) if not isinstance(llm_response, str) else llm_response

        # Prefer finish result if present
        blocks = self._parse_function_blocks(text)
        if blocks:
            for block in blocks:
                fn_name, params = self._parse_function_call(block)
                if fn_name.lower() in {"finish", "submit"}:
                    action_line = params.get("result", "").split("\n", 1)[0].split("---", 1)[0].strip()
                    action_content = action_line
                    break
            else:
                action_content = text
        else:
            # Try <answer>...</answer>
            import re as _re
            m = _re.search(r"<answer>(.*?)</answer>", text, flags=_re.DOTALL)
            if m:
                action_content = m.group(1).strip()
            else:
                action_content = text.strip()

        # Normalize separators around '||'
        # Replace variations like ' | | ' or multiple spaces with ' || '
        action_content = action_content.replace("| |", "||")
        action_content = action_content.replace("|||", "||")
        # Ensure we split on the configured separator token
        parts = [p.strip() for p in action_content.split(self.action_separator) if p.strip()]

        # Limit to per-turn maximum
        if len(parts) > self.max_actions_per_turn:
            parts = parts[: self.max_actions_per_turn]

        normalized = " || ".join(parts)
        processed = f"<answer>{normalized}</answer>"
        return processed, parts


    def get_env_outputs(self, llm_response: Union[str, List[str]]):
        """Process LLM outputs and get environment outputs from a pure action string.
        Accepts an action sequence (e.g., "Right || Up || Left"), parses it,
        executes the actions in the Sokoban env, and records the trajectory.
        """
        llm_raw_response = llm_response

        # Store raw response for debugging
        self.raw_response_list.append(llm_raw_response)

        self.cur_turn += 1

        # Parse actions (supports either plain action string or <answer> format)
        processed_llm_response, actions = self.parse_llm_response(
            str(llm_raw_response), enable_think=self.enable_think
        )

        # Log assistant content
        self.messages.append({"role": "assistant", "content": processed_llm_response})
        self._cache_log(f"[get_env_outputs] received: {processed_llm_response}")

        obs = self.env.render()
        total_reward = 0.0
        done = False
        executed_actions: List[int] = []
        info: Dict[str, Any] = {}

        action_lookup_reverse = {v: k for k, v in self.env_config['action_lookup'].items()}
        action_lookup_reverse_lower = {v.lower(): k for k, v in self.env_config['action_lookup'].items()}

        valid_actions: List[int] = []
        invalid_actions: List[str] = []

        for action_str in actions:
            try:
                action_str_clean = action_str.strip()
                if action_str_clean in action_lookup_reverse:
                    action = action_lookup_reverse[action_str_clean]
                    if action in self.env_config['action_lookup']:
                        valid_actions.append(action)
                    else:
                        invalid_actions.append(action_str)
                elif action_str_clean.lower() in action_lookup_reverse_lower:
                    action = action_lookup_reverse_lower[action_str_clean.lower()]
                    if action in self.env_config['action_lookup']:
                        valid_actions.append(action)
                    else:
                        invalid_actions.append(action_str)
                else:
                    action = int(action_str_clean)
                    if action in self.env_config['action_lookup']:
                        valid_actions.append(action)
                    else:
                        invalid_actions.append(action_str)
            except (ValueError, KeyError, TypeError):
                invalid_actions.append(action_str)
                continue

        # Apply penalty for invalid actions or empty
        if len(actions) == 0 or invalid_actions or len(valid_actions) != len(actions):
            self.penalty += self.format_penalty

        # Execute valid actions
        for a in valid_actions:
            try:
                obs, reward, done, step_info = self.env.step(a)
                total_reward += reward
                executed_actions.append(a)
                info.update(step_info)
                if done:
                    break
            except Exception as e:
                print(f"Warning: Agent {self.agent_id} step failed for action {a}: {e}")
                continue
        self._cache_log(f"[get_env_outputs] actions_parsed={actions} executed={executed_actions} reward={total_reward} done={done}")

        # Update counters
        self.total_actions_consumed += len(executed_actions)
        actions_left = max(0, self.max_actions_all_turns - self.total_actions_consumed)

        # Done conditions
        if self.cur_turn >= self.max_turns or self.total_actions_consumed >= self.max_actions_all_turns:
            done = True

        # Record trajectory
        self.trajectory_history.add(SingleTurnTrajectory(
            state=obs,
            actions_left=actions_left,
            actions=executed_actions,
            reward=total_reward,
            info=info,
            llm_response=processed_llm_response,
            llm_raw_response=str(llm_raw_response)
        ))

        return EnvOutput(
            truncated=done,
            terminated=done,
            state=obs,
            reward=total_reward,
            info=info
        )

    def iterate_function_calls(self, llm_response: str, max_tool_calls: int = 20) -> Tuple[bool, Optional[EnvOutput]]:
        """Execute non-finish tool calls in an LLM response. If a finish call is present,
        execute final actions via get_env_outputs and return (True, EnvOutput).
        Otherwise execute tools, append feedback to messages, and return (False, None).
        """
        function_blocks = self._parse_function_blocks(llm_response)
        if not function_blocks:
            # Nothing to do
            return False, None

        calls = 0
        for block in function_blocks:
            if calls >= max_tool_calls:
                break
            calls += 1
            fn_name, params = self._parse_function_call(block)
            if not fn_name:
                continue

            # Log assistant call
            self.messages.append({"role": "assistant", "content": block})
            self._cache_log(f"[tool] call {fn_name} with params_keys={list(params.keys())}")

            if fn_name.lower() in {"finish", "submit"}:
                # Extract the action sequence from result and execute via get_env_outputs
                result_text = params.get("result", "")
                action_line = result_text.split("\n", 1)[0].split("---", 1)[0].strip()
                self._cache_log(f"[tool] finish with actions='{action_line}'")
                env_out = self.get_env_outputs(action_line)
                return True, env_out

            # Execute regular tool
            tool_out: Dict[str, Any] = {"output": "", "exit_code": "0"}
            try:
                if getattr(self, "tool_manager", None) is not None:
                    tool_out = self.tool_manager.execute(fn_name, params)
                else:
                    tool_out = {"output": f"Tool manager unavailable for {fn_name}.", "exit_code": "-1"}
            except Exception as e:
                tool_out = {"output": f"Error executing tool {fn_name}: {e}", "exit_code": "-1"}

            feedback = self._format_tool_observation(fn_name, tool_out)
            self.messages.append({"role": "user", "content": feedback})
            self._cache_log(f"[tool] feedback {fn_name} exit={tool_out.get('exit_code')} bytes={len(str(tool_out.get('output','')))}")

        # If we iterated tools but did not finish, mark a small penalty
        self.penalty += self.format_penalty
        return False, None