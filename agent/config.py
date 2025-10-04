
workspace_absolute_path = "/workspace/SokobanCodingAgent/workspace"

# Inline configuration for sokobanCodingAgent_6_6_dim_1_box
# Replaces YAML-based config and pulls prompts from agent.prompts
from agent.prompts import system_prompt as sokoban_system_prompt, prompt as sokoban_user_prompt


def get_sokoban_coding_agent_config():
    """
    Return the configuration dict equivalent to `sokobanCodingAgent_6_6_dim_1_box` in config.yaml,
    but with system_prompt and prompt replaced by the ones in agent.prompts.
    """
    return {
        "agent_type": "sokobanAgent",
        "agent_config": {
            "system_prompt": sokoban_system_prompt,
            "prompt": sokoban_user_prompt,
            "enable_think": False,
            "max_tokens": 100,
            "max_turns": 6,
            "max_actions_per_turn": 100,
            "max_actions_all_turns": 100,
            "format_penalty": -0.1,
            "action_separator": "||",
        },
        "env_config": {
            "dim_room": [10, 10],
            "num_boxes": 5,
            "max_steps": 100,
            "search_depth": 100,
            "grid_lookup": {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"},
            "grid_vocab": {"#": "wall", "_": "empty", "O": "target", "√": "box on target", "X": "box", "P": "player", "S": "player on target"},
            "action_lookup": {1: "Up", 2: "Down", 3: "Left", 4: "Right"},
            "render_mode": "text",
        },
    }


