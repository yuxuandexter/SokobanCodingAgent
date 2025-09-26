from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ------------------------------
# Core tool abstractions
# ------------------------------


class tool:
    """
    SkyRL-style descriptor for registering methods as tools.
    Extended to optionally carry a JSON schema and description for LLM tool-calling.
    """

    def __init__(self, func: Optional[Callable] = None, *, schema: Optional[dict] = None, description: Optional[str] = None):
        self.func = func
        self.name = func.__name__ if func else None
        self.schema = schema or {}
        self.description = description or ""

    def __call__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        return self

    def __get__(self, instance, owner):
        if instance is None:
            return self
        # Bind to instance to pass self automatically when called
        return lambda *args, **kwargs: self.func(instance, *args, **kwargs)


class ToolGroup:
    """
    Groups related tools and auto-registers methods decorated with @tool.
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Callable[..., Any]] = {}
        self._schemas: Dict[str, dict] = {}
        self._register_tools()

    def get_name(self) -> str:
        return self.name

    def _register_tools(self) -> None:
        for attr in dir(self):
            raw = getattr(type(self), attr, None)
            if isinstance(raw, tool) and raw.func is not None:
                if raw.name in self._registry:
                    raise ValueError(f"Duplicate tool name detected: {raw.name}")
                self._registry[raw.name] = getattr(self, attr)
                # Build OpenAI function-calling compatible schema object
                schema_obj = {
                    "type": "function",
                    "function": {
                        "name": raw.name,
                        "description": raw.description or "",
                        "parameters": raw.schema or {"type": "object", "properties": {}, "required": []},
                    },
                }
                self._schemas[raw.name] = schema_obj

    def get_tool(self, name: str) -> Optional[Callable[..., Any]]:
        return self._registry.get(name)

    def get_tool_names(self) -> List[str]:
        return list(self._registry.keys())

    def get_schemas(self) -> List[dict]:
        return [self._schemas[name] for name in self.get_tool_names()]

    def get_tool_to_group_mapping(self) -> Dict[str, str]:
        return {name: self.name for name in self._registry}

    def execute(self, name: str, *args, **kwargs) -> Any:
        func = self.get_tool(name)
        if not func:
            raise ValueError(f"Tool '{name}' not found in group '{self.name}'.")
        return func(*args, **kwargs)


class ToolManager:
    """
    Aggregates multiple ToolGroup instances, provides unified schema and execution dispatch.
    Maintains optional edit history cache for editor tools.
    """

    def __init__(self):
        self._groups: List[ToolGroup] = []
        self._tool_to_group: Dict[str, ToolGroup] = {}
        # Persistent file edit history across multi-turn sessions (path -> List[str])
        self._editor_state_path = Path("/tmp/sokoban_editor_state.json")

    def add_group(self, group: ToolGroup) -> None:
        self._groups.append(group)
        for name in group.get_tool_names():
            if name in self._tool_to_group:
                raise ValueError(f"Tool '{name}' already registered by another group.")
            self._tool_to_group[name] = group

    def get_schemas(self) -> List[dict]:
        schemas: List[dict] = []
        for group in self._groups:
            schemas.extend(group.get_schemas())
        return schemas

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        group = self._tool_to_group.get(name)
        if not group:
            raise ValueError(f"Unknown tool: {name}")
        # Route calls, passing editor state path when needed
        if name == "file_editor":
            return group.execute(name, arguments, editor_state_path=self._editor_state_path)
        return group.execute(name, arguments)
 
__all__ = [
    "tool",
    "ToolGroup",
    "ToolManager",
]


