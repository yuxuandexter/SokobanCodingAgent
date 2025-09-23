from .agent_utils import EnvOutput, SingleTurnTrajectory, MultiTurnTrajectory

# Minimal registry and decorator to register agents by name.
AGENT_REGISTRY = {}

def register_agent(name):
    """Decorator to register an agent class under a given name."""
    def decorator(cls):
        AGENT_REGISTRY[name] = cls
        return cls
    return decorator

__all__ = [
    "EnvOutput",
    "SingleTurnTrajectory",
    "MultiTurnTrajectory",
    "register_agent",
    "AGENT_REGISTRY",
]
