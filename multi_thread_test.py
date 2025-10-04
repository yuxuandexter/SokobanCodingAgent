import os
import asyncio
from typing import List, Tuple, Dict, Any

from agent.async_agent_group import AsyncAgentGroupBuilder


async def main_async() -> List[Tuple[int, Dict[str, Any]]]:
    builder = AsyncAgentGroupBuilder(seeds=[101, 202, 303, 404], agent_name="sokobanAgent")
    results = await builder.run(max_workers=min(4, os.cpu_count() or 2))
    return results


def main() -> None:
    results = asyncio.run(main_async())
    for seed, res in results:
        print(f"seed={seed} finished={res.get('finished')} reward={res.get('reward')} actions={res.get('actions')}")


if __name__ == "__main__":
    main()

