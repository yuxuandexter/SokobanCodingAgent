# SokobanCodingAgent
a lightweight, tool-calling coding agent that plans with ReAct and writes/executes code to solve classic Sokoban box-pushing puzzles.

## Setup

### Create conda environment

```bash
conda create -y -n agent_hub_env python=3.12
conda activate agent_hub_env
```

### Install dependencies (editable mode)

You can install via editable mode to develop locally:

```bash
pip install -e .
```

Alternatively, to install directly from requirements:

```bash
pip install -r requirements.txt
```

This will install:
- gymnasium
- gym-sokoban
- numpy
- PyYAML
- pytest (for tests)

### Run tests

```bash
pytest -q
```

Test runs will write simple logs under a root-level `cache` directory.

### Development Track

[O] test sokoban coding agent with gpt api
[ ] refine sokoban agent with tool calling
    [ ] coding agent trajectory support
    [ ] multi threads support 

