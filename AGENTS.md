# Project: REx — Reputation-regulated Exploration for LLM Agents

## Goal

Build a minimal but extensible Python research prototype for studying Reputation-regulated Exploration in LLM-agent evolutionary games.

The system should simulate multiple LLM agents in repeated social dilemma games. Each agent has a reputation score. The reputation dynamically regulates the agent's exploration behavior, including sampling temperature, reflection depth, and strategic risk tendency.

## Strict scope

- Only implement LLM agents.
- Do not implement RL agents, MARL agents, or ABM-style agents as the core.
- Do not train model parameters.
- Do not use gradient updates.
- This is a gradient-free multi-agent simulation framework.
- The default backend must be a mock LLM backend so the whole project can run without external API calls.
- A real OpenAI-compatible backend can be added later, but it must not be used by default.

## Coding standards

- Python 3.10+.
- Use type hints.
- Use dataclasses where appropriate.
- Keep modules small and focused.
- Avoid putting all logic in one file.
- Every random process must be seedable and reproducible.
- Prefer simple, readable research code over over-engineered abstractions.
- Do not call external APIs in default tests or baseline runs.
- Write CSV logs for agent-level and round-level results.
- Include plotting scripts using matplotlib only.
- Avoid seaborn.
- Each plot should be a separate figure, not subplots.

## Required runnable commands

The following commands must work:

```bash
python scripts/run_baseline.py
python scripts/run_reputation_attack.py
python scripts/run_malicious_agents.py
python scripts/plot_results.py
pytest
```

## Implementation priority

First implement:
- MockBackend
- Prisoner's Dilemma
- ReputationModel
- ReputationExplorationController
- LLMAgent
- SimulationRunner
- baseline experiments
- CSV logging
- plotting

Then implement:
- ReputationAttack
- MaliciousAgentInjection
- recovery metrics
- tests

Optional later:
- TrustGame
- PublicGoodsGame
- OpenAI-compatible backend

## Research logic

The main experimental comparison should include:

1. fixed_low_temperature
2. fixed_mid_temperature
3. fixed_high_temperature
4. reputation_linear
5. reputation_piecewise
6. reputation_piecewise_with_attack
7. reputation_piecewise_with_malicious_agents

The most important method is reputation_piecewise:

```python
If reputation < 0.3:
    temperature = 0.2
    reflection_depth = 3
    risk_tendency = "low"
    mode = "recovery"

If 0.3 <= reputation < 0.7:
    temperature = 0.5
    reflection_depth = 2
    risk_tendency = "medium"
    mode = "normal"

If reputation >= 0.7:
    temperature = 0.75
    reflection_depth = 1
    risk_tendency = "medium_high"
    mode = "exploration"
```

Also implement reputation shock response:
If the agent defected within the last 3 rounds, temporarily reduce its temperature by multiplying it by 0.6.

## Validation

After each major implementation step, run the relevant command and fix errors before continuing.
Do not leave TODO placeholders in core logic.
When done, summarize:
- files created
- commands run
- tests passed or failed
- known limitations
