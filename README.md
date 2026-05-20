# REx: Reputation-regulated Exploration for LLM Agents in Evolutionary Games

REx is a minimal Python research prototype for studying whether an LLM agent's
reputation can regulate its exploration behavior in repeated social dilemma
games. Exploration is represented by sampling temperature, reflection depth, and
strategic risk tendency.

## Research Question

Can reputation-regulated exploration help LLM agents recover from low-trust
states, sustain cooperation, and remain robust under reputation attacks or
malicious-agent injection?

## Method

The prototype simulates repeated Prisoner's Dilemma interactions among mock LLM
agents. Each agent has a reputation in `[0, 1]`, memory of recent interactions,
and a current exploration state. The main comparison is between fixed-temperature
conditions and reputation-regulated conditions:

- `fixed_low_temperature`
- `fixed_mid_temperature`
- `fixed_high_temperature`
- `reputation_linear`
- `reputation_piecewise`
- `reputation_piecewise_with_attack`
- `reputation_piecewise_with_malicious_agents`

The default backend is `MockLLMBackend`, so baseline runs and tests do not call
external APIs.

## Project Structure

```text
config/                 YAML defaults and experiment overrides
src/agents/             LLM agent, memory, reputation, exploration controller
src/backends/           backend interface and mock backend
src/games/              Prisoner's Dilemma
src/simulation/         runner and interventions
src/metrics/            metrics and recovery analysis helpers
src/visualization/      matplotlib plotting
scripts/                runnable experiment scripts
tests/                  pytest coverage
outputs/results/        generated CSV outputs
outputs/figures/        generated figures
```

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is required.

## Running Experiments

Run the baseline fixed-temperature and reputation-regulated comparisons:

```bash
python scripts/run_baseline.py
```

Run the reputation attack experiment:

```bash
python scripts/run_reputation_attack.py
```

Run the malicious-agent injection experiment:

```bash
python scripts/run_malicious_agents.py
```

Generate plots from all available round-level CSV files:

```bash
python scripts/plot_results.py
```

Run tests:

```bash
pytest
```

## Output Files

Each simulation writes two CSV files to `outputs/results/`:

- `{condition}_agent_results.csv`: one row per agent interaction.
- `{condition}_round_results.csv`: one row per simulation round.

Plots are written to `outputs/figures/`:

- `cooperation_rate_over_time.png`
- `avg_reputation_over_time.png`
- `avg_temperature_over_time.png`
- `social_welfare_over_time.png`
- `action_entropy_over_time.png`
- `reputation_attack_recovery.png` when attack outputs exist

## Key Metrics

- `cooperation_rate`: share of actions that are cooperation.
- `avg_reputation`: mean agent reputation after each round.
- `avg_temperature`: mean exploration temperature.
- `social_welfare`: total payoff produced in a round.
- `action_entropy`: diversity of cooperation and defection actions.
- Recovery helpers in `src/metrics/analysis.py` estimate time to recover after
  an attack, welfare drop, and betrayal cascade size.

## Future Extensions

- Add Trust Game and Public Goods Game implementations.
- Add an OpenAI-compatible backend that is opt-in and never default.
- Add richer prompt templates and structured reflection traces.
- Add more intervention designs and recovery diagnostics.
- Add experiment sweeps over population size, attack intensity, and reputation
  update parameters.
