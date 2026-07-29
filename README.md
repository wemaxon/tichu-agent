# tichu-ai

Repo to test different machine learning approaches at playing the card game Tichu.

## Game Environment
The basic [game environment](/tichu/) implements a standard four-player parnership tichu.

Agents or humans can connect using the following API:
```
player = round_.current_player
observation = round_.observation(player)
legal_actions = round_.legal_actions()
action = agent.choose_action(observation, legal_actions)
round_ = round_.apply(action)
```

#### Run a simulation with random agents
```bash
python -m scripts.tichu_cli play --seed 10 --rounds 1 --verbose
```

Select agents per player:

```bash
python -m scripts.tichu_cli play --agents random random greedy greedy
```

#### Play from the console
```bash
python -m scripts.tichu_cli console --seat 0 --agent random
```

This controls one human seat and assigns agents to the other three seats,
including the human player's partner.


## Agents

Available agents:

- `random`: random legal-action baseline.
- `greedy`: deterministic baseline that declines Tichu calls, exchanges low cards,
  avoids bombs unless they finish the hand, and otherwise plays the cheapest useful
  legal combination.

### Evaluation

Benchmark two agents with paired rounds:

```bash
python -m scripts.benchmark_agents --agent-a greedy --agent-b random --round-pairs 100
```

Each round pair uses the same seed and dealer twice, swapping which team uses
agent A and agent B. This reduces seat and deal bias. Per-round data can be
saved for later analysis:

```bash
python -m scripts.benchmark_agents --agent-a greedy --agent-b random --round-pairs 100 --json results.json --csv rounds.csv
```
