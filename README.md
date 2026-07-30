# tichu-ai

Repo to test different machine learning approaches at playing the card game Tichu.

## Game Engine
The [game engine](/tichu/) implements a standard four-player parnership tichu.

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
python -m scripts.tichu_cli play --agents random random greedy heuristic
```

Bomb combinations are enabled by default and follow normal turn order. Enable
standard Tichu-style interrupt windows explicitly:

```bash
python -m scripts.tichu_cli play --out-of-turn-bombs
```

#### Play from the console
```bash
python -m scripts.tichu_cli console --seat 0 --agent random
```

This controls one human seat and assigns agents to the other three seats,
including the human player's partner.

## RL Environment

```mermaid
  flowchart TB
      subgraph APPLICATION["Training / Evaluation"]
          direction TB
          TRAINER["Rollout Collector / Trainer"]
          POLICY["Policy + Value Network"]

          TRAINER -->|"observation and legal-action features"| POLICY
          POLICY -->|"action index, value and log probability"| TRAINER
      end

      subgraph ML["ML Environment — ml/"]
          direction TB

          ENV["TichuEnv<br/>reset • state • step"]
          POLICY_VIEW(["PolicyObservation<br/>actor-relative and player-safe"])
          OBS_ENCODER["Observation Encoder"]
          ACTION_ENCODER["Legal-Action Encoder"]
          ENV_STATE(["TichuEnvState<br/>features • candidates • rewards"])

          subgraph ENGINE["Tichu Engine — tichu/"]
              direction TB

              DOMAIN["Domain Model<br/>cards • combinations • actions • config"]
              ROUND["TichuRound<br/>immutable state machine"]
              OBSERVATION(["RoundObservation<br/>observable information"])
              LEGAL_ACTIONS(["Legal actions<br/>tuple of Action objects"])

              DOMAIN -->|"defines game concepts used by"| ROUND
              ROUND -->|"observation(player)"| OBSERVATION
              ROUND -->|"legal_actions()"| LEGAL_ACTIONS
          end

          ENV -->|"creates, owns and advances"| ROUND
          OBSERVATION -->|"policy_view()"| POLICY_VIEW
          POLICY_VIEW -->|"encode_policy_observation()"| OBS_ENCODER
          LEGAL_ACTIONS -->|"encode_legal_actions(observer=player)"| ACTION_ENCODER
          OBS_ENCODER -->|"observation_features"| ENV_STATE
          ACTION_ENCODER -->|"encoded_actions"| ENV_STATE
          ENV -->|"adds player, terminal state and rewards"| ENV_STATE
      end

      TRAINER -->|"reset() or step(action_index)"| ENV
      ENV_STATE -->|"returned after reset, state or step"| TRAINER

      style ENGINE fill:#fff4d6,stroke:#d39b24,stroke-width:3px
      style ML fill:#e8f4ff,stroke:#3278a8,stroke-width:3px
      style APPLICATION fill:#e9f8ec,stroke:#3c8c4c,stroke-width:3px
```

`TichuEnv` exposes actor-relative policy observations and variable legal-action
features without exposing the underlying round or opponents' hands:

```python
from ml import TichuEnv

env = TichuEnv(seed=0)
state = env.state()

assert state.policy_observation.current_player == 0

action_features = [candidate.features for candidate in state.encoded_actions]
state = env.step(action_index=0)
```

Relative seats are ordered as the observer, next seat, partner, and previous
seat. Action targets use the same coordinates:
  - 0: self
  - 1: next player
  - 2: partner
  - 3: previous player
  
Full engine state remains available explicitly as `env.round` for diagnostics, but it is not part of the
policy-facing `TichuEnvState`.


## Agents

- `random`: random legal-action baseline.
- `greedy`: deterministic baseline that declines Tichu calls, exchanges low cards,
  avoids bombs unless they finish the hand, and otherwise plays the cheapest useful
  legal combination.
- `heuristic`: deterministic rule-based baseline with hand-strength Tichu calls,
  partner-aware exchange/pass decisions, and conservative tactical bomb use.

## Evaluation

Benchmark two agents with paired rounds:

```bash
python -m scripts.benchmark_agents --agent-a greedy --agent-b random --round-pairs 100 --json results.json --csv rounds.csv
```

Each round pair uses the same seed and dealer twice, swapping which team uses
agent A and agent B. This reduces seat and deal bias.
