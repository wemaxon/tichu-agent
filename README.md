# tichu-ai

Repo to test different machine learning approaches at playing the card game Tichu.

## Python Game Engine
The basic [tichu game engine](/tichu/) implements a standard four-player parnership tichu.

Agents or humans can connect using the following API:
```
player = round_.current_player
observation = round_.observation(player)
legal_actions = round_.legal_actions()
round_.apply(selected_action)
```

#### Run a simulation with random agents
```bash
python -m scripts.tichu_cli play --seed 10 --rounds 1 --verbose
```

Select agents per player:

```bash
python -m scripts.tichu_cli play --agents random random random random
```