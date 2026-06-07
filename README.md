# RL for Chess960

Applying reinforcement learning to Chess960 (Fischer Random Chess) to explore whether an agent trained from scratch on randomised starting positions develops different strategic behaviour compared to classical engines, and whether the absence of opening theory narrows the performance gap.

Chess960 randomises the back rank piece positions across 960 possible configurations, nullifying opening books entirely. Classical engines like Stockfish derive significant advantage from memorised opening theory, so Chess960 levels the playing field theoretically and makes it a more honest test of learned strategic reasoning.

---

## Research Question

> Does an RL agent trained purely from self-play on Chess960 develop emergent strategic behaviour that differs from classical engines? And does the absence of opening theory reduce the performance gap?

---

## Tech Stack

- **Python** — core language
- **python-chess** — chess logic and Chess960 support
- **PyTorch** — neural network and training
- **Stable-Baselines3 / sb3-contrib** — MaskablePPO implementation
- **Gymnasium** — RL environment interface
- **Matplotlib** — training visualisation
- **Wandb** — experiment tracking

---

## Project Structure

```
RL-FOR-Chess960/
├── game/
│   ├── environment.py          # Gym-compatible Chess960 environment
│   └── board_representation.py # Board to tensor conversion (8x8x12)
├── engines/
│   ├── random/                 # Random agent
│   ├── minimax/                # Minimax agent with material evaluation
│   ├── stockfish/              # Stockfish wrapper agent
│   └── rl/                     # PPO-based RL agent, policy network, training
├── evaluation/
│   ├── elo_tracker.py          # Elo rating tracking across agents
│   ├── evaluator.py            # Game-playing evaluation loop
│   ├── training_logger.py      # Per-run training log (JSON)
│   └── plot_training.py        # Training curve visualisation
├── utils/
│   └── action_masks.py         # Legal move masking for MaskablePPO
├── models/                     # Saved model weights and training logs (gitignored)
├── visualisation/              # Training curve graphs (auto-generated)
└── tests/                      # Full test suite
```

---

## Board Representation

The board is encoded as an **8x8x12 tensor**:
- 8x8 for board squares
- 12 layers: 6 piece types x 2 colours
- White pieces: layers 0-5, Black pieces: layers 6-11
- Each cell is binary (0 or 1)

## Action Space

Actions are integers 0-4095 encoding all 64x64 from/to square combinations:
- `from_square = action // 64`
- `to_square = action % 64`

Action masking (via `MaskablePPO`) ensures only legal moves are sampled during training and inference.

---

## Training Curriculum

The agent is trained using a staged curriculum:

1. **Random opponent** — learns basic legal play
2. **Minimax (depth 2)** — learns to avoid simple blunders and take pieces
3. **Stockfish (depth 1)** — learns from a tactically consistent opponent
4. **Self-play** — develops emergent strategy through iterative improvement

Training metrics (ep_rew_mean) and Elo ratings are logged per run and updated automatically.

---
## Improvements in v2

Discovered a large bug present for all of v1 training, where opponent model wasn't loading, meaning self play was actually against the random agent.
Applied fix, and as it gave opportunity to train new agent implemented some architecture improvements.

- **Bug fix** — opponent model loading was failing silently during self-play, meaning v1's "self-play" training.
- **More input planes** — expanded board representation from 8x8x12 to 8x8x20, adding turn indicator, castling rights (4 planes), en passant, repetition, and move count threshold.
- **Fictitious Self-Play** — opponent in self-play sampled 20% of the time from a pool of past snapshots, addressing strategy cycling. 
- **PPO hyperparameter tuning** — gamma=0.995 (default 0.99) for long chess games, ent_coef=0.01 (default 0) for explicit exploration regularisation, n_steps=4096 for more samples per update.
- **Draw penalty** — small negative reward (-0.1) for draws to discourage repetition-based stalling.
- **Random colour assignment** — agent trains as both white and black, balanced 50/50.
- **Opponent temperature with decay** — opponent occasionally plays random moves (0.2 → 0.05) to encourage exploration of novel positions during self-play.

---

## Training Results

### v1 (baseline — 12-plane representation)

#### Performance vs Minimax
<img src="visualisation/rl_agent_v1/MinimaxAgent.png" width="500"/>

#### Performance vs Stockfish
<img src="visualisation/rl_agent_v1/StockfishAgent.png" width="500"/>

### v2 (richer representation + FSP)

#### Performance vs Minimax
<img src="visualisation/rl_agent_v2/MinimaxAgent.png" width="500"/>

#### Performance vs Stockfish
<img src="visualisation/rl_agent_v2/StockfishAgent.png" width="500"/>

*Graphs show mean episode reward over total timesteps trained. Orange line is 5-run rolling average. Above 0 = net positive reward.*


## Current Elo Ratings

ELo only based on interactions between these agents, not a true FIDE elo rating.

| Agent | Elo |
|---|---|
| RandomAgent | ~50 |
| rlAgent | ~125 (improving) |
| MinimaxAgent | ~200 |
| StockfishAgent (depth 1) | ~820 |

*Elo tracked dynamically using evaluation games between agents after each training phase.*

---

## Reward Function

- Win: **+1**
- Loss: **-1**
- Draw: **0**
- Illegal move: **-1** (episode terminates)
- Midgame move: **0**

Reward shaping (e.g. material advantage bonuses) is intentionally omitted to avoid encoding human chess knowledge into the agent. The goal is purely emergent learning.

---

## Known Limitations

- **Queen-only promotion** — the agent always promotes to queen. (In some cases a knight promotion can be better).
- **Endgame conversion** — the agent struggles to convert winning endgames, often drawing by repetition due to sparse rewards.
- **Training scale** — significantly below production RL chess systems (AlphaZero used ~56M timesteps on 5,000 TPUs).
- **White only (training)** — the agent trains as white. Self-play exposes it to both colours at inference time.
- **No MCTS** — inference uses greedy policy sampling rather than Monte Carlo Tree Search, limiting tactical depth at play time.

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run tests
```bash
pytest
```

### Train a model
```bash
python -m engines.rl.train
```

### Play a game (outputs PGN for lichess/chess.com analysis)
```bash
python self_play_game_pgn.py
```

### Plot training curves
```bash
python -m evaluation.plot_training --model models/rl_agent_v1
```

---

## Roadmap

### Phase 1 — Environment 
- Chess960 Gym-compatible environment, board tensor, reward function, CI

### Phase 2 — Baseline Agents 
- Random agent, Minimax agent with material evaluation

### Phase 3 — RL Agent
- MaskablePPO with custom CNN policy, action masking, self-play loop

### Phase 4 — Training and Evaluation
- Elo tracking, Stockfish benchmarking, training curves, per-run logging

### Phase 5 — Interactive Play (planned)
- CLI/GUI interface to play against the trained agent

---

## Potential Future Improvements

- **Reward shaping** — small intermediate rewards for material gain to speed up tactical learning while preserving the emergent learning premise
- **Longer survival reward** — rewarding the agent for surviving more moves to discourage early collapse
- **MCTS at inference** — replacing greedy policy sampling with Monte Carlo Tree Search for stronger play at inference time without retraining
- **ResNet architecture** — replacing the CNN with a residual network for richer positional feature learning
- **Meta-learning** — training a reward function that maximises learning speed rather than hand-designing rewards, inspired by the idea of self-improving reward mechanisms