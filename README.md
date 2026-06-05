# Current plan

### Step 1
1. Setup basic chess960, use python-chess
2. Wrap chess environment as a Gym compatible environment: reset, step, render
3. Implement board state representation as an 8x8x12 tensor
4. Define reward function

### Step 2
1. Implement a random agent
2. Simple minimax with basic eval
3. Play random vs minimax to confirm environment working correctly

### Step 3
1. Implement PPO (Proximal Policy Optimisation) using Stable-Baselines3 or from scratch with pytorch
2. Define policy netwwork: CNN over board tensor
3. Train agent against random/ minimax opponent first
4. Add self play loop, to generate training data

### Step 4
1. Track elo rating over training iterations
2. Benchmark against stockfish at progressivly lower depth settings
3. Log/ visualise training curves
4. Compare move distributions/ piece mobility patterns

### Step 5
1. GUI to play against the agent
 - python-chess has this built in i believe.
2. Could do website too? flask? 


*To train*
python -m engines.rl.train