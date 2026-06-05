"""
Handle training
"""
from game.environment import ChessEnvironment
from engines.minimax.minimax_agent import MinimaxAgent
from engines.random.random_agent import RandomAgent
from engines.rl.rl_agent import rlAgent
from wandb.integration.sb3 import WandbCallback
import wandb

def train(opponent, total_timesteps, model_path=None, use_wandb=True):
    """
    Run the training loop
    """
    # Create environment with engine opponent
    environment = ChessEnvironment(opponent)
    
    # Create reinforcement learning agent
    agent = rlAgent(environment)
    
    # Load existing model
    if model_path:
        agent.load(model_path)
        agent.model.set_env(environment)
    
    callback = None
    if use_wandb:
        wandb.init(project="rl-chess960", sync_tensorboard=True)
        callback = WandbCallback()
    
    # Train, try/ finally to ensure model save incase issue during train.
    try:
        agent.train(total_timesteps, callback)
    finally:
        # Save model
        agent.save()
        if use_wandb:
            wandb.finish()
    
def self_play(total_timesteps, model_path, use_wandb=True):
    """
    Holds self play loop
    """
    # Create opponent agent
    temp_environment = ChessEnvironment(RandomAgent())
    opponent_agent = rlAgent(temp_environment)
    
    # Create training agent
    environment = ChessEnvironment(opponent_agent)
    agent = rlAgent(environment)
    
    # Load models
    agent.load(model_path)
    opponent_agent.load(model_path)
    
    agent.model.set_env(environment)
    opponent_agent.model.set_env(temp_environment)
    
    callback = None
    if use_wandb:
        wandb.init(project="rl-chess960", sync_tensorboard=True)
        callback = WandbCallback()
    
    try:
        agent.train(total_timesteps, callback)
    finally:
        agent.save()
        if use_wandb:
            wandb.finish()
    
if __name__=="__main__":
    train(RandomAgent(), 5000, use_wandb=True)
    train(MinimaxAgent(depth=2), 50000, model_path="models/rl_agent",  use_wandb=True)
    for _ in range(0,100):
        self_play(10000, model_path="models/rl_agent", use_wandb=True)