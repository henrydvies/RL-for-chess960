"""
Handle training
"""
from game.environment import ChessEnvironment
from engines.minimax.minimax_agent import MinimaxAgent
from engines.random.random_agent import RandomAgent
from engines.rl.rl_agent import rlAgent

def train(opponent, total_timesteps, model_path=None):
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
    
    # Train, try/ finally to ensure model save incase issue during train.
    try:
        agent.train(total_timesteps)
    finally:
        # Save model
        agent.save()
    
    
    
if __name__=="__main__":
    train(MinimaxAgent(depth=1), 100000)