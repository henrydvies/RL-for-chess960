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
    
def self_play(total_timesteps, model_path):
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
    
    try:
        agent.train(total_timesteps)
    finally:
        agent.save()
    
if __name__=="__main__":
    self_play(20000, "models/rl_agent")