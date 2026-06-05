"""
Handle training
"""
from game.environment import ChessEnvironment
from engines.minimax.minimax_agent import MinimaxAgent
from engines.random.random_agent import RandomAgent
from engines.rl.rl_agent import rlAgent

def train(opponent, total_timesteps):
    """
    Run the training loop
    """
    # Create environment with engine opponent
    environment = ChessEnvironment(opponent)
    
    # Create reinforcement learning agent
    agent = rlAgent(environment)
    
    # Train
    agent.train(total_timesteps)
    
    # Save model
    agent.save()
    
    
    
if __name__=="__main__":
    train(MinimaxAgent(), 120)