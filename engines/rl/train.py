"""
Handle training
"""
from game.environment import ChessEnvironment
from engines.minimax.minimax_agent import MinimaxAgent
from engines.random.random_agent import RandomAgent
from engines.rl.rl_agent import rlAgent
from wandb.integration.sb3 import WandbCallback
from evaluation.elo_tracker import EloTracker
from evaluation.evaluator import evaluate
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
        agent.save(model_path)
        if use_wandb:
            wandb.finish()
    
def self_play_train(total_timesteps, model_path, use_wandb=True):
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
        agent.save(model_path=model_path)
        if use_wandb:
            wandb.finish()

def handle_training(config=[(RandomAgent, 0), (MinimaxAgent, 0), (rlAgent, 10000)], model_path="models/rl_agent", use_wandb=True):
    """
    Handle the training loop, along with evaluation.
    """
    elo_tracker = EloTracker()
    for agent, timesteps in config:
        if isinstance(agent, (RandomAgent, MinimaxAgent)):
            train(agent, timesteps, model_path, use_wandb=use_wandb)
            rl_agent_instance = rlAgent(ChessEnvironment(RandomAgent()))
            rl_agent_instance.load(model_path)
            elo_tracker = evaluate(rl_agent_instance, agent, n_games=20, tracker=elo_tracker)
                
        
        if agent in [rlAgent]:
            # For self play break down into 25k timesteps in order to let model update every so often
            while timesteps > 25000:
                self_play_train(25000, model_path, use_wandb=use_wandb)
                timesteps -= 25000
                
                # Evaluate after loop
                rl_agent_instance = rlAgent(ChessEnvironment(RandomAgent()))
                rl_agent_instance.load(model_path)
                opponent_agent_instance = rlAgent(ChessEnvironment(RandomAgent()))
                opponent_agent_instance.load(model_path)
                elo_tracker = evaluate(rl_agent_instance, opponent_agent_instance, n_games=20, tracker=elo_tracker)
                
                # Evaluate random vs minimax to keep ratings updated and avoid plummet to 0
                elo_tracker = evaluate(RandomAgent(), MinimaxAgent(depth=3), n_games=20, tracker=elo_tracker)
                
            self_play_train(timesteps, model_path, use_wandb=use_wandb)
        elo_tracker.save()
    
    
if __name__=="__main__":
    config = [
        (RandomAgent(), 0),
        (MinimaxAgent(depth=2), 10000),
        (rlAgent, 1000)
    ]
    while True:
        handle_training(config=config, use_wandb=False, model_path="models/rl_agent_minimax")