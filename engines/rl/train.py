"""
Handle training
"""
from game.environment import ChessEnvironment
from engines.minimax.minimax_agent import MinimaxAgent
from engines.random.random_agent import RandomAgent
from engines.rl.rl_agent import rlAgent
from engines.stockfish.stockfish_agent import StockfishAgent
from wandb.integration.sb3 import WandbCallback
from evaluation.elo_tracker import EloTracker
from evaluation.evaluator import evaluate
import wandb
from evaluation.training_logger import TrainingLogger
import numpy as np

def run_training(agent_class, opponent, agent_model_folder="models/rl_agent", opponent_agent_model_path=None, total_timesteps=0, use_wandb=False):
    """
    Run the training loop vs any opponent
    """
    # Extract file locations
    agent_model_path = f"{agent_model_folder}/{agent_model_folder.split('/')[-1]}"
    log_path = f"{agent_model_folder}/training_log.json"
    
    # Create environment
    environment = ChessEnvironment(opponent)
    
    agent = agent_class(environment)
    
    # Load agent
    if agent_model_path:
        agent.load(agent_model_path)
        agent.model.set_env(environment)
        
    if opponent_agent_model_path:
        opponent.load(opponent_agent_model_path)
        
    # Handle wandb logging
    callback = None
    if use_wandb:
        wandb.init(project="rl-chess960", sync_tensorboard=True)
        callback = WandbCallback()
    
    logger = TrainingLogger(log_path)
    ep_rew_mean = None
    # Run training
    try:
        agent.train(total_timesteps, callback=callback)
        
        # Get ep_mean for last few
        if agent.model.ep_info_buffer:
            ep_rew_mean = np.mean([ep["r"] for ep in agent.model.ep_info_buffer])
        
    finally:
        agent.save(agent_model_path)
        
        # Logging
        logger.update_log(total_timesteps, opponent, ep_rew_mean)
        logger.save()
        if use_wandb:
            wandb.finish()

def handle_training(agent_class=rlAgent, config=[(RandomAgent, 0, None), (MinimaxAgent, 0, None), (rlAgent, 10000, "models/rl_agent")], model_path="models/rl_agent", use_wandb=True):
    """
    Handle the training loop, along with evaluation.
    """
    elo_tracker = EloTracker()
    timesteps_iteration_cap = 20000
    for opponent_agent, timesteps, opponent_model_path in config:
        # Handle rl agent
        if opponent_agent == rlAgent:
            temp_env = ChessEnvironment(RandomAgent())
            opponent_instance = rlAgent(temp_env)
        else:
            temp_env = None
            opponent_instance = opponent_agent()
            
        # Loop to run timesteps_iteration_cap before reloading model, to ensure it trains against up to date model
        while timesteps > timesteps_iteration_cap:
            run_training(agent_class=agent_class, opponent=opponent_instance, agent_model_folder=model_path, opponent_agent_model_path=opponent_model_path, total_timesteps=timesteps_iteration_cap, use_wandb=use_wandb)
            timesteps -= timesteps_iteration_cap
        
        run_training(agent_class=agent_class, opponent=opponent_instance, agent_model_folder=model_path, opponent_agent_model_path=opponent_model_path, total_timesteps=timesteps, use_wandb=use_wandb)
        
        # Update elo
        if not(temp_env):
            temp_env = ChessEnvironment(RandomAgent())
        elo_agent = rlAgent(temp_env)
        elo_agent.load(f"{model_path}/{model_path.split('/')[-1]}")
        
        elo_tracker = evaluate(elo_agent, opponent_instance, n_games=10, tracker=elo_tracker)

        # For self-play also evaluate random vs minimax
        if opponent_agent == rlAgent:
            elo_tracker = evaluate(RandomAgent(), MinimaxAgent(), n_games=10, tracker=elo_tracker)
        
        elo_tracker.save()
        
        # For stockfish to close correctly
        if hasattr(opponent_instance, 'close'):
            opponent_instance.close()
    
if __name__=="__main__":
    config = [
        (RandomAgent, 0, None),
        (MinimaxAgent, 100, None),
        (StockfishAgent, 1000, None),
        (rlAgent, 1000, "models/rl_agent")
    ]
    #while True:
    #    handle_training(agent_class=rlAgent, config=config, use_wandb=False, model_path="models/rl_agent_minimax")
    handle_training(agent_class=rlAgent, config=config, use_wandb=False, model_path="models/rl_agent_test")