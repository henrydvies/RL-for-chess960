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
from evaluation.evaluator import play_single_game
import shutil
import os
import random

def run_training(agent_class, opponent, agent_model_folder="models/rl_agent", opponent_agent_model_path=None, total_timesteps=0, use_wandb=False):
    """
    Run the training loop vs any opponent
    """
    agent_model_path = f"{agent_model_folder}/{agent_model_folder.split('/')[-1]}"
    log_path = f"{agent_model_folder}/training_log.json"
    
    environment = ChessEnvironment(opponent)
    
    agent = agent_class(environment)
    
    if agent_model_path:
        agent.load(agent_model_path)
        agent.model.set_env(environment)
        
    if opponent_agent_model_path:
        if os.path.exists(f"{opponent_agent_model_path}.zip"):
            opponent.load(opponent_agent_model_path)
        else:
            derived_opponent_path = f"{opponent_agent_model_path}/{opponent_agent_model_path.split('/')[-1]}"
            opponent.load(derived_opponent_path)
        
    callback = None
    if use_wandb:
        wandb.init(project="rl-chess960", sync_tensorboard=True)
        callback = WandbCallback()
    
    logger = TrainingLogger(log_path)
    colour_distribution = {"white": 0, "black": 0}
    ep_rew_mean = None
    
    try:
        agent.train(total_timesteps, callback=callback)
        
        colour_distribution = {"white": environment.white_episodes, "black": environment.black_episodes}
        ep_rew_mean = None
        
    finally:
        agent.save(agent_model_path)
        
        logger.update_log(total_timesteps, opponent, ep_rew_mean, colour_distribution)
        logger.save()
        if use_wandb:
            wandb.finish()

def _evaluate_all_opponents(agent_class, model_path):
    """
    Evaluate current model vs all opponents and log as 0-timestep entries.
    """
    log_path = f"{model_path}/training_log.json"
    logger = TrainingLogger(log_path)

    eval_agent = agent_class(ChessEnvironment(RandomAgent()))
    eval_agent.load(f"{model_path}/{model_path.split('/')[-1]}")

    for eval_opp in [RandomAgent(), MinimaxAgent(), StockfishAgent()]:
        score = get_ep_rew_mean(eval_agent, eval_opp, n_games=15)
        logger.update_log(0, eval_opp, score)
        logger.save()
        if hasattr(eval_opp, 'close'):
            eval_opp.close()

    self_play_opp = agent_class(ChessEnvironment(RandomAgent()))
    self_play_opp.load(f"{model_path}/{model_path.split('/')[-1]}")
    score = get_ep_rew_mean(eval_agent, self_play_opp, n_games=15)
    logger.update_log(0, self_play_opp, score)
    logger.save()

def handle_training(agent_class=rlAgent, config=[(RandomAgent, 0, None), (MinimaxAgent, 0, None), (rlAgent, 10000, "models/rl_agent")], model_path="models/rl_agent", use_wandb=True):
    """
    Handle the training loop, along with evaluation.
    """
    elo_tracker = EloTracker()
    timesteps_iteration_cap = 50000
    for opponent_agent, timesteps, opponent_model_path in config:
        if opponent_agent == rlAgent:
            temp_env = ChessEnvironment(RandomAgent())
            opponent_instance = rlAgent(temp_env)
        else:
            temp_env = None
            opponent_instance = opponent_agent()
        
        fst_chance = 0.2
        chosen_opponent_path = opponent_model_path
        if opponent_agent == rlAgent and os.path.exists(f"{model_path}/snapshots"):
            snapshots = os.listdir(f"{model_path}/snapshots")
            if snapshots and random.random() < fst_chance:
                chosen = random.choice(snapshots).replace(".zip", "")
                chosen_opponent_path = f"{model_path}/snapshots/{chosen}"
        
        while timesteps > timesteps_iteration_cap:
            run_training(agent_class=agent_class, opponent=opponent_instance, agent_model_folder=model_path, opponent_agent_model_path=chosen_opponent_path, total_timesteps=timesteps_iteration_cap, use_wandb=use_wandb)
            timesteps -= timesteps_iteration_cap
            _evaluate_all_opponents(agent_class, model_path)
        
        run_training(agent_class=agent_class, opponent=opponent_instance, agent_model_folder=model_path, opponent_agent_model_path=chosen_opponent_path, total_timesteps=timesteps, use_wandb=use_wandb)
        _evaluate_all_opponents(agent_class, model_path)
        
        if not(temp_env):
            temp_env = ChessEnvironment(RandomAgent())
        elo_agent = rlAgent(temp_env)
        elo_agent.load(f"{model_path}/{model_path.split('/')[-1]}")
        
        elo_tracker = evaluate(elo_agent, opponent_instance, n_games=10, tracker=elo_tracker)

        if opponent_agent == rlAgent:
            elo_tracker = evaluate(RandomAgent(), MinimaxAgent(), n_games=10, tracker=elo_tracker)
        
        elo_tracker.save()
        
        if hasattr(opponent_instance, 'close'):
            opponent_instance.close()
    os.makedirs(f"{model_path}/snapshots", exist_ok=True)
    snapshot_count = len(os.listdir(f"{model_path}/snapshots"))
    shutil.copy(
        f"{model_path}/{model_path.split('/')[-1]}.zip",
        f"{model_path}/snapshots/snapshot_{snapshot_count + 1}.zip"
    )
    
def get_ep_rew_mean(agent, opponent, n_games=10):
    """
    Play n_games between agent and opponent.
    Returns (wins - losses) / n_games. Win=+1, Draw=0, Loss=-1.
    Cleaner than SB3 episode buffer which mixes opponents and includes draw penalty.
    """
    wins, draws, losses = 0, 0, 0
    for i in range(n_games):
        if i % 2 == 0:
            result = play_single_game(agent, opponent)
            agent_won = result is True
            agent_lost = result is False
        else:
            result = play_single_game(opponent, agent)
            agent_won = result is False
            agent_lost = result is True
        
        if agent_won: wins += 1
        elif agent_lost: losses += 1
        else: draws += 1
    
    return (wins - losses) / n_games
    
if __name__=="__main__":
    """
    Dynamic loop that selects training opponents based of performance.
    """


    while True:
        # Dynamic training designed to be ran and left, only plays opponent if it will learn from it, so no stockfish when it just looses every game etc
        eval_agent = rlAgent(ChessEnvironment(RandomAgent()))
        eval_agent.load("models/rl_agent_v3/rl_agent_v3")
        
        random_score = get_ep_rew_mean(eval_agent, RandomAgent(), n_games=10)
        minimax_score = get_ep_rew_mean(eval_agent, MinimaxAgent(), n_games=10)
        stockfish_score = get_ep_rew_mean(eval_agent, StockfishAgent(), n_games=10)
        StockfishAgent().close()
        
        # If more opponents then less time on self play for equal loop length
        # This also means that early on when it arguably needs more random/ minimax it does it more frequently due to less self play.
        opponents_added = sum([
            random_score <= 0.5,
            -0.8 <= minimax_score <= 0.5,
            stockfish_score > -0.8
        ])
        self_play_timesteps = 250000 - (50000 * opponents_added)
        
        main_config = [
            (rlAgent, self_play_timesteps, "models/rl_agent_v3"),
        ]
        
        if random_score <= 0.5:
            main_config.insert(0, (RandomAgent, 50000, None))
        
        if -0.7 <= minimax_score <= 0.6:
            main_config.insert(0, (MinimaxAgent, 50000, None))
        
        if stockfish_score > -0.8:
            main_config.append((StockfishAgent, 100000, None))
        
        handle_training(agent_class=rlAgent, config=main_config, use_wandb=False, model_path="models/rl_agent_v3")