"""
Handle training
"""
from game.environment import ChessEnvironment
from engines.minimax.minimax_agent import MinimaxAgent
from engines.random.random_agent import RandomAgent
from engines.rl.rl_agent import rlAgent
from engines.stockfish.stockfish_agent import StockfishAgent
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
from evaluation.elo_tracker import EloTracker
from evaluation.evaluator import evaluate
import wandb
from evaluation.training_logger import TrainingLogger
from evaluation.evaluator import play_single_game
import shutil
import os
import random

EVAL_EVERY_TIMESTEPS = 50000
SELF_PLAY_TEMPERATURE = 0.2


class PeriodicEvaluationCallback(BaseCallback):
    """
    Every eval_every timesteps during a single learn() call:
    checkpoint the model, log the chunk to the training log, and run evaluation
    games vs all opponents. Replaces the old tear-down-and-reload-every-50k loop,
    so optimizer state, env (and its temperature decay) persist across the whole run.
    """
    def __init__(self, agent, opponent, environment, model_folder, agent_model_path, eval_every=EVAL_EVERY_TIMESTEPS, opponent_reload_path=None):
        super().__init__()
        self.agent = agent
        self.opponent = opponent
        self.environment = environment
        self.model_folder = model_folder
        self.agent_model_path = agent_model_path
        self.eval_every = eval_every
        self.opponent_reload_path = opponent_reload_path
        self.last_eval_timesteps = 0
        self.last_colour_counts = {"white": 0, "black": 0}

    def _on_step(self):
        """
        Check if it is time to evaluate opponent
        """
        if self.num_timesteps - self.last_eval_timesteps >= self.eval_every:
            self.checkpoint(self.num_timesteps)
            # Refresh the self-play opponent
            if self.opponent_reload_path:
                self.opponent.load(self.opponent_reload_path)
            _evaluate_all_opponents(self.agent, self.model_folder)
        return True

    def checkpoint(self, current_timesteps):
        """
        Save the model and log the timesteps/colour distribution since the last checkpoint.
        """
        chunk_timesteps = current_timesteps - self.last_eval_timesteps
        self.last_eval_timesteps = current_timesteps

        self.agent.save(self.agent_model_path)

        colour_counts = {
            "white": self.environment.white_episodes - self.last_colour_counts["white"],
            "black": self.environment.black_episodes - self.last_colour_counts["black"],
        }
        self.last_colour_counts = {
            "white": self.environment.white_episodes,
            "black": self.environment.black_episodes,
        }

        logger = TrainingLogger(f"{self.model_folder}/training_log.json")
        logger.update_log(chunk_timesteps, self.opponent, None, colour_counts)
        logger.save()


def run_training(agent, opponent, agent_model_folder="models/rl_agent", opponent_agent_model_path=None, total_timesteps=0, use_wandb=False, self_play=False):
    """
    Run one training phase vs a single opponent with a single learn() call.
    Periodic checkpointing/evaluation happens in-process via callback.
    """
    agent_model_path = f"{agent_model_folder}/{agent_model_folder.split('/')[-1]}"

    # Random opponent moves (exploration temperature) only apply during self-play,
    # so the benchmark opponents (random/minimax/stockfish) stay a clean signal.
    environment = ChessEnvironment(opponent, temperature=SELF_PLAY_TEMPERATURE if self_play else 0.0)
    agent.model.set_env(environment)

    resolved_opponent_path = None
    if opponent_agent_model_path:
        if os.path.exists(f"{opponent_agent_model_path}.zip"):
            resolved_opponent_path = opponent_agent_model_path
        else:
            resolved_opponent_path = f"{opponent_agent_model_path}/{opponent_agent_model_path.split('/')[-1]}"
        opponent.load(resolved_opponent_path)

    eval_callback = PeriodicEvaluationCallback(agent, opponent, environment, agent_model_folder, agent_model_path, opponent_reload_path=resolved_opponent_path)
    callbacks = [eval_callback]
    if use_wandb:
        wandb.init(project="rl-chess960", sync_tensorboard=True)
        callbacks.append(WandbCallback())

    try:
        agent.train(total_timesteps, callback=callbacks)
    finally:
        # Checkpoint and log whatever is left since the last periodic checkpoint
        eval_callback.checkpoint(agent.model.num_timesteps)
        if use_wandb:
            wandb.finish()

    _evaluate_all_opponents(agent, agent_model_folder)

def _evaluate_all_opponents(agent, model_folder):
    """
    Evaluate the live agent vs all opponents and log as 0-timestep entries.
    """
    log_path = f"{model_folder}/training_log.json"
    logger = TrainingLogger(log_path)

    for eval_opp in [RandomAgent(), MinimaxAgent(), StockfishAgent()]:
        score = get_ep_rew_mean(agent, eval_opp, n_games=15)
        logger.update_log(0, eval_opp, score)
        logger.save()
        if hasattr(eval_opp, 'close'):
            eval_opp.close()

    # Self-play evaluation vs the latest checkpoint on disk
    agent_model_path = f"{model_folder}/{model_folder.split('/')[-1]}"
    self_play_opp = type(agent)(ChessEnvironment(RandomAgent()))
    self_play_opp.load(agent_model_path)
    score = get_ep_rew_mean(agent, self_play_opp, n_games=15)
    logger.update_log(0, self_play_opp, score)
    logger.save()

def handle_training(agent_class=rlAgent, config=[(RandomAgent, 0, None), (MinimaxAgent, 0, None), (rlAgent, 10000, "models/rl_agent")], model_path="models/rl_agent", use_wandb=True):
    """
    Handle the training loop, along with evaluation.
    """
    elo_tracker = EloTracker()
    agent_model_path = f"{model_path}/{model_path.split('/')[-1]}"

    # Create the agent once and keep training it in process
    agent = agent_class(ChessEnvironment(RandomAgent()))
    if os.path.exists(f"{agent_model_path}.zip"):
        agent.load(agent_model_path)
    else:
        print(f"No existing model at {agent_model_path}.zip - starting from fresh weights.")
        os.makedirs(model_path, exist_ok=True)
        agent.save(agent_model_path)

    for opponent_agent, timesteps, opponent_model_path in config:
        self_play = opponent_agent == rlAgent
        if self_play:
            opponent_instance = opponent_agent(ChessEnvironment(RandomAgent()))
        else:
            opponent_instance = opponent_agent()

        # Chacnce to play a random older version of the model for self play
        fsp_chance = 0.2
        chosen_opponent_path = opponent_model_path
        if self_play and os.path.exists(f"{model_path}/snapshots"):
            snapshots = os.listdir(f"{model_path}/snapshots")
            if snapshots and random.random() < fsp_chance:
                chosen = random.choice(snapshots).replace(".zip", "")
                chosen_opponent_path = f"{model_path}/snapshots/{chosen}"

        run_training(
            agent=agent,
            opponent=opponent_instance,
            agent_model_folder=model_path,
            opponent_agent_model_path=chosen_opponent_path,
            total_timesteps=timesteps,
            use_wandb=use_wandb,
            self_play=self_play,
        )

        elo_tracker = evaluate(agent, opponent_instance, n_games=10, tracker=elo_tracker)

        if self_play:
            elo_tracker = evaluate(RandomAgent(), MinimaxAgent(), n_games=10, tracker=elo_tracker)

        elo_tracker.save()

        if hasattr(opponent_instance, 'close'):
            opponent_instance.close()

    os.makedirs(f"{model_path}/snapshots", exist_ok=True)
    snapshot_count = len(os.listdir(f"{model_path}/snapshots"))
    shutil.copy(
        f"{agent_model_path}.zip",
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
        model_file = "models/rl_agent_v3/rl_agent_v3"
        if os.path.exists(f"{model_file}.zip"):
            eval_agent.load(model_file)
        else:
            print(f"No existing model at {model_file}.zip - evaluating fresh weights.")
        
        random_score = get_ep_rew_mean(eval_agent, RandomAgent(), n_games=10)
        minimax_score = get_ep_rew_mean(eval_agent, MinimaxAgent(), n_games=10)
        stockfish_opponent = StockfishAgent()
        stockfish_score = get_ep_rew_mean(eval_agent, stockfish_opponent, n_games=10)
        stockfish_opponent.close()
        
        # If more opponents then less time on self play for equal loop length
        # This also means that early on when it arguably needs more random/ minimax it does it more frequently due to less self play.
        opponents_added = sum([
            random_score <= 0.5,
            -0.7 <= minimax_score <= 0.6,
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
