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
from stable_baselines3.common.vec_env import SubprocVecEnv
from evaluation.elo_tracker import EloTracker
from evaluation.evaluator import evaluate
import wandb
from evaluation.training_logger import TrainingLogger
from evaluation.evaluator import play_single_game
import shutil
import os
import random

EVAL_EVERY_TIMESTEPS = 100000
EVAL_N_GAMES = 10
INCLUDE_SELF_PLAY_EVAL = False
SELF_PLAY_TEMPERATURE = 0.2
TEMPERATURE_DECAY_STEPS = 40000
N_ENVS = 8
ENDGAME_CURRICULUM_PROB = 0.4
ENDGAME_CURRICULUM_PROB_FINAL = 0.15
ENDGAME_CURRICULUM_DECAY_EPISODES = 4000
MCTS_SIMS_TRAIN = 15
MCTS_C_PUCT = 1.25
BENCHMARK_OPPONENT_CLASSES = (RandomAgent, MinimaxAgent, StockfishAgent)


class MCTSOpponentWrapper:
    """
    Self-play opponent that searches with MCTS in take_turn.
    """

    def __init__(self, agent, n_sims, c_puct=MCTS_C_PUCT, root_deterministic=False):
        self._agent = agent
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.root_deterministic = root_deterministic

    def take_turn(self, board):
        return self._agent.take_turn(
            board,
            n_sims=self.n_sims,
            c_puct=self.c_puct,
            root_deterministic=self.root_deterministic,
        )

    def load(self, model_path):
        self._agent.load(model_path)

    def close(self):
        if hasattr(self._agent, "close"):
            self._agent.close()


def _make_env_fn(opponent_factory, temperature, use_mcts=False):
    """
    Returns a function that builds a fresh env with its own opponent instance.
    """
    def _init():
        return ChessEnvironment(
            opponent_factory(),
            temperature=temperature,
            temperature_decay_steps=max(1, TEMPERATURE_DECAY_STEPS // N_ENVS),
            endgame_probability=ENDGAME_CURRICULUM_PROB,
            endgame_probability_final=ENDGAME_CURRICULUM_PROB_FINAL,
            endgame_decay_episodes=ENDGAME_CURRICULUM_DECAY_EPISODES,
            use_mcts=use_mcts,
        )
    return _init


def _self_play_opponent_factory(opponent_model_path, mcts_sims=MCTS_SIMS_TRAIN):
    """
    Builds the self-play opponent inside a worker process. On cpu to stop it using up gpu.
    """
    def _factory():
        opponent = rlAgent(ChessEnvironment(RandomAgent()), device="cpu")
        opponent.load(opponent_model_path)
        if mcts_sims <= 0:
            return opponent
        return MCTSOpponentWrapper(
            opponent,
            n_sims=mcts_sims,
            c_puct=MCTS_C_PUCT,
            root_deterministic=False,
        )
    return _factory


def _update_elo_from_game(elo_tracker, agent, opponent, result, agent_played_white):
    """
    Update Elo from a single game result (True=white wins, False=black wins, None=draw).
    """
    agent_name = agent.__class__.__name__
    opponent_name = opponent.__class__.__name__

    if agent_name not in elo_tracker.elo_map:
        elo_tracker.elo_map[agent_name] = 600
    if opponent_name not in elo_tracker.elo_map:
        elo_tracker.elo_map[opponent_name] = 600

    if result is None:
        agent_outcome = None
    elif agent_played_white:
        agent_outcome = result
    else:
        agent_outcome = not result

    elo_tracker.update(agent_name, opponent_name, agent_outcome)


def get_ep_rew_mean(
    agent,
    opponent,
    n_games=10,
    n_sims=MCTS_SIMS_TRAIN,
    root_deterministic=True,
    elo_tracker=None,
):
    """
    Play n_games between agent and opponent.
    Returns (wins - losses) / n_games. Win=+1, Draw=0, Loss=-1.
    Cleaner than SB3 episode buffer which mixes opponents and includes draw penalty.
    """
    wins, draws, losses = 0, 0, 0
    opponent_n_sims = n_sims if isinstance(opponent, rlAgent) else 0
    for i in range(n_games):
        if i % 2 == 0:
            result = play_single_game(
                agent,
                opponent,
                white_n_sims=n_sims,
                black_n_sims=opponent_n_sims,
                root_deterministic=root_deterministic,
            )
            agent_won = result is True
            agent_lost = result is False
            if elo_tracker is not None:
                _update_elo_from_game(elo_tracker, agent, opponent, result, agent_played_white=True)
        else:
            result = play_single_game(
                opponent,
                agent,
                white_n_sims=opponent_n_sims,
                black_n_sims=n_sims,
                root_deterministic=root_deterministic,
            )
            agent_won = result is False
            agent_lost = result is True
            if elo_tracker is not None:
                _update_elo_from_game(elo_tracker, agent, opponent, result, agent_played_white=False)

        if agent_won:
            wins += 1
        elif agent_lost:
            losses += 1
        else:
            draws += 1

    return (wins - losses) / n_games


def run_benchmark_suite(
    agent,
    model_folder,
    *,
    n_games=EVAL_N_GAMES,
    include_self_play=INCLUDE_SELF_PLAY_EVAL,
    log_timesteps=0,
    logger=None,
    elo_tracker=None,
    n_sims=MCTS_SIMS_TRAIN,
    root_deterministic=True,
):
    """
    Run one evaluation pass vs standard benchmarks. Returns scores keyed by opponent class.
    Optionally logs to training_log.json and updates Elo from the same games.

    Self-play eval (agent vs its own checkpoint) is off by default; set
    include_self_play=True or INCLUDE_SELF_PLAY_EVAL to re-enable.
    """
    if logger is None and model_folder is not None:
        logger = TrainingLogger(f"{model_folder}/training_log.json")

    scores = {}
    for opponent_cls in BENCHMARK_OPPONENT_CLASSES:
        opponent = opponent_cls()
        score = get_ep_rew_mean(
            agent,
            opponent,
            n_games=n_games,
            n_sims=n_sims,
            root_deterministic=root_deterministic,
            elo_tracker=elo_tracker,
        )
        scores[opponent_cls] = score
        if logger is not None:
            logger.update_log(log_timesteps, opponent, score)
            logger.save()
        if hasattr(opponent, "close"):
            opponent.close()

    if include_self_play and model_folder is not None:
        agent_model_path = f"{model_folder}/{model_folder.split('/')[-1]}"
        self_play_opponent = type(agent)(ChessEnvironment(RandomAgent()))
        self_play_opponent.load(agent_model_path)
        score = get_ep_rew_mean(
            agent,
            self_play_opponent,
            n_games=n_games,
            n_sims=n_sims,
            root_deterministic=root_deterministic,
            elo_tracker=elo_tracker,
        )
        scores[type(agent)] = score
        if logger is not None:
            logger.update_log(log_timesteps, self_play_opponent, score)
            logger.save()

    if elo_tracker is not None:
        elo_tracker.save()

    return scores


def _evaluate_all_opponents(agent, model_folder, n_games=EVAL_N_GAMES):
    """
    Evaluate the live agent vs benchmark opponents and log as 0-timestep entries.
    """
    run_benchmark_suite(
        agent,
        model_folder,
        n_games=n_games,
        log_timesteps=0,
    )


class PeriodicEvaluationCallback(BaseCallback):
    """
    Every eval_every timesteps during a single learn() call:
    checkpoint the model, log the chunk to the training log, and run evaluation
    games vs all opponents. Replaces the old tear-down-and-reload-every-50k loop,
    so optimizer state, env (and its temperature decay) persist across the whole run.
    """
    def __init__(
        self,
        agent,
        opponent,
        environment,
        model_folder,
        agent_model_path,
        eval_every=EVAL_EVERY_TIMESTEPS,
        opponent_reload_path=None,
        elo_tracker=None,
    ):
        super().__init__()
        self.agent = agent
        self.opponent = opponent
        self.environment = environment
        self.model_folder = model_folder
        self.agent_model_path = agent_model_path
        self.eval_every = eval_every
        self.opponent_reload_path = opponent_reload_path
        self.elo_tracker = elo_tracker
        self.last_eval_timesteps = 0
        self.last_eval_at_timesteps = None
        self.last_colour_counts = {"white": 0, "black": 0}

    def _on_step(self):
        """
        Check if it is time to evaluate opponent
        """
        if self.num_timesteps - self.last_eval_timesteps >= self.eval_every:
            self.checkpoint(self.num_timesteps)
            # Refresh the self-play opponents in every worker process
            if self.opponent_reload_path:
                self.environment.env_method("reload_opponent", self.opponent_reload_path)
            run_benchmark_suite(
                self.agent,
                self.model_folder,
                n_games=EVAL_N_GAMES,
                elo_tracker=self.elo_tracker,
            )
            self.last_eval_at_timesteps = self.num_timesteps
        return True

    def checkpoint(self, current_timesteps):
        """
        Save the model and log the timesteps/colour distribution since the last checkpoint.
        """
        chunk_timesteps = current_timesteps - self.last_eval_timesteps
        self.last_eval_timesteps = current_timesteps

        self.agent.save(self.agent_model_path)

        # Colour episode counters summed across all worker envs
        colour_totals = {
            "white": sum(self.environment.get_attr("white_episodes")),
            "black": sum(self.environment.get_attr("black_episodes")),
        }
        colour_counts = {
            "white": colour_totals["white"] - self.last_colour_counts["white"],
            "black": colour_totals["black"] - self.last_colour_counts["black"],
        }
        self.last_colour_counts = colour_totals

        logger = TrainingLogger(f"{self.model_folder}/training_log.json")
        logger.update_log(chunk_timesteps, self.opponent, None, colour_counts)
        logger.save()


def run_training(
    agent,
    opponent,
    agent_model_folder="models/rl_agent",
    opponent_agent_model_path=None,
    total_timesteps=0,
    use_wandb=False,
    self_play=False,
    elo_tracker=None,
):
    """
    Run one training phase vs a single opponent with a single learn() call over
    N_ENVS parallel environments. Periodic checkpointing/evaluation happens
    in-process via callback.
    """
    agent_model_path = f"{agent_model_folder}/{agent_model_folder.split('/')[-1]}"

    resolved_opponent_path = None
    if opponent_agent_model_path:
        if os.path.exists(f"{opponent_agent_model_path}.zip"):
            resolved_opponent_path = opponent_agent_model_path
        else:
            resolved_opponent_path = f"{opponent_agent_model_path}/{opponent_agent_model_path.split('/')[-1]}"
        # Fail loudly in the main process rather than inside a worker
        if not os.path.exists(f"{resolved_opponent_path}.zip"):
            raise FileNotFoundError(f"Opponent model not found at '{resolved_opponent_path}.zip'")

    # Learner uses MCTS in every training phase; self-play also searches on both sides
    use_mcts = MCTS_SIMS_TRAIN > 0
    if use_mcts:
        agent.mcts_sims = MCTS_SIMS_TRAIN
        agent.mcts_c_puct = MCTS_C_PUCT
        agent.mcts_root_deterministic = False
        temperature = 0.0
    else:
        agent.mcts_sims = 0
        agent.mcts_root_deterministic = False
        temperature = SELF_PLAY_TEMPERATURE if self_play else 0.0

    if self_play:
        opponent_factory = _self_play_opponent_factory(
            resolved_opponent_path,
            mcts_sims=MCTS_SIMS_TRAIN if use_mcts else 0,
        )
    else:
        opponent_factory = type(opponent)

    # One env (and opponent) per worker process
    vec_env = SubprocVecEnv([
        _make_env_fn(opponent_factory, temperature, use_mcts=use_mcts)
        for _ in range(N_ENVS)
    ])

    agent.load(agent_model_path, env=vec_env)

    eval_callback = PeriodicEvaluationCallback(
        agent,
        opponent,
        vec_env,
        agent_model_folder,
        agent_model_path,
        opponent_reload_path=resolved_opponent_path if self_play else None,
        elo_tracker=elo_tracker,
    )
    callbacks = [eval_callback]
    if use_wandb:
        wandb.init(project="rl-chess960", sync_tensorboard=True)
        callbacks.append(WandbCallback())

    try:
        agent.train(total_timesteps, callback=callbacks)
    finally:
        # Checkpoint and log whatever is left since the last periodic checkpoint
        eval_callback.checkpoint(agent.model.num_timesteps)
        vec_env.close()
        if use_wandb:
            wandb.finish()

    final_timesteps = agent.model.num_timesteps
    if eval_callback.last_eval_at_timesteps != final_timesteps:
        run_benchmark_suite(
            agent,
            agent_model_folder,
            n_games=EVAL_N_GAMES,
            elo_tracker=elo_tracker,
        )

def handle_training(agent_class=rlAgent, config=[(RandomAgent, 0, None), (MinimaxAgent, 0, None), (rlAgent, 10000, "models/rl_agent")], model_path="models/rl_agent", use_wandb=True):
    """
    Handle the training loop, along with evaluation.
    """
    elo_tracker = EloTracker()
    agent_model_path = f"{model_path}/{model_path.split('/')[-1]}"

    # Create the agent once
    agent = agent_class(ChessEnvironment(RandomAgent()), n_steps=4096 // N_ENVS)
    if not os.path.exists(f"{agent_model_path}.zip"):
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
            elo_tracker=elo_tracker,
        )

        if self_play:
            elo_tracker = evaluate(RandomAgent(), MinimaxAgent(), n_games=10, tracker=elo_tracker)

        if hasattr(opponent_instance, 'close'):
            opponent_instance.close()

    os.makedirs(f"{model_path}/snapshots", exist_ok=True)
    snapshot_count = len(os.listdir(f"{model_path}/snapshots"))
    shutil.copy(
        f"{agent_model_path}.zip",
        f"{model_path}/snapshots/snapshot_{snapshot_count + 1}.zip"
    )


if __name__=="__main__":
    """
    Dynamic loop that selects training opponents based of performance.
    """


    while True:
        # Dynamic training designed to be ran and left, only plays opponent if it will learn from it, so no stockfish when it just looses every game etc
        eval_agent = rlAgent(ChessEnvironment(RandomAgent()))
        model_file = "models/rl_agent_v4/rl_agent_v4"
        if os.path.exists(f"{model_file}.zip"):
            eval_agent.load(model_file)
        else:
            print(f"No existing model at {model_file}.zip - evaluating fresh weights.")
        
        benchmark_scores = run_benchmark_suite(
            eval_agent,
            "models/rl_agent_v4",
            n_games=EVAL_N_GAMES,
            log_timesteps=0,
        )
        random_score = benchmark_scores[RandomAgent]
        minimax_score = benchmark_scores[MinimaxAgent]
        stockfish_score = benchmark_scores[StockfishAgent]
        
        # If more opponents then less time on self play for equal loop length
        # This also means that early on when it arguably needs more random/ minimax it does it more frequently due to less self play.
        opponents_added = sum([
            random_score <= 0.5,
            -1 <= minimax_score <= 0.6,
            stockfish_score > -0.8
        ])
        self_play_timesteps = 250000 - (50000 * opponents_added)
        
        main_config = [
            (rlAgent, self_play_timesteps, "models/rl_agent_v4"),
        ]
        
        if random_score <= 0.5:
            main_config.insert(0, (RandomAgent, 50000, None))
        
        if -1 <= minimax_score <= 0.6:
            main_config.insert(0, (MinimaxAgent, 50000, None))
        
        if stockfish_score > -0.8:
            main_config.append((StockfishAgent, 100000, None))
        
        handle_training(agent_class=rlAgent, config=main_config, use_wandb=False, model_path="models/rl_agent_v4")
