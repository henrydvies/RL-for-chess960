"""
Tests for unified benchmark evaluation in engines/rl/train.py
"""
from unittest.mock import MagicMock, patch

from engines.minimax.minimax_agent import MinimaxAgent
from engines.random.random_agent import RandomAgent
from engines.stockfish.stockfish_agent import StockfishAgent
from engines.rl import train as train_module
from engines.rl.train import run_benchmark_suite, _update_elo_from_game
from evaluation.elo_tracker import EloTracker


def test_run_benchmark_suite_returns_scores_for_curriculum_opponents():
    """
    Curriculum eval should score Random, Minimax, and Stockfish in one pass.
    """
    agent = MagicMock()
    with patch.object(train_module, "get_ep_rew_mean", side_effect=[0.2, -0.4, -0.9]) as mock_eval:
        scores = run_benchmark_suite(
            agent,
            "models/rl_agent_test",
            n_games=10,
            include_self_play=False,
            log_timesteps=0,
        )

    assert scores[RandomAgent] == 0.2
    assert scores[MinimaxAgent] == -0.4
    assert scores[StockfishAgent] == -0.9
    assert mock_eval.call_count == 3


def test_run_training_skips_end_eval_when_callback_just_ran():
    """
    End-of-phase eval should not duplicate a periodic eval at the same timestep.
    """
    agent = MagicMock()
    agent.model = MagicMock()
    agent.model.device = "cpu"
    agent.model.num_timesteps = 50000

    eval_callback = MagicMock()
    eval_callback.last_eval_at_timesteps = 50000
    eval_callback.checkpoint = MagicMock()

    with patch.object(train_module, "SubprocVecEnv") as mock_vec_env, patch.object(
        train_module, "PeriodicEvaluationCallback", return_value=eval_callback
    ), patch.object(train_module, "run_benchmark_suite") as mock_suite, patch.object(
        agent, "load"
    ), patch.object(
        agent, "train"
    ):
        mock_vec_env.return_value.close = MagicMock()
        train_module.run_training(
            agent=agent,
            opponent=RandomAgent(),
            agent_model_folder="models/rl_agent_test",
            total_timesteps=50000,
            self_play=False,
        )

    mock_suite.assert_not_called()


def test_run_training_runs_end_eval_when_no_recent_callback_eval():
    """
    End-of-phase eval should still run if the callback has not evaluated yet.
    """
    agent = MagicMock()
    agent.model = MagicMock()
    agent.model.device = "cpu"
    agent.model.num_timesteps = 50000

    eval_callback = MagicMock()
    eval_callback.last_eval_at_timesteps = None
    eval_callback.checkpoint = MagicMock()

    with patch.object(train_module, "SubprocVecEnv") as mock_vec_env, patch.object(
        train_module, "PeriodicEvaluationCallback", return_value=eval_callback
    ), patch.object(train_module, "run_benchmark_suite") as mock_suite, patch.object(
        agent, "load"
    ), patch.object(
        agent, "train"
    ):
        mock_vec_env.return_value.close = MagicMock()
        train_module.run_training(
            agent=agent,
            opponent=RandomAgent(),
            agent_model_folder="models/rl_agent_test",
            total_timesteps=50000,
            self_play=False,
        )

    mock_suite.assert_called_once()


def test_update_elo_from_game_when_agent_plays_black():
    """
    Alternating-colour benchmark games should update Elo from the agent's perspective.
    """
    class StubAgent:
        pass

    tracker = EloTracker(elo_map={"StubAgent": 600, "RandomAgent": 600})
    agent = StubAgent()
    opponent = RandomAgent()

    _update_elo_from_game(tracker, agent, opponent, False, agent_played_white=False)

    assert tracker.elo_map["StubAgent"] > 600
