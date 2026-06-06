"""
Tests for TrainingLogger in evaluation/training_logger.py
"""
import json
import pytest
import os
from evaluation.training_logger import TrainingLogger
from engines.random.random_agent import RandomAgent
from engines.minimax.minimax_agent import MinimaxAgent
from engines.rl.rl_agent import rlAgent
from game.environment import ChessEnvironment


@pytest.fixture
def tmp_logger(tmp_path):
    """
    TrainingLogger with a temporary log file path
    """
    path = str(tmp_path / "training_log.json")
    return TrainingLogger(log_file_path=path)


## Testing all three opponent types update the correct summary field

def test_random_opponent_updates_correct_summary(tmp_logger):
    """
    Training vs RandomAgent should update timesteps_vs_random
    """
    tmp_logger.update_log(500, RandomAgent(), None)
    assert tmp_logger.logs["summary"]["timesteps_vs_random"] == 500


def test_minimax_opponent_updates_correct_summary(tmp_logger):
    """
    Training vs MinimaxAgent should update timesteps_vs_minimax
    """
    tmp_logger.update_log(500, MinimaxAgent(), None)
    assert tmp_logger.logs["summary"]["timesteps_vs_minimax"] == 500


def test_self_play_updates_correct_summary(tmp_logger, tmp_path):
    """
    Training vs rlAgent should update timesteps_self_play
    """
    temp_env = ChessEnvironment(opponent=RandomAgent())
    opponent = rlAgent(temp_env)
    tmp_logger.update_log(500, opponent, None)
    assert tmp_logger.logs["summary"]["timesteps_self_play"] == 500


## Testing summary accumulates not replaces

def test_summary_accumulates_across_updates(tmp_logger):
    """
    Multiple updates should accumulate total_timesteps, not replace
    """
    tmp_logger.update_log(500, RandomAgent(), None)
    tmp_logger.update_log(300, RandomAgent(), None)
    assert tmp_logger.logs["summary"]["total_timesteps"] == 800
    assert tmp_logger.logs["summary"]["timesteps_vs_random"] == 800


def test_mixed_opponents_accumulate_separately(tmp_logger):
    """
    Updates vs different opponents should accumulate in separate fields
    """
    tmp_logger.update_log(500, RandomAgent(), None)
    tmp_logger.update_log(300, MinimaxAgent(), None)
    assert tmp_logger.logs["summary"]["timesteps_vs_random"] == 500
    assert tmp_logger.logs["summary"]["timesteps_vs_minimax"] == 300
    assert tmp_logger.logs["summary"]["total_timesteps"] == 800


## Testing log output shape

def test_run_entry_has_correct_keys(tmp_logger):
    """
    Each run entry should have timestamp, opponent, timesteps, ep_rew_mean
    """
    tmp_logger.update_log(100, RandomAgent(), -0.5)
    run = tmp_logger.logs["runs"][0]
    assert "timestamp" in run
    assert "opponent" in run
    assert "timesteps" in run
    assert "ep_rew_mean" in run


def test_run_entry_values_correct(tmp_logger):
    """
    Run entry values should match what was passed in
    """
    tmp_logger.update_log(200, MinimaxAgent(), -0.2)
    run = tmp_logger.logs["runs"][0]
    assert run["timesteps"] == 200
    assert run["opponent"] == "MinimaxAgent"
    assert run["ep_rew_mean"] == -0.2


def test_multiple_runs_appended(tmp_logger):
    """
    Multiple updates should append multiple run entries
    """
    tmp_logger.update_log(100, RandomAgent(), None)
    tmp_logger.update_log(200, MinimaxAgent(), None)
    assert len(tmp_logger.logs["runs"]) == 2


## Testing save and load roundtrip

def test_save_creates_file(tmp_logger):
    """
    Save should create the log file
    """
    tmp_logger.update_log(100, RandomAgent(), None)
    tmp_logger.save()
    assert os.path.exists(tmp_logger.log_file_path)


def test_save_load_roundtrip(tmp_logger):
    """
    Save then load should produce identical logs
    """
    tmp_logger.update_log(500, RandomAgent(), -0.3)
    tmp_logger.save()

    loaded = TrainingLogger(log_file_path=tmp_logger.log_file_path)
    assert loaded.logs["summary"]["total_timesteps"] == 500
    assert len(loaded.logs["runs"]) == 1
    assert loaded.logs["runs"][0]["ep_rew_mean"] == -0.3


def test_ep_rew_mean_none_serialises(tmp_logger):
    """
    None ep_rew_mean should serialise to null in JSON and load back as None
    """
    tmp_logger.update_log(100, RandomAgent(), None)
    tmp_logger.save()

    loaded = TrainingLogger(log_file_path=tmp_logger.log_file_path)
    assert loaded.logs["runs"][0]["ep_rew_mean"] is None