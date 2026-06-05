"""
Tests for evaluate function in evaluation/evaluator.py
"""
import pytest
from unittest.mock import patch, MagicMock
from evaluation.evaluator import evaluate, play_single_game
from evaluation.elo_tracker import EloTracker
from engines.random.random_agent import RandomAgent


@pytest.fixture
def tracker():
    """
    Fresh EloTracker with known starting ratings
    """
    return EloTracker(elo_map={"rlAgent": 600, "RandomAgent": 600})


## Testing evaluate updates the tracker

def test_evaluate_updates_rl_agent_elo(tracker):
    """
    evaluate should change rlAgent Elo after playing games
    """
    random_opponent = RandomAgent()

    # Mock rl_agent with a take_turn that returns a legal move
    mock_rl = MagicMock()
    mock_rl.__class__.__name__ = "rlAgent"
    mock_rl.take_turn.return_value = 0  # Will trigger illegal move path

    before = tracker.elo_map["rlAgent"]
    evaluate(mock_rl, random_opponent, n_games=3, tracker=tracker)
    # Elo should have changed
    assert tracker.elo_map["rlAgent"] != before or tracker.elo_map["RandomAgent"] != before


def test_evaluate_returns_tracker(tracker):
    """
    evaluate should return the updated tracker
    """
    mock_rl = MagicMock()
    mock_rl.__class__.__name__ = "rlAgent"
    mock_rl.take_turn.return_value = 0

    result = evaluate(mock_rl, RandomAgent(), n_games=3, tracker=tracker)
    assert isinstance(result, EloTracker)


## Testing correct number of games played

def test_evaluate_plays_correct_number_of_games(tracker):
    """
    evaluate should play exactly n_games games
    """
    mock_rl = MagicMock()
    mock_rl.__class__.__name__ = "rlAgent"
    mock_rl.take_turn.return_value = 0

    with patch("evaluation.evaluator.play_single_game") as mock_game:
        mock_game.return_value = True
        evaluate(mock_rl, RandomAgent(), n_games=5, tracker=tracker)
        assert mock_game.call_count == 5


def test_evaluate_plays_correct_number_large(tracker):
    """
    evaluate should play exactly n_games for a larger number
    """
    mock_rl = MagicMock()
    mock_rl.__class__.__name__ = "rlAgent"
    mock_rl.take_turn.return_value = 0

    with patch("evaluation.evaluator.play_single_game") as mock_game:
        mock_game.return_value = None
        evaluate(mock_rl, RandomAgent(), n_games=20, tracker=tracker)
        assert mock_game.call_count == 20