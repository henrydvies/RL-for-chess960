"""
Tests for rlAgent in engines/rl/rl_agent.py
"""
import chess
import numpy as np
import pytest
from unittest.mock import patch
from game.environment import ChessEnvironment
from engines.random.random_agent import RandomAgent
from engines.rl.rl_agent import rlAgent
from utils.action_masks import action_to_move


@pytest.fixture
def environment():
    """
    Shared ChessEnvironment instance with random opponent
    """
    return ChessEnvironment(opponent=RandomAgent())


@pytest.fixture
def agent(environment):
    """
    Shared rlAgent instance
    """
    return rlAgent(environment)


## Testing instantiation

def test_agent_instantiates(environment):
    """
    rlAgent should instantiate without errors
    """
    agent = rlAgent(environment)
    assert agent is not None


def test_agent_has_model(agent):
    """
    rlAgent should have a model attribute after instantiation
    """
    assert agent.model is not None


## Testing take_turn

def test_take_turn_returns_move(agent):
    """
    take_turn should return a chess.Move
    """
    board = chess.Board()
    move = agent.take_turn(board)
    assert isinstance(move, chess.Move)


def test_take_turn_returns_legal_move(agent):
    """
    take_turn should return a legal move after masking
    """
    board = chess.Board()
    move = agent.take_turn(board)
    assert move in board.legal_moves


def test_take_turn_legal_as_black(agent):
    """
    take_turn should return a legal move when playing black (mirrored frame)
    """
    board = chess.Board()
    board.push_san("e4")
    move = agent.take_turn(board)
    assert move in board.legal_moves


def test_take_turn_legal_chess960(agent):
    """
    take_turn should return a legal move from a Chess960 starting position
    """
    board = chess.Board.from_chess960_pos(42)
    move = agent.take_turn(board)
    assert move in board.legal_moves


## Testing take_turn with MCTS

def test_take_turn_n_sims_zero_skips_search(agent):
    """
    n_sims=0 should use greedy predict without calling MCTS
    """
    board = chess.Board()
    with patch("engines.rl.rl_agent.mcts_search") as mock_search:
        move = agent.take_turn(board, n_sims=0)
        mock_search.assert_not_called()
    assert move in board.legal_moves


def test_take_turn_mcts_returns_legal_move(agent):
    """
    take_turn with search should return a legal move
    """
    board = chess.Board()
    move = agent.take_turn(board, n_sims=8, root_deterministic=True)
    assert move in board.legal_moves


def test_take_turn_mcts_legal_as_black(agent):
    """
    MCTS take_turn should return a legal move when black is to move
    """
    board = chess.Board()
    board.push_san("e4")
    move = agent.take_turn(board, n_sims=8, root_deterministic=True)
    assert move in board.legal_moves


def test_take_turn_mcts_finds_mate_in_one(agent):
    """
    MCTS should find a forced mate with enough sims
    """
    board = chess.Board("6k1/8/6KQ/8/8/8/8/8 w - - 0 1")
    move = agent.take_turn(board, n_sims=50, root_deterministic=True)
    board.push(move)
    assert board.is_checkmate()


## Testing save and load

def test_save_creates_file(agent, tmp_path):
    """
    save should create a model file
    """
    path = str(tmp_path / "test_model")
    agent.save(model_path=path)
    assert (tmp_path / "test_model.zip").exists()


def test_load_restores_model(agent, tmp_path):
    """
    load should restore the model without errors
    """
    path = str(tmp_path / "test_model")
    agent.save(model_path=path)
    agent.load(path)
    assert agent.model is not None


def test_load_missing_file_raises(agent, tmp_path):
    """
    load should raise loudly when the file does not exist
    """
    with pytest.raises(FileNotFoundError):
        agent.load(str(tmp_path / "does_not_exist"))


## Testing get_policy_value

def test_get_policy_value_priors_sum_to_one(agent):
    """
    Priors over legal moves should sum to 1
    """
    board = chess.Board()
    priors, _ = agent.get_policy_value(board)
    assert abs(sum(priors.values()) - 1.0) < 1e-5


def test_get_policy_value_only_legal_actions(agent):
    """
    Priors should only be returned for legal actions
    """
    board = chess.Board()
    priors, _ = agent.get_policy_value(board)
    legal = {move for move in board.legal_moves}
    for action in priors:
        assert action_to_move(action, board) in legal


def test_get_policy_value_returns_scalar(agent):
    """
    Value should be a plain float
    """
    board = chess.Board()
    _, value = agent.get_policy_value(board)
    assert isinstance(value, float)


def test_get_policy_value_as_black(agent):
    """
    get_policy_value should work when black is to move
    """
    board = chess.Board()
    board.push_san("e4")
    priors, value = agent.get_policy_value(board)
    assert abs(sum(priors.values()) - 1.0) < 1e-5
    assert isinstance(value, float)


def test_get_policy_value_chess960(agent):
    """
    get_policy_value should work from a Chess960 starting position
    """
    board = chess.Board.from_chess960_pos(42)
    priors, value = agent.get_policy_value(board)
    assert len(priors) == len(list(board.legal_moves))
    assert abs(sum(priors.values()) - 1.0) < 1e-5
    assert isinstance(value, float)
