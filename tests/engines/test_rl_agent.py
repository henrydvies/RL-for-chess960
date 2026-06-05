"""
Tests for rlAgent in engines/rl/rl_agent.py
"""
import chess
import numpy as np
import pytest
from game.environment import ChessEnvironment
from engines.random.random_agent import RandomAgent
from engines.rl.rl_agent import rlAgent


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

def test_take_turn_returns_integer(agent):
    """
    take_turn should return an integer action
    """
    board = chess.Board()
    action = agent.take_turn(board)
    assert isinstance(action, (int, np.integer))


def test_take_turn_within_range(agent):
    """
    take_turn should return an action within valid range 0-4095
    """
    board = chess.Board()
    action = agent.take_turn(board)
    assert 0 <= action <= 4095


def test_take_turn_returns_legal_move(agent):
    """
    take_turn should return a legal move after masking
    """
    board = chess.Board()
    action = agent.take_turn(board)
    from_square = int(action) // 64
    to_square = int(action) % 64
    move = chess.Move(from_square, to_square)
    assert move in board.legal_moves


## Testing save and load

def test_save_creates_file(agent, tmp_path):
    """
    save should create a model file
    """
    path = str(tmp_path / "test_model")
    agent.save(model_folder=path)
    assert (tmp_path / "test_model.zip").exists()


def test_load_restores_model(agent, tmp_path):
    """
    load should restore the model without errors
    """
    path = str(tmp_path / "test_model")
    agent.save(model_folder=path)
    agent.load(path)
    assert agent.model is not None