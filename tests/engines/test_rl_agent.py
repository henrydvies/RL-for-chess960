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
