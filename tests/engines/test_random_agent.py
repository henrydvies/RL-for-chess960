"""
Tests for RandomAgent in engines/random/random_agent.py
"""
import chess
import pytest
from engines.random.random_agent import RandomAgent


@pytest.fixture
def agent():
    """
    Shared RandomAgent instance across tests
    """
    return RandomAgent()


## Testing action is a valid integer

def test_action_is_integer(agent):
    """
    Action returned should be a plain integer
    """
    board = chess.Board()
    action = agent.take_turn(board)
    assert isinstance(action, int)


def test_action_within_range_start(agent):
    """
    Action should be within valid range 0-4095 from starting position
    """
    board = chess.Board()
    action = agent.take_turn(board)
    assert 0 <= action <= 4095


def test_action_within_range_midgame(agent):
    """
    Action should be within valid range 0-4095 from a midgame position
    """
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    action = agent.take_turn(board)
    assert 0 <= action <= 4095


## Testing action corresponds to a legal move

def test_action_is_legal_start(agent):
    """
    Action should decode to a legal move from the starting position
    """
    board = chess.Board()
    action = agent.take_turn(board)
    from_square = action // 64
    to_square = action % 64
    move = chess.Move(from_square, to_square)
    assert move in board.legal_moves


def test_action_is_legal_midgame(agent):
    """
    Action should decode to a legal move from a midgame position
    """
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    action = agent.take_turn(board)
    from_square = action // 64
    to_square = action % 64
    move = chess.Move(from_square, to_square)
    assert move in board.legal_moves


def test_action_is_legal_endgame(agent):
    """
    Action should decode to a legal move from an endgame position
    """
    board = chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")
    action = agent.take_turn(board)
    from_square = action // 64
    to_square = action % 64
    move = chess.Move(from_square, to_square)
    assert move in board.legal_moves


## Testing agent never produces an illegal move across many calls

def test_never_illegal_many_calls(agent):
    """
    Agent should never return an illegal move across 200 calls from the starting position
    """
    board = chess.Board()
    for _ in range(200):
        action = agent.take_turn(board)
        from_square = action // 64
        to_square = action % 64
        move = chess.Move(from_square, to_square)
        assert move in board.legal_moves


def test_never_illegal_chess960(agent):
    """
    Agent should never return an illegal move across 200 calls from a Chess960 starting position
    """
    board = chess.Board.from_chess960_pos(42)
    for _ in range(200):
        action = agent.take_turn(board)
        from_square = action // 64
        to_square = action % 64
        move = chess.Move(from_square, to_square)
        assert move in board.legal_moves