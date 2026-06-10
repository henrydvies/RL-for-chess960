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


## Testing a move object is returned

def test_returns_move(agent):
    """
    take_turn should return a chess.Move
    """
    board = chess.Board()
    move = agent.take_turn(board)
    assert isinstance(move, chess.Move)


## Testing the move is legal

def test_move_is_legal_start(agent):
    """
    Move should be legal from the starting position
    """
    board = chess.Board()
    move = agent.take_turn(board)
    assert move in board.legal_moves


def test_move_is_legal_midgame(agent):
    """
    Move should be legal from a midgame position
    """
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    move = agent.take_turn(board)
    assert move in board.legal_moves


def test_move_is_legal_endgame(agent):
    """
    Move should be legal from an endgame position
    """
    board = chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")
    move = agent.take_turn(board)
    assert move in board.legal_moves


## Testing agent never produces an illegal move across many calls

def test_never_illegal_many_calls(agent):
    """
    Agent should never return an illegal move across 200 calls from the starting position
    """
    board = chess.Board()
    for _ in range(200):
        assert agent.take_turn(board) in board.legal_moves


def test_never_illegal_chess960(agent):
    """
    Agent should never return an illegal move across 200 calls from a Chess960 starting position
    """
    board = chess.Board.from_chess960_pos(42)
    for _ in range(200):
        assert agent.take_turn(board) in board.legal_moves
