"""
Tests for MinimaxAgent in engines/minimax/minimax_agent.py
"""
import chess
import pytest
from engines.minimax.minimax_agent import MinimaxAgent


@pytest.fixture
def agent():
    """
    Shared MinimaxAgent instance at depth 3
    """
    return MinimaxAgent(depth=3)


@pytest.fixture
def agent_depth_1():
    """
    MinimaxAgent at depth 1 for fast deterministic tests
    """
    return MinimaxAgent(depth=1)

@pytest.fixture
def agent_depth_2():
    """
    MinimaxAgent at depth 2 for fast deterministic tests
    """
    return MinimaxAgent(depth=2)


## Testing action is a valid integer

def test_action_is_integer(agent):
    """
    Action returned should be a plain integer
    """
    board = chess.Board()
    action = agent.take_turn(board)
    assert isinstance(action, int)


def test_action_within_range(agent):
    """
    Action should be within valid range 0-4095
    """
    board = chess.Board()
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


## Testing agent picks best move

def test_captures_free_queen(agent_depth_1):
    """
    White pawn can capture a black queen — agent should always take it.
    Position: white pawn on d5, black queen on e6, white king on e1, black king on e8.
    """
    board = chess.Board("4k3/8/4q3/3P4/8/8/8/4K3 w - - 0 1")
    action = agent_depth_1.take_turn(board)
    from_square = action // 64
    to_square = action % 64
    move = chess.Move(from_square, to_square)
    # d5xe6 — pawn on d5 (square 35) captures queen on e6 (square 44)
    assert move == chess.Move(chess.D5, chess.E6)


def test_takes_free_rook(agent_depth_1):
    """
    White queen can capture an undefended black rook — agent should take it.
    Position: white queen on a1, black rook on a8, white king on e1, black king on e8.
    """
    board = chess.Board("r3k3/8/8/8/8/8/8/Q3K3 w - - 0 1")
    action = agent_depth_1.take_turn(board)
    from_square = action // 64
    to_square = action % 64
    move = chess.Move(from_square, to_square)
    # Qa1xa8
    assert move == chess.Move(chess.A1, chess.A8)


## Testing agent finds checkmate

def test_finds_mate_in_one_white(agent_depth_2):
    """
    White to move with a forced mate in one — agent should play it.
    Position: white queen on h6, white king on f6, black king on g8.
    Only legal mate: Qh6-g7#
    """
    board = chess.Board("6k1/8/6KQ/8/8/8/8/8 w - - 0 1")
    action = agent_depth_2.take_turn(board)
    from_square = action // 64
    to_square = action % 64
    move = chess.Move(from_square, to_square)
    # After the move, black should be in checkmate
    board.push(move)
    assert board.is_checkmate()


## Testing agent handles draw position

def test_handles_stalemate_position(agent_depth_1):
    """
    Agent should still return a legal move even in a position close to stalemate.
    Position: white king and queen vs black king with legal moves remaining.
    """
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    action = agent_depth_1.take_turn(board)
    from_square = action // 64
    to_square = action % 64
    move = chess.Move(from_square, to_square)
    assert move in board.legal_moves


## Testing agent plays as black

def test_action_is_legal_as_black(agent):
    """
    Agent should return a legal move when it is black's turn
    """
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4")
    action = agent.take_turn(board)
    from_square = action // 64
    to_square = action % 64
    move = chess.Move(from_square, to_square)
    assert move in board.legal_moves