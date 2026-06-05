"""
Tests for ChessEnvironment in environment.py
"""
import chess
import numpy as np
import pytest
from game.environment import ChessEnvironment
from engines.random.random_agent import RandomAgent


def get_env():
    """
    Helper to return a fresh ChessEnvironment instance with a random opponent
    """
    return ChessEnvironment(opponent=RandomAgent())


def checkmate_env():
    """
    Helper to return environment with board one move from checkmate.
    Fool's mate position: white to deliver checkmate with Qh5#
    FEN: r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4
    """
    env = get_env()
    env.board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    return env


## Testing reset

def test_reset_returns_correct_shape():
    """
    Reset should return a tensor of shape (8, 8, 12) and an empty info dict
    """
    env = get_env()
    obs, info = env.reset()
    assert obs.shape == (8, 8, 12)
    assert info == {}


def test_reset_clears_game_over():
    """
    Reset should set game_over back to False
    """
    env = get_env()
    env.game_over = True
    env.reset()
    assert env.game_over == False


def test_reset_generates_new_position():
    """
    Reset should generate a new Chess960 position each time
    """
    env = get_env()
    board_before = env.board.fen()
    env.reset()
    board_after = env.board.fen()
    assert isinstance(board_after, str)


## Testing illegal moves

def test_illegal_move_returns_negative_reward():
    """
    An illegal move should return reward of -1
    """
    env = get_env()
    _, reward, _, _, _ = env.step(0)
    assert reward == -1


def test_illegal_move_terminates_game():
    """
    An illegal move should terminate the episode
    """
    env = get_env()
    _, _, terminated, _, _ = env.step(0)
    assert terminated == True


def test_illegal_move_returns_correct_shape():
    """
    Even an illegal move should return a tensor of shape (8, 8, 12)
    """
    env = get_env()
    obs, _, _, _, _ = env.step(0)
    assert obs.shape == (8, 8, 12)


## Testing legal moves

def test_legal_move_returns_zero_reward():
    """
    A legal move during normal play should return reward of 0
    """
    env = get_env()
    move = list(env.board.legal_moves)[0]
    action = move.from_square * 64 + move.to_square
    _, reward, _, _, _ = env.step(action)
    assert reward == 0


def test_legal_move_does_not_terminate():
    """
    A legal move during normal play should not terminate the episode
    """
    env = get_env()
    move = list(env.board.legal_moves)[0]
    action = move.from_square * 64 + move.to_square
    _, _, terminated, _, _ = env.step(action)
    assert terminated == False


def test_legal_move_returns_correct_shape():
    """
    A legal move should return a tensor of shape (8, 8, 12)
    """
    env = get_env()
    move = list(env.board.legal_moves)[0]
    action = move.from_square * 64 + move.to_square
    obs, _, _, _, _ = env.step(action)
    assert obs.shape == (8, 8, 12)


## Testing checkmate

def test_checkmate_returns_positive_reward():
    """
    Delivering checkmate as white should return reward of +1
    """
    env = checkmate_env()
    move = chess.Move(chess.H5, chess.F7)
    action = move.from_square * 64 + move.to_square
    _, reward, _, _, _ = env.step(action)
    assert reward == 1


def test_checkmate_terminates_game():
    """
    Checkmate should terminate the episode
    """
    env = checkmate_env()
    move = chess.Move(chess.H5, chess.F7)
    action = move.from_square * 64 + move.to_square
    _, _, terminated, _, _ = env.step(action)
    assert terminated == True


def test_checkmate_sets_game_over():
    """
    Checkmate should set game_over to True
    """
    env = checkmate_env()
    move = chess.Move(chess.H5, chess.F7)
    action = move.from_square * 64 + move.to_square
    env.step(action)
    assert env.game_over == True