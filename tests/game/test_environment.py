"""
Tests for ChessEnvironment in environment.py
"""
import chess
import numpy as np
import pytest
from unittest.mock import patch
from game.environment import ChessEnvironment
from engines.random.random_agent import RandomAgent
from utils.action_masks import move_to_action


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
    Reset should return a tensor of shape (8, 8, 20) and an empty info dict
    """
    env = get_env()
    obs, info = env.reset()
    assert obs.shape == (8, 8, 20)
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


def test_endgame_curriculum_forces_minimal_position():
    """
    With probability 1.0, reset should start from a KQ/KR vs K endgame
    """
    env = ChessEnvironment(opponent=RandomAgent(), endgame_probability=1.0)
    env.reset()
    assert len(env.board.piece_map()) == 3
    assert env.endgame_episodes == 1


def test_endgame_curriculum_resets_step_counter():
    env = ChessEnvironment(opponent=RandomAgent(), endgame_probability=1.0)
    env.step_counter = 50
    env.reset()
    assert env.step_counter == 0


## Testing MCTS disables temperature

def test_mcts_disables_temperature_random_moves():
    """
    use_mcts=True should never take random opponent moves even at high temperature
    """
    class TrackingOpponent:
        def take_turn(self, board):
            return list(board.legal_moves)[0]

    env = ChessEnvironment(TrackingOpponent(), temperature=1.0, use_mcts=True)
    env.reset()
    move = list(env.board.legal_moves)[0]
    with patch.object(env.random_agent, "take_turn") as mock_random:
        env.step(move_to_action(move))
        mock_random.assert_not_called()


def test_mcts_skips_temperature_decay():
    """
    use_mcts=True should leave temperature unchanged across steps
    """
    env = ChessEnvironment(RandomAgent(), temperature=0.2, use_mcts=True)
    env.reset()
    start_temp = env.temperature
    move = list(env.board.legal_moves)[0]
    env.step(move_to_action(move))
    assert env.temperature == start_temp


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
    Even an illegal move should return a tensor of shape (8, 8, 20)
    """
    env = get_env()
    obs, _, _, _, _ = env.step(0)
    assert obs.shape == (8, 8, 20)


## Testing legal moves

def test_legal_move_returns_zero_reward():
    """
    A legal move during normal play should return reward of 0
    """
    env = get_env()
    move = list(env.board.legal_moves)[0]
    action = move_to_action(move)
    _, reward, _, _, _ = env.step(action)
    assert reward == 0


def test_legal_move_does_not_terminate():
    """
    A legal move during normal play should not terminate the episode
    """
    env = get_env()
    move = list(env.board.legal_moves)[0]
    action = move_to_action(move)
    _, _, terminated, _, _ = env.step(action)
    assert terminated == False


def test_legal_move_returns_correct_shape():
    """
    A legal move should return a tensor of shape (8, 8, 20)
    """
    env = get_env()
    move = list(env.board.legal_moves)[0]
    action = move_to_action(move)
    obs, _, _, _, _ = env.step(action)
    assert obs.shape == (8, 8, 20)


## Testing checkmate

def test_checkmate_returns_positive_reward():
    """
    Delivering checkmate as white should return reward of +1
    """
    env = checkmate_env()
    action = move_to_action(chess.Move(chess.H5, chess.F7))
    _, reward, _, _, _ = env.step(action)
    assert reward >= 1


def test_checkmate_terminates_game():
    """
    Checkmate should terminate the episode
    """
    env = checkmate_env()
    action = move_to_action(chess.Move(chess.H5, chess.F7))
    _, _, terminated, _, _ = env.step(action)
    assert terminated == True


def test_checkmate_sets_game_over():
    """
    Checkmate should set game_over to True
    """
    env = checkmate_env()
    action = move_to_action(chess.Move(chess.H5, chess.F7))
    env.step(action)
    assert env.game_over == True