"""
Tests for MCTS eval wiring in evaluation/evaluator.py
"""
import chess
from unittest.mock import MagicMock, patch

from evaluation.evaluator import play_single_game, _take_turn
from engines.rl.rl_agent import rlAgent


def test_take_turn_passes_n_sims_to_rl_agent():
    """
    _take_turn should forward n_sims to rlAgent
    """
    agent = MagicMock(spec=rlAgent)
    board = chess.Board()
    _take_turn(agent, board, n_sims=12, root_deterministic=True)
    agent.take_turn.assert_called_once_with(
        board,
        n_sims=12,
        c_puct=1.25,
        root_deterministic=True,
    )


def test_take_turn_ignores_n_sims_for_non_rl_agent():
    """
    Non-rl agents should receive a plain take_turn call
    """
    from engines.random.random_agent import RandomAgent

    agent = RandomAgent()
    board = chess.Board()
    move = _take_turn(agent, board, n_sims=25)
    assert move in board.legal_moves


def test_play_single_game_forwards_mcts_kwargs():
    """
    play_single_game should pass n_sims through to rlAgent moves
    """
    white = MagicMock(spec=rlAgent)
    black = MagicMock(spec=rlAgent)
    legal = chess.Move.from_uci("e2e4")
    white.take_turn.return_value = legal
    black.take_turn.return_value = legal

    with patch("evaluation.evaluator.random.randint", return_value=0):
        play_single_game(
            white,
            black,
            white_n_sims=7,
            black_n_sims=9,
            root_deterministic=True,
        )

    assert white.take_turn.call_args.kwargs["n_sims"] == 7
    assert black.take_turn.call_args.kwargs["n_sims"] == 9
