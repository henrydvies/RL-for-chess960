"""
Tests for MCTS search in engines/mcts/search.py
"""
import chess
import numpy as np
import pytest
from utils.action_masks import action_masks, action_to_move, move_to_action
from engines.mcts.search import _terminal_value, mcts_search


def _uniform_policy_value(board):
    """
    Mock network: uniform priors over legal moves, neutral leaf value.
    """
    mask = action_masks(board)
    legal = np.flatnonzero(mask)
    prior = 1.0 / len(legal)
    priors = {int(action): prior for action in legal}
    return priors, 0.0


def _mate_chasing_policy_value(board):
    """
    Mock network that up-weights moves which immediately win for the side to move.
    """
    mask = action_masks(board)
    legal = np.flatnonzero(mask)
    mover = board.turn
    mate_actions = []
    for action in legal:
        move = action_to_move(int(action), board)
        board.push(move)
        outcome = board.outcome(claim_draw=True)
        board.pop()
        if outcome is not None and outcome.winner == mover:
            mate_actions.append(int(action))

    priors = {}
    if mate_actions:
        mate_prior = 0.9 / len(mate_actions)
        other_count = len(legal) - len(mate_actions)
        other_prior = 0.1 / other_count if other_count else 0.0
        for action in legal:
            priors[int(action)] = mate_prior if int(action) in mate_actions else other_prior
    else:
        prior = 1.0 / len(legal)
        priors = {int(action): prior for action in legal}

    total = sum(priors.values())
    priors = {action: p / total for action, p in priors.items()}
    return priors, 0.0


## Testing terminal values

def test_terminal_value_side_to_move_lost():
    """
    Side to move is checkmated — value should be -1
    """
    board = chess.Board("6k1/8/6KQ/8/8/8/8/8 w - - 0 1")
    board.push(chess.Move(chess.H6, chess.G7))
    assert board.is_checkmate()
    assert _terminal_value(board) == -1.0


def test_terminal_value_draw():
    """
    Stalemate should score 0
    """
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    assert _terminal_value(board) == 0.0


## Testing search returns legal actions

def test_mcts_returns_legal_action_start():
    """
    Root action should decode to a legal move from the starting position
    """
    board = chess.Board()
    action = mcts_search(board, _uniform_policy_value, n_sims=8, root_deterministic=True)
    move = action_to_move(action, board)
    assert move in board.legal_moves


def test_mcts_returns_legal_action_chess960():
    """
    Search should respect Chess960 legality
    """
    board = chess.Board.from_chess960_pos(42)
    action = mcts_search(board, _uniform_policy_value, n_sims=8, root_deterministic=True)
    move = action_to_move(action, board)
    assert move in board.legal_moves


def test_mcts_invalid_n_sims_raises():
    """
    n_sims must be positive
    """
    board = chess.Board()
    with pytest.raises(ValueError):
        mcts_search(board, _uniform_policy_value, n_sims=0)


## Testing backup prefers winning move

def test_mcts_finds_mate_in_one():
    """
    With enough sims MCTS should pick the mate in one even with a neutral value head
    """
    board = chess.Board("6k1/8/6KQ/8/8/8/8/8 w - - 0 1")
    action = mcts_search(
        board,
        _uniform_policy_value,
        n_sims=50,
        c_puct=1.25,
        root_deterministic=True,
    )
    move = action_to_move(action, board)
    board.push(move)
    assert board.is_checkmate()


def test_mcts_finds_mate_with_policy_hint():
    """
    Biased priors plus search should reliably choose the mating move
    """
    board = chess.Board("6k1/8/6KQ/8/8/8/8/8 w - - 0 1")
    action = mcts_search(
        board,
        _mate_chasing_policy_value,
        n_sims=10,
        root_deterministic=True,
    )
    assert action_to_move(action, board) == chess.Move(chess.H6, chess.G7)


## Testing root move selection modes

def test_mcts_deterministic_picks_mate():
    """
    Deterministic root should pick the mating move in a forced line
    """
    board = chess.Board("6k1/8/6KQ/8/8/8/8/8 w - - 0 1")
    mate_action = move_to_action(chess.Move(chess.H6, chess.G7))
    action = mcts_search(
        board,
        _uniform_policy_value,
        n_sims=40,
        root_deterministic=True,
    )
    assert action == mate_action


def test_mcts_stochastic_root_returns_legal_move():
    """
    Stochastic root should still return a legal action
    """
    board = chess.Board()
    rng = np.random.default_rng(0)
    action = mcts_search(
        board,
        _uniform_policy_value,
        n_sims=8,
        root_deterministic=False,
        rng=rng,
    )
    assert action_to_move(action, board) in board.legal_moves
