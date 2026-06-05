"""
Tests for action_masks in utils/action_masks.py
"""
import chess
import numpy as np
import pytest
from utils.action_masks import action_masks


## Testing output shape and type

def test_mask_shape():
    """
    Mask should always be of shape (4096,)
    """
    board = chess.Board()
    mask = action_masks(board)
    assert mask.shape == (4096,)


def test_mask_is_boolean():
    """
    Mask should be a boolean numpy array
    """
    board = chess.Board()
    mask = action_masks(board)
    assert mask.dtype == bool


## Testing legal moves are marked True

def test_legal_moves_are_true():
    """
    Every legal move should be True in the mask
    """
    board = chess.Board()
    mask = action_masks(board)
    for move in board.legal_moves:
        action = move.from_square * 64 + move.to_square
        assert mask[action] == True


def test_illegal_moves_are_false():
    """
    Action 0 (a1 to a1) should always be False
    """
    board = chess.Board()
    mask = action_masks(board)
    assert mask[0] == False


def test_number_of_true_values():
    """
    Number of True values should match number of legal moves
    """
    board = chess.Board()
    mask = action_masks(board)
    assert mask.sum() == len(list(board.legal_moves))


## Testing with different board states

def test_mask_midgame_position():
    """
    Mask should correctly reflect legal moves in a midgame position
    """
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    mask = action_masks(board)
    assert mask.sum() == len(list(board.legal_moves))


def test_mask_chess960_position():
    """
    Mask should correctly reflect legal moves in a Chess960 starting position
    """
    board = chess.Board.from_chess960_pos(42)
    mask = action_masks(board)
    assert mask.sum() == len(list(board.legal_moves))