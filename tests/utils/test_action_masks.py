"""
Tests for action_masks in utils/action_masks.py
"""
import chess
import numpy as np
import pytest
from utils.action_masks import mirror_action, mirror_square, action_masks

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

def test_mirror_square_rank_0_to_rank_7():
    """A1 (file 0, rank 0) should mirror to A8 (file 0, rank 7)"""
    assert mirror_square(chess.A1) == chess.A8


def test_mirror_square_rank_7_to_rank_0():
    """H8 (file 7, rank 7) should mirror to H1 (file 7, rank 0)"""
    assert mirror_square(chess.H8) == chess.H1


def test_mirror_square_middle_rank():
    """E4 (rank 3) should mirror to E5 (rank 4)"""
    assert mirror_square(chess.E4) == chess.E5


def test_mirror_square_preserves_file():
    """Mirroring should never change the file, only the rank"""
    for square in chess.SQUARES:
        original_file = chess.square_file(square)
        mirrored = mirror_square(square)
        assert chess.square_file(mirrored) == original_file


def test_mirror_square_is_involutive():
    """Mirroring twice should return the original square"""
    for square in chess.SQUARES:
        assert mirror_square(mirror_square(square)) == square


## mirror_action tests

def test_mirror_action_pawn_push():
    """e2-e4 (action) should mirror to e7-e5"""
    original = chess.E2 * 64 + chess.E4
    expected = chess.E7 * 64 + chess.E5
    assert mirror_action(original) == expected


def test_mirror_action_castling_kingside():
    """e1-g1 (white kingside castle) should mirror to e8-g8"""
    original = chess.E1 * 64 + chess.G1
    expected = chess.E8 * 64 + chess.G8
    assert mirror_action(original) == expected


def test_mirror_action_is_involutive():
    """Mirroring an action twice should return the original"""
    # Test a few specific actions
    test_actions = [
        chess.A1 * 64 + chess.A8,
        chess.E2 * 64 + chess.E4,
        chess.H1 * 64 + chess.H8,
        chess.D4 * 64 + chess.E5,
    ]
    for action in test_actions:
        assert mirror_action(mirror_action(action)) == action


def test_mirror_action_all_actions_involutive():
    """Any action 0-4095 should be involutive under mirroring"""
    for action in range(4096):
        assert mirror_action(mirror_action(action)) == action


## action_masks tests with mirroring

def test_action_masks_white_no_mirror():
    """Starting position with white to move: mask should match white's legal moves directly"""
    board = chess.Board()
    mask = action_masks(board)
    for move in board.legal_moves:
        action = move.from_square * 64 + move.to_square
        assert mask[action] == True


def test_action_masks_black_mirrored():
    """When black to move, mask should reflect mirrored actions"""
    board = chess.Board()
    board.push_san("e4")  # Now black to move
    mask = action_masks(board)
    for move in board.legal_moves:
        action = move.from_square * 64 + move.to_square
        mirrored_action = mirror_action(action)
        assert mask[mirrored_action] == True


def test_action_masks_correct_count():
    """Number of True values in mask should equal number of legal moves"""
    board = chess.Board()
    mask = action_masks(board)
    assert mask.sum() == len(list(board.legal_moves))
