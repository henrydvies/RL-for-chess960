"""
Tests for board_to_tensor function in board_representation.py
"""
import chess
import numpy as np
import pytest
from game.board_representation import board_to_tensor


def empty_board():
    """
    Helper to return empty board
    """
    return chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")


## Testing output shape

def test_output_shape():
    """
    Tensor should always be (8, 8, 20) regardless of board state
    """
    board = chess.Board()
    result = board_to_tensor(board)
    assert result.shape == (8, 8, 20)


def test_output_shape_empty_board():
    """
    Shape should be (8, 8, 20) even for an empty board
    """
    result = board_to_tensor(empty_board())
    assert result.shape == (8, 8, 20)

## Testing with white pieces

def test_white_pawn_correct_layer():
    """
    White pawn should appear in layer 0 (piece_type 1 - 1 = 0)
    """
    board = empty_board()
    board.set_piece_at(chess.A1, chess.Piece(chess.PAWN, chess.WHITE))
    result = board_to_tensor(board)

    # Layer 0 at rank 0, file 0 should be 1
    assert result[chess.square_rank(chess.A1), chess.square_file(chess.A1), 0] == 1

    # All other layers at that square should be 0
    for layer in range(1, 12):
        assert result[chess.square_rank(chess.A1), chess.square_file(chess.A1), layer] == 0


def test_white_queen_correct_layer():
    """White queen should appear in layer 4 (piece_type 5 - 1 = 4)"""
    board = empty_board()
    board.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.WHITE))
    result = board_to_tensor(board)

    assert result[chess.square_rank(chess.D4), chess.square_file(chess.D4), 4] == 1


## Testing with black pieces

def test_black_pawn_correct_layer():
    """
    Black pawn should appear in layer 6 (piece_type 1 - 1 + 6 = 6)
    """
    board = empty_board()
    board.set_piece_at(chess.H8, chess.Piece(chess.PAWN, chess.BLACK))
    result = board_to_tensor(board)

    assert result[chess.square_rank(chess.H8), chess.square_file(chess.H8), 6] == 1

    # All other layers at that square should be 0
    for layer in range(12):
        if layer != 6:
            assert result[chess.square_rank(chess.H8), chess.square_file(chess.H8), layer] == 0


def test_black_king_correct_layer():
    """
    Black king should appear in layer 11 (piece_type 6 - 1 + 6 = 11)
    """
    board = empty_board()
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    result = board_to_tensor(board)

    assert result[chess.square_rank(chess.E8), chess.square_file(chess.E8), 11] == 1


## Testing with white and black pieces

def test_mixed_colours_correct_layers():
    """
    White and black pieces on the same board should land in correct layers
    """
    board = empty_board()
    board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))   # layer 3
    board.set_piece_at(chess.H8, chess.Piece(chess.ROOK, chess.BLACK))   # layer 9
    result = board_to_tensor(board)

    assert result[chess.square_rank(chess.A1), chess.square_file(chess.A1), 3] == 1
    assert result[chess.square_rank(chess.H8), chess.square_file(chess.H8), 9] == 1


def test_mixed_colours_no_bleed():
    """
    A white piece should not appear in black layers and vice versa
    """
    board = empty_board()
    board.set_piece_at(chess.A1, chess.Piece(chess.KNIGHT, chess.WHITE))  # layer 1
    board.set_piece_at(chess.A2, chess.Piece(chess.KNIGHT, chess.BLACK))  # layer 7
    result = board_to_tensor(board)

    # White knight should not appear in black layers
    for layer in range(6, 12):
        assert result[chess.square_rank(chess.A1), chess.square_file(chess.A1), layer] == 0

    # Black knight should not appear in white layers
    for layer in range(0, 6):
        assert result[chess.square_rank(chess.A2), chess.square_file(chess.A2), layer] == 0


## Testing only 1 per square

def test_only_one_layer_set_per_square():
    """
    Each occupied square should have exactly one 1 across all 12 layers
    """
    board = empty_board()
    board.set_piece_at(chess.E4, chess.Piece(chess.BISHOP, chess.WHITE))
    result = board_to_tensor(board)

    rank, file = chess.square_rank(chess.E4), chess.square_file(chess.E4)
    assert result[rank, file, :12].sum() == 1