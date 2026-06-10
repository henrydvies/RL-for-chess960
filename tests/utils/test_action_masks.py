"""
Tests for the 8x8x73 action encoding in utils/action_masks.py
"""
import random

import chess
import numpy as np
import pytest
from utils.action_masks import (
    ACTION_SPACE_SIZE,
    action_masks,
    action_to_move,
    mirror_move,
    mirror_square,
    move_to_action,
)

## Testing output shape and type

def test_mask_shape():
    """
    Mask should always be of shape (4672,)
    """
    board = chess.Board()
    mask = action_masks(board)
    assert mask.shape == (ACTION_SPACE_SIZE,)


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
        assert mask[move_to_action(move)] == True


def test_number_of_true_values():
    """
    Number of True values should match number of legal moves (one action per move,
    including distinct actions per promotion piece)
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


## mirror_square tests

def test_mirror_square_rank_0_to_rank_7():
    """A1 (file 0, rank 0) should mirror to A8 (file 0, rank 7)"""
    assert mirror_square(chess.A1) == chess.A8


def test_mirror_square_rank_7_to_rank_0():
    """H8 (file 7, rank 7) should mirror to H1 (file 7, rank 0)"""
    assert mirror_square(chess.H8) == chess.H1


def test_mirror_square_preserves_file():
    """Mirroring should never change the file, only the rank"""
    for square in chess.SQUARES:
        assert chess.square_file(mirror_square(square)) == chess.square_file(square)


def test_mirror_square_is_involutive():
    """Mirroring twice should return the original square"""
    for square in chess.SQUARES:
        assert mirror_square(mirror_square(square)) == square


## mirror_move tests

def test_mirror_move_pawn_push():
    """e2-e4 should mirror to e7-e5"""
    assert mirror_move(chess.Move(chess.E2, chess.E4)) == chess.Move(chess.E7, chess.E5)


def test_mirror_move_preserves_promotion():
    """Mirroring should keep the promotion piece"""
    move = chess.Move(chess.E7, chess.E8, promotion=chess.KNIGHT)
    mirrored = mirror_move(move)
    assert mirrored == chess.Move(chess.E2, chess.E1, promotion=chess.KNIGHT)


def test_mirror_move_is_involutive():
    """Mirroring a move twice should return the original"""
    move = chess.Move(chess.D4, chess.E5)
    assert mirror_move(mirror_move(move)) == move


## Encode/decode roundtrip

def test_encode_decode_roundtrip_white():
    """Every legal white move should encode then decode back to itself"""
    board = chess.Board()
    for move in board.legal_moves:
        action = move_to_action(move)
        assert 0 <= action < ACTION_SPACE_SIZE
        assert action_to_move(action, board) == move


def test_encode_decode_roundtrip_black_mirrored():
    """Black moves are encoded in the mirrored frame and must decode back"""
    board = chess.Board()
    board.push_san("e4")  # Now black to move
    for move in board.legal_moves:
        action = move_to_action(mirror_move(move))
        assert action_to_move(action, board) == move


def test_encode_decode_roundtrip_random_games():
    """
    Property test: across random Chess960 games, every legal move for either
    colour must roundtrip through the encoding exactly.
    """
    rng = random.Random(0)
    for _ in range(4):
        board = chess.Board.from_chess960_pos(rng.randint(0, 959))
        for _ in range(80):
            if board.is_game_over():
                break
            for move in board.legal_moves:
                framed = mirror_move(move) if board.turn == chess.BLACK else move
                action = move_to_action(framed)
                assert action_to_move(action, board) == move
            board.push(rng.choice(list(board.legal_moves)))


## Promotions

def test_promotions_have_distinct_actions():
    """
    All four promotion pieces should map to distinct actions and decode back
    with the right promotion piece.
    """
    board = chess.Board("7k/4P3/8/8/8/8/8/4K3 w - - 0 1")
    promo_moves = [m for m in board.legal_moves if m.promotion]
    assert len(promo_moves) == 4  # Q, R, B, N
    actions = {move_to_action(m) for m in promo_moves}
    assert len(actions) == 4
    for move in promo_moves:
        assert action_to_move(move_to_action(move), board) == move


def test_black_underpromotion_roundtrip():
    """Black underpromotions must roundtrip through the mirrored frame"""
    board = chess.Board("4k3/8/8/8/8/8/4p3/6K1 b - - 0 1")
    promo_moves = [m for m in board.legal_moves if m.promotion]
    assert len(promo_moves) == 4
    for move in promo_moves:
        action = move_to_action(mirror_move(move))
        assert action_to_move(action, board) == move


def test_queen_promotion_attached_automatically():
    """A queen-plane pawn move to the last rank should decode with promotion=QUEEN"""
    board = chess.Board("7k/4P3/8/8/8/8/8/4K3 w - - 0 1")
    queen_promo = chess.Move(chess.E7, chess.E8, promotion=chess.QUEEN)
    decoded = action_to_move(move_to_action(queen_promo), board)
    assert decoded.promotion == chess.QUEEN


## Decode safety

def test_off_board_action_decodes_to_null_move():
    """An action whose geometry leaves the board should decode to a null move"""
    board = chess.Board()
    # From h1 (square 7), queen-move East distance 1 goes off the board.
    # Plane: East is direction index 2, distance 1 -> plane 2*7+0 = 14
    action = 7 * 73 + 14
    move = action_to_move(action, board)
    assert move == chess.Move.null()
    assert move not in board.legal_moves
