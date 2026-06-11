"""
Tests for game/endgame_positions.py
"""
import chess
from game.endgame_positions import generate_endgame, generate_kq_vs_k, generate_kr_vs_k


def _piece_counts(board):
    white = sum(1 for sq in chess.SQUARES if board.piece_at(sq) and board.color_at(sq) == chess.WHITE)
    black = sum(1 for sq in chess.SQUARES if board.piece_at(sq) and board.color_at(sq) == chess.BLACK)
    return white, black


def test_generate_endgame_is_valid():
    for colour in (chess.WHITE, chess.BLACK):
        for _ in range(50):
            board = generate_endgame(colour)
            assert board.is_valid()
            assert len(board.piece_map()) == 3
            assert not board.is_checkmate()
            assert not board.is_stalemate()


def test_agent_holds_winning_material():
    for colour in (chess.WHITE, chess.BLACK):
        board = generate_endgame(colour)
        agent_pieces = [
            board.piece_at(sq)
            for sq in chess.SQUARES
            if board.piece_at(sq) and board.color_at(sq) == colour
        ]
        assert len(agent_pieces) == 2
        types = {p.piece_type for p in agent_pieces}
        assert chess.KING in types
        assert types & {chess.QUEEN, chess.ROOK}


def test_opponent_is_lone_king():
    board = generate_kq_vs_k(chess.WHITE)
    assert _piece_counts(board) == (2, 1)
    board = generate_kr_vs_k(chess.BLACK)
    assert _piece_counts(board) == (1, 2)


def test_agent_to_move():
    board = generate_endgame(chess.WHITE)
    assert board.turn == chess.WHITE
    board = generate_endgame(chess.BLACK)
    assert board.turn == chess.BLACK
