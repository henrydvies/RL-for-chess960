"""
Random minimal-piece winning endgame positions for curriculum training.
"""
import random
import chess


def _kings_valid(white_king: chess.Square, black_king: chess.Square) -> bool:
    """
    Check if the kings are valid on the board.
    """
    wr, wf = chess.square_rank(white_king), chess.square_file(white_king)
    br, bf = chess.square_rank(black_king), chess.square_file(black_king)
    return max(abs(wr - br), abs(wf - bf)) > 1


def _build_endgame(
    white_king: chess.Square,
    black_king: chess.Square,
    extra_square: chess.Square,
    extra_piece: chess.PieceType,
    extra_colour: chess.Color,
    turn: chess.Color,
) -> chess.Board | None:
    """
    Build a chess board with the given kings and extra piece.
    """
    if not _kings_valid(white_king, black_king):
        return None
    if extra_square in (white_king, black_king):
        return None

    board = chess.Board(None)
    board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(extra_square, chess.Piece(extra_piece, extra_colour))
    board.turn = turn
    board.clear_stack()

    if not board.is_valid():
        return None
    if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
        return None
    return board


def _generate_with_piece(extra_piece: chess.PieceType, agent_colour: chess.Color, max_attempts=100) -> chess.Board:
    """
    Generate a random endgame position with the given piece type and agent colour.
    """
    for _ in range(max_attempts):
        winning_king = random.randint(chess.A1, chess.H8)
        losing_king = random.randint(chess.A1, chess.H8)
        extra_square = random.randint(chess.A1, chess.H8)

        if agent_colour == chess.WHITE:
            board = _build_endgame(
                winning_king, losing_king, extra_square, extra_piece, chess.WHITE, chess.WHITE
            )
        else:
            board = _build_endgame(
                losing_king, winning_king, extra_square, extra_piece, chess.BLACK, chess.BLACK
            )

        if board is not None:
            return board

    raise RuntimeError(f"Failed to generate endgame after {max_attempts} attempts")


def generate_kq_vs_k(agent_colour: chess.Color) -> chess.Board:
    """
    King + queen vs lone king, with the agent on the winning side.
    """
    return _generate_with_piece(chess.QUEEN, agent_colour)


def generate_kr_vs_k(agent_colour: chess.Color) -> chess.Board:
    """
    King + rook vs lone king, with the agent on the winning side.
    """
    return _generate_with_piece(chess.ROOK, agent_colour)


def generate_endgame(agent_colour: chess.Color) -> chess.Board:
    """
    Random KQ or KR vs K endgame with the agent holding the extra piece.
    """
    if random.random() < 0.5:
        return generate_kq_vs_k(agent_colour)
    return generate_kr_vs_k(agent_colour)
