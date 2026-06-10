"""
8*8*73 action encoding and legal move masking.
"""
import numpy as np
import chess

N_PLANES = 73
ACTION_SPACE_SIZE = 64 * N_PLANES  # 4672

# (rank delta, file delta) unit vectors: N, NE, E, SE, S, SW, W, NW
QUEEN_DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
KNIGHT_DELTAS = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
_KNIGHT_PLANE = {delta: i for i, delta in enumerate(KNIGHT_DELTAS)}
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def mirror_square(square):
    """
    Vertically flip a square
    """
    return chess.square(chess.square_file(square), 7 - chess.square_rank(square))


def mirror_move(move):
    """
    Vertically flip a move's squares.
    """
    return chess.Move(mirror_square(move.from_square), mirror_square(move.to_square), promotion=move.promotion)


def move_to_plane(move):
    """
    Map a current-player-frame move to its movement plane.
    """
    dr = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    df = chess.square_file(move.to_square) - chess.square_file(move.from_square)

    if move.promotion is not None and move.promotion != chess.QUEEN:
        return 64 + UNDERPROMOTION_PIECES.index(move.promotion) * 3 + (df + 1)

    if (dr, df) in _KNIGHT_PLANE:
        return 56 + _KNIGHT_PLANE[(dr, df)]

    direction = ((dr > 0) - (dr < 0), (df > 0) - (df < 0))
    distance = max(abs(dr), abs(df))
    return QUEEN_DIRECTIONS.index(direction) * 7 + (distance - 1)


def move_to_action(move):
    """
    Encode a current-player-frame move as an action integer.
    """
    return move.from_square * N_PLANES + move_to_plane(move)


def action_masks(board):
    """
    Boolean mask of legal actions, in the current player's frame.
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for move in board.legal_moves:
        if board.turn == chess.BLACK:
            move = mirror_move(move)
        mask[move_to_action(move)] = True
    return mask


def action_to_move(action, board):
    """
    Decode plane to a move.
    """
    from_square = action // N_PLANES
    plane = action % N_PLANES
    
    promotion = None
    if plane >= 64:
        underpromotion = plane - 64
        promotion = UNDERPROMOTION_PIECES[underpromotion // 3]
        dr, df = 1, (underpromotion % 3) - 1
    elif plane >= 56:
        dr, df = KNIGHT_DELTAS[plane - 56]
    else:
        direction = QUEEN_DIRECTIONS[plane // 7]
        distance = (plane % 7) + 1
        dr, df = direction[0] * distance, direction[1] * distance

    to_rank = chess.square_rank(from_square) + dr
    to_file = chess.square_file(from_square) + df
    if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
        return chess.Move.null()

    move = chess.Move(from_square, chess.square(to_file, to_rank), promotion=promotion)
    if board.turn == chess.BLACK:
        move = mirror_move(move)

    if move.promotion is None:
        piece = board.piece_at(move.from_square)
        last_rank = 7 if board.turn == chess.WHITE else 0
        if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(move.to_square) == last_rank:
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    return move
