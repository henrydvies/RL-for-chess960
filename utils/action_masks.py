"""
Helper to apply mask
"""
import numpy as np
import chess

def action_masks(board):
    """
    Apply action mask
    """
    mask = np.zeros(4096, dtype=bool)
    for move in board.legal_moves:
        action = move.from_square * 64 + move.to_square
        if board.turn == chess.BLACK:
            action = mirror_action(action)
        mask[action] = True
    return mask

def mirror_square(square):
    """Vertically flip a square"""
    return chess.square(chess.square_file(square), 7 - chess.square_rank(square))

def mirror_action(action):
    """Mirror an action's from and to squares."""
    from_sq = action // 64
    to_sq = action % 64
    return mirror_square(from_sq) * 64 + mirror_square(to_sq)