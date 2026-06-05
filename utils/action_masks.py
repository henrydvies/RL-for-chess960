"""
Helper to apply mask
"""
import numpy as np

def action_masks(board):
    """
    Apply action mask
    """
    mask = np.zeros(4096, dtype=bool)
    for move in board.legal_moves:
        action = move.from_square * 64 + move.to_square
        mask[action] = True
    return mask