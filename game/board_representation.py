"""
Tensor converstion: 8*8*12, as 8*8 board, 6 pieces * 2 colours = 12 total layers to represent pieces
"""
import numpy as np
import chess

def board_to_tensor(board):
    """
    Take in a board and convert to tensor representation
    """
    tensor_board = np.zeros((8, 8, 12), dtype=np.int8)
    
    # Loop over all squares converting piece to tensor array.
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        # Get type and colour
        piece_type = piece.piece_type
        piece_colour = piece.color
        
        # White pieces layers 0-5, black pieces layers 6-11
        layer = piece_type - 1 if piece_colour else piece_type - 1 + 6
        
        # Set the value to 1 on the tensor board
        tensor_board[chess.square_rank(square), chess.square_file(square), layer] = 1
    
    return tensor_board
        
        
            
            