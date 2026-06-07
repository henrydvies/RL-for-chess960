"""
Tensor conversion for chess board state: 8*8*20

Originally (v1_agent), 8*8*12 with 12 layers representing 6 pieces, 2 colours.

Now includes further layers to represent turn indicator, castling rights, en passant, repetition, and move count (to show when its late game).
Idea seen on AlphaZero input representation paper, in which they use 119 planes including this and more.

Originally network had to infer game state, like whose turn, castling, repetitions implicitly from move history within weight, now those can be used immediately, in theory speeding up training, and improving positional understanding late game.
"""
import numpy as np
import chess

def board_to_tensor(board):
    """
    Take in a board and convert to tensor representation
    """
    tensor_board = np.zeros((8, 8, 20), dtype=np.int8)
    
    # Loop over all squares converting piece to tensor array.
    # L0-11: Piece positions
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
        
    # After iteration add new planes: Turn indicator, White/black king/queenside castling rights, en passant square, repetition indicator, move count threshold
    
    # L12: Turn indicator
    if board.turn == chess.WHITE:
        tensor_board[:, :, 12] = 1

    # 13-16: Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor_board[:, :, 13] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor_board[:, :, 14] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor_board[:, :, 15] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor_board[:, :, 16] = 1

    # L17: En passant square
    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        tensor_board[rank, file, 17] = 1

    # L18: Repetition
    if board.is_repetition(count=2):
        tensor_board[:, :, 18] = 1

    # L19 Move count threshold (1s if past move 50)
    if board.fullmove_number >= 50:
        tensor_board[:, :, 19] = 1
    
    return tensor_board
        
        
            
            