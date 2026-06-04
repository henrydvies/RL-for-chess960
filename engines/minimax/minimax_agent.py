"""
Basic minimax chess playing engine. 
"""
import chess
from numpy import inf as np_inf
class MinimaxAgent:
    def __init__(self, depth=3):
        # Depth for move depth of the agent
        self.depth = depth
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.KING: 100000,
            chess.QUEEN: 8,
            chess.ROOK: 3,
        }
        
    def take_turn(self, board):
        pass
    
    def _minimax(self, board, depth, is_maximising):
        """
        Recursively perform minimax algorithm to a defined depth
        """
        # Base case of depth 0, or game over
        if depth == 0:
            return [None, self._evaluate(board)]
        if board.outcome: 
            if board.is_checkmate():
                return [None, -np_inf if board.turn == chess.WHITE else np_inf]
            return [None, 0]
        
        # Loop through legal moves checking each ones evaluation affect
        best_case = [None, {False: np_inf, True: -np_inf}[is_maximising]] # [action, best score(based on max/min)]
        for move in list(board.legal_moves):
            board.push(move)
            
            # Check if move maximising/ minimises better than current best
            evaluation_score = self._minimax(board, depth - 1, not(is_maximising))[1] # The score
            if is_maximising:
                if evaluation_score > best_case[1]:
                    best_case = [move, evaluation_score]
            else:
                if evaluation_score < best_case[1]:
                    best_case = [move, evaluation_score]
            # Remove move after testing it
            board.pop()
            
        return best_case
                    

                
        
    def _evaluate(self, board):
        """
        Evaluate a chess board returning evaluation of white/ black.
        """
        pieces = board.piece_map().items()
        # Positive for white, negative for black
        eval_val = 0
        colour = {chess.WHITE: 1, chess.BLACK: -1}
        
        for _, piece in pieces:
            value = self.piece_values[piece.piece_type]
            colour_weight = colour[piece.color]
            
            eval_val += colour_weight * value
            
        
        
        return eval_val
        