"""
Basic minimax chess playing engine. 
"""
import chess
import random
from numpy import inf as np_inf
class MinimaxAgent:
    def __init__(self, depth=2):
        # Depth for move depth of the agent
        self.depth = depth
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.QUEEN: 9,
            chess.ROOK: 5,
        }
        
    def take_turn(self, board):
        # Determine whether to start max/ min
        is_maximising = True if board.turn == chess.WHITE else False

        # Get best move from _minimax
        return self._minimax(board, self.depth, is_maximising, -np_inf, np_inf)[0]
        
    def _minimax(self, board, depth, is_maximising, alpha, beta):
        """
        Recursively perform minimax algorithm to a defined depth, with alpha-beta pruning.
        """
        # Game-over check must come before the depth cutoff,
        # otherwise checkmate at the horizon is scored as material instead of mate.
        outcome = board.outcome()
        if outcome:
            if outcome.winner is None:
                return [None, 0]
            return [None, np_inf if outcome.winner == chess.WHITE else -np_inf]
        if depth == 0:
            return [None, self._evaluate(board)]
        
        # Loop through legal moves checking each ones evaluation affect
        best_case = [None, {False: np_inf, True: -np_inf}[is_maximising]] # [action, best score(based on max/min)]
        for move in self._ordered_moves(board):
            board.push(move)
            
            # Check if move maximising/ minimises better than current best
            evaluation_score = self._minimax(board, depth - 1, not(is_maximising), alpha, beta)[1] # The score
            board.pop()
            
            if is_maximising:
                if best_case[0] is None or evaluation_score > best_case[1]:
                    best_case = [move, evaluation_score]
                alpha = max(alpha, evaluation_score)
            else:
                if best_case[0] is None or evaluation_score < best_case[1]:
                    best_case = [move, evaluation_score]
                beta = min(beta, evaluation_score)
            
            # Alpha-beta cutoff
            if beta <= alpha:
                break
            
        return best_case
    
    def _ordered_moves(self, board):
        """
        Order moves for the search: shuffle for random tiebreaking between equal moves
        (stops the agent being deterministic and exploitable), then captures first
        which makes alpha-beta cutoffs happen much earlier.
        """
        moves = list(board.legal_moves)
        random.shuffle(moves)
        moves.sort(key=board.is_capture, reverse=True)
        return moves
                    

                
        
    def _evaluate(self, board):
        """
        Evaluate a chess board returning evaluation of white/ black.
        """
        pieces = board.piece_map().items()
        # Positive for white, negative for black
        eval_val = 0
        colour = {chess.WHITE: 1, chess.BLACK: -1}
        
        for _, piece in pieces:
            value = self.piece_values.get(piece.piece_type, 0)
            colour_weight = colour[piece.color]
            
            eval_val += colour_weight * value
            
        
        
        return eval_val
        