"""
Engine that picks a random legal move. 
"""
from random import choice as random_choice
class RandomAgent:
    """
    Represents a random move playing chess engine. 
    """
    def __init__(self):
        pass
    
    def take_turn(self, board):
        # Take a turn on the board
        legal_moves = list(board.legal_moves)
        
        # Pick a random move
        move = random_choice(legal_moves)
        
        # Convert move to single int
        action = (move.from_square * 64) + move.to_square
        
        return action
        
        
        