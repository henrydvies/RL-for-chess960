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
        return random_choice(legal_moves)
        
        
        