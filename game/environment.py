"""
Wrapper for python-chess representing the gym environment
"""
import chess
import gymnasium as gym
from random import randint
import numpy as np

class ChessEnvironment(gym.Env):
    """
    Class to represent chess gym environment, used for training. 
    """

    def __init__(self):
        # 960 position seed random by default to ensure always trained on random chess960 setup.
        pos_seed = randint(0, 959)
        self.board = chess.Board.from_chess960_pos(pos_seed)
        
        # Attribute to track game status
        self.game_over = False
        
        # All possible moves: 64 squares map to 64 squares: Hence 64*64 = 4096
        self.action_space = gym.spaces.Discrete(4096)
        
        # Set of valid data that can be recieved: 8*8 for the board, 12 layers for 6 piece types * 2 piece colours
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.int8)
    
    def reset(self):
        # Reset the chess board.
        pass
    
    def step(self):
        # Take the next step, making a move, returning new state/ reward.
        pass
    
    def render(self):
        # Render chess board
        pass
