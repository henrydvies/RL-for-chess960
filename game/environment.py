"""
Wrapper for python-chess representing the gym environment
"""
# Dependencies
import chess
import gymnasium as gym
from random import randint
import numpy as np

# Modules
from .board_representation import board_to_tensor


def pos_seed():
    # Random seed for chess960 setup
    return randint(0, 959)

class ChessEnvironment(gym.Env):
    """
    Class to represent chess gym environment, used for training. 
    """

    def __init__(self, opponent):
        # 960 position seed random by default to ensure always trained on random chess960 setup.
        self.board = chess.Board.from_chess960_pos(pos_seed())
        
        # Attribute to track game status
        self.game_over = False
        
        # All possible moves: 64 squares map to 64 squares: Hence 64*64 = 4096
        self.action_space = gym.spaces.Discrete(4096)
        
        # Set of valid data that can be recieved: 8*8 for the board, 12 layers for 6 piece types * 2 piece colours
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.int8)
        
        # Player colour, white only for now
        self.player_colour = chess.WHITE
        
        # Set opponent agent
        self.opponent = opponent
    
    def reset(self, seed=None, options=None):
        # Reset the chess board.
        self.board = chess.Board.from_chess960_pos(pos_seed())
        self.game_over = False
        
        return board_to_tensor(self.board), {}
    
    def step(self, action):
        """
        Takes the next step, then takes opponents move.
        """
        move = self._convert_to_move(action)
        
        # Handle illegal move
        if move not in self.board.legal_moves:
            return (board_to_tensor(self.board), -1, True, False, {})
        
        # Make the move
        self.board.push(move)
        
        # Calculate reward and if game is over
        outcome = self.board.outcome()
        reward = self._handle_outcome(outcome)

        if not(self.game_over):     
            # Take opponent move
            opponent_action = self.opponent.take_turn(self.board)
            
            # Make opponent move
            opponent_move = self._convert_to_move(opponent_action)
            self.board.push(opponent_move)
            
            # Check if that results in game over
            outcome = self.board.outcome()
            reward = self._handle_outcome(outcome)            
        
        # Calculate reward
        return(board_to_tensor(self.board), reward, self.game_over, False, {})
        
    
    def render(self):
        # Render chess board
        print(self.board)

    def _convert_to_move(self, action):
        """
        Converts action to a chess move object.
        """
        from_square = action // 64
        to_square = action % 64
        
        # Make the move object
        move = chess.Move(from_square, to_square)
        
        return move
    def _handle_outcome(self, outcome):
        """
        Helper to calculate reward based of outcome, and handle game over
        """
        
        if outcome:
            # Handle game over
            if outcome.winner == self.player_colour:
                reward = 1
            elif outcome.winner is None:
                reward = 0
            else:
                reward = -1
            
            # Game over 
            self.game_over = True
        else:
            # Game continues
            reward = 0
        return reward