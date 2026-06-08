"""
Wrapper for python-chess representing the gym environment
"""
import chess
import gymnasium as gym
from random import randint
import numpy as np
from .board_representation import board_to_tensor
from utils.action_masks import action_masks as action_masks_helper
from utils.action_masks import mirror_action, mirror_square
import random
from engines.random.random_agent import RandomAgent


def pos_seed():
    # Random seed for chess960 setup
    return randint(0, 959)

class ChessEnvironment(gym.Env):
    """
    Class to represent chess gym environment, used for training. 
    """

    def __init__(self, opponent, temperature = 0.1):
        # 960 position seed random by default to ensure always trained on random chess960 setup.
        self.board = chess.Board.from_chess960_pos(pos_seed())
        
        # Attribute to track game status
        self.game_over = False
        
        # All possible moves: 64 squares map to 64 squares: Hence 64*64 = 4096
        self.action_space = gym.spaces.Discrete(4096)
        
        # Set of valid data
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8, 8, 20), dtype=np.int8)
        
        # Player colour, white only for now
        self.player_colour = chess.WHITE
        
        # Colour tracking
        self.white_episodes = 0
        self.black_episodes = 0
        
        # Set opponent agent
        self.opponent = opponent
        
        # Set chance opponent moves randomly for exploration of moves
        self.temperature = temperature
        
        # Random agent instance
        self.random_agent = RandomAgent()
        
        # For temperature decay
        self.step_counter = 0
    
    def action_masks(self):
        """
        Returns a boolean mask of legal actions for the current board state.
        """
        mask = action_masks_helper(self.board)
        return mask
    
    def reset(self, seed=None, options=None):
        # Reset the chess board.
        self.board = chess.Board.from_chess960_pos(pos_seed())
        self.game_over = False
        self.player_colour = random.choice([chess.WHITE, chess.BLACK])
        
        if self.player_colour == chess.WHITE:
            self.white_episodes += 1
        else:
            self.black_episodes += 1
        
        # Handle Black side
        if self.player_colour == chess.BLACK:
            # Push opponent move
            self.board.push(self._convert_to_move(self.opponent.take_turn(self.board)))
        
        return board_to_tensor(self.board), {}
    
    def step(self, action):
        """
        Takes the next step, then takes opponents move.
        """
        if self.board.turn == chess.BLACK:
            action = mirror_action(int(action))
    

        move = self._convert_to_move(action)
        self.step_counter += 1
        # Handle illegal move
        if move not in self.board.legal_moves:
            return (board_to_tensor(self.board), -1, True, False, {})
        
        # Make the move
        self.board.push(move)
        
        # Calculate reward and if game is over
        outcome = self.board.outcome()
        reward = self._handle_outcome(outcome)

        if not(self.game_over):     
            # Take opponent move, with chance for random move, defaulted to 10%
            # This also applies vs random/ minimax/ stockfish, intend to add only vs self play.
            if random.random() < self.temperature:
                opponent_action = self.random_agent.take_turn(self.board)
            else:
                opponent_action = self.opponent.take_turn(self.board)
            
            # Decay temperature
            self.temperature = max(0.05, 0.2 - (self.step_counter / 40000) * 0.15) # Linear decay, goes from 0.1 -> 0.05 in 100k steps.
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
        Converts action to a chess move object. Also handle promotions.
        """
        from_square = action // 64
        to_square = action % 64
        
        # Make the move object
        move = chess.Move(from_square, to_square)
        
        # Check for promotions, only queen promote for now
        piece = self.board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(to_square) == {chess.WHITE: 7, chess.BLACK: 0}[self.board.turn]:
            move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
        return move
    def _handle_outcome(self, outcome):
        """
        Helper to calculate reward based of outcome, and handle game over
        """
        
        if outcome:
            # Handle game over
            if outcome.winner == self.player_colour:
                reward = 1 + max(0, (200 - self.board.fullmove_number) * 0.001)
            elif outcome.winner is None:
                reward = -0.1 # Slight penalty on draws
            else:
                reward = -1
            
            # Game over 
            self.game_over = True
        else:
            # Game continues
            reward = 0
        return reward