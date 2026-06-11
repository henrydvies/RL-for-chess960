"""
Wrapper for python-chess representing the gym environment
"""
import chess
import gymnasium as gym
from random import randint
import numpy as np
from .board_representation import board_to_tensor
from .endgame_positions import generate_endgame
from utils.action_masks import action_masks as action_masks_helper
from utils.action_masks import action_to_move, ACTION_SPACE_SIZE
import random
from engines.random.random_agent import RandomAgent


def pos_seed():
    # Random seed for chess960 setup
    return randint(0, 959)

class ChessEnvironment(gym.Env):
    """
    Class to represent chess gym environment, used for training. 
    """

    def __init__(
        self,
        opponent,
        temperature=0.0,
        temperature_decay_steps=40000,
        endgame_probability=0.0,
        endgame_probability_final=0.0,
        endgame_decay_episodes=0,
    ):
        # 960 position seed random by default to ensure always trained on random chess960 setup.
        self.board = chess.Board.from_chess960_pos(pos_seed())
        
        # Attribute to track game status
        self.game_over = False
        
        # AlphaZero-style encoding: 64 from-squares x 73 movement planes = 4672
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)
        
        # Set of valid data
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8, 8, 20), dtype=np.int8)
        
        # Player colour, white only for now
        self.player_colour = chess.WHITE
        
        # Colour tracking
        self.white_episodes = 0
        self.black_episodes = 0
        self.endgame_episodes = 0
        
        # Set opponent agent
        self.opponent = opponent
        
        # Chance opponent moves randomly for exploration.
        self.initial_temperature = temperature
        self.temperature = temperature
        # Per-env steps over which temperature decays to 0.05. 
        self.temperature_decay_steps = temperature_decay_steps

        # Endgame curriculum: start high while the agent draw-farms, decay as mating improves
        self.endgame_probability = endgame_probability
        self.endgame_probability_final = endgame_probability_final
        self.endgame_decay_episodes = endgame_decay_episodes
        
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

    def _current_endgame_probability(self):
        if self.endgame_probability <= 0:
            return 0.0
        if self.endgame_decay_episodes <= 0 or self.endgame_probability_final <= 0:
            return self.endgame_probability
        episodes = self.white_episodes + self.black_episodes
        progress = min(1.0, episodes / self.endgame_decay_episodes)
        return self.endgame_probability + progress * (
            self.endgame_probability_final - self.endgame_probability
        )
    
    def reset(self, seed=None, options=None):
        self.game_over = False
        self.step_counter = 0
        self.player_colour = random.choice([chess.WHITE, chess.BLACK])

        if random.random() < self._current_endgame_probability():
            self.board = generate_endgame(self.player_colour)
            self.endgame_episodes += 1
        else:
            self.board = chess.Board.from_chess960_pos(pos_seed())
        
        if self.player_colour == chess.WHITE:
            self.white_episodes += 1
        else:
            self.black_episodes += 1
        
        # Handle Black side
        if self.player_colour == chess.BLACK:
            # Push opponent move
            self.board.push(self.opponent.take_turn(self.board))
        
        return board_to_tensor(self.board, self.player_colour), {}
    
    def step(self, action):
        """
        Takes the next step, then takes opponents move.
        """
        move = action_to_move(int(action), self.board)
        self.step_counter += 1
        # Handle illegal move
        if move not in self.board.legal_moves:
            return (board_to_tensor(self.board, self.player_colour), -1, True, False, {})
        
        # Make the move
        self.board.push(move)
        
        # Calculate reward and if game is over (claim_draw so threefold/fifty-move end the game)
        outcome = self.board.outcome(claim_draw=True)
        reward = self._handle_outcome(outcome)

        if not(self.game_over):     
            # Take opponent move, with chance for random move (temperature > 0 in self-play only)
            if self.temperature > 0 and random.random() < self.temperature:
                opponent_move = self.random_agent.take_turn(self.board)
            else:
                opponent_move = self.opponent.take_turn(self.board)
            
            # Linear decay from initial_temperature to 0.05 over temperature_decay_steps
            if self.initial_temperature > 0:
                self.temperature = max(0.05, self.initial_temperature - (self.step_counter / self.temperature_decay_steps) * (self.initial_temperature - 0.05))
            # Make opponent move
            self.board.push(opponent_move)
            
            # Check if that results in game over
            outcome = self.board.outcome(claim_draw=True)
            reward = self._handle_outcome(outcome)            
        
        # Calculate reward
        return(board_to_tensor(self.board, self.player_colour), reward, self.game_over, False, {})
        
    
    def render(self):
        # Render chess board
        print(self.board)

    def reload_opponent(self, model_path):
        """
        Reload the opponent model from disk.
        """
        if hasattr(self.opponent, "load"):
            self.opponent.load(model_path)

    def close(self):
        # Shut down opponent subprocesses when a worker terminates
        if hasattr(self.opponent, "close"):
            self.opponent.close()

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