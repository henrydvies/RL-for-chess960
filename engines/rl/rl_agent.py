"""
Reinforcement learning based engine.
"""
from engines.rl.policy_network import PolicyNetwork
from game.board_representation import board_to_tensor
from utils.action_masks import action_masks as action_masks_helper
from numpy import newaxis as new_axis
from sb3_contrib import MaskablePPO
import os
import chess
from utils.action_masks import mirror_action
class rlAgent:
    def __init__(self, environment):
        """
        Create model
        """
        policy_kwargs = dict(
            features_extractor_class=PolicyNetwork,
            features_extractor_kwargs=dict(features_dim=256)
        )
        self.model = MaskablePPO(
            "CnnPolicy", 
            environment, 
            policy_kwargs=policy_kwargs, 
            verbose=0,
            gamma=0.995, # Discount factor, longer chess games are devalued with defaults.
            ent_coef=0.01, # entropy bonus, forces exploration
            n_steps=4096, # More samples per update - not applied to agent_v1
        )
        
    def train(self, total_timesteps, callback=None):
        """
        Train model over total_timesteps iterations
        """
        # Learn for timesteps time
        self.model.learn(total_timesteps, callback=callback)
        
    
    def save(self, model_path=None):
        """
        Save state
        """
        if model_path is None:
            model_path = "models/rl_agent"
        self.model.save(model_path)
    
    def load(self, model_path):
        """
        Load model
        """
        if os.path.exists(model_path + ".zip"):
            self.model = self.model.load(model_path)
            self.model.gamme = 0.995
            self.model.ent_coef = 0.01
        
    def take_turn(self, board):
        """
        Takes a turn by converting board to 8*8*12 format then passing in
        """
        tensor_board = board_to_tensor(board)
        masks = action_masks_helper(board)
        # Get action 
        action = self.model.predict(tensor_board[new_axis], action_masks=masks)[0][0]#
        
        if board.turn == chess.BLACK:
            action = mirror_action(action)
        
        return action
        