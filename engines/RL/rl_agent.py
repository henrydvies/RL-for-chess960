"""
Reinforcement learning based engine.
"""
from stable_baselines3 import PPO
from engines.rl.policy_network import PolicyNetwork
from game.board_representation import board_to_tensor
from numpy import newaxis as new_axis
class rlAgent:
    def __init__(self, environment):
        """
        Create model
        """
        policy_kwargs = dict(
            features_extractor_class=PolicyNetwork,
            features_extractor_kwargs=dict(features_dim=256)
        )
        self.model = PPO("CnnPolicy", environment, policy_kwargs=policy_kwargs)
    
    def train(self, total_timesteps):
        """
        Train model over total_timesteps iterations
        """
        # Learn for timesteps time
        self.model.learn(total_timesteps)
        
    
    def save(self, model_folder="models/rl_agent"):
        """
        Save state
        """
        self.model.save(model_folder)
    
    def load(self, model_path):
        """
        Load model
        """
        self.model = self.model.load(model_path)
        
    def take_turn(self, board):
        """
        Takes a turn by converting board to 8*8*12 format then passing in
        """
        tensor_board = board_to_tensor(board)
        # Get action 
        action = self.model.predict(tensor_board[new_axis])[0] 
        
        return action
        