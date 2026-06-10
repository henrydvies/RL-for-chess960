"""
Reinforcement learning based engine.
"""
from engines.rl.policy_network import PolicyNetwork, ChessPolicy
from game.board_representation import board_to_tensor
from utils.action_masks import action_masks as action_masks_helper
from utils.action_masks import action_to_move
from numpy import newaxis as new_axis
from sb3_contrib import MaskablePPO
import os
class rlAgent:
    def __init__(self, environment, n_steps=4096, device="auto"):
        """
        Create model.
        n_steps is per environment: with N parallel envs pass 4096 // N to keep
        the same total samples per PPO update.
        device="cpu" is used for opponent copies in vectorised workers, so they
        don't each allocate a CUDA context on the training GPU.
        """
        policy_kwargs = dict(
            features_extractor_class=PolicyNetwork,
            net_arch=[],  # heads attach directly to the trunk's spatial features
        )
        self.model = MaskablePPO(
            ChessPolicy, 
            environment, 
            policy_kwargs=policy_kwargs, 
            verbose=0,
            gamma=0.995, # Discount factor, longer chess games are devalued with defaults.
            ent_coef=0.01, # entropy bonus, forces exploration
            n_steps=n_steps, # Samples per env per update - not applied to agent_v1
            device=device,
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
    
    def load(self, model_path, env=None):
        """
        Load model. Raises loudly if the file is missing.
        Pass env so buffer built for right number of environemnts
        """
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(
                f"No model found at '{model_path}.zip' - refusing to continue with randomly initialised weights. "
                "Check the path, or skip the load explicitly if a fresh model is intended."
            )
        # custom_objects overrides hyperparameters baked into older saves
        self.model = MaskablePPO.load(
            model_path,
            env=env,
            device=self.model.device,
            custom_objects={"gamma": 0.995, "ent_coef": 0.01, "n_steps": self.model.n_steps},
        )
        
    def take_turn(self, board):
        """
        Takes a turn: board tensor in, chess.Move out.
        """
        tensor_board = board_to_tensor(board)
        masks = action_masks_helper(board)
        # Get action 
        action = self.model.predict(tensor_board[new_axis], action_masks=masks)[0][0]
        
        # Decode network-frame action to a real-board move (handles mirroring and promotion)
        return action_to_move(int(action), board)
        