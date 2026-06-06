"""
Holds what is needed to track model logs in a clean way.
"""
import json
from engines.minimax.minimax_agent import MinimaxAgent
from engines.random.random_agent import RandomAgent
from engines.rl.rl_agent import rlAgent
from datetime import datetime

class TrainingLogger:
    """
    Used to update logs
    """
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.logs = self.load()
        
    def update_log(self, timesteps, opponent, ep_rew_mean=None):
        """
        Update summary and append a per run log
        """
        # Update summary
        new_summary = self.logs["summary"]
        new_summary["total_timesteps"] += timesteps
        opponent_type = {MinimaxAgent: "timesteps_vs_minimax", RandomAgent: "timesteps_vs_random", rlAgent: "timesteps_self_play"}[type(opponent)]
        new_summary[opponent_type] += timesteps
        
        self.logs["summary"] = new_summary
        
        # Append per run log
        self.logs["runs"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "opponent": opponent.__class__.__name__,
                "timesteps": timesteps,
                "ep_rew_mean": ep_rew_mean
            }
        )
        
    def save(self):
        """
        Save current log state to json
        """
        try:
            with open(self.log_file_path, 'w') as f: json.dump(self.logs, f)
        except:
            print(f"Saving log to {self.log_file_path} failed.")
        
    def load(self):
        """
        Load existing logs, return none if none found
        """
        try:
            with open(self.log_file_path, 'r') as f: logs = json.load(f)
        except:
            logs = {
                "summary": {
                    "total_timesteps": 0,
                    "timesteps_vs_random": 0,
                    "timesteps_vs_minimax": 0,
                    "timesteps_self_play": 0
                },
                "runs": []
            }
            
        return logs