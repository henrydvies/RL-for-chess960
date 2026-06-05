"""
For tracking elo of agents
"""
import json

class EloTracker:
    """
    Represents the elo tracker
    """
    def __init__(self, elo_map=None,elo_tracking_path="elo_tracking/elo_tracker.json"):
        self.elo_map = elo_map
        self.elo_tracking_path = elo_tracking_path
        # If elo data present use it.
        elo_data = self.load()
        if elo_data:
            self.elo_map = elo_data
            
    def update(self, agent_one, agent_two, outcome):
        """
        Updates the elo based of the outcome.
        """
        # Get ratings
        agent_one_elo = self.elo_map[agent_one]
        agent_two_elo = self.elo_map[agent_two]
        
        # Determine outcome value
        agent_one_outcome_value = {True: 1, False: 0, None: 0.5}[outcome]
        agent_two_outcome_value = {True: 0, False: 1, None: 0.5}[outcome]
        
        
        # Determine new elos
        new_agent_one_elo = self._calculate_elo(agent_two_elo, agent_one_elo, agent_one_outcome_value)
        new_agent_two_elo = self._calculate_elo(agent_one_elo, agent_two_elo, agent_two_outcome_value)
        
        # Update elo map
        self.elo_map[agent_one] = new_agent_one_elo
        self.elo_map[agent_two] = new_agent_two_elo
        
    def _calculate_elo(self, opponent_rating, agent_rating, actual_score):
        """
        Helper to use elo formula: 
        E = 1 / (1 + 10^((opponent_rating - your_rating) / 400))
        Actual is Win: 1, Draw: 0.5, Loss: 0
        """
        k = 32
        expected_score = 1 / (1 + 10**((opponent_rating - agent_rating) / 400))
        new_elo = agent_rating + k * (actual_score - expected_score)
        
        return new_elo
        
    def save(self):
        """
        Save updated elo ratings
        """
        try:
            with open(self.elo_tracking_path, "w") as f:
                json.dump(self.elo_map, f)
        except:
            print("Failed to write new elo dict. ")
    
    def load(self):
        """
        Load elo ratings from tracker folder
        """
        try:
            with open(self.elo_tracking_path, "r") as f:
                data = json.load(f)
        except:
            data = None
        return data