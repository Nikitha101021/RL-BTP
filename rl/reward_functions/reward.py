from configs.hyperparameters import REWARD_WEIGHTS

class RewardFunction:
    """
    Modular reward calculation for autonomous driving.
    """
    def __init__(self, weights=None):
        self.weights = weights if weights is not None else REWARD_WEIGHTS

    def calculate_default_reward(self, step_reward):
        """
        Calculates the standard reward provided by the environment (e.g., track progress).
        """
        return step_reward * self.weights["track_progress"]

    def calculate_risk_aware_reward(self, distance_to_obstacle, collision_warning=False):
        """
        Risk-aware reward adjustment.
        """
        reward = 0.0
        if collision_warning or distance_to_obstacle < 18:
            reward += self.weights["collision_risk"]
        else:
            # Small positive reinforcement for maintaining safe distance
            reward += self.weights["safe_distance"] * 0.1 
            
        return reward

    def get_total_reward(self, base_reward, info):
        """
        Aggregates all components of the reward equation.
        Reward = Track Progress + Lane Keeping + Smooth Steering + Safe Distance - Collision Risk - Off-road Penalty - Excessive Steering
        """
        total = self.calculate_default_reward(base_reward)
        
        # In a fully integrated setup, info dict would provide these values:
        if info.get("opponent_warning", False):
            total += self.calculate_risk_aware_reward(0, True)

        # Placeholders for future values passed through info dictionary
        if info.get("off_road", False):
            total += self.weights["off_road_penalty"]
            
        return total
