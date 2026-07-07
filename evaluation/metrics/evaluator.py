class Evaluator:
    """
    Calculates custom metrics for autonomous driving evaluation.
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.total_reward = 0
        self.lane_keeping_score = 0
        self.collision_count = 0
        self.off_road_steps = 0
        self.total_steps = 0
        self.steering_actions = []
        
    def update(self, reward, info, action):
        self.total_reward += reward
        self.total_steps += 1
        
        # Action is typically [steering, gas, brake]
        self.steering_actions.append(action[0])
        
        if info.get("collision_warning", False):
            self.collision_count += 1
            
        if info.get("off_road", False):
            self.off_road_steps += 1
            
    def get_metrics(self):
        off_road_percentage = (self.off_road_steps / self.total_steps * 100) if self.total_steps > 0 else 0
        
        # Calculate steering smoothness (variance of steering diffs)
        steering_smoothness = 0
        if len(self.steering_actions) > 1:
            diffs = [abs(self.steering_actions[i] - self.steering_actions[i-1]) for i in range(1, len(self.steering_actions))]
            steering_smoothness = sum(diffs) / len(diffs)
            
        return {
            "Average Reward": self.total_reward, # Per episode
            "Collision Count": self.collision_count,
            "Off-road Percentage": off_road_percentage,
            "Steering Smoothness": steering_smoothness,
            "Episode Length": self.total_steps
        }
