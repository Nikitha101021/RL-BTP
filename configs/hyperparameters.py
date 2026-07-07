# PPO Hyperparameters
PPO_HYPERPARAMS = {
    "policy": "CnnPolicy",
    "learning_rate": 5e-5,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 8,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "ent_coef": 0.0005,
    "clip_range": 0.15,
    "verbose": 1,
    "device": "auto"
}

# Reward Weights
REWARD_WEIGHTS = {
    "track_progress": 1.0,
    "lane_keeping": 0.5,
    "smooth_steering": 0.2,
    "safe_distance": 0.8,
    "collision_risk": -8.0,
    "off_road_penalty": -2.0,
    "excessive_steering": -0.1
}
