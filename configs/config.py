import os

# =========================
# Base Paths
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
GRAPHS_DIR = os.path.join(BASE_DIR, "evaluation", "graphs")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
VIDEOS_DIR = os.path.join(OUTPUTS_DIR, "videos")
SCREENSHOTS_DIR = os.path.join(OUTPUTS_DIR, "screenshots")

# =========================
# Create Required Folders
# =========================

for folder in [
    CHECKPOINTS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    GRAPHS_DIR,
    LOGS_DIR,
    VIDEOS_DIR,
    SCREENSHOTS_DIR,
]:
    os.makedirs(folder, exist_ok=True)

# =========================
# Environment Configuration
# =========================

ENV_CONFIG = {
    "env_id": "CarRacing-v3",
    "num_agents": 1,
    "continuous": True,
    "render_mode": None,
    "use_multi_agent": False,
}

# =========================
# PPO Training Configuration
# =========================

TRAINING_CONFIG = {
    "model_name_prefix": "ppo_autonomous_driving",
    "track_seed": 42,

    # Training length
    "total_timesteps": 3_000_000,
    "quick_test_timesteps": 1_000,

    # Hardware
    "device": "cuda",

    # Parallel environments
    "num_envs": 8,
    "quick_test_num_envs": 1,

    # PPO hyperparameters
    "learning_rate": 5e-5,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 8,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "ent_coef": 0.0005,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    # Saving/logging
    "save_freq": 50_000,
    "graph_save_freq": 25_000,
    "log_interval": 10,
    "tensorboard_log": LOGS_DIR,
}
