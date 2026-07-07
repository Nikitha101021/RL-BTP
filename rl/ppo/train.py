import argparse
import os

import gymnasium as gym
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

from configs.config import CHECKPOINTS_DIR, ENV_CONFIG, MODELS_DIR, TRAINING_CONFIG
from configs.hyperparameters import PPO_HYPERPARAMS
from rl.callbacks.callbacks import get_callbacks
from rl.ppo.model import load_or_create_model

try:
    import rl.environments.car_racing  # noqa: F401
except Exception as exc:
    print(f"[WARN] Custom single-agent environment import failed: {exc}")

try:
    import rl.environments.multi_agent_environment  # noqa: F401
except Exception as exc:
    print(f"[WARN] Multi-agent environment import failed; using CarRacing-v3 fallback: {exc}")


class FixedTrackWrapper(gym.Wrapper):
    def __init__(self, env, seed=42):
        super().__init__(env)
        self.fixed_seed = seed

    def reset(self, **kwargs):
        kwargs["seed"] = self.fixed_seed
        return self.env.reset(**kwargs)


def resolve_device(requested_device):
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return requested_device or "auto"


def get_training_hyperparams():
    hyperparams = PPO_HYPERPARAMS.copy()
    for key in [
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "ent_coef",
        "clip_range",
        "verbose",
        "device",
    ]:
        if key in TRAINING_CONFIG:
            hyperparams[key] = TRAINING_CONFIG[key]

    hyperparams["device"] = resolve_device(hyperparams.get("device", "auto"))
    return hyperparams


def make_env(env_id="CarRacing-v3", seed=42, rank=0):
    def _init():
        env = gym.make(
            env_id,
            render_mode=ENV_CONFIG.get("render_mode"),
            continuous=ENV_CONFIG.get("continuous", True),
        )
        env = FixedTrackWrapper(env, seed=seed + rank)
        return env

    return _init


def train(quick_test=False):
    num_agents = ENV_CONFIG.get("num_agents", 1)
    num_envs = (
        TRAINING_CONFIG.get("quick_test_num_envs", 1)
        if quick_test
        else TRAINING_CONFIG.get("num_envs", num_agents)
    )
    model_name_prefix = TRAINING_CONFIG.get("model_name_prefix", "ppo_autonomous_driving")
    model_name = f"{model_name_prefix}_{num_agents}_agent"
    if quick_test:
        model_name += "_quick_test"

    model_path = os.path.join(MODELS_DIR, model_name)
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, f"{num_agents}_agent")
    os.makedirs(checkpoint_dir, exist_ok=True)

    env_id = ENV_CONFIG.get("env_id", "CarRacing-v3")
    track_seed = TRAINING_CONFIG.get("track_seed", 42)
    env_fns = [make_env(env_id, track_seed, rank) for rank in range(num_envs)]

    env = DummyVecEnv(env_fns)
    env = VecMonitor(env)
    env = VecTransposeImage(env)

    hyperparams = get_training_hyperparams()
    model = load_or_create_model(
        model_path=model_path,
        env=env,
        hyperparams=hyperparams,
        checkpoint_dir=checkpoint_dir,
    )

    callbacks = get_callbacks(
        checkpoint_dir=checkpoint_dir,
        save_freq=TRAINING_CONFIG.get("save_freq", 10000),
        graph_save_freq=TRAINING_CONFIG.get("graph_save_freq", 25000),
        model_prefix=model_name,
    )

    requested_timesteps = (
        TRAINING_CONFIG.get("quick_test_timesteps", 1000)
        if quick_test
        else TRAINING_CONFIG.get("total_timesteps", 300000)
    )
    try:
        print(f"[INFO] Training on {hyperparams['device']} with {num_envs} env(s).")
        print(
            f"[INFO] Current model timesteps: {model.num_timesteps}. "
            f"Adding {requested_timesteps} timestep(s)."
        )
        model.learn(
            total_timesteps=requested_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            log_interval=TRAINING_CONFIG.get("log_interval", 10),
        )
    except KeyboardInterrupt:
        print("[INFO] Training stopped manually.")
    finally:
        model.save(model_path)
        env.close()
        print(f"[SUCCESS] Model saved as: {model_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a short training pass using quick_test_timesteps from config.",
    )
    args = parser.parse_args()
    train(quick_test=args.quick_test)
