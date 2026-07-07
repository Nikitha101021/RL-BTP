import argparse
import glob
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

from configs.config import ENV_CONFIG, MODELS_DIR, TRAINING_CONFIG
from rl.environments.weather_env import WeatherFrictionWrapper
from rl.ppo.shield import apply_physics_shield


class FixedTrackWrapper(gym.Wrapper):
    def __init__(self, env, seed=42):
        super().__init__(env)
        self.fixed_seed = seed

    def reset(self, **kwargs):
        kwargs["seed"] = self.fixed_seed
        return self.env.reset(**kwargs)


def configure_carracing_render_size():
    try:
        import gymnasium.envs.box2d.car_racing as car_racing
    except Exception as exc:
        print(f"[WARN] Could not configure CarRacing render size: {exc}")
        return

    car_racing.VIDEO_W = int(TRAINING_CONFIG.get("render_width", 1000))
    car_racing.VIDEO_H = int(TRAINING_CONFIG.get("render_height", 800))


def make_eval_env(render_mode=None, seed=42):
    def _init():
        configure_carracing_render_size()
        env = gym.make(
            ENV_CONFIG.get("env_id", "CarRacing-v3"),
            render_mode=render_mode,
            continuous=ENV_CONFIG.get("continuous", True),
        )
        env = FixedTrackWrapper(env, seed=seed)
        env = WeatherFrictionWrapper(env)
        return env

    return _init


def find_latest_model(model_path=None):
    if model_path:
        return model_path

    model_files = glob.glob(os.path.join(MODELS_DIR, "*.zip"))
    if not model_files:
        return None

    return max(model_files, key=os.path.getmtime)


def get_base_env(vec_env):
    current = vec_env.envs[0]
    while hasattr(current, "env"):
        current = current.env
    return current


def extract_physics_telemetry(vec_env, info):
    base_env = get_base_env(vec_env)
    car = getattr(base_env.unwrapped, "car", None)
    hull = getattr(car, "hull", None)

    speed = 0.0
    angular_velocity = 0.0
    if hull is not None:
        linear_velocity = getattr(hull, "linearVelocity", None)
        if linear_velocity is not None:
            speed = float(linear_velocity.length)
        angular_velocity = float(getattr(hull, "angularVelocity", 0.0))

    return {
        "speed": speed,
        "true_speed": speed,
        "angular_velocity": angular_velocity,
        "is_on_slick_surface": bool(info.get("is_on_slick_surface", False)),
    }


def evaluate(model_path=None, timesteps=1000, render_mode=None):
    print("[INFO] Starting shielded PPO evaluation...")

    latest_model_path = find_latest_model(model_path)
    if not latest_model_path:
        print("[FAIL] No models found to evaluate.")
        return

    print(f"[INFO] Evaluating model: {latest_model_path}")

    env = DummyVecEnv(
        [
            make_eval_env(
                render_mode=render_mode,
                seed=TRAINING_CONFIG.get("track_seed", 42),
            )
        ]
    )
    env = VecMonitor(env)
    env = VecTransposeImage(env)

    try:
        model = PPO.load(latest_model_path, env=env)
    except Exception as exc:
        env.close()
        print(f"[FAIL] Could not load model: {exc}")
        return

    print(f"[INFO] Model loaded. Running shielded evaluation for {timesteps} timesteps.")

    obs = env.reset()
    infos = [{"is_on_slick_surface": False}]
    total_reward = 0.0
    shield_interventions = 0

    for step in range(1, timesteps + 1):
        action, _states = model.predict(obs, deterministic=True)

        telemetry = extract_physics_telemetry(env, infos[0])
        safe_action = apply_physics_shield(action, telemetry)

        if not np.allclose(action, safe_action):
            shield_interventions += 1

        obs, rewards, dones, infos = env.step(safe_action)
        total_reward += float(rewards[0])

        if dones[0]:
            print(
                "[INFO] Episode finished "
                f"at step {step}. Reward: {total_reward:.2f}. "
                f"Shield interventions: {shield_interventions}."
            )
            obs = env.reset()
            infos = [{"is_on_slick_surface": False}]
            total_reward = 0.0

    print(
        "[SUCCESS] Shielded evaluation completed. "
        f"Final rolling reward: {total_reward:.2f}. "
        f"Shield interventions: {shield_interventions}."
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None, help="Optional explicit model zip path.")
    parser.add_argument("--timesteps", type=int, default=1000, help="Evaluation timesteps.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation in a human window. Leave disabled for headless evaluation.",
    )
    args = parser.parse_args()
    evaluate(
        model_path=args.model_path,
        timesteps=args.timesteps,
        render_mode="human" if args.render else ENV_CONFIG.get("render_mode", "rgb_array"),
    )
