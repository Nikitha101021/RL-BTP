import os
import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList


# ===============================
# FIXED SAME TRACK WRAPPER
# ===============================
class FixedTrackWrapper(gym.Wrapper):
    def __init__(self, env, seed=42):
        super().__init__(env)
        self.fixed_seed = seed

    def reset(self, **kwargs):
        return self.env.reset(seed=self.fixed_seed)


# ===============================
# GRAPH CALLBACK
# ===============================
class GraphCheckpointCallback(BaseCallback):
    def __init__(self, save_every=25000):
        super().__init__()
        self.save_every = save_every
        self.rewards = []
        self.timesteps = []
        os.makedirs("graphs", exist_ok=True)

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            reward = self.model.ep_info_buffer[-1]["r"]
            self.rewards.append(reward)
            self.timesteps.append(self.num_timesteps)

        if self.num_timesteps % self.save_every == 0 and len(self.rewards) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(self.timesteps, self.rewards)
            plt.xlabel("Timesteps")
            plt.ylabel("Episode Reward")
            plt.title(f"Multi-Car CNN PPO Learning Curve at {self.num_timesteps}")
            plt.grid(True)
            plt.savefig(f"graphs/multicar_reward_graph_{self.num_timesteps}.png")
            plt.close()

            print(f"📊 Graph saved at {self.num_timesteps} timesteps")

        return True


# ===============================
# SETTINGS
# ===============================
MODEL_PATH = "ppo_multicar_same_track"
CHECKPOINT_DIR = "checkpoints_multicar"
TRACK_SEED = 42
NUM_CARS = 3

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ===============================
# CREATE ONE CAR ENV
# ===============================
def make_env():
    def _init():
        env = gym.make(
            "CarRacing-v3",
            render_mode="human",
            continuous=True
        )

        env = FixedTrackWrapper(env, seed=TRACK_SEED)

        return env

    return _init


# ===============================
# 3 PARALLEL CARS SAME TRACK
# ===============================
env = DummyVecEnv([
    make_env(),
    make_env(),
    make_env()
])

env = VecMonitor(env)
env = VecTransposeImage(env)


# ===============================
# LOAD OLD MODEL OR CREATE NEW
# ===============================
if os.path.exists(MODEL_PATH + ".zip"):
    print("✅ Previous multi-car model found. Continuing training...")

    model = PPO.load(
        MODEL_PATH,
        env=env,
        device="auto"
    )

else:
    print("🆕 Starting new 3-car CNN PPO training...")

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=5e-5,
        n_steps=1024,
        batch_size=64,
        n_epochs=8,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.0005,
        clip_range=0.15,
        verbose=1,
        device="auto"
    )


# ===============================
# CALLBACKS
# ===============================
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=CHECKPOINT_DIR,
    name_prefix="ppo_multicar_checkpoint"
)

graph_callback = GraphCheckpointCallback(
    save_every=25000
)

callbacks = CallbackList([
    checkpoint_callback,
    graph_callback
])


# ===============================
# TRAIN
# ===============================
try:
    model.learn(
        total_timesteps=900000,
        callback=callbacks,
        reset_num_timesteps=False
    )

except KeyboardInterrupt:
    print("⏸️ Training stopped manually.")

finally:
    model.save(MODEL_PATH)
    env.close()
    print("✅ Multi-car model saved as:", MODEL_PATH + ".zip")