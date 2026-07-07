import os
import cv2
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList


# =====================================================
# CUSTOM TWO-agent VISUAL WRAPPER
# Adds a blue traffic agent into agentRacing observation
# =====================================================
class DynamicTrafficWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        obs, info = self.env.reset(**kwargs)
        obs = self.add_traffic_agent(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.step_count += 1

        obs = self.add_traffic_agent(obs)

        traffic_x, traffic_y = self.get_traffic_position()

        # Ego agent is usually near bottom-center in agentRacing image
        ego_x = 48
        ego_y = 75

        distance = np.sqrt(
            (traffic_x - ego_x) ** 2 +
            (traffic_y - ego_y) ** 2
        )

        # Collision / close-following penalty
        if distance < 18:
            reward -= 8.0
            info["traffic_warning"] = True
        else:
            info["traffic_warning"] = False

        return obs, reward, terminated, truncated, info

    def get_traffic_position(self):
        # traffic moves visually on the same screen
        x = int(48 + 18 * np.sin(self.step_count / 35))
        y = int(30 + 35 * abs(np.sin(self.step_count / 80)))

        return x, y

    def add_traffic_agent(self, obs):
        img = obs.copy()

        x, y = self.get_traffic_position()

        # Draw blue traffic agent
        cv2.rectangle(
            img,
            (x - 5, y - 9),
            (x + 5, y + 9),
            (0, 0, 255),
            -1
        )

        # Draw small white border
        cv2.rectangle(
            img,
            (x - 5, y - 9),
            (x + 5, y + 9),
            (255, 255, 255),
            1
        )

        return img


# =====================================================
# GRAPH CALLBACK
# =====================================================
class GraphCheckpointCallback(BaseCallback):
    def __init__(self, save_every=25000):
        super().__init__()
        self.save_every = save_every
        self.rewards = []
        self.timesteps = []

        os.makedirs("graphs_two_agent", exist_ok=True)

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
            plt.title(f"Two-agent CNN-PPO Learning Curve at {self.num_timesteps}")
            plt.grid(True)
            plt.savefig(f"graphs_two_agent/two_agent_graph_{self.num_timesteps}.png")
            plt.close()

            print(f"📊 Graph saved at {self.num_timesteps} timesteps")

        return True


# =====================================================
# SETTINGS
# =====================================================
MODEL_PATH = "ppo_custom_two_agent"
CHECKPOINT_DIR = "checkpoints_two_agent_custom"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =====================================================
# ENVIRONMENT
# =====================================================
env = gym.make(
    "agentRacing-v3",
    render_mode="human",
    continuous=True
)

env = DynamicTrafficWrapper(env)


# =====================================================
# LOAD OLD MODEL OR CREATE NEW
# =====================================================
if os.path.exists(MODEL_PATH + ".zip"):
    print("✅ Previous two-agent model found. Continuing training...")

    model = PPO.load(
        MODEL_PATH,
        env=env,
        device="auto"
    )

else:
    print("🆕 Starting new custom two-agent CNN-PPO training...")

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


# =====================================================
# CALLBACKS
# =====================================================
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=CHECKPOINT_DIR,
    name_prefix="two_agent_custom_checkpoint"
)

graph_callback = GraphCheckpointCallback(
    save_every=25000
)

callbacks = CallbackList([
    checkpoint_callback,
    graph_callback
])


# =====================================================
# TRAINING
# =====================================================
try:
    model.learn(
        total_timesteps=300000,
        callback=callbacks,
        reset_num_timesteps=False
    )

except KeyboardInterrupt:
    print("⏸️ Training stopped manually.")

finally:
    model.save(MODEL_PATH)
    env.close()
    print("✅ Model saved as:", MODEL_PATH + ".zip")
