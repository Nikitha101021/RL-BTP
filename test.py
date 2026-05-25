import os
import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList


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
            plt.figure()
            plt.plot(self.timesteps, self.rewards)
            plt.xlabel("Timesteps")
            plt.ylabel("Episode Reward")
            plt.title(f"PPO Learning Curve at {self.num_timesteps} Timesteps")
            plt.savefig(f"graphs/reward_graph_{self.num_timesteps}.png")
            plt.close()

            print(f"📊 Graph saved at {self.num_timesteps} timesteps")

        return True


MODEL_PATH = "ppo_carracing_continue"
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

env = gym.make(
    "CarRacing-v3",
    render_mode="human",
    continuous=True
)

if os.path.exists(MODEL_PATH + ".zip"):
    print("✅ Previous model found. Continuing training...")
    model = PPO.load(
        MODEL_PATH,
        env=env,
        device="auto"
    )
else:
    print("🆕 No old model found. Starting new training...")
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

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=CHECKPOINT_DIR,
    name_prefix="ppo_carracing_checkpoint"
)

graph_callback = GraphCheckpointCallback(
    save_every=25000
)

callbacks = CallbackList([
    checkpoint_callback,
    graph_callback
])

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