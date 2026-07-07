import os

import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from configs.config import GRAPHS_DIR


class GraphCheckpointCallback(BaseCallback):
    def __init__(self, save_every=25000):
        super().__init__()
        self.save_every = save_every
        self.rewards = []
        self.timesteps = []
        os.makedirs(GRAPHS_DIR, exist_ok=True)

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            reward = self.model.ep_info_buffer[-1]["r"]
            self.rewards.append(reward)
            self.timesteps.append(self.num_timesteps)

        if self.save_every > 0 and self.num_timesteps % self.save_every == 0:
            self._save_reward_graph()

        return True

    def _save_reward_graph(self):
        if not self.rewards:
            return

        os.makedirs(GRAPHS_DIR, exist_ok=True)
        graph_path = os.path.join(GRAPHS_DIR, f"reward_graph_{self.num_timesteps}.png")

        plt.figure(figsize=(10, 5))
        plt.plot(self.timesteps, self.rewards)
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.title(f"Autonomous Driving PPO Learning Curve at {self.num_timesteps}")
        plt.grid(True)
        plt.savefig(graph_path)
        plt.close()

        print(f"[INFO] Reward graph saved: {graph_path}")


def get_callbacks(checkpoint_dir, save_freq, graph_save_freq, model_prefix):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix=model_prefix,
    )

    graph_callback = GraphCheckpointCallback(save_every=graph_save_freq)

    return CallbackList([checkpoint_callback, graph_callback])
