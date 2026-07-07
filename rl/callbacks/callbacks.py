import os
from collections import deque

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from configs.config import GRAPHS_DIR, SCREENSHOTS_DIR, VIDEOS_DIR


class TrainingOutputCallback(BaseCallback):
    def __init__(self, save_every=20000, video_clip_frames=300, video_fps=30):
        super().__init__()
        self.save_every = int(save_every)
        self.video_clip_frames = int(video_clip_frames)
        self.video_fps = int(video_fps)
        self.rewards = []
        self.timesteps = []
        self.frame_buffer = deque(maxlen=max(1, self.video_clip_frames))
        self.next_save_step = None

        os.makedirs(GRAPHS_DIR, exist_ok=True)
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        os.makedirs(VIDEOS_DIR, exist_ok=True)

    def _on_training_start(self):
        if self.save_every <= 0:
            self.next_save_step = None
            return

        completed_intervals = self.num_timesteps // self.save_every
        self.next_save_step = (completed_intervals + 1) * self.save_every

    def _on_step(self):
        self._record_reward()
        self._record_observation_frame()

        if (
            self.save_every > 0
            and self.next_save_step is not None
            and self.num_timesteps >= self.next_save_step
        ):
            self._save_outputs(self.next_save_step)
            while self.num_timesteps >= self.next_save_step:
                self.next_save_step += self.save_every

        return True

    def _record_reward(self):
        if len(self.model.ep_info_buffer) == 0:
            return

        reward = self.model.ep_info_buffer[-1].get("r")
        if reward is None:
            return

        self.rewards.append(reward)
        self.timesteps.append(self.num_timesteps)

    def _record_observation_frame(self):
        frame = self._render_training_frame()
        if frame is not None:
            self.frame_buffer.append(frame)
            return

        observation = self.locals.get("new_obs")
        frame = self._observation_to_frame(observation)
        if frame is not None:
            self.frame_buffer.append(frame)

    def _render_training_frame(self):
        try:
            rendered_frames = self.training_env.env_method("render", indices=0)
        except Exception:
            return None

        if not rendered_frames:
            return None

        return self._normalize_frame(rendered_frames[0])

    def _observation_to_frame(self, observation):
        if observation is None:
            return None

        frame = np.asarray(observation)
        if frame.ndim == 4:
            frame = frame[0]

        return self._normalize_frame(frame)

    def _normalize_frame(self, frame):
        if frame is None:
            return None

        frame = np.asarray(frame)
        if frame.ndim != 3:
            return None

        if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = np.transpose(frame, (1, 2, 0))

        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        elif frame.shape[-1] != 3:
            return None

        if frame.dtype != np.uint8:
            frame_min = float(np.min(frame))
            frame_max = float(np.max(frame))
            if frame_max <= 1.0 and frame_min >= 0.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        return frame

    def _save_outputs(self, step):
        self._save_reward_graph(step)
        self._save_screenshot(step)
        self._save_video_clip(step)

    def _save_reward_graph(self, step):
        if not self.rewards:
            return

        graph_path = os.path.join(GRAPHS_DIR, f"reward_graph_{step}.png")

        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.timesteps, self.rewards)
            plt.xlabel("Timesteps")
            plt.ylabel("Episode Reward")
            plt.title(f"Autonomous Driving PPO Learning Curve at {step}")
            plt.grid(True)
            plt.savefig(graph_path)
            plt.close()
            print(f"[INFO] Reward graph saved: {graph_path}")
        except Exception as exc:
            plt.close("all")
            print(f"[WARN] Reward graph generation failed; training will continue: {exc}")

    def _save_screenshot(self, step):
        if not self.frame_buffer:
            return

        screenshot_path = os.path.join(SCREENSHOTS_DIR, f"training_step_{step}.png")

        try:
            Image.fromarray(self.frame_buffer[-1]).save(screenshot_path)
            print(f"[INFO] Screenshot saved: {screenshot_path}")
        except Exception as exc:
            print(f"[WARN] Screenshot generation failed; training will continue: {exc}")

    def _save_video_clip(self, step):
        if not self.frame_buffer:
            return

        video_path = os.path.join(VIDEOS_DIR, f"training_clip_step_{step}.mp4")
        frames = list(self.frame_buffer)
        height, width = frames[0].shape[:2]

        try:
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.video_fps,
                (width, height),
            )

            if not writer.isOpened():
                print(f"[WARN] Video writer could not open: {video_path}")
                return

            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            writer.release()
            print(f"[INFO] Video clip saved: {video_path}")
        except Exception as exc:
            print(f"[WARN] Video clip generation failed; training will continue: {exc}")


def get_callbacks(
    checkpoint_dir,
    save_freq,
    graph_save_freq,
    model_prefix,
    media_save_freq=None,
    video_clip_frames=300,
    video_fps=30,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix=model_prefix,
    )

    output_callback = TrainingOutputCallback(
        save_every=media_save_freq or graph_save_freq,
        video_clip_frames=video_clip_frames,
        video_fps=video_fps,
    )

    return CallbackList([checkpoint_callback, output_callback])
