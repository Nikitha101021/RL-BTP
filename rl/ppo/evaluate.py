import os
import glob
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from configs.config import MODELS_DIR

def evaluate():
    print("[INFO] Starting evaluation...")
    
    # Find the most recently modified zip file in the models directory
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.zip"))
    if not model_files:
        print("[FAIL] No models found to evaluate.")
        return
        
    latest_model_path = max(model_files, key=os.path.getctime)
    print(f"[INFO] Evaluating model: {latest_model_path}")
    
    # Create the environment
    env_id = "CarRacing-v3"
    def make_env():
        return gym.make(env_id, render_mode="human", continuous=True)
        
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    
    # Load the model
    try:
        model = PPO.load(latest_model_path, env=env)
    except Exception as e:
        print(f"[FAIL] Could not load model: {e}")
        return
        
    print("[INFO] Model loaded successfully. Running evaluation for 1000 timesteps...")
    
    obs = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        total_reward += rewards[0]
        
        if dones[0]:
            print(f"[INFO] Episode finished. Total reward: {total_reward}")
            obs = env.reset()
            total_reward = 0
            
    print("[SUCCESS] Evaluation completed.")
    env.close()

if __name__ == "__main__":
    evaluate()
