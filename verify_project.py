import os
import sys

def check(condition, message):
    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")
        sys.exit(1)

def verify():
    print("Starting project verification...")
    
    # 1. Imports
    try:
        import configs
        import rl.ppo.train
        import rl.environments.multi_agent_environment
        check(True, "All necessary modules imported successfully.")
    except Exception as e:
        check(False, f"Failed to import modules: {e}")

    # 2. Configs & Output folders
    try:
        from configs.config import TRAINING_CONFIG, ENV_CONFIG, CHECKPOINTS_DIR, MODELS_DIR, OUTPUTS_DIR, GRAPHS_DIR
        check(os.path.exists(CHECKPOINTS_DIR), "Checkpoints directory exists.")
        check(os.path.exists(MODELS_DIR), "Models directory exists.")
        check(os.path.exists(OUTPUTS_DIR), "Outputs directory exists.")
        check(os.path.exists(GRAPHS_DIR), "Graphs directory exists.")
    except Exception as e:
        check(False, f"Failed config and directory checks: {e}")
        
    # 3. Create Environment
    try:
        import gymnasium as gym
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        check(True, "CarRacing-v3 environment created successfully.")
    except Exception as e:
        check(False, f"Failed to create environment: {e}")
        
    # 4. Create PPO model
    try:
        from stable_baselines3 import PPO
        model = PPO("CnnPolicy", env, n_steps=100, verbose=0, device="cpu")
        check(True, "PPO model created successfully.")
    except Exception as e:
        check(False, f"Failed to create PPO model: {e}")

    # 5. Run 10 test timesteps
    try:
        obs, info = env.reset()
        for _ in range(10):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        check(True, "Ran 10 test timesteps successfully.")
        env.close()
    except Exception as e:
        check(False, f"Failed to run test timesteps: {e}")
        
    print("\n[SUCCESS] Verification completed successfully. All checks passed!")

if __name__ == "__main__":
    verify()
