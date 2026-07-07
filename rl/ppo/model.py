import os
import glob

from stable_baselines3 import PPO


def _latest_checkpoint(checkpoint_dir):
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    if not checkpoint_files:
        return None

    return max(checkpoint_files, key=os.path.getmtime)


def load_or_create_model(model_path, env, hyperparams, checkpoint_dir=None):
    """
    Loads an existing final model first, then the latest checkpoint if present.
    Creates a new PPO model only when no resumable model exists.
    """
    if os.path.exists(model_path + ".zip"):
        print(f"[INFO] Previous final model found: {model_path}.zip")
        print("[INFO] Continuing training from the saved model.")
        model = PPO.load(
            model_path,
            env=env,
            device=hyperparams.get("device", "auto")
        )
    else:
        checkpoint_path = _latest_checkpoint(checkpoint_dir)
        if checkpoint_path:
            print(f"[INFO] Previous checkpoint found: {checkpoint_path}")
            print("[INFO] Continuing training from the latest checkpoint.")
            return PPO.load(
                checkpoint_path,
                env=env,
                device=hyperparams.get("device", "auto")
            )

        print("[INFO] Starting new PPO training...")
        model = PPO(
            policy=hyperparams["policy"],
            env=env,
            learning_rate=hyperparams["learning_rate"],
            n_steps=hyperparams["n_steps"],
            batch_size=hyperparams["batch_size"],
            n_epochs=hyperparams["n_epochs"],
            gamma=hyperparams["gamma"],
            gae_lambda=hyperparams["gae_lambda"],
            ent_coef=hyperparams["ent_coef"],
            clip_range=hyperparams["clip_range"],
            verbose=hyperparams["verbose"],
            device=hyperparams.get("device", "auto")
        )
        
    return model
