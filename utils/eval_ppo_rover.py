import time
import sys
from pathlib import Path
import yaml

import pygame
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.rover_env import RoverEnv

'''
making this a separate file to avoid reloading the environment and model during training
'''
def make_env(render_mode="human"):
    def _init():
        env = RoverEnv(render_mode=render_mode)
        return env
    return _init

def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "ppo_rover_training.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_artifact_paths(config_path=None):
    """Resolve the final model and VecNormalize files written by training."""
    config = load_config(config_path)

    try:
        output_cfg = config["output"]
    except Exception as e:
        raise RuntimeError(f"Invalid or incomplete config file: {e}") from e

    model_path = Path(output_cfg.get("final_model_name", "ppo_rover_final"))
    vecnorm_path = Path(output_cfg.get("vecnormalize_stats_name", "ppo_rover_vecnormalize"))

    if model_path.suffix != ".zip":
        model_path = model_path.with_suffix(".zip")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not vecnorm_path.exists():
        fallback_vecnorm_path = vecnorm_path.with_suffix(".pkl")
        if fallback_vecnorm_path.exists():
            vecnorm_path = fallback_vecnorm_path
        else:
            raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")

    return str(model_path), str(vecnorm_path)

def handle_pygame_events():
    """Handle pygame events to keep window responsive"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True

def main(config_path=None):
    # Load the latest saved training artifacts
    try:
        model_path, vecnorm_path = resolve_artifact_paths(config_path)
        print("Loading the latest saved training artifacts...")
        print(f"Model: {model_path}")
        print(f"VecNormalize: {vecnorm_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # vectorized environment
    env = DummyVecEnv([make_env(render_mode="human")])

    # normalized observations and rewards
    env = VecNormalize.load(vecnorm_path, env)

    # load the trained agent
    model = PPO.load(model_path, env=env)

    # evaluate the agent
    print("\nStarting evaluation... (close the window to exit)")
    obs = env.reset()
    window_open = True
    episode = 0
    
    for _ in range(50):
        if not window_open:
            break
        episode += 1
        done = False
        steps = 0
        
        while not done:
            # Handle pygame events to keep window responsive
            window_open = handle_pygame_events()
            if not window_open:
                break
            
            action, _states = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            steps += 1
            time.sleep(0.01)  # reduce sleep time to allow more frequent event handling
        
        if window_open:
            obs = env.reset()
            print(f"Episode {episode} completed ({steps} steps)")

    print("\nEvaluation finished!")
    env.close()


if __name__ == "__main__":
    main()

