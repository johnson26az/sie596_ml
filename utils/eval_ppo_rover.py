import time
import sys
from pathlib import Path
import yaml

import pygame
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.rover_env import RoverEnv

'''
making this a separate file to avoid reloading the environment and model during training
'''
def make_env(render_mode="human", env_params=None):
    def _init():
        if env_params is None:
            env = RoverEnv(render_mode=render_mode)
        else:
            env = RoverEnv(render_mode=render_mode, **env_params)
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
        config = load_config(config_path)
        model_path, vecnorm_path = resolve_artifact_paths(config_path)
        print("Loading the latest saved training artifacts...")
        print(f"Model: {model_path}")
        print(f"VecNormalize: {vecnorm_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Extract environment parameters from config
    env_cfg = config.get('environment', {})
    env_params = {
        'l': env_cfg.get('l', 25.0),
        'W': env_cfg.get('W', 2.0),
        'N': env_cfg.get('N', 6),
        'R': env_cfg.get('R', 1.0),
        'r': env_cfg.get('r', 0.25),
        'rW': env_cfg.get('rW', 2.0),
        'omega_max': env_cfg.get('omega_max', 2.0),
        'delta_t': env_cfg.get('delta_t', 0.1),
        'max_time_steps': env_cfg.get('max_time', 200.0),
    }
    
    # vectorized environment
    env = DummyVecEnv([make_env(render_mode="human", env_params=env_params)])

    # normalized observations and rewards
    env = VecNormalize.load(vecnorm_path, env)

    # load the trained agent
    model = PPO.load(model_path, env=env)

    # evaluate the agent
    print("\nStarting evaluation... (close the window to exit)")
    obs = env.reset()
    window_open = True
    episode = 0
    episode_rewards = []
    ep_reward = 0.0
    
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
            obs, reward, done, _ = env.step(action)
            # reward may be an array (VecEnv); convert to scalar
            try:
                r = float(np.asarray(reward).reshape(-1)[0])
            except Exception:
                r = float(reward)
            ep_reward += r
            steps += 1
            time.sleep(0.01)  # reduce sleep time to allow more frequent event handling

        if window_open:
            episode_rewards.append(ep_reward)
            print(f"Episode {episode} completed ({steps} steps) reward={ep_reward:.3f}")
            ep_reward = 0.0
            obs = env.reset()

    if len(episode_rewards) > 0:
        print(f"\nEvaluation reward mean: {np.mean(episode_rewards):.3f}, std: {np.std(episode_rewards):.3f}")

    print("\nEvaluation finished!")
    env.close()


if __name__ == "__main__":
    main()

