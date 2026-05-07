import time
import sys
from pathlib import Path
import re

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

def find_latest_checkpoint(checkpoint_dir="ppo_rover_checkpoints"):
    """Find the latest checkpoint by step count"""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory '{checkpoint_dir}' not found")
    
    # Find all model files and extract step counts
    model_files = list(checkpoint_path.glob("ppo_rover_model_*_steps.zip"))
    
    if not model_files:
        raise FileNotFoundError(f"No checkpoint files found in '{checkpoint_dir}'")
    
    # Extract step count from filename and find the latest
    def get_step_count(file_path):
        match = re.search(r'(\d+)_steps\.zip$', file_path.name)
        return int(match.group(1)) if match else 0
    
    latest_model = max(model_files, key=get_step_count)
    step_count = get_step_count(latest_model)
    
    # Corresponding vecnormalize file
    vecnorm_file = checkpoint_path / f"ppo_rover_model_vecnormalize_{step_count}_steps.pkl"
    
    if not vecnorm_file.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_file}")
    
    return str(latest_model), str(vecnorm_file), step_count

def handle_pygame_events():
    """Handle pygame events to keep window responsive"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True

def main(checkpoint_dir="ppo_rover_checkpoints"):
    # Find the latest checkpoint
    try:
        model_path, vecnorm_path, step_count = find_latest_checkpoint(checkpoint_dir)
        print(f"Loading checkpoint from {step_count} training steps...")
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

