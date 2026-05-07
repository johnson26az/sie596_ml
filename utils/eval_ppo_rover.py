import time
import sys
from pathlib import Path

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

def main():
    # vectorized environment
    env = DummyVecEnv([make_env(render_mode="human")])

    # normalized observations and rewards
    env = VecNormalize.load("ppo_rover_checkpoints/ppo_rover_model_500000_vecnormalize.pkl", env)

    # load the trained agent
    model = PPO.load("ppo_rover_checkpoints/ppo_rover_model_500000.zip", env=env)

    # evaluate the agent
    obs = env.reset()
    for _ in range(50):
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            time.sleep(0.05)  # slow down the rendering
        obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()

