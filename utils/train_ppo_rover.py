import sys
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.rover_env import RoverEnv

def make_env(render_mode=None):
    def _init():
        env = RoverEnv(render_mode=render_mode)
        return env
    return _init

def main():
    
    # vectorized environment
    env = DummyVecEnv([make_env(render_mode=None)])

    # normalized observations and rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )

    # instantiate the agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="auto",
    )

    # checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./ppo_rover_checkpoints/",
        name_prefix="ppo_rover_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # train the agent
    total_timesteps = 1_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # save the final model and vecnormalize stats
    model.save("ppo_rover_final")
    env.save("ppo_rover_vecnormalize")

    env.close()


if __name__ == "__main__":
    main()