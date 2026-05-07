import sys
from pathlib import Path
import yaml
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

def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "ppo_rover_training.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path=None):
    # Load configuration
    config = load_config(config_path)
    
    # vectorized environment
    env = DummyVecEnv([make_env(render_mode=None)])

    # normalized observations and rewards
    env = VecNormalize(
        env,
        norm_obs=config['vec_normalize']['norm_obs'],
        norm_reward=config['vec_normalize']['norm_reward'],
        clip_obs=config['vec_normalize']['clip_obs'],
        clip_reward=config['vec_normalize']['clip_reward']
    )

    # instantiate the agent
    model = PPO(
        config['ppo_model']['policy'],
        env,
        verbose=config['training']['verbose'],
        learning_rate=config['ppo_model']['learning_rate'],
        n_steps=config['ppo_model']['n_steps'],
        batch_size=config['ppo_model']['batch_size'],
        n_epochs=config['ppo_model']['n_epochs'],
        gamma=config['ppo_model']['gamma'],
        gae_lambda=config['ppo_model']['gae_lambda'],
        clip_range=config['ppo_model']['clip_range'],
        ent_coef=config['ppo_model']['ent_coef'],
        vf_coef=config['ppo_model']['vf_coef'],
        max_grad_norm=config['ppo_model']['max_grad_norm'],
        device=config['training']['device'],
    )

    # checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config['checkpoint']['save_freq'],
        save_path=config['checkpoint']['save_path'],
        name_prefix=config['checkpoint']['name_prefix'],
        save_replay_buffer=config['checkpoint']['save_replay_buffer'],
        save_vecnormalize=config['checkpoint']['save_vecnormalize'],
    )

    # train the agent
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=checkpoint_callback,
        progress_bar=config['training']['progress_bar'],
    )

    # save the final model and vecnormalize stats
    model.save(config['output']['final_model_name'])
    env.save(config['output']['vecnormalize_stats_name'])

    env.close()


if __name__ == "__main__":
    main()