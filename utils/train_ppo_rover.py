import sys
from pathlib import Path
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.rover_env import RoverEnv

def make_env(render_mode=None):
    def _init():
        env = RoverEnv(render_mode=render_mode)
        return env
    return _init

'''
Loading the configuration from a YAML file allows us to easily tweak hyperparameters and training settings without modifying the code.
'''
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

    # Basic validation of expected sections
    try:
        ppo_cfg = config['ppo_model']
        training_cfg = config['training']
        vec_cfg = config['vec_normalize']
        checkpoint_cfg = config['checkpoint']
        output_cfg = config['output']
    except Exception as e:
        raise RuntimeError(f"Invalid or incomplete config file: {e}") from e

    # Parse learning rate: allow numeric or a string lambda (e.g. "lambda progress: 1e-4 * progress")
    lr_cfg = ppo_cfg.get('learning_rate')
    if isinstance(lr_cfg, str):
        try:
            # safe-ish eval: disable builtins
            learning_rate = eval(lr_cfg, {"__builtins__": {}}, {})
        except Exception:
            try:
                learning_rate = float(lr_cfg)
            except Exception as e:
                raise ValueError(f"Could not parse learning_rate from config: {lr_cfg}") from e
    else:
        learning_rate = lr_cfg

    if not (callable(learning_rate) or isinstance(learning_rate, (float, int))):
        raise ValueError("`learning_rate` must be a float/int or a string lambda expression in the config")

    # Ensure total_timesteps is an int
    try:
        total_timesteps = int(training_cfg.get('total_timesteps'))
    except Exception as e:
        raise ValueError(f"Invalid total_timesteps in config: {training_cfg.get('total_timesteps')}") from e

    # for parallel environments, we need to ensure the total timesteps is divisible by the number of environments
    # Get number of environments and execution mode
    num_envs = training_cfg.get('num_envs', 1)
    num_envs = max(1, min(8, int(num_envs)))  # clamp to 1-8
    env_mode = training_cfg.get('env_mode', 'subproc').lower()

    # Create multiple environments
    env_fns = [make_env(render_mode=None) for _ in range(num_envs)]
    
    # Use SubprocVecEnv for parallel execution or DummyVecEnv for synchronous
    if num_envs > 1 and env_mode == 'subproc':
        print(f"Creating {num_envs} parallel environments (async mode)...")
        env = SubprocVecEnv(env_fns)
    else:
        print(f"Creating {num_envs} environment(s) (sync mode)...")
        env = DummyVecEnv(env_fns)

    # normalized observations and rewards
    env = VecNormalize(
        env,
        norm_obs=vec_cfg.get('norm_obs', True),
        norm_reward=vec_cfg.get('norm_reward', True),
        clip_obs=vec_cfg.get('clip_obs', 10.0),
        clip_reward=vec_cfg.get('clip_reward', 10.0)
    )

    # instantiate the agent
    model = PPO(
        ppo_cfg.get('policy', 'MlpPolicy'),
        env,
        verbose=training_cfg.get('verbose', 1),
        learning_rate=learning_rate,
        n_steps=ppo_cfg.get('n_steps', 2048),
        batch_size=ppo_cfg.get('batch_size', 64),
        n_epochs=ppo_cfg.get('n_epochs', 10),
        gamma=ppo_cfg.get('gamma', 0.99),
        gae_lambda=ppo_cfg.get('gae_lambda', 0.95),
        clip_range=ppo_cfg.get('clip_range', 0.2),
        ent_coef=ppo_cfg.get('ent_coef', 0.0),
        vf_coef=ppo_cfg.get('vf_coef', 0.5),
        max_grad_norm=ppo_cfg.get('max_grad_norm', 0.5),
        device=training_cfg.get('device', 'auto'),
    )

    # checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_cfg.get('save_freq', 50000),
        save_path=checkpoint_cfg.get('save_path', './ppo_rover_checkpoints/'),
        name_prefix=checkpoint_cfg.get('name_prefix', 'ppo_rover_model'),
        save_replay_buffer=checkpoint_cfg.get('save_replay_buffer', False),
        save_vecnormalize=checkpoint_cfg.get('save_vecnormalize', True),
    )

    # train the agent
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=training_cfg.get('progress_bar', True),
        )
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}") from e

    # save the final model and vecnormalize stats
    model.save(config['output']['final_model_name'])
    env.save(config['output']['vecnormalize_stats_name'])

    env.close()


if __name__ == "__main__":
    main()