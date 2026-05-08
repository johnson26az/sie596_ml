import os, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

'''
loading this from the event files written by training allows us to avoid having to save intermediate reward logs during training, and also ensures we are plotting the same data that TensorBoard would show.
'''
def load_scalar_from_eventdir(event_dir, tag=None):
    files = glob.glob(os.path.join(event_dir, '**', 'events*'), recursive=True)
    if not files:
        raise FileNotFoundError(f'No events files found under {event_dir}')
    files.sort(key=os.path.getmtime)
    ea = EventAccumulator(files[-1], size_guidance={EventAccumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    if tag is None:
        for candidate in ['rollout/ep_rew_mean', 'episode_reward', 'episode_reward_mean', 'eval/mean_reward', 'eval/episode_reward']:
            if candidate in tags:
                tag = candidate
                break
        if tag is None:
            raise ValueError('No tag provided and common tags not found. Available: ' + ','.join(tags))
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.int64)
    values = np.array([e.value for e in events], dtype=np.float64)
    return steps, values, tag


def find_event_files(event_dir):
    files = glob.glob(os.path.join(event_dir, '**', 'events*'), recursive=True)
    files.sort(key=os.path.getmtime)
    return files


def list_scalar_tags(event_dir):
    files = find_event_files(event_dir)
    if not files:
        return []
    ea = EventAccumulator(files[-1], size_guidance={EventAccumulator.SCALARS: 0})
    ea.Reload()
    return ea.Tags().get('scalars', [])


'''
Plot the trend of reward over training steps.
'''
def plot_reward_trend(event_dir, out_path='reward_trend.png', tag=None, smooth_window=1):
    steps, vals, tag = load_scalar_from_eventdir(event_dir, tag)
    if smooth_window and smooth_window > 1:
        vals = pd.Series(vals).rolling(smooth_window, min_periods=1, center=True).mean().values
    plt.figure(figsize=(10,5))
    plt.plot(steps, vals, linewidth=1.5)
    plt.xlabel('Training steps')
    plt.ylabel('Mean episode reward')
    plt.title('Mean episode reward vs training steps')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'Saved plot to {out_path} (tag={tag})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a scalar (mean episode reward) from TensorBoard event files')
    parser.add_argument('logdir', nargs='?', default='runs/', help='TensorBoard log directory (default: runs/)')
    parser.add_argument('--tag', '-t', help='Scalar tag to plot (if omitted, will try common names)')
    parser.add_argument('--out', '-o', default='mean_reward.png', help='Output file path (png/jpg/pdf)')
    parser.add_argument('--smooth', '-s', type=int, default=5, help='Smoothing window (moving average); 1=no smoothing')
    parser.add_argument('--list-tags', action='store_true', help='List available scalar tags and exit')
    args = parser.parse_args()

    if args.list_tags:
        tags = list_scalar_tags(args.logdir)
        if tags:
            print('Available scalar tags:')
            for t in tags:
                print(' -', t)
        else:
            print('No event files or no scalar tags found under', args.logdir)
    else:
        plot_reward_trend(args.logdir, out_path=args.out, tag=args.tag, smooth_window=args.smooth)