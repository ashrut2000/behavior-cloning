"""
Data Visualization Script

Inspect and visualize collected demonstration data.
"""

import numpy as np
import argparse
import json
from pathlib import Path

# Try to import matplotlib (optional dependency)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not found. Install it for visualization plots.")


def load_and_summarize(data_path: str):
    """Load demonstration data and print summary statistics."""
    print("=" * 60)
    print("DEMONSTRATION DATA SUMMARY")
    print("=" * 60)
    
    data = np.load(data_path, allow_pickle=True)
    
    observations = data['observations']
    actions = data['actions']
    episode_starts = data['episode_starts']
    episode_lengths = data['episode_lengths']
    
    # Try to get metadata
    metadata = data.get('metadata', None)
    if metadata is not None:
        metadata = metadata.item() if hasattr(metadata, 'item') else metadata
    
    print(f"\nFile: {data_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total transitions: {len(observations):,}")
    print(f"  Number of episodes: {len(episode_lengths)}")
    print(f"  Observation dim: {observations.shape[1]}")
    print(f"  Action dim: {actions.shape[1]}")
    
    print(f"\nEpisode Length Statistics:")
    print(f"  Min:  {episode_lengths.min()}")
    print(f"  Max:  {episode_lengths.max()}")
    print(f"  Mean: {episode_lengths.mean():.1f}")
    print(f"  Std:  {episode_lengths.std():.1f}")
    
    print(f"\nObservation Statistics:")
    obs_mean = observations.mean(axis=0)
    obs_std = observations.std(axis=0)
    obs_min = observations.min(axis=0)
    obs_max = observations.max(axis=0)
    
    print(f"  {'Dim':<5} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*45}")
    for i in range(min(observations.shape[1], 12)):
        print(f"  {i:<5} {obs_mean[i]:>10.4f} {obs_std[i]:>10.4f} {obs_min[i]:>10.4f} {obs_max[i]:>10.4f}")
    if observations.shape[1] > 12:
        print(f"  ... ({observations.shape[1] - 12} more dimensions)")
    
    print(f"\nAction Statistics:")
    act_mean = actions.mean(axis=0)
    act_std = actions.std(axis=0)
    act_min = actions.min(axis=0)
    act_max = actions.max(axis=0)
    
    print(f"  {'Dim':<5} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*45}")
    for i in range(actions.shape[1]):
        print(f"  {i:<5} {act_mean[i]:>10.4f} {act_std[i]:>10.4f} {act_min[i]:>10.4f} {act_max[i]:>10.4f}")
    
    # Action distribution analysis
    print(f"\nAction Usage Analysis:")
    for i in range(actions.shape[1]):
        nonzero = np.sum(np.abs(actions[:, i]) > 0.01)
        pct = nonzero / len(actions) * 100
        print(f"  Joint {i}: {pct:.1f}% active")
    
    if metadata:
        print(f"\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    return {
        'observations': observations,
        'actions': actions,
        'episode_starts': episode_starts,
        'episode_lengths': episode_lengths,
        'metadata': metadata
    }


def visualize_data(data_path: str, save_plots: bool = False):
    """Create visualization plots for the demonstration data."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Install with: pip install matplotlib")
        return
    
    data = load_and_summarize(data_path)
    observations = data['observations']
    actions = data['actions']
    episode_lengths = data['episode_lengths']
    episode_starts = data['episode_starts']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Demonstration Data Analysis', fontsize=14)
    
    # 1. Episode length distribution
    ax = axes[0, 0]
    ax.hist(episode_lengths, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(episode_lengths.mean(), color='red', linestyle='--', label=f'Mean: {episode_lengths.mean():.1f}')
    ax.set_xlabel('Episode Length (steps)')
    ax.set_ylabel('Count')
    ax.set_title('Episode Length Distribution')
    ax.legend()
    
    # 2. Action distributions (histogram for each joint)
    ax = axes[0, 1]
    for i in range(actions.shape[1]):
        ax.hist(actions[:, i], bins=30, alpha=0.5, label=f'Joint {i}')
    ax.set_xlabel('Action Value')
    ax.set_ylabel('Count')
    ax.set_title('Action Value Distributions')
    ax.legend(fontsize=8)
    
    # 3. Action usage over time (sample episode)
    ax = axes[0, 2]
    # Find a medium-length episode
    median_idx = np.argmin(np.abs(episode_lengths - np.median(episode_lengths)))
    start = episode_starts[median_idx]
    length = episode_lengths[median_idx]
    episode_actions = actions[start:start+length]
    
    for i in range(episode_actions.shape[1]):
        ax.plot(episode_actions[:, i], label=f'Joint {i}', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Action Value')
    ax.set_title(f'Sample Episode Actions (Episode {median_idx})')
    ax.legend(fontsize=8)
    
    # 4. End-effector position (if available in observations)
    ax = axes[1, 0]
    if observations.shape[1] >= 9:
        # Assuming obs format: [joint_pos (6), ee_pos (3), rel_target (3)]
        ee_positions = observations[:, 6:9]
        ax.scatter(ee_positions[:, 0], ee_positions[:, 1], 
                  c=range(len(ee_positions)), cmap='viridis', s=1, alpha=0.3)
        ax.set_xlabel('EE X Position')
        ax.set_ylabel('EE Y Position')
        ax.set_title('End-Effector XY Positions')
        ax.set_aspect('equal')
    else:
        ax.text(0.5, 0.5, 'EE position not available', ha='center', va='center')
        ax.set_title('End-Effector Positions')
    
    # 5. Target relative positions
    ax = axes[1, 1]
    if observations.shape[1] >= 12:
        rel_targets = observations[:, 9:12]
        distances = np.linalg.norm(rel_targets, axis=1)
        ax.hist(distances, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(distances.mean(), color='red', linestyle='--', label=f'Mean: {distances.mean():.3f}')
        ax.set_xlabel('Distance to Target')
        ax.set_ylabel('Count')
        ax.set_title('Distance to Target Distribution')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Target distance not available', ha='center', va='center')
        ax.set_title('Distance to Target')
    
    # 6. Action correlations
    ax = axes[1, 2]
    action_corr = np.corrcoef(actions.T)
    im = ax.imshow(action_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(actions.shape[1]))
    ax.set_yticks(range(actions.shape[1]))
    ax.set_xticklabels([f'J{i}' for i in range(actions.shape[1])])
    ax.set_yticklabels([f'J{i}' for i in range(actions.shape[1])])
    ax.set_title('Action Correlation Matrix')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = data_path.replace('.npz', '_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
    else:
        plt.show()


def replay_episode(data_path: str, episode_idx: int = 0):
    """Replay a specific episode in the environment."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from envs.reach_env import ReachEnv
    import time
    
    data = np.load(data_path, allow_pickle=True)
    actions = data['actions']
    episode_starts = data['episode_starts']
    episode_lengths = data['episode_lengths']
    
    if episode_idx >= len(episode_lengths):
        print(f"Episode {episode_idx} not found. Max episode: {len(episode_lengths) - 1}")
        return
    
    start = episode_starts[episode_idx]
    length = episode_lengths[episode_idx]
    episode_actions = actions[start:start+length]
    
    print(f"Replaying episode {episode_idx} ({length} steps)...")
    
    env = ReachEnv(render=True)
    obs = env.reset(seed=episode_idx)
    
    try:
        for step, action in enumerate(episode_actions):
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.02)  # Slow down for visualization
            
            if step % 20 == 0:
                print(f"  Step {step}: distance={info['distance']:.3f}")
            
            if terminated:
                print(f"  Episode reached target at step {step}!")
                break
        
        if not terminated:
            print(f"  Episode ended without reaching target (distance={info['distance']:.3f})")
        
        # Keep window open briefly
        time.sleep(1)
        
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize demonstration data")
    parser.add_argument(
        "--data_path", type=str, default="data/demos.npz",
        help="Path to demonstration data"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Create visualization plots"
    )
    parser.add_argument(
        "--save_plots", action="store_true",
        help="Save plots to file instead of displaying"
    )
    parser.add_argument(
        "--replay", type=int, default=None,
        help="Replay a specific episode index in the environment"
    )
    
    args = parser.parse_args()
    
    if args.replay is not None:
        replay_episode(args.data_path, args.replay)
    elif args.plot:
        visualize_data(args.data_path, save_plots=args.save_plots)
    else:
        load_and_summarize(args.data_path)


if __name__ == "__main__":
    main()
