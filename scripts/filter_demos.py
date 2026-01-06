"""
Filter Demonstrations Script

Keep only the shorter, cleaner demonstrations.
"""

import numpy as np
import argparse


def filter_demonstrations(
    input_path: str,
    output_path: str,
    max_length: int = 100,
    min_length: int = 15
):
    """
    Filter demonstrations to keep only episodes within length bounds.

    Args:
        input_path: Path to original demos
        output_path: Path to save filtered demos
        max_length: Maximum episode length to keep
        min_length: Minimum episode length to keep
    """
    print("=" * 60)
    print("FILTERING DEMONSTRATIONS")
    print("=" * 60)

    # Load data
    data = np.load(input_path, allow_pickle=True)
    observations = data['observations']
    actions = data['actions']
    episode_starts = data['episode_starts']
    episode_lengths = data['episode_lengths']

    print(f"\nOriginal data:")
    print(f"  Episodes: {len(episode_lengths)}")
    print(f"  Total transitions: {len(observations)}")
    print(f"  Episode lengths: min={episode_lengths.min()}, max={episode_lengths.max()}, mean={episode_lengths.mean():.1f}")

    # Filter episodes
    new_observations = []
    new_actions = []
    new_episode_starts = []
    new_episode_lengths = []

    kept = 0
    discarded = 0

    for i, (start, length) in enumerate(zip(episode_starts, episode_lengths)):
        if min_length <= length <= max_length:
            new_episode_starts.append(len(new_observations))
            new_observations.extend(observations[start:start+length])
            new_actions.extend(actions[start:start+length])
            new_episode_lengths.append(length)
            kept += 1
        else:
            discarded += 1
            print(f"  Discarding episode {i}: {length} steps")

    if kept == 0:
        print(f"\nNo episodes match criteria (length {min_length}-{max_length})!")
        print("Try adjusting --max_length or --min_length")
        return

    # Save filtered data
    new_observations = np.array(new_observations, dtype=np.float32)
    new_actions = np.array(new_actions, dtype=np.float32)
    new_episode_starts = np.array(new_episode_starts, dtype=np.int64)
    new_episode_lengths = np.array(new_episode_lengths, dtype=np.int64)

    np.savez(
        output_path,
        observations=new_observations,
        actions=new_actions,
        episode_starts=new_episode_starts,
        episode_lengths=new_episode_lengths
    )

    print(f"\nFiltered data:")
    print(f"  Episodes kept: {kept} / {kept + discarded}")
    print(f"  Episodes discarded: {discarded}")
    print(f"  Total transitions: {len(new_observations)}")
    print(f"  New avg length: {new_episode_lengths.mean():.1f}")
    print(f"\nSaved to: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Filter demonstrations by length")
    parser.add_argument(
        "--input", type=str, default="data/demos.npz",
        help="Input demonstration file"
    )
    parser.add_argument(
        "--output", type=str, default="data/demos_filtered.npz",
        help="Output filtered file"
    )
    parser.add_argument(
        "--max_length", type=int, default=80,
        help="Maximum episode length to keep (default: 80)"
    )
    parser.add_argument(
        "--min_length", type=int, default=15,
        help="Minimum episode length to keep (default: 15)"
    )

    args = parser.parse_args()

    filter_demonstrations(
        input_path=args.input,
        output_path=args.output,
        max_length=args.max_length,
        min_length=args.min_length
    )


if __name__ == "__main__":
    main()
