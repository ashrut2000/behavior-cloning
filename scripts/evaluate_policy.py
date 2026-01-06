"""
Policy Evaluation Script

Deploy trained policy in closed loop and measure performance.
"""

import numpy as np
import torch
import argparse
import sys
from pathlib import Path
import time

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.reach_env import ReachEnv
from train_policy import MLPPolicy


def load_policy(model_path: str) -> MLPPolicy:
    """Load a trained policy from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = MLPPolicy(
        observation_dim=checkpoint['observation_dim'],
        action_dim=checkpoint['action_dim'],
        hidden_dims=checkpoint['hidden_dims']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded policy from: {model_path}")
    print(f"  Observation dim: {checkpoint['observation_dim']}")
    print(f"  Action dim: {checkpoint['action_dim']}")
    print(f"  Hidden dims: {checkpoint['hidden_dims']}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    return model


def evaluate_policy(
    model_path: str,
    num_episodes: int = 20,
    render: bool = True,
    max_steps: int = 200,
    seed: int = 42,
    action_noise: float = 0.0,
    slow_motion: float = 1.0
):
    """
    Evaluate a trained policy in the environment.
    
    Args:
        model_path: Path to trained model checkpoint
        num_episodes: Number of evaluation episodes
        render: Whether to render the simulation
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
        action_noise: Standard deviation of Gaussian noise to add to actions
        slow_motion: Factor to slow down rendering (1.0 = normal, 2.0 = half speed)
    """
    np.random.seed(seed)
    
    print("=" * 60)
    print("POLICY EVALUATION")
    print("=" * 60)
    
    # Load policy
    policy = load_policy(model_path)
    
    # Create environment
    env = ReachEnv(render=render, max_steps=max_steps)
    
    # Evaluation metrics
    successes = []
    episode_lengths = []
    final_distances = []
    total_rewards = []
    
    print(f"\nRunning {num_episodes} evaluation episodes...")
    if action_noise > 0:
        print(f"Adding action noise with std={action_noise}")
    print("-" * 40)
    
    try:
        for episode in range(num_episodes):
            obs = env.reset(seed=seed + episode)
            episode_reward = 0.0
            
            for step in range(max_steps):
                # Get action from policy
                action = policy.get_action(obs)
                
                # Add noise if specified
                if action_noise > 0:
                    action = action + np.random.normal(0, action_noise, action.shape)
                    action = np.clip(action, -1, 1)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                # Slow down for visualization
                if render and slow_motion > 1.0:
                    time.sleep(0.01 * slow_motion)
                
                if terminated or truncated:
                    break
            
            # Record metrics
            success = info['success']
            successes.append(success)
            episode_lengths.append(step + 1)
            final_distances.append(info['distance'])
            total_rewards.append(episode_reward)
            
            status = "✓ SUCCESS" if success else f"✗ FAIL (dist={info['distance']:.3f})"
            print(f"Episode {episode + 1:2d}: {status} | Steps: {step + 1:3d} | Reward: {episode_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")
    
    finally:
        env.close()
    
    # Compute statistics
    if len(successes) > 0:
        success_rate = np.mean(successes) * 100
        avg_length = np.mean(episode_lengths)
        avg_distance = np.mean(final_distances)
        avg_reward = np.mean(total_rewards)
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Episodes completed: {len(successes)}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average episode length: {avg_length:.1f} steps")
        print(f"Average final distance: {avg_distance:.4f}")
        print(f"Average total reward: {avg_reward:.2f}")
        
        # Success episodes stats
        if any(successes):
            success_lengths = [l for l, s in zip(episode_lengths, successes) if s]
            print(f"Avg steps to success: {np.mean(success_lengths):.1f}")
        
        print("=" * 60)
        
        return {
            'success_rate': success_rate,
            'avg_episode_length': avg_length,
            'avg_final_distance': avg_distance,
            'avg_reward': avg_reward,
            'successes': successes,
            'episode_lengths': episode_lengths,
            'final_distances': final_distances
        }
    
    return None


def compare_with_random(
    model_path: str,
    num_episodes: int = 20,
    seed: int = 42
):
    """Compare trained policy with random baseline."""
    print("\n" + "=" * 60)
    print("COMPARISON: Trained Policy vs Random Baseline")
    print("=" * 60)
    
    # Evaluate trained policy
    print("\n--- Trained Policy ---")
    trained_results = evaluate_policy(
        model_path=model_path,
        num_episodes=num_episodes,
        render=False,
        seed=seed
    )
    
    # Evaluate random policy
    print("\n--- Random Policy ---")
    np.random.seed(seed)
    env = ReachEnv(render=False)
    
    random_successes = []
    random_distances = []
    
    for episode in range(num_episodes):
        obs = env.reset(seed=seed + episode)
        
        for step in range(200):
            action = np.random.uniform(-1, 1, env.action_space_dim)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        random_successes.append(info['success'])
        random_distances.append(info['distance'])
        status = "✓" if info['success'] else "✗"
        print(f"Episode {episode + 1:2d}: {status} | dist={info['distance']:.3f}")
    
    env.close()
    
    random_success_rate = np.mean(random_successes) * 100
    random_avg_dist = np.mean(random_distances)
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'Trained':>12} {'Random':>12}")
    print("-" * 50)
    print(f"{'Success Rate (%)':<25} {trained_results['success_rate']:>12.1f} {random_success_rate:>12.1f}")
    print(f"{'Avg Final Distance':<25} {trained_results['avg_final_distance']:>12.4f} {random_avg_dist:>12.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policy")
    parser.add_argument(
        "--model_path", type=str, default="models/policy.pt",
        help="Path to trained model"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=20,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render", action="store_true", default=False,
        help="Render the simulation"
    )
    parser.add_argument(
        "--no_render", dest="render", action="store_false",
        help="Don't render (for faster evaluation)"
    )
    parser.add_argument(
        "--max_steps", type=int, default=200,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--noise", type=float, default=0.0,
        help="Action noise standard deviation"
    )
    parser.add_argument(
        "--slow", type=float, default=1.0,
        help="Slow motion factor for visualization"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare with random baseline"
    )
    
    parser.set_defaults(render=True)
    args = parser.parse_args()
    
    if args.compare:
        compare_with_random(
            model_path=args.model_path,
            num_episodes=args.num_episodes,
            seed=args.seed
        )
    else:
        evaluate_policy(
            model_path=args.model_path,
            num_episodes=args.num_episodes,
            render=args.render,
            max_steps=args.max_steps,
            seed=args.seed,
            action_noise=args.noise,
            slow_motion=args.slow
        )


if __name__ == "__main__":
    main()
