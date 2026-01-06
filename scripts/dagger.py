"""
DAgger (Dataset Aggregation) Collection Script

The policy controls the robot, but you can intervene at any time
to provide corrections. These corrections are added to the dataset.

This addresses the "distribution shift" problem in behavior cloning:
the policy makes small errors → enters unseen states → doesn't know what to do.
DAgger teaches the policy how to recover from its own mistakes.
"""

import numpy as np
import argparse
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import pybullet as p
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.reach_env import ReachEnv
from train_policy import MLPPolicy


class DAggerCollector:
    """
    DAgger collection: policy runs, human provides corrections.
    
    Controls:
        - Policy runs automatically
        - W/S/A/D/Q/E: Override with your correction (recorded as expert action)
        - SPACE: Mark episode complete / continue to next
        - R: Reset episode
        - ESC: Quit
    """
    
    def __init__(self, env, policy, num_episodes, save_path, existing_data_path=None):
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.save_path = save_path
        
        # Load existing data if provided
        if existing_data_path and os.path.exists(existing_data_path):
            print(f"Loading existing data from {existing_data_path}")
            data = np.load(existing_data_path, allow_pickle=True)
            self.all_observations = list(data['observations'])
            self.all_actions = list(data['actions'])
            self.all_episode_starts = list(data['episode_starts'])
            self.all_episode_lengths = list(data['episode_lengths'])
            print(f"  Loaded {len(self.all_observations)} existing transitions")
        else:
            self.all_observations = []
            self.all_actions = []
            self.all_episode_starts = []
            self.all_episode_lengths = []
        
        # Current episode
        self.episode_observations = []
        self.episode_actions = []
        
        # Stats
        self.successful_episodes = 0
        self.human_interventions = 0
        self.total_steps = 0
        
        # Display
        self.text_id = None
    
    def show_text(self, message, color=[1, 1, 1]):
        """Display text on screen."""
        if self.text_id is not None:
            try:
                p.removeUserDebugItem(self.text_id)
            except:
                pass
        
        self.text_id = p.addUserDebugText(
            message,
            textPosition=[0.3, 0, 0.6],
            textColorRGB=color,
            textSize=1.5,
            lifeTime=0
        )
    
    def wait_for_space(self, message, color=[1, 1, 0]):
        """Wait for SPACE key. Returns False if ESC pressed."""
        self.show_text(message, color)
        
        while True:
            keys = p.getKeyboardEvents()
            if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                return True
            if 27 in keys:
                return False
            time.sleep(0.01)
    
    def get_human_override(self):
        """Check if human is providing a correction via keyboard."""
        keys = p.getKeyboardEvents()
        
        cartesian_vel = np.zeros(3)
        is_override = False
        
        # Check movement keys
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            cartesian_vel[0] = 0.8
            is_override = True
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            cartesian_vel[0] = -0.8
            is_override = True
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            cartesian_vel[1] = 0.8
            is_override = True
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            cartesian_vel[1] = -0.8
            is_override = True
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
            cartesian_vel[2] = 0.8
            is_override = True
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
            cartesian_vel[2] = -0.8
            is_override = True
        
        # Convert to joint velocities if override
        if is_override:
            action = self.env.cartesian_to_joint_velocity(cartesian_vel)
            return action, True
        
        return None, False
    
    def run_episode(self):
        """
        Run one DAgger episode.
        Policy controls, human can override anytime.
        Returns: 'success', 'timeout', 'reset', or 'quit'
        """
        obs = self.episode_observations[-1]
        
        while True:
            if not self.env.is_connected:
                return 'quit'
            
            keys = p.getKeyboardEvents()
            
            # Check ESC
            if 27 in keys:
                return 'quit'
            
            # Check R (reset)
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                return 'reset'
            
            # Check for human override
            human_action, is_human = self.get_human_override()
            
            if is_human:
                # Human is providing correction
                action = human_action
                self.human_interventions += 1
                action_source = "HUMAN"
            else:
                # Policy controls
                action = self.policy.get_action(obs)
                action_source = "POLICY"
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.total_steps += 1
            
            # Record transition (observation + expert action)
            # Key insight: we record the ACTION that was taken, whether human or policy
            # But in DAgger, we specifically want human corrections
            self.episode_observations.append(next_obs.copy())
            self.episode_actions.append(action.copy())
            
            # Update display
            dist = info['distance']
            steps = len(self.episode_actions)
            
            if is_human:
                self.show_text(f"[HUMAN] Steps: {steps} | Dist: {dist:.3f}", [0, 1, 0])
            else:
                if dist < 0.15:
                    self.show_text(f"[POLICY] Steps: {steps} | Dist: {dist:.3f} | Close!", [1, 1, 0])
                else:
                    self.show_text(f"[POLICY] Steps: {steps} | Dist: {dist:.3f}", [1, 1, 1])
            
            obs = next_obs
            
            # Check termination
            if terminated:
                return 'success'
            if truncated:
                return 'timeout'
            
            time.sleep(0.02)  # Slightly slower for human reaction time
    
    def save_episode(self):
        """Save current episode data."""
        if len(self.episode_actions) > 5:
            self.all_episode_starts.append(len(self.all_observations))
            self.all_observations.extend(self.episode_observations[:-1])
            self.all_actions.extend(self.episode_actions)
            self.all_episode_lengths.append(len(self.episode_actions))
            self.successful_episodes += 1
            return True
        return False
    
    def reset_episode(self):
        """Reset for new episode."""
        self.episode_observations = []
        self.episode_actions = []
        obs = self.env.reset()
        self.episode_observations.append(obs.copy())
    
    def collect(self):
        """Main DAgger collection loop."""
        
        # Initial reset
        obs = self.env.reset()
        self.episode_observations.append(obs.copy())
        
        print("\n" + "=" * 60)
        print("DAGGER COLLECTION")
        print("=" * 60)
        print("  The POLICY controls the robot automatically")
        print("  YOU provide corrections when it makes mistakes")
        print("")
        print("  W/S/A/D/Q/E : Override with your correction")
        print("  SPACE       : Continue to next episode")
        print("  R           : Reset episode")
        print("  ESC         : Quit and save")
        print("=" * 60)
        
        while self.successful_episodes < self.num_episodes:
            
            # Ready state
            ep_num = self.successful_episodes + 1
            if not self.wait_for_space(
                f"DAgger Ep {ep_num}/{self.num_episodes} - SPACE to start",
                [0, 1, 1]
            ):
                break
            
            self.show_text("Policy running... Override if needed!", [1, 1, 0])
            time.sleep(0.5)
            
            # Run episode
            result = self.run_episode()
            
            if result == 'quit':
                break
            
            elif result == 'success':
                steps = len(self.episode_actions)
                self.save_episode()
                print(f"  ✓ Episode {self.successful_episodes}: SUCCESS in {steps} steps")
                
                if self.successful_episodes < self.num_episodes:
                    if not self.wait_for_space(
                        f"SUCCESS! {steps} steps - SPACE for next",
                        [0, 1, 0]
                    ):
                        break
                
                self.reset_episode()
            
            elif result == 'timeout':
                # Still save the episode - corrections during failure are valuable!
                steps = len(self.episode_actions)
                self.save_episode()
                print(f"  ✗ Episode timed out ({steps} steps) - corrections saved")
                
                if not self.wait_for_space("TIMEOUT (data saved) - SPACE to continue", [1, 0.5, 0]):
                    break
                
                self.reset_episode()
            
            elif result == 'reset':
                print(f"  ↺ Episode reset")
                self.reset_episode()
        
        # Cleanup and save
        if self.text_id:
            try:
                p.removeUserDebugItem(self.text_id)
            except:
                pass
        
        return self.save_data()
    
    def save_data(self):
        """Save aggregated dataset."""
        if len(self.all_observations) == 0:
            print("\nNo data collected.")
            return False
        
        # Ensure directory exists
        save_dir = os.path.dirname(self.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Convert to arrays
        observations = np.array(self.all_observations, dtype=np.float32)
        actions = np.array(self.all_actions, dtype=np.float32)
        episode_starts = np.array(self.all_episode_starts, dtype=np.int64)
        episode_lengths = np.array(self.all_episode_lengths, dtype=np.int64)
        
        # Save
        np.savez(
            self.save_path,
            observations=observations,
            actions=actions,
            episode_starts=episode_starts,
            episode_lengths=episode_lengths,
            metadata={
                'num_episodes': len(episode_lengths),
                'total_transitions': len(observations),
                'observation_dim': observations.shape[1],
                'action_dim': actions.shape[1],
                'collection_date': datetime.now().isoformat(),
                'dagger_interventions': self.human_interventions,
                'dagger_total_steps': self.total_steps
            }
        )
        
        intervention_rate = self.human_interventions / max(self.total_steps, 1) * 100
        
        print("\n" + "=" * 60)
        print("DAGGER COLLECTION COMPLETE")
        print("=" * 60)
        print(f"Total episodes: {len(episode_lengths)}")
        print(f"Total transitions: {len(observations)}")
        print(f"Human interventions: {self.human_interventions} ({intervention_rate:.1f}%)")
        print(f"Saved to: {self.save_path}")
        print("=" * 60)
        
        return True


def load_policy(model_path):
    """Load trained policy."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    policy = MLPPolicy(
        observation_dim=checkpoint['observation_dim'],
        action_dim=checkpoint['action_dim'],
        hidden_dims=checkpoint['hidden_dims']
    )
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    return policy


def main():
    parser = argparse.ArgumentParser(description="DAgger data collection")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to trained policy (e.g., models/policy_fast.pt)"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=20,
        help="Number of DAgger episodes to collect"
    )
    parser.add_argument(
        "--existing_data", type=str, default=None,
        help="Path to existing demo data to append to"
    )
    parser.add_argument(
        "--save_path", type=str, default="data/demos_dagger.npz",
        help="Path to save aggregated dataset"
    )
    parser.add_argument(
        "--max_steps", type=int, default=200,
        help="Max steps per episode"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DAGGER - Dataset Aggregation")
    print("=" * 60)
    print(f"Loading policy from: {args.model_path}")
    
    # Load policy
    policy = load_policy(args.model_path)
    print("Policy loaded!")
    
    # Create environment
    env = ReachEnv(render=True, max_steps=args.max_steps)
    
    try:
        collector = DAggerCollector(
            env=env,
            policy=policy,
            num_episodes=args.num_episodes,
            save_path=args.save_path,
            existing_data_path=args.existing_data
        )
        collector.collect()
    finally:
        env.close()


if __name__ == "__main__":
    main()
