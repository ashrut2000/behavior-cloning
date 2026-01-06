"""
Demonstration Collection Script

Collect expert demonstrations via keyboard teleoperation for behavior cloning.
ALL CONTROLS IN PYBULLET WINDOW - no terminal interaction needed.
"""

import numpy as np
import argparse
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import pybullet as p

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.reach_env import ReachEnv


class DemoCollector:
    """Handles demonstration collection with proper state management."""
    
    def __init__(self, env, num_episodes, save_path):
        self.env = env
        self.num_episodes = num_episodes
        self.save_path = save_path
        
        # Data storage
        self.all_observations = []
        self.all_actions = []
        self.all_episode_starts = []
        self.all_episode_lengths = []
        
        # Current episode
        self.episode_observations = []
        self.episode_actions = []
        
        # Counters
        self.successful_episodes = 0
        self.total_attempts = 0
        
        # State: "ready" (looking around) or "moving" (controlling robot)
        self.state = "ready"
        
        # For on-screen text
        self.text_id = None
    
    def show_text(self, message, color=[1, 1, 1]):
        """Display text on the PyBullet screen."""
        if self.text_id is not None:
            try:
                p.removeUserDebugItem(self.text_id)
            except:
                pass
        
        self.text_id = p.addUserDebugText(
            message,
            textPosition=[0.3, 0, 0.6],
            textColorRGB=color,
            textSize=2.0,
            lifeTime=0  # Persistent until removed
        )
    
    def clear_text(self):
        """Remove on-screen text."""
        if self.text_id is not None:
            try:
                p.removeUserDebugItem(self.text_id)
            except:
                pass
            self.text_id = None
    
    def wait_for_space(self, message, color=[1, 1, 0]):
        """Wait until user presses SPACE. Returns False if ESC pressed."""
        self.show_text(message, color)
        
        while True:
            keys = p.getKeyboardEvents()
            
            # Check SPACE
            if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                return True
            
            # Check ESC
            if 27 in keys:
                return False
            
            time.sleep(0.01)
    
    def run_episode(self):
        """Run a single episode. Returns: 'success', 'timeout', 'reset', or 'quit'."""
        
        while True:
            # Check connection
            if not self.env.is_connected:
                return 'quit'
            
            # Get keyboard input
            keys = p.getKeyboardEvents()
            
            # Check ESC
            if 27 in keys:
                return 'quit'
            
            # Check R (reset)
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                return 'reset'
            
            # Get action from keyboard
            action, _ = self.env.get_action_from_keyboard()
            
            # Only step if moving
            if np.any(action != 0):
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Record
                self.episode_actions.append(action.copy())
                self.episode_observations.append(obs.copy())
                
                # Update display
                dist = info['distance']
                steps = len(self.episode_actions)
                
                if dist < 0.07:
                    self.show_text(f"Steps: {steps} | Dist: {dist:.3f} | ALMOST!", [0, 1, 0])
                elif dist < 0.15:
                    self.show_text(f"Steps: {steps} | Dist: {dist:.3f} | Close!", [1, 1, 0])
                else:
                    self.show_text(f"Steps: {steps} | Dist: {dist:.3f}", [1, 1, 1])
                
                # Check success
                if terminated:
                    return 'success'
                
                # Check timeout
                if truncated:
                    return 'timeout'
            
            time.sleep(0.005)
    
    def save_episode(self):
        """Save current episode to storage."""
        if len(self.episode_actions) > 5:
            self.all_episode_starts.append(len(self.all_observations))
            self.all_observations.extend(self.episode_observations[:-1])
            self.all_actions.extend(self.episode_actions)
            self.all_episode_lengths.append(len(self.episode_actions))
            self.successful_episodes += 1
            return True
        return False
    
    def reset_episode(self):
        """Reset for a new episode."""
        self.episode_observations = []
        self.episode_actions = []
        obs = self.env.reset()
        self.episode_observations.append(obs.copy())
        self.total_attempts += 1
    
    def collect(self):
        """Main collection loop."""
        
        # Initial reset
        obs = self.env.reset()
        self.episode_observations.append(obs.copy())
        
        print("\n" + "=" * 60)
        print("ALL CONTROLS IN PYBULLET WINDOW")
        print("=" * 60)
        print("  SPACE  : Start episode / Confirm")
        print("  W/S/A/D/Q/E : Move robot")
        print("  R      : Reset current episode")
        print("  ESC    : Quit and save")
        print("=" * 60)
        
        while self.successful_episodes < self.num_episodes:
            
            # === READY STATE: Look around, press SPACE to start ===
            ep_num = self.successful_episodes + 1
            if not self.wait_for_space(
                f"Ep {ep_num}/{self.num_episodes} - SPACE to start",
                [0, 1, 1]  # Cyan
            ):
                break  # ESC pressed
            
            # === GO! ===
            self.show_text("GO!", [0, 1, 0])
            time.sleep(0.3)
            
            # === MOVING STATE: Control the robot ===
            result = self.run_episode()
            
            if result == 'quit':
                break
            
            elif result == 'success':
                steps = len(self.episode_actions)
                self.save_episode()
                print(f"  ✓ Episode {self.successful_episodes}: SUCCESS in {steps} steps")
                
                # Wait for user to see result and press SPACE
                if self.successful_episodes < self.num_episodes:
                    if not self.wait_for_space(
                        f"SUCCESS! {steps} steps - SPACE for next",
                        [0, 1, 0]  # Green
                    ):
                        break
                
                self.reset_episode()
            
            elif result == 'timeout':
                print(f"  ✗ Episode timed out")
                
                if not self.wait_for_space("TIMEOUT - SPACE to retry", [1, 0, 0]):
                    break
                
                self.reset_episode()
            
            elif result == 'reset':
                print(f"  ↺ Episode reset ({len(self.episode_actions)} steps discarded)")
                self.reset_episode()
        
        # Cleanup
        self.clear_text()
        return self.save_data()
    
    def save_data(self):
        """Save all collected data to file."""
        if len(self.all_observations) == 0:
            print("\nNo demonstrations collected.")
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
                'num_episodes': self.successful_episodes,
                'total_transitions': len(observations),
                'observation_dim': observations.shape[1],
                'action_dim': actions.shape[1],
                'collection_date': datetime.now().isoformat()
            }
        )
        
        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)
        print(f"Successful episodes: {self.successful_episodes}")
        print(f"Total transitions: {len(observations)}")
        print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
        print(f"Saved to: {self.save_path}")
        print("=" * 60)
        
        return True


def collect_demonstrations(num_episodes, save_path, max_steps_per_episode=200):
    """Main entry point for demonstration collection."""
    
    print("=" * 60)
    print("DEMONSTRATION COLLECTION")
    print("=" * 60)
    print("\nControls (all in PyBullet window):")
    print("  W/S    : Forward / Backward")
    print("  A/D    : Left / Right")
    print("  Q/E    : Up / Down")
    print("  SPACE  : Start episode / Continue")
    print("  R      : Reset episode")
    print("  ESC    : Quit and save")
    print("\nGoal: Move CYAN ball into RED ball!")
    print("=" * 60)
    
    # Create environment
    env = ReachEnv(render=True, max_steps=max_steps_per_episode)
    
    try:
        collector = DemoCollector(env, num_episodes, save_path)
        collector.collect()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Collect demonstrations for behavior cloning")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="data/demos.npz")
    parser.add_argument("--max_steps", type=int, default=200)
    
    args = parser.parse_args()
    
    collect_demonstrations(
        num_episodes=args.num_episodes,
        save_path=args.save_path,
        max_steps_per_episode=args.max_steps
    )


if __name__ == "__main__":
    main()
