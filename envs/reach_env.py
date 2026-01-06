"""
Custom PyBullet Reaching Environment

A simple manipulation environment where a robot arm must reach a target position.
Designed for behavior cloning demonstrations.
"""

import numpy as np
import pybullet as p
import pybullet_data
import time
from typing import Tuple, Optional, Dict, Any


class ReachEnv:
    """
    A reaching task environment using PyBullet.
    
    The robot arm must move its end-effector to a randomly placed target.
    
    Observation space (12-dim):
        - Joint positions (6)
        - End-effector position (3)
        - Target position relative to end-effector (3)
        
    Action space (6-dim):
        - Joint velocity commands for 6 joints
        - Continuous, normalized to [-1, 1]
    """
    
    def __init__(
        self,
        render: bool = True,
        max_steps: int = 200,
        target_radius: float = 0.07,  # Slightly larger target for easier success
        action_scale: float = 0.5,  # Increased for more responsive control
    ):
        """
        Initialize the reaching environment.
        
        Args:
            render: Whether to render the simulation visually
            max_steps: Maximum steps per episode
            target_radius: Distance threshold for success
            action_scale: Multiplier for action magnitudes
        """
        self.render_mode = render
        self.max_steps = max_steps
        self.target_radius = target_radius
        self.action_scale = action_scale
        
        # Track connection state
        self.physics_client = None
        self.is_connected = False
        
        # Connect to PyBullet
        self._connect()
        
        # Environment state
        self.robot_id = None
        self.target_id = None
        self.target_pos = None
        self.step_count = 0
        self.joint_indices = None
        self.num_joints = 6
        
        # Workspace bounds for target placement
        # Adjusted to be in front of robot and within reachable area
        self.workspace_low = np.array([0.35, -0.25, 0.15])
        self.workspace_high = np.array([0.55, 0.25, 0.45])
        
        # Debug visualization
        self.debug_line_id = None
        
        # Initialize the environment
        self._setup_scene()
    
    def _connect(self):
        """Connect to PyBullet physics server."""
        if self.is_connected:
            return
            
        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
            # Disable GUI panels for cleaner view
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        self.is_connected = True
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
    def _setup_scene(self):
        """Set up the simulation scene with robot and objects."""
        if not self.is_connected:
            self._connect()
            
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        
        # Disable real-time simulation - we'll step manually
        p.setRealTimeSimulation(0)
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        # Load robot arm (using Kuka IIWA as example)
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        
        # Get joint information
        self.joint_indices = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
        
        self.num_joints = min(len(self.joint_indices), 6)
        self.joint_indices = self.joint_indices[:self.num_joints]
        
        # Set joint damping for smoother motion
        for joint_idx in self.joint_indices:
            p.changeDynamics(self.robot_id, joint_idx, linearDamping=0.1, angularDamping=0.1)
        
        # Create target sphere
        target_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.target_radius,
            rgbaColor=[1, 0, 0, 0.8]
        )
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual,
            basePosition=[0.4, 0, 0.3]
        )
        
        # Create a larger, bright cyan sphere to show end-effector position
        ee_marker_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.035,  # Larger
            rgbaColor=[0, 1, 1, 1.0]  # Cyan - stands out more
        )
        self.ee_marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=ee_marker_visual,
            basePosition=[0, 0, 0]
        )
        
        # Set camera for better view - more top-down to see front/back clearly
        if self.render_mode:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.9,
                cameraYaw=90,      # Looking from the side
                cameraPitch=-45,   # More top-down angle
                cameraTargetPosition=[0.45, 0, 0.25]
            )
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reconnect if needed
        if not self.is_connected:
            self._connect()
            self._setup_scene()
        
        self.step_count = 0
        
        # Reset robot to home position - facing forward toward target area
        home_position = [0, 0.3, 0, -1.5, 0, 0.8, 0]
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(
                self.robot_id,
                joint_idx,
                home_position[i] if i < len(home_position) else 0,
                targetVelocity=0
            )
        
        # Randomize target position
        self.target_pos = np.random.uniform(
            self.workspace_low,
            self.workspace_high
        )
        p.resetBasePositionAndOrientation(
            self.target_id,
            self.target_pos,
            [0, 0, 0, 1]
        )
        
        # Step simulation to settle
        for _ in range(50):
            p.stepSimulation()
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Joint velocity commands (6-dim, normalized to [-1, 1])
            
        Returns:
            observation: Current state
            reward: Step reward
            terminated: Whether episode ended due to success/failure
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to physics server")
            
        self.step_count += 1
        
        # Clip and scale action
        action = np.clip(action, -1, 1) * self.action_scale
        
        # Apply velocity control with higher force for more responsive movement
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(action):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.VELOCITY_CONTROL,
                    targetVelocity=action[i],
                    force=200  # Increased force
                )
        
        # Step simulation multiple times for smoother motion
        for _ in range(8):
            p.stepSimulation()
            if self.render_mode:
                time.sleep(1/480)  # Slow down for visualization
        
        # Get observation and compute reward
        obs = self._get_observation()
        ee_pos = self._get_end_effector_position()
        distance = np.linalg.norm(ee_pos - self.target_pos)
        
        # Update end-effector marker position
        if self.render_mode:
            p.resetBasePositionAndOrientation(
                self.ee_marker_id,
                ee_pos.tolist(),
                [0, 0, 0, 1]
            )
        
        # Draw debug line from end-effector to target
        if self.render_mode:
            # Color: green when close, red when far
            if distance < self.target_radius:
                color = [0, 1, 0]  # Green - success!
            elif distance < self.target_radius * 2:
                color = [1, 1, 0]  # Yellow - getting close
            else:
                color = [1, 0.5, 0]  # Orange - keep going
            
            if self.debug_line_id is not None:
                p.removeUserDebugItem(self.debug_line_id)
            self.debug_line_id = p.addUserDebugLine(
                ee_pos.tolist(),
                self.target_pos.tolist(),
                lineColorRGB=color,
                lineWidth=3
            )
        
        # Reward shaping
        reward = -distance  # Negative distance as reward
        
        # Check termination conditions
        terminated = distance < self.target_radius
        truncated = self.step_count >= self.max_steps
        
        info = {
            "distance": distance,
            "success": terminated,
            "ee_position": ee_pos,
            "target_position": self.target_pos
        }
        
        if terminated:
            reward += 10.0  # Bonus for reaching target
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Joint positions
        joint_positions = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            joint_positions.append(joint_state[0])
        
        # End-effector position
        ee_pos = self._get_end_effector_position()
        
        # Relative target position
        rel_target = self.target_pos - ee_pos
        
        obs = np.concatenate([
            np.array(joint_positions),
            ee_pos,
            rel_target
        ]).astype(np.float32)
        
        return obs
    
    def _get_end_effector_position(self) -> np.ndarray:
        """Get end-effector position."""
        # Use the actual last link of the robot (link 6 for Kuka IIWA)
        # This is the tip, not just the last controlled joint
        num_joints = p.getNumJoints(self.robot_id)
        ee_link = num_joints - 1  # Last link in the chain
        ee_state = p.getLinkState(self.robot_id, ee_link)
        return np.array(ee_state[0])
    
    def _get_jacobian(self) -> np.ndarray:
        """Compute Jacobian matrix for IK velocity control."""
        # Get ALL joint positions (not just controlled ones)
        num_all_joints = p.getNumJoints(self.robot_id)
        joint_positions = []
        for i in range(num_all_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] != p.JOINT_FIXED:  # Skip fixed joints
                joint_positions.append(p.getJointState(self.robot_id, i)[0])
        
        # Zero velocities and accelerations for Jacobian computation
        zero_vec = [0.0] * len(joint_positions)
        
        # Get Jacobian - use the actual last link (true end-effector)
        ee_link = num_all_joints - 1
        jac_linear, jac_angular = p.calculateJacobian(
            self.robot_id,
            ee_link,
            localPosition=[0, 0, 0],
            objPositions=joint_positions,
            objVelocities=zero_vec,
            objAccelerations=zero_vec
        )
        
        # Return only columns for our controlled joints
        jac_linear = np.array(jac_linear)
        return jac_linear[:, :self.num_joints]
    
    def cartesian_to_joint_velocity(self, cartesian_vel: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian velocity to joint velocities using Jacobian pseudoinverse.
        
        Args:
            cartesian_vel: [vx, vy, vz] desired end-effector velocity
            
        Returns:
            joint_vel: Joint velocities to achieve the Cartesian motion
        """
        jacobian = self._get_jacobian()
        
        # Use pseudoinverse for redundant manipulator
        jacobian_pinv = np.linalg.pinv(jacobian)
        
        # Compute joint velocities
        joint_vel = jacobian_pinv @ cartesian_vel
        
        # Clip to reasonable range
        joint_vel = np.clip(joint_vel, -1.0, 1.0)
        
        return joint_vel
    
    def get_action_from_keyboard(self) -> Tuple[np.ndarray, str]:
        """
        Get action from keyboard input for teleoperation.
        Uses CARTESIAN control - keys move the end-effector in X/Y/Z.
        
        Returns:
            action: 6-dim joint velocity array (computed from Cartesian input)
            key_pressed: String indicating special keys pressed
        """
        cartesian_vel = np.zeros(3)  # [vx, vy, vz]
        key_pressed = ""
        
        keys = p.getKeyboardEvents()
        
        # Check for special keys first
        if 27 in keys:  # ESC
            key_pressed = "escape"
            return np.zeros(self.num_joints), key_pressed
            
        if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
            key_pressed = "space"
            return np.zeros(self.num_joints), key_pressed
            
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            key_pressed = "reset"
            return np.zeros(self.num_joints), key_pressed
        
        # CARTESIAN CONTROLS - Increased speed for faster demos
        # W/S: Forward/Backward (X axis)
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            cartesian_vel[0] = 0.8
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            cartesian_vel[0] = -0.8
            
        # A/D: Left/Right (Y axis)
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            cartesian_vel[1] = 0.8
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            cartesian_vel[1] = -0.8
            
        # Q/E: Up/Down (Z axis)
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
            cartesian_vel[2] = 0.8
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
            cartesian_vel[2] = -0.8
        
        # Arrow keys as alternative
        if 65297 in keys and keys[65297] & p.KEY_IS_DOWN:  # UP
            cartesian_vel[0] = 0.8
        if 65298 in keys and keys[65298] & p.KEY_IS_DOWN:  # DOWN
            cartesian_vel[0] = -0.8
        if 65295 in keys and keys[65295] & p.KEY_IS_DOWN:  # LEFT
            cartesian_vel[1] = 0.8
        if 65296 in keys and keys[65296] & p.KEY_IS_DOWN:  # RIGHT
            cartesian_vel[1] = -0.8
        
        # Convert Cartesian velocity to joint velocities
        if np.any(cartesian_vel != 0):
            joint_vel = self.cartesian_to_joint_velocity(cartesian_vel)
        else:
            joint_vel = np.zeros(self.num_joints)
            
        return joint_vel, key_pressed
    
    @property
    def observation_space_dim(self) -> int:
        """Dimension of observation space."""
        return self.num_joints + 3 + 3  # joints + ee_pos + rel_target
    
    @property
    def action_space_dim(self) -> int:
        """Dimension of action space."""
        return self.num_joints
    
    def close(self):
        """Clean up the environment."""
        if self.is_connected:
            try:
                p.disconnect()
            except:
                pass
            self.is_connected = False


# Test the environment
if __name__ == "__main__":
    print("Testing ReachEnv with CARTESIAN keyboard control...")
    print("\nControls (moves the tip directly):")
    print("  W/S    : Move tip forward/backward")
    print("  A/D    : Move tip left/right")
    print("  Q/E    : Move tip up/down")
    print("  Arrows : Alternative for forward/back/left/right")
    print("  SPACE  : Mark success")
    print("  R      : Reset")
    print("  ESC    : Quit")
    print("\nClick on the PyBullet window to give it focus!")
    print("-" * 50)
    
    env = ReachEnv(render=True)
    obs = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print("Running interactive test... Press ESC to quit.")
    
    step = 0
    try:
        while True:
            action, key = env.get_action_from_keyboard()
            
            if key == "escape":
                print("\nESC pressed, exiting...")
                break
            elif key == "reset":
                print("Resetting environment...")
                obs = env.reset()
                step = 0
                continue
            elif key == "space":
                print("SPACE pressed - would mark as success")
                obs = env.reset()
                step = 0
                continue
            
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # Print status every 50 steps
            if step % 50 == 0:
                print(f"Step {step}: distance={info['distance']:.3f}")
            
            if terminated:
                print(f"Target reached! Distance: {info['distance']:.3f}")
                obs = env.reset()
                step = 0
                
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C")
    finally:
        env.close()
    
    print("Test complete!")
