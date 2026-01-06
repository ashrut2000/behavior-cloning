# Behavior Cloning for Robotic Manipulation

A complete imitation learning pipeline: collect demonstrations via teleoperation, train a neural network policy, and deploy it in closed-loop control.

![Demo](assets/demo.gif)

## Overview

This project implements **behavior cloning** — a form of imitation learning where a robot learns to perform tasks by mimicking human demonstrations. The pipeline includes:

1. **Custom simulation environment** (PyBullet)
2. **Teleoperation interface** for demonstration collection
3. **Policy training** via supervised learning
4. **Closed-loop evaluation** of learned policies
5. **DAgger implementation** for iterative improvement

## Results

| Metric | Value |
|--------|-------|
| Task | 3D reaching (move end-effector to target) |
| Success Rate | **85%** |
| Training Demos | 30 episodes (~1,200 transitions) |
| Avg Episode Length | 45 steps |
| Training Time | ~2 minutes (CPU) |

## Project Structure

```
behavior_cloning/
├── envs/
│   └── reach_env.py          # PyBullet reaching environment
├── scripts/
│   ├── collect_demos.py      # Teleoperation data collection
│   ├── train_policy.py       # Behavior cloning training
│   ├── evaluate_policy.py    # Policy evaluation
│   ├── dagger.py             # DAgger interactive correction
│   ├── filter_demos.py       # Data filtering utilities
│   └── visualize_data.py     # Dataset analysis
├── models/                   # Saved policy checkpoints
├── data/                     # Demonstration datasets
└── README.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/behavior-cloning.git
cd behavior-cloning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyBullet
- PyTorch
- NumPy

## Quick Start

### 1. Test the Environment
```bash
python envs/reach_env.py
```

### 2. Collect Demonstrations
```bash
python scripts/collect_demos.py --num_episodes 30 --save_path data/demos.npz
```

**Controls (in PyBullet window):**
| Key | Action |
|-----|--------|
| W/S | Move forward/backward |
| A/D | Move left/right |
| Q/E | Move up/down |
| SPACE | Start episode / Continue |
| R | Reset episode |
| ESC | Quit and save |

### 3. Train Policy
```bash
python scripts/train_policy.py --data_path data/demos.npz --save_path models/policy.pt --epochs 200
```

### 4. Evaluate Policy
```bash
python scripts/evaluate_policy.py --model_path models/policy.pt --render
```

## Technical Details

### Environment

- **Simulator:** PyBullet
- **Robot:** Kuka IIWA 7-DOF arm
- **Task:** Reach randomly spawned targets in 3D space
- **Observation space (12-dim):**
  - Joint positions (6)
  - End-effector position (3)
  - Relative target position (3)
- **Action space (6-dim):** Joint velocities
- **Control:** Cartesian velocity commands converted via Jacobian pseudoinverse

### Policy Architecture

```
Input (12) → Linear (64) → ReLU → Linear (64) → ReLU → Linear (6) → Tanh
```

- **Parameters:** 5,382
- **Training:** Supervised learning (MSE loss)
- **Optimizer:** Adam with cosine annealing LR

### Behavior Cloning

The policy is trained via supervised learning on state-action pairs:

```
Loss = MSE(π(s), a_expert)
```

Where `π(s)` is the policy's predicted action and `a_expert` is the demonstrated action.

### DAgger (Dataset Aggregation)

To address **distribution shift** — where the policy encounters states not seen during training — this project includes a DAgger implementation:

1. Run the trained policy
2. Human provides corrections when policy fails
3. Corrections added to dataset
4. Retrain on aggregated data

```bash
python scripts/dagger.py --model_path models/policy.pt --num_episodes 20
```

## Key Learnings

1. **Demonstration quality matters more than quantity** — 30 clean, direct demonstrations outperformed 50 wandering ones

2. **Cartesian control improves teleoperation** — Converting joint control to end-effector control made data collection intuitive

3. **Visual feedback is essential** — Real-time distance display and colored indicators improved demonstration quality

4. **Distribution shift is real** — The 15% failure rate occurred in states underrepresented in training data

#
## References

- [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686) (DAgger paper)
- [PyBullet Quickstart Guide](https://pybullet.org)
- [End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/abs/1504.00702)


