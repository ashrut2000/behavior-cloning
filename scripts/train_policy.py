"""
Policy Training Script

Train a neural network policy via behavior cloning (supervised learning).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from pathlib import Path
import json
from datetime import datetime


class DemonstrationDataset(Dataset):
    """PyTorch dataset for demonstration data."""
    
    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class MLPPolicy(nn.Module):
    """
    Simple MLP policy network for behavior cloning.
    
    Architecture: obs -> FC -> ReLU -> FC -> ReLU -> FC -> action
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: list = [64, 64],
        activation: str = "relu"
    ):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Build network layers
        layers = []
        prev_dim = observation_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            prev_dim = hidden_dim
        
        # Output layer (no activation - continuous actions)
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Bound actions to [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy network."""
        return self.network(obs)
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action for a single observation (numpy interface)."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = self.forward(obs_tensor)
            return action.squeeze(0).numpy()


def load_demonstrations(data_path: str) -> tuple:
    """Load demonstration data from file."""
    data = np.load(data_path, allow_pickle=True)
    
    observations = data['observations']
    actions = data['actions']
    
    # Print dataset statistics
    print(f"Loaded {len(observations)} transitions")
    print(f"Observation shape: {observations.shape}")
    print(f"Action shape: {actions.shape}")
    
    # Compute basic statistics
    print(f"\nObservation stats:")
    print(f"  Mean: {observations.mean(axis=0)[:3]}...")
    print(f"  Std:  {observations.std(axis=0)[:3]}...")
    print(f"\nAction stats:")
    print(f"  Mean: {actions.mean(axis=0)}")
    print(f"  Std:  {actions.std(axis=0)}")
    
    return observations, actions


def train_policy(
    data_path: str,
    save_path: str,
    hidden_dims: list = [64, 64],
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    validation_split: float = 0.1,
    seed: int = 42
):
    """
    Train a policy via behavior cloning.
    
    Args:
        data_path: Path to demonstration data
        save_path: Path to save trained model
        hidden_dims: List of hidden layer dimensions
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization strength
        validation_split: Fraction of data for validation
        seed: Random seed for reproducibility
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("=" * 60)
    print("BEHAVIOR CLONING TRAINING")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading demonstrations from: {data_path}")
    observations, actions = load_demonstrations(data_path)
    
    # Split into train/validation
    n_samples = len(observations)
    n_val = int(n_samples * validation_split)
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    train_obs, train_act = observations[train_indices], actions[train_indices]
    val_obs, val_act = observations[val_indices], actions[val_indices]
    
    print(f"\nTrain samples: {len(train_obs)}")
    print(f"Validation samples: {len(val_obs)}")
    
    # Create datasets and dataloaders
    train_dataset = DemonstrationDataset(train_obs, train_act)
    val_dataset = DemonstrationDataset(val_obs, val_act)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Create model
    observation_dim = observations.shape[1]
    action_dim = actions.shape[1]
    
    model = MLPPolicy(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        
        for batch_obs, batch_act in train_loader:
            optimizer.zero_grad()
            
            pred_actions = model(batch_obs)
            loss = criterion(pred_actions, batch_act)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch_obs, batch_act in val_loader:
                pred_actions = model(batch_obs)
                loss = criterion(pred_actions, batch_act)
                epoch_val_loss += loss.item()
        
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1:3d}/{epochs} | "
                  f"Train Loss: {epoch_train_loss:.6f} | "
                  f"Val Loss: {epoch_val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Save checkpoint with metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'observation_dim': observation_dim,
        'action_dim': action_dim,
        'hidden_dims': hidden_dims,
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'seed': seed,
        },
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'training_date': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Model saved to: {save_path}")
    
    # Save training curves
    curves_path = save_path.replace('.pt', '_curves.json')
    with open(curves_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses
        }, f)
    print(f"Training curves saved to: {curves_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train policy via behavior cloning")
    parser.add_argument(
        "--data_path", type=str, default="data/demos.npz",
        help="Path to demonstration data"
    )
    parser.add_argument(
        "--save_path", type=str, default="models/policy.pt",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--hidden_dims", type=int, nargs="+", default=[64, 64],
        help="Hidden layer dimensions (default: 64 64)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay (default: 1e-4)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    train_policy(
        data_path=args.data_path,
        save_path=args.save_path,
        hidden_dims=args.hidden_dims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
