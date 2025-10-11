"""
Training script for PPO agent on Gin Rummy environment with action masking.

Requirements:
pip install stable-baselines3[extra] pettingzoo gymnasium wandb

Usage:
python train_ppo.py --train
"""

import os
import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor
import torch
import wandb
from typing import Union,Optional
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Import your custom components
from gym_wrapper import GinRummySB3Wrapper
from agents.random_agent import RandomAgent

# ============================================
# W&B Configuration - Works on Colab & Local
# ============================================
WANDB_API_KEY = "41fe78a601dfc0909950ad6ec7e6c4fb042d032a"  # Replace with your actual API key
WANDB_PROJECT = "Adversarial-CoEvolution"

# Login to W&B
wandb.login(key=WANDB_API_KEY)


class MaskedGinRummyPolicy(ActorCriticPolicy):
    """
    PPO-compatible masked MLP policy for PettingZoo Gin Rummy.
    Works with dict observation: {'observation': ..., 'action_mask': ...}
    
    FIXES:
    - Properly extracts action masks from observations
    - Overrides evaluate_actions() to apply masks during training
    - Handles batched observations correctly
    """

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # Set default network architecture
        kwargs.setdefault("net_arch", [dict(pi=[256, 256], vf=[256, 256])])
        
        super(MaskedGinRummyPolicy, self).__init__(
            observation_space, action_space, lr_schedule, **kwargs
        )

    def _extract_obs_and_mask(self, obs):
        """
        Extract observation vector and action mask from dict observation.
        Handles both single and batched observations.
        """
        if isinstance(obs, dict):
            # Dict observation from environment
            obs_tensor = obs['observation']
            mask_tensor = obs.get('action_mask', None)
        else:
            # Already a tensor (SB3 might flatten dict obs)
            obs_tensor = obs
            mask_tensor = None
        
        # Ensure tensors are on correct device
        if not isinstance(obs_tensor, th.Tensor):
            obs_tensor = th.as_tensor(obs_tensor, device=self.device).float()
        if mask_tensor is not None and not isinstance(mask_tensor, th.Tensor):
            mask_tensor = th.as_tensor(mask_tensor, device=self.device)
            
        return obs_tensor, mask_tensor

    def _apply_action_mask(self, logits, action_mask):
        """
        Apply action mask to logits by setting invalid actions to -inf.
        """
        if action_mask is None:
            return logits
        
        # Ensure mask is boolean tensor on same device
        mask = action_mask.to(dtype=th.bool, device=logits.device)
        
        # Use -inf for invalid actions (safer than finfo.min)
        logits = th.where(mask, logits, th.tensor(float('-inf'), device=logits.device, dtype=logits.dtype))
        
        return logits
    
    # def extract_features(  # type: ignore[override]
    #     self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None
    #     ) -> Union[th.Tensor, tuple[th.Tensor, th.Tensor]]:
    #     """
    #     Flatten all tensors in the observation dictionary and concatenate them.
        
    #     Args:
    #         obs: Dictionary of tensors
            
    #     Returns:
    #         Flattened tensor of shape (batch_size, total_features)
    #     """
    #     # Collect all tensors and flatten each one
    #     flattened_tensors = []
        
    #     for key in sorted(obs.keys()):  # Sort keys for consistent ordering
    #         tensor = obs[key]
    #         # Flatten all dimensions except the batch dimension (first dimension)
    #         flattened = tensor.flatten(start_dim=1)
    #         flattened_tensors.append(flattened)
        
    #     # Concatenate all flattened tensors along the feature dimension
    #     flatted_obs = torch.cat(flattened_tensors, dim=1)
        
    #     return flatted_obs

    def forward(self, obs, deterministic=False):
        """
        Forward pass for action selection.
        Returns actions, values, and log probabilities.
        """
        # Extract observation and mask
        _, action_mask = self._extract_obs_and_mask(obs)
        
        # Get features and latent vectors
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get action logits and apply mask
        logits = self.action_net(latent_pi)
        logits = self._apply_action_mask(logits, action_mask)
        
        # Create distribution
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        print(logits.shape)
        # Sample actions
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
        
        # Get log probabilities and values
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        """
        CRITICAL: This method is called during PPO training to evaluate actions.
        Must apply action masking here for training to work correctly.
        """
        # Extract observation and mask
        _, action_mask = self._extract_obs_and_mask(obs)
        
        # obs_tensor = torch.flatten(obs)
        # Get features and latent vectors
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get action logits and apply mask
        logits = self.action_net(latent_pi)
        logits = self._apply_action_mask(logits, action_mask)
        
        # Create distribution and evaluate
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)
        
        return values, log_prob, entropy

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Predict action from observation (used for inference).
        Handles single observations from the environment.
        """
        # Convert to tensor if needed
        if isinstance(observation, dict):
            obs_tensor = th.as_tensor(observation["observation"], device=self.device).float().unsqueeze(0)
            mask_tensor = th.as_tensor(observation["action_mask"], device=self.device).unsqueeze(0)
            obs_dict = {'observation': obs_tensor, 'action_mask': mask_tensor}
        else:
            obs_dict = th.as_tensor(observation, device=self.device).float().unsqueeze(0)
        
        with th.no_grad():
            actions, _, _ = self.forward(obs_dict, deterministic=deterministic)
        
        return actions.cpu().numpy(), state


class WandbCallback(BaseCallback):
    """
    Custom callback for logging to Weights & Biases.
    """
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log episode data when episodes end
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                wandb.log({
                    "train/episode_reward": info['r'],
                    "train/episode_length": info['l'],
                    "train/timesteps": self.num_timesteps,
                })
        
        return True
    
    def _on_rollout_end(self) -> bool:
        # Log training metrics after each rollout
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            wandb.log({
                "train/timesteps": self.num_timesteps,
                "train/fps": self.model.logger.name_to_value.get("time/fps", 0),
            })
        return True


class WandbBestModelCallback(BaseCallback):
    """
    Callback to log when a new best model is found during evaluation.
    """
    def __init__(self, verbose=0):
        super(WandbBestModelCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        wandb.log({"eval/new_best_model": 1, "eval/timesteps": self.num_timesteps})
        return True


def make_env():
    """Create and wrap the environment."""
    env = GinRummySB3Wrapper(opponent_policy=RandomAgent, randomize_position=True)
    env = Monitor(env)
    return env


def train_ppo(
    total_timesteps=500_000,
    save_path='./artifacts/models/ppo_gin_rummy',
    log_path='./logs/',
    checkpoint_freq=50_000,
    eval_freq=10_000,
    n_eval_episodes=10,
    randomize_position=True,
    wandb_project=WANDB_PROJECT,
    wandb_run_name=None,
    wandb_config=None,
):
    """
    Train a PPO agent on Gin Rummy with W&B logging.
    
    Args:
        total_timesteps: Total number of training steps
        save_path: Path to save the trained model
        log_path: Path for logs
        checkpoint_freq: Frequency (in timesteps) to save checkpoints
        eval_freq: Frequency to evaluate the model
        n_eval_episodes: Number of episodes for evaluation
        randomize_position: Whether to randomize training agent position each episode
        wandb_run_name: W&B run name (optional)
        wandb_config: Additional config dict for W&B (optional)
    """
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Initialize Weights & Biases
    config = {
        "algorithm": "PPO",
        "policy": "MaskedGinRummyPolicy",
        "total_timesteps": total_timesteps,
        "learning_rate": 3e-4,
        "n_steps": 512,          # Reduced
        "batch_size": 256,        # Increased
        "n_epochs": 5,            # Reduced slightly
        "gamma": 1,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.03,         # Increased for exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "randomize_position": randomize_position,
    }
    
    if wandb_config:
        config.update(wandb_config)
    
    wandb.init(
        project=WANDB_PROJECT,
        name=wandb_run_name,
        config=config,
        sync_tensorboard=False,  # We're not using tensorboard
        monitor_gym=True,
    )
    
    # Create training environment
    print("Creating training environment...")
    print(f"Position randomization: {'ENABLED' if randomize_position else 'DISABLED'}")
    train_env = DummyVecEnv([make_env])
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env])
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_path,
        name_prefix='ppo_gin_rummy_checkpoint'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        callback_on_new_best=WandbBestModelCallback()
    )
    
    wandb_callback = WandbCallback()
    
    # Create PPO model
    print("Initializing PPO model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = PPO(
        MaskedGinRummyPolicy,
        train_env,
        verbose=1,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        tensorboard_log=None,  # Disabled tensorboard
        device=device,
        policy_kwargs={'features_extractor_class': CombinedExtractor}
    )
    print ('____________MODEL CREATED SUCCESSFULLY______________')
    
    # Log model architecture to W&B
    wandb.watch(model.policy, log="all", log_freq=1000)
    
    print(f"Training on device: {model.device}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("Starting training...")
    
    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, wandb_callback],
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(save_path, 'ppo_gin_rummy_final')
        model.save(final_path)
        print(f"\nTraining complete! Model saved to {final_path}")
        
        # Log final model to W&B
        wandb.save(final_path + ".zip")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        interrupted_path = os.path.join(save_path, 'ppo_gin_rummy_interrupted')
        model.save(interrupted_path)
        print(f"Model saved to {interrupted_path}")
        wandb.save(interrupted_path + ".zip")
    
    finally:
        train_env.close()
        eval_env.close()
        wandb.finish()
    
    return model


def test_trained_model(model_path, num_episodes=10, log_to_wandb=False):
    """
    Test a trained model.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to test
        log_to_wandb: Whether to log results to W&B
    """
    print(f"\nTesting model: {model_path}")
    
    if log_to_wandb:
        wandb.init(project=WANDB_PROJECT, name="model_test", job_type="evaluation")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create test environment
    env = make_env()
    
    total_rewards = []
    wins = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        if episode_reward > 0:
            wins += 1
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        if log_to_wandb:
            wandb.log({
                "test/episode_reward": episode_reward,
                "test/episode": episode + 1,
            })
    
    env.close()
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    win_rate = wins / num_episodes * 100
    
    print(f"\n=== Test Results ===")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Max Reward: {max(total_rewards):.2f}")
    print(f"Min Reward: {min(total_rewards):.2f}")
    
    if log_to_wandb:
        wandb.log({
            "test/avg_reward": avg_reward,
            "test/win_rate": win_rate,
            "test/max_reward": max(total_rewards),
            "test/min_reward": min(total_rewards),
        })
        wandb.finish()


def setup_wandb_colab(api_key=None):
    """
    Setup W&B for Google Colab environment.
    
    Args:
        api_key: Your W&B API key (optional, will check environment variable)
    """
    if api_key:
        wandb.login(key=api_key)
    else:
        # Try to get from environment
        api_key = os.environ.get('WANDB_API_KEY')
        if api_key:
            wandb.login(key=api_key)
        else:
            print("⚠️  W&B API key not found!")
            print("Please provide your API key:")
            print("1. Set environment variable: os.environ['WANDB_API_KEY'] = 'your_key_here'")
            print("2. Or call: setup_wandb_colab(api_key='your_key_here')")
            print("3. Or use: wandb.login()")
            raise ValueError("W&B API key required")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO on Gin Rummy')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--test', type=str, help='Test a trained model (provide path)')
    parser.add_argument('--timesteps', type=int, default=500_000, help='Training timesteps')
    parser.add_argument('--save-path', type=str, default='./artifacts/models/ppo_gin_rummy', 
                       help='Path to save models')
    parser.add_argument('--no-randomize', action='store_true',
                       help='Disable position randomization during training')
    parser.add_argument('--wandb-project', type=str, default='gin-rummy-ppo',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='Weights & Biases run name')
    parser.add_argument('--wandb-key', type=str, default=None,
                       help='Weights & Biases API key')
    
    args = parser.parse_args()
    
    # Setup W&B login if key provided
    if args.wandb_key:
        setup_wandb_colab(api_key=args.wandb_key)
    
    if args.train:
        # Train model
        train_ppo(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            randomize_position=not args.no_randomize,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )
        
        # Test the trained model
        final_model = os.path.join(args.save_path, 'ppo_gin_rummy_final')
        if os.path.exists(final_model + '.zip'):
            test_trained_model(final_model, num_episodes=10, log_to_wandb=True)
    
    elif args.test:
        # Test existing model
        test_trained_model(args.test, num_episodes=20, log_to_wandb=True)
    
    else:
        print("Please specify --train or --test <model_path>")
        print("\nExamples:")
        print("  python train_ppo.py --train")
        print("  python train_ppo.py --train --timesteps 1000000")
        print("  python train_ppo.py --train --no-randomize")
        print("  python train_ppo.py --train --wandb-run-name experiment-1")
        print("  python train_ppo.py --test ./artifacts/models/ppo_gin_rummy/ppo_gin_rummy_final")