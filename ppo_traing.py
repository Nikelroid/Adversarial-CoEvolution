"""
Training script for PPO agent on Gin Rummy environment.

Requirements:
pip install stable-baselines3[extra] pettingzoo gymnasium

Usage:
python train_ppo.py
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from agents.random_agent import RandomAgent
import torch

# Import the wrapper
from gym_wrapper import GinRummySB3Wrapper


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
    randomize_position=True
):
    """
    Train a PPO agent on Gin Rummy.
    
    Args:
        total_timesteps: Total number of training steps
        save_path: Path to save the trained model
        log_path: Path for tensorboard logs
        checkpoint_freq: Frequency (in timesteps) to save checkpoints
        eval_freq: Frequency to evaluate the model
        n_eval_episodes: Number of episodes for evaluation
        randomize_position: Whether to randomize training agent position each episode
    """
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
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
        render=False
    )
    
    # Create PPO model
    print("Initializing PPO model...")
    model = PPO(
        MaskedMLPPolicy,
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=log_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Training on device: {model.device}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("Starting training...")
    
    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(save_path, 'ppo_gin_rummy_final')
        model.save(final_path)
        print(f"\nTraining complete! Model saved to {final_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        interrupted_path = os.path.join(save_path, 'ppo_gin_rummy_interrupted')
        model.save(interrupted_path)
        print(f"Model saved to {interrupted_path}")
    
    finally:
        train_env.close()
        eval_env.close()
    
    return model


def test_trained_model(model_path, num_episodes=10):
    """
    Test a trained model.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to test
    """
    print(f"\nTesting model: {model_path}")
    
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
    
    env.close()
    
    print(f"\n=== Test Results ===")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"Win Rate: {wins / num_episodes * 100:.1f}%")
    print(f"Max Reward: {max(total_rewards):.2f}")
    print(f"Min Reward: {min(total_rewards):.2f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO on Gin Rummy')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--test', type=str, help='Test a trained model (provide path)')
    parser.add_argument('--timesteps', type=int, default=500_000, help='Training timesteps')
    parser.add_argument('--save-path', type=str, default='./models/ppo_gin_rummy', 
                       help='Path to save models')
    parser.add_argument('--no-randomize', action='store_true',
                       help='Disable position randomization during training')
    
    args = parser.parse_args()
    
    if args.train:
        # Train model
        train_ppo(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            randomize_position=not args.no_randomize
        )
        
        # Test the trained model
        final_model = os.path.join(args.save_path, 'ppo_gin_rummy_final')
        if os.path.exists(final_model + '.zip'):
            test_trained_model(final_model, num_episodes=10)
    
    elif args.test:
        # Test existing model
        test_trained_model(args.test, num_episodes=20)
    
    else:
        print("Please specify --train or --test <model_path>")
        print("\nExamples:")
        print("  python train_ppo.py --train")
        print("  python train_ppo.py --train --timesteps 1000000")
        print("  python train_ppo.py --train --no-randomize  # Train without position randomization")
        print("  python train_ppo.py --test ./models/ppo_gin_rummy/ppo_gin_rummy_final")




import torch as th
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy


class MaskedMLPPolicy(ActorCriticPolicy):
    """
    Custom PPO policy with MLP architecture and action masking.
    Works with discrete action spaces.
    """

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(MaskedMLPPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        # You can adjust hidden layers here if needed
        self.net_arch = [dict(pi=[128, 128], vf=[128, 128])]
        self._build(lr_schedule)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, action_mask=None):
        """
        Override distribution creation to apply masking to logits.
        """
        logits = self.action_net(latent_pi)

        if action_mask is not None:
            # Convert mask to bool tensor
            mask = action_mask.bool()
            # Assign -inf to invalid actions
            logits = logits.masked_fill(~mask, -1e9)

        # Return a categorical distribution with masked logits
        return self.action_dist.proba_distribution(logits=logits)

    def forward(self, obs: th.Tensor, action_mask=None, deterministic=False):
        """
        Forward pass for action selection (with optional mask).
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi, action_mask)
        value = self.value_net(latent_vf)

        if deterministic:
            actions = distribution.get_mode()
        else:
            actions = distribution.sample()

        log_prob = distribution.log_prob(actions)
        return actions, value, log_prob

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        """
        Override predict() to handle dict observations (e.g. with masks).
        """
        if isinstance(obs, dict) and "observation" in obs and "action_mask" in obs:
            obs_tensor = th.as_tensor(obs["observation"], device=self.device).float()
            mask_tensor = th.as_tensor(obs["action_mask"], device=self.device).float()
        else:
            obs_tensor = th.as_tensor(obs, device=self.device).float()
            mask_tensor = None

        actions, _, _ = self.forward(obs_tensor, action_mask=mask_tensor, deterministic=deterministic)
        return actions.cpu().numpy(), None