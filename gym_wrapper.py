import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo.classic import gin_rummy_v4
from agents import Agent
import random
import torch


class GinRummySB3Wrapper(gym.Env):
    """
    Wrapper to make PettingZoo Gin Rummy compatible with Stable-Baselines3.
    Converts the multi-agent environment to single-agent by having the opponent play randomly.
    Training agent position is randomized each episode for fair learning.
    """
    
    def __init__(self, opponent_policy, randomize_position=True):
        super().__init__()
        
        self.env = gin_rummy_v4.env(render_mode=None)
        self.opponent_policy: Agent = opponent_policy(self.env)
        self.randomize_position = randomize_position
        
        # Get a sample observation to determine spaces
        self.env.reset()
        agent = self.env.agents[0]
        sample_obs, _, _, _, _ = self.env.last()
        
        # Define observation and action spaces
        obs_shape = sample_obs['observation'].shape
        self.observation_space = spaces.Box(
            low=0, high=1, shape=obs_shape, dtype=np.float32
        )
        
        # Action space is discrete (number of possible actions)
        action_space_size = self.env.action_space(agent).n
        self.action_space = spaces.Discrete(action_space_size)
        
        self.agents = ['player_0', 'player_1']
        
        # These will be set in reset()
        self.training_agent = None
        self.opponent_agent = None
        
        self.env.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            self.env.reset(seed=seed)
        else:
            self.env.reset()
        
        # Randomly assign training agent position each episode
        if self.randomize_position and random.random() < 0.5:
            self.training_agent = 'player_1'
            self.opponent_agent = 'player_0'
            self.opponent_policy.set_player('player_0')
        else:
            self.training_agent = 'player_0'
            self.opponent_agent = 'player_1'
            self.opponent_policy.set_player('player_1')
        
        # Play until it's the training agent's turn
        while True:
            agent = self.env.agent_selection
            if agent == self.training_agent:
                obs, _, _, _, _ = self.env.last()
                return obs['observation'], {}
            else:
                # Opponent plays
                self._opponent_step()
    
    def _opponent_step(self):
        """Have the opponent take an action."""
        obs, reward, termination, truncation, info = self.env.last()
        
        if termination or truncation:
            self.env.step(None)
        else:
            action = self.opponent_policy.do_action()
            self.env.step(action)
    
    def step(self, action):
        """Take a step in the environment with PPO correction for invalid actions."""
        obs, reward, termination, truncation, info = self.env.last()
        if not termination and not truncation:
            mask = obs['action_mask']


            ############
            # Get probability distribution from PPO policy
            obs_array = obs['observation']
            obs_tensor = torch.as_tensor(obs_array).float().unsqueeze(0)

            with torch.no_grad():
                dist = self.agent.model.policy.get_distribution(obs_tensor)
                policy_probs = dist.distribution.probs.cpu().numpy().squeeze()
            ###########
    
            # Apply mask
            masked_probs = policy_probs * mask
            if masked_probs.sum() == 0:
                # fallback if all invalid (rare)
                masked_probs = mask / mask.sum()
            else:
                masked_probs /= masked_probs.sum()
            
            # Sample valid action
            action = np.random.choice(len(masked_probs), p=masked_probs)

        # Apply the (corrected) action
        self.env.step(action)

        # Check if episode ended
        if termination or truncation:
            next_obs, _, _, _, _ = self.env.last()
            return next_obs['observation'], reward, True, False, info

        # Handle opponent turns
        while True:
            agent = self.env.agent_selection

            if agent == self.training_agent:
                obs, reward, termination, truncation, info = self.env.last()
                done = termination or truncation
                return obs['observation'], reward, done, False, info
            else:
                self._opponent_step()
                _, _, termination, truncation, _ = self.env.last()
                if termination or truncation:
                    obs, reward, _, _, info = self.env.last()
                    return obs['observation'], reward, True, False, info
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()