from pettingzoo.classic import gin_rummy_v4
from typing import Optional, Tuple, Dict, Any


class GinRummyEnvAPI:
    """
    API wrapper for PettingZoo Gin Rummy environment.
    
    Usage:
        env = GinRummyEnvAPI(render_mode="human")
        env.reset(seed=42)
        
        for step_data in env.play_game():
            agent, obs, reward, done, info = step_data
            # Your custom logic here
        
        env.close()
    """
    
    def __init__(self, render_mode: str = "ansi"):
        """
        Initialize the Gin Rummy environment.
        
        Args:
            render_mode: Rendering mode for the environment (default: "human")
        """
        if render_mode == 'human':
            self.env = gin_rummy_v4.env(render_mode=render_mode, screen_height=600)
        else:
            self.env = gin_rummy_v4.env(render_mode=render_mode)
    
    def reset(self, seed: Optional[int] = None):
        """
        Reset the environment.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.env.reset(seed=seed)
    
    def step(self, action: Optional[int] = None) -> None:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (None if agent is done)
        """
        self.env.step(action)
    
    def get_current_state(self) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Get the current state of the environment.
        
        Returns:
            Tuple of (observation, reward, termination, truncation, info)
        """
        return self.env.last()
    
    def get_valid_action(self, agent: str) -> int:
        """
        Sample a valid action for the given agent.
        
        Args:
            agent: Agent identifier
            
        Returns:
            Valid action index
        """
        observation, _, _, _, _ = self.env.last()
        mask = observation["action_mask"]
        return self.env.action_space(agent).sample(mask)
    
    def agent_iter(self):
        """
        Iterate over agents in the environment.
        
        Yields:
            Agent identifier
        """
        for agent in self.env.agent_iter():
            yield agent
    
    def play_step(self, agent: str, auto_action: bool = True) -> Tuple[str, Any, float, bool, bool, Dict]:
        """
        Play a single step for the given agent.
        
        Args:
            agent: Current agent identifier
            auto_action: Whether to automatically sample a valid action (default: True)
            
        Returns:
            Tuple of (agent, observation, reward, termination, truncation, info)
        """
        observation, reward, termination, truncation, info = self.get_current_state()
        
        if termination or truncation:
            action = None
        else:
            if auto_action:
                action = self.get_valid_action(agent)
            else:
                # Return state without taking action, allowing custom action selection
                return agent, observation, reward, termination, truncation, info
        
        self.step(action)
        
        return agent, observation, reward, termination, truncation, info
    
    def play_game(self, auto_action: bool = True):
        """
        Generator that plays through an entire game.
        
        Args:
            auto_action: Whether to automatically sample valid actions (default: True)
            
        Yields:
            Tuple of (agent, observation, reward, termination, truncation, info)
        """
        for agent in self.agent_iter():
            yield self.play_step(agent, auto_action=auto_action)
    
    def close(self):
        """Clean up environment resources."""
        self.env.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()



    
