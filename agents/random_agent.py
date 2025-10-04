from .agent import Agent

class RandomAgent(Agent):

    def __init__(self, env):
        self.env = env

    def do_action(self):
        observation, _, _, _, _ = self.env.last()
        mask = observation["action_mask"]
        return self.env.action_space(self.player).sample(mask)

