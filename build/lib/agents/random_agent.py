from agent import Agent

class RandomAgent(Agent):

    def __init__(self, env):
        self.env = env

    def do_action(self):
        return self.env.get_valid_action(self.player)

