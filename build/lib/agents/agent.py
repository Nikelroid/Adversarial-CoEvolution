
from abc import abstractmethod

class Agent:

    def __init__(self, env):
        self.env = env

    @abstractmethod    
    def do_action(self):
        pass
    
    def set_player(self, player):
        self.player = player

