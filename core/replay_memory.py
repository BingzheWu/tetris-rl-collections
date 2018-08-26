from collections import namedtuple
from itertools import count
import random
import sys
sys.path.append('/home/bingzhe/tetrisRL')
from engine import TetrisEngine
Transition = namedtuple('Transition', 
                    ('state0', 'action', 'state1', 'reward', 'done'))

class BaseReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        raise NotImplementedError
    def sample(self, batch_size):
        raise NotImplementedError
    def __len__(self):
        return len(self.memory)

class ReplayMemory(BaseReplayMemory):

    def __init__(self, capacity):
        super(ReplayMemory, self).__init__(capacity)
    
    def push(self, *args):
        """
        store transitions
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

def test_replay_buffer():
    replay_buffer = ReplayMemory(500)
    width, height = 10, 20
    env = TetrisEngine(width, height)
    obs = env.clear()
    while True:
        action = 0
        obs, reward, done = env.step(action)
        print(obs.shape)
        if done:
            break
if __name__ == '__main__':
    test_replay_buffer()
