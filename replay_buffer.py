from collections import deque
import random


class ReplayBuffer:
    def __init__(self, memory_capacity, minibatch_size):
        self.replay_memory_capacity = memory_capacity  # capacity of experience replay memory
        self.minibatch_size = minibatch_size  # size of minibatch from experience replay memory for updates
        self.replay_memory = deque(maxlen=self.replay_memory_capacity)

    def add_to_memory(self, experience):
        self.replay_memory.append(experience)

    def sample_from_memory(self):
        return random.sample(self.replay_memory, self.minibatch_size)

    def erase(self):
        self.replay_memory.popleft()
